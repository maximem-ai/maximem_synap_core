"""Worker for Stage 4: Organization and Persistence."""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict

from synap.cloud.agents.ingestion.models import (
    IngestDocumentRequest,
    Stage3Output,
    PipelineStage,
    IngestStatus,
)
from synap.cloud.agents.ingestion.pipeline.stage4_organization import Stage4Organizer
from synap.cloud.shared import edition
from synap.cloud.observability.tracing import start_span, set_current_span_attributes
from .base import BaseIngestionWorker

logger = logging.getLogger(__name__)


class Stage4Worker(BaseIngestionWorker):
    """Worker for Stage 4: Organization and Persistence."""

    @property
    def tracing_span_name(self) -> str:
        return "ingestion.stage4"

    @property
    def input_queue(self) -> str:
        return self.QUEUE_STAGE4

    @property
    def output_queue(self) -> None:
        # Final stage - no output queue
        return None

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.STAGE_4_ORGANIZATION

    def __init__(
        self,
        organizer: Stage4Organizer,
        config_loader=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._organizer = organizer
        # None when no usage tracker is registered (Core): tracking is skipped.
        self._usage_tracker = edition.call("ingestion_usage_tracking")
        # Shared per-instance MACA loader (same _DBConfigLoader instance the
        # stage1/stage23 workers hold; 300s TTL). Needed by the user-profile
        # hook — the organizer's constructor config is only the startup
        # DEFAULT, never per-instance. Optional so existing constructions
        # keep working; the hook lazily builds one (shared module cache).
        self._config_loader = config_loader

    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process Stage 4: Organize and Persist."""
        from uuid import UUID

        request = IngestDocumentRequest.model_validate(payload["request"])
        stage3_output = Stage3Output.model_validate(payload["stage3_output"])
        ingestion_id = UUID(payload["ingestion_id"])
        instance_id = str(request.scope.instance_id)
        logger.info(
            "ingestion_stage4_start ingestion_id=%s request_id=%s extracted_memories=%s extracted_entities=%s",
            ingestion_id,
            request.request_id,
            len(stage3_output.memories),
            len(stage3_output.entities),
        )

        # Organize and persist — sub-spans are emitted inside stage4_organization.py
        set_current_span_attributes({
            "synap.memories_in": len(stage3_output.memories),
            "synap.entities_in": len(stage3_output.entities),
            "synap.relationships_in": len(stage3_output.relationships),
        })
        t0 = time.monotonic()
        # Pass content for raw chunk embedding (Stage 4 splits into windows)
        raw_content = request.content if isinstance(request.content, str) else request.content.decode("utf-8", errors="replace")
        doc_id = request.document_id or str(ingestion_id)
        # Extract the HTTP correlation id stashed on the request by the SDK
        # memory route (sdk_memories._to_ingest_document_request) so Stage 4
        # can stamp memory_actions rows with it for the Requests-page changelog.
        http_correlation_id = None
        if isinstance(request.metadata, dict):
            trace_meta = request.metadata.get("_trace")
            if isinstance(trace_meta, dict):
                http_correlation_id = trace_meta.get("correlation_id")

        stage4_output = await self._organizer.organize_and_persist(
            stage3_output=stage3_output,
            scope=request.scope,
            document_id=doc_id,
            content=raw_content,
            request_id=request.request_id,
            http_correlation_id=http_correlation_id,
            ingestion_id=str(ingestion_id),
        )
        set_current_span_attributes({
            "synap.rdbms_memories": len(stage4_output.rdbms_memory_ids),
            "synap.rdbms_candidates": len(stage4_output.rdbms_candidate_ids),
            "synap.graph_entities": len(stage4_output.graph_entity_ids),
            "synap.graph_relationships": len(stage4_output.graph_relationship_ids),
            "synap.vectors": len(stage4_output.vector_ids),
            "synap.partial_failure": stage4_output.partial_failure,
            "synap.persist_latency_ms": round((time.monotonic() - t0) * 1000, 2),
        })

        logger.info(
            "ingestion_stage4_persisted ingestion_id=%s request_id=%s rdbms_memories=%s rdbms_candidates=%s graph_entities=%s graph_relationships=%s vectors=%s",
            ingestion_id,
            request.request_id,
            len(stage4_output.rdbms_memory_ids),
            len(stage4_output.rdbms_candidate_ids),
            len(stage4_output.graph_entity_ids),
            len(stage4_output.graph_relationship_ids),
            len(stage4_output.vector_ids),
        )

        # Update final status (PARTIAL_SUCCESS if any graph/vector store failed)
        final_status = (
            IngestStatus.PARTIAL_SUCCESS if stage4_output.partial_failure
            else IngestStatus.COMPLETED
        )
        async with start_span("ingestion.stage4.update_status", {
            "synap.ingestion_id": str(ingestion_id),
            "synap.final_status": final_status.value,
        }):
            t0 = time.monotonic()
            await self._status.update(
                ingestion_id,
                final_status,
                # Customer- and client-scope facts live in their own tables, so counting
                # only rdbms_memory_ids reported 0 created and an empty id list for an
                # ingest that stored thousands. Usage tracking below still counts memories only.
                memories_created=(
                    len(stage4_output.rdbms_memory_ids) + len(stage4_output.rdbms_knowledge_ids)
                ),
                candidates_created=len(stage4_output.rdbms_candidate_ids),
                entities_created=len(stage4_output.graph_entity_ids),
                memory_ids=stage4_output.rdbms_memory_ids + stage4_output.rdbms_knowledge_ids,
                completed_at=datetime.now(timezone.utc),
                error_message="; ".join(stage4_output.failure_details) if stage4_output.partial_failure else None,
            )
            set_current_span_attributes({
                "synap.status_update_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })

        # A forget made before this batch was stored may be waiting for it. Runs only for a
        # conversation batch, costs one lookup unless a matching request exists, and never
        # raises (memory_forget/marker.py).
        try:
            from synap.cloud.application.memory_forget import marker as forget_marker
            await forget_marker.after_persist(
                scope=request.scope, metadata=request.metadata,
                memory_ids=stage4_output.rdbms_memory_ids)
        except Exception:  # noqa: BLE001
            logger.debug("forget marker hook raised (non-fatal)", exc_info=True)

        memory_count = len(stage4_output.rdbms_memory_ids)
        if memory_count > 0 and self._usage_tracker is not None:
            async with start_span("ingestion.stage4.track_usage", {
                "synap.ingestion_id": str(ingestion_id),
                "synap.memory_count": memory_count,
                "synap.client_id": str(request.scope.client_id),
                "synap.instance_id": instance_id,
            }):
                try:
                    t0 = time.monotonic()
                    await self._usage_tracker.track_memory_stored(
                        client_id=str(request.scope.client_id),
                        instance_id=instance_id,
                        memory_count=memory_count,
                        user_id=str(request.scope.user_id) if request.scope.user_id else None,
                        conversation_id=request.metadata.get("conversation_id"),
                        endpoint="ingestion.stage4_persist",
                        correlation_id=request.request_id,
                    )

                    # v2 ingestion charge — single per-document credit
                    # debit at the pipeline boundary — see the flush call
                    # below, which is deliberately OUTSIDE this block.
                    set_current_span_attributes({
                        "synap.track_success": True,
                        "synap.track_latency_ms": round((time.monotonic() - t0) * 1000, 2),
                    })
                    logger.info(
                        "cloud_usage_track_memories_stored source=ingestion.stage4_persist memory_count=%s client_id=%s instance_id=%s user_id=%s correlation_id=%s",
                        memory_count,
                        request.scope.client_id,
                        request.scope.instance_id,
                        request.scope.user_id,
                        request.request_id,
                    )
                except Exception as e:
                    set_current_span_attributes({
                        "synap.track_success": False,
                        "synap.error": True,
                        "synap.error_type": type(e).__name__,
                    })
                    logger.warning(
                        "cloud_usage_track_memories_stored_failed source=ingestion.stage4_persist memory_count=%s client_id=%s instance_id=%s correlation_id=%s error=%s",
                        memory_count,
                        request.scope.client_id,
                        request.scope.instance_id,
                        request.request_id,
                        e,
                    )

        if memory_count > 0:
            # Activation analytics emit. First-memory site.
            # The seam self-acquires a control-plane session, resolves the owner,
            # arbitrates first-ness via the activation_milestones ledger, scrubs
            # PII, and fans out to analytics. persist_unit_id = doc_id so
            # multi-doc/multi-chunk persists under one correlation_id are NOT
            # collapsed to a single event. transport/document_type
            # are not reliably in scope on the Redis-decoupled worker, so they
            # are omitted rather than fabricated. Never raises.
            #
            # Cloud-only: Core routes this through an edition hook that does nothing. Wrapped defensively (as
            # the retrieval/short-term sites are) so a seam fault — or its
            # strict-mode re-raise under tests — never fails a persist.
            try:
                await edition.acall(
                    "ingestion_analytics",
                    "memory_stored",
                    client_id=str(request.scope.client_id),
                    instance_id=instance_id,
                    scope_user_id=str(request.scope.user_id) if request.scope.user_id else None,
                    conversation_id=request.metadata.get("conversation_id"),
                    correlation_id=request.request_id,
                    persist_unit_id=doc_id,
                    milestone_key="first_memory_stored",
                    properties={"memory_count": memory_count, "mode": "pipeline"},
                )
            except Exception:  # noqa: BLE001 — analytics never fails a persist
                logger.debug("stage4 memory_stored activation emit raised (non-fatal)", exc_info=True)
        elif stage4_output.partial_failure:
            # Persist boundary reached but nothing landed → store failure, so a
            # true store failure RATE is computable.
            #
            # Logged as well as emitted, so a failing customer write shows in the logs
            # and not only in the product analytics.
            try:
                edition.call(
                    "vector_failure_log",
                    "stage4_persist",
                    "; ".join(stage4_output.failure_details)[:200]
                    if stage4_output.failure_details else "unknown",
                    collection=str(instance_id) if instance_id else None,
                )
            except Exception:  # noqa: BLE001 — a log line never fails a persist
                logger.debug("vector persistence log raised (non-fatal)", exc_info=True)
            try:
                await edition.acall(
                    "ingestion_analytics",
                    "memory_store_failed",
                    client_id=str(request.scope.client_id),
                    instance_id=instance_id,
                    scope_user_id=str(request.scope.user_id) if request.scope.user_id else None,
                    conversation_id=request.metadata.get("conversation_id"),
                    correlation_id=request.request_id,
                    persist_unit_id=doc_id,
                    properties={
                        "stage": "persist",
                        "failure_reason": "; ".join(stage4_output.failure_details)[:200]
                        if stage4_output.failure_details else "unknown",
                        "mode": "pipeline",
                    },
                )
            except Exception:  # noqa: BLE001 — analytics never fails a persist
                logger.debug("stage4 memory_store_failed activation emit raised (non-fatal)", exc_info=True)

        # Invalidate retrieval cache to ensure fresh results
        async with start_span("ingestion.stage4.invalidate_cache", {
            "synap.ingestion_id": str(ingestion_id),
            "synap.instance_id": instance_id,
        }):
            try:
                from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
                from synap.cloud.application.shared.providers import get_cached_providers

                t0 = time.monotonic()
                providers = get_cached_providers()
                retrieval_manager = RetrievalManager(
                    cache_provider=providers["redis"],
                    vector_provider=providers["vector"],
                    graph_provider=providers["graph"],
                    postgres_provider=providers["postgres"],
                    embedding_client=providers.get("embedding"),
                )

                keys_deleted = await retrieval_manager.invalidate_user_cache(
                    instance_id=instance_id,
                    user_id=str(request.scope.user_id) if request.scope.user_id else None,
                    client_id=str(request.scope.client_id) if request.scope.client_id else "",
                    customer_id=str(request.scope.customer_id) if request.scope.customer_id else "",
                )
                set_current_span_attributes({
                    "synap.l1_l2_keys_deleted": keys_deleted,
                })

                logger.info(
                    "ingestion_stage4_cache_invalidated ingestion_id=%s request_id=%s keys_deleted=%s",
                    ingestion_id,
                    request.request_id,
                    keys_deleted,
                )

                # Also invalidate L3 graph cache (forces adjacency rebuild on next query)
                try:
                    from synap.cloud.shared.cache_keys import CacheKeyGenerator
                    user_key = str(request.scope.user_id) if request.scope.user_id else "global"
                    client_id = str(request.scope.client_id)
                    customer_id = str(request.scope.customer_id)
                    l3_data_key = CacheKeyGenerator.l3_graph_data_key(client_id, customer_id, user_key)
                    l3_meta_key = CacheKeyGenerator.l3_graph_meta_key(client_id, customer_id, user_key)
                    await providers["redis"].delete(l3_data_key, l3_meta_key)
                    set_current_span_attributes({
                        "synap.l3_keys_deleted": 2,
                        "synap.invalidate_latency_ms": round((time.monotonic() - t0) * 1000, 2),
                    })
                    logger.info(
                        "ingestion_stage4_l3_cache_invalidated instance_id=%s user_key=%s",
                        instance_id,
                        user_key,
                    )

                    # Mark L2 in-memory cache as dirty (module-level set shared with singleton RM)
                    try:
                        from synap.cloud.application.client_api.retrieval_manager import _l2_dirty_keys
                        l2_key = f"{client_id}:{customer_id}:{user_key}"
                        _l2_dirty_keys.add(l2_key)
                        logger.info(
                            "ingestion_stage4_l2_dirty_flag_set instance_id=%s l2_key=%s",
                            instance_id,
                            l2_key,
                        )
                    except Exception as l2_err:
                        logger.warning("L2 dirty flag set failed (non-fatal): %s", l2_err)

                except Exception as l3_err:
                    set_current_span_attributes({"synap.l3_invalidate_error": True})
                    logger.warning("L3 cache invalidation failed (non-fatal): %s", l3_err)

                # D-007 (plan A4): apply-locally happened above; publish globally so
                # OTHER processes (consolidation worker, future replicas) apply too.
                try:
                    await edition.acall(
                        "consolidation_links",
                        "invalidation",
                        providers["redis"],
                        client_id=str(request.scope.client_id),
                        customer_id=str(request.scope.customer_id),
                        user_id=str(request.scope.user_id) if request.scope.user_id else None,
                        instance_id=instance_id,
                        reason="ingest_stage4",
                    )
                except Exception:  # noqa: BLE001 — publish never fails a persist
                    logger.debug("stage4 invalidation publish failed (non-fatal)", exc_info=True)

            except Exception as e:
                set_current_span_attributes({
                    "synap.cache_invalidate_error": True,
                    "synap.error_type": type(e).__name__,
                })
                # Never fail ingestion due to cache invalidation
                logger.warning(f"Cache invalidation failed (non-fatal): {e} ingestion_id={ingestion_id}")

        logger.info(
            "ingestion_stage4_completed ingestion_id=%s request_id=%s",
            ingestion_id,
            request.request_id,
        )

        # Post-ingest user-profile merge (spec §5.3). Runs after the retrieval
        # cache invalidation so the merge call never delays caches, and now
        # BEFORE the ingestion-charge flush, because the merge became a
        # billable endpoint — see the flush below. It stamps the plain
        # request_id; the derived "{request_id}:profile" id it used to carry
        # was a workaround for a charge query that had no endpoint filter, and
        # that query now has an explicit one. Guarded like the analytics
        # blocks: never fails a persist.
        # No-op unless the instance's MACA sets user_profile.enabled.
        try:
            from synap.cloud.application.ingestion.startup import _get_llm_client
            from synap.cloud.memory_config.loader import get_config_loader
            from synap.cloud.application.profile.store import ProfileStore
            from synap.cloud.application.profile.updater import (
                on_ingestion_complete as _profile_on_ingestion_complete,
            )

            # Fresh _DBConfigLoader instances share the module-level 300s
            # TTL cache, so a lazily-built loader is equivalent to the one
            # startup wires in.
            _profile_config_loader = self._config_loader or get_config_loader()
            _profile_result = await _profile_on_ingestion_complete(
                request=request,
                stage4_output=stage4_output,
                ingestion_id=ingestion_id,
                config_loader=_profile_config_loader,
                llm_client=_get_llm_client("ingestion_agent_light"),
                store=ProfileStore(),
            )
            logger.info(
                "profile_update result=%s ingestion_id=%s request_id=%s instance_id=%s",
                _profile_result,
                ingestion_id,
                request.request_id,
                instance_id,
            )
        except Exception:  # noqa: BLE001 — profile merge never fails a persist
            logger.warning(
                "profile_update result=error ingestion_id=%s request_id=%s instance_id=%s",
                ingestion_id,
                request.request_id,
                instance_id,
                exc_info=True,
            )

        # v2 ingestion charge — one credit debit per document, and the LAST
        # thing Stage 4 does, so every billable LLM call for this request has
        # already written its usage_events rows (they are awaited, not
        # fire-and-forget, so they are committed by the time we read them).
        #
        # ⚠ TWO placement rules, both of which cost real money when broken:
        #
        # 1. It must NOT sit inside `if memory_count > 0:`. That block guards
        #    `track_memory_stored`, an analytics event whose whole content is
        #    "N memories were stored" — gating THAT on N > 0 is obviously
        #    right. The charge was added inside it later and silently
        #    inherited a condition that was never a pricing decision. Before
        #    this moved, a large share of documents ran the full
        #    four-call Stage 3 extraction, the model answered "nothing here
        #    worth keeping", and nobody was charged for work we had paid for.
        #
        # 2. It must run AFTER the profile merge. The merge is a billable
        #    endpoint now, so charging before it would silently drop its
        #    tokens from every bill — the same failure as (1), one step later.
        try:
            await edition.acall(
                "billing_charge",
                client_id=str(request.scope.client_id),
                correlation_id=request.request_id,
                instance_id=instance_id,
                user_id=(
                    str(request.scope.user_id)
                    if request.scope.user_id
                    else None
                ),
                conversation_id=request.metadata.get("conversation_id"),
            )
        except Exception:
            logger.warning(
                "ingestion_charge_flush_failed correlation_id=%s memory_count=%s",
                request.request_id, memory_count, exc_info=True,
            )

        # NOTE: Post-bootstrap deduplication
        # For batch ingestion, deduplication should be triggered after the entire
        # batch completes, not per-document. This is coordinated at the SDK level
        # by calling ConsolidationManager.check_and_eliminate_duplicates() with
        # the appropriate scope after batch ingestion completes.
        # See: IngestionManager.trigger_post_batch_consolidation() for the hook.

        # No next stage
        return None
