"""Stage 4: Organization & Persistence.

Stage 4 is responsible for:
- Organizing extracted memories, entities, and relationships
- Persisting to RDBMS (memories/candidates tables)
- Persisting to Graph store (entities/relationships)
- Persisting to Vector store (embeddings)
- Tracking S3 original path reference
"""

import asyncio
import os
from typing import TYPE_CHECKING, Optional, List, Dict
import logging
import time

if TYPE_CHECKING:
    from synap.cloud.agents.ingestion.resolution.entity_resolver import (
        EntityResolutionService,
    )
    from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
        KnowledgeUpdateDetector,
    )
    from synap.cloud.agents.ingestion.pipeline.stage5_hqe import (
        Stage5HQEGenerator,
    )

from synap.cloud.agents.ingestion.models import (
    Stage3Output,
    Stage4Output,
    MemoryDestination,
    ExtractedMemory,
    Scope,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from synap.cloud.agents.ingestion.storage import StorageCoordinator
from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
from synap.cloud.agents.ingestion.services import anchor_resolver
from synap.cloud.shared.logging_utils import log_data_sample, truncate_string
from synap.cloud.shared import edition
from synap.cloud.observability.tracing import start_span, set_current_span_attributes
from synap.cloud.shared.hook_signals import inc_counter

logger = logging.getLogger(__name__)

# ── A failed vector write must not be a silent loss ──────────────────────────
#
# When the Chroma container is killed mid-ingest and comes back seconds later,
# every vector write in that window raises ``[Errno 111] Connection refused``.
# Stage 4 did what it
# was built to do: it logged each leg at ERROR, set ``partial_failure=True``,
# and the status tracker wrote PARTIAL_SUCCESS with the reason to Redis.
#
# And then nothing happened. Nothing retried the write, nothing alerted, and
# nothing repaired it. Memories stayed in Postgres with no vector,
# which means no query could reach them, and they were only found later by
# counting one store against the other. The detection was never the problem.
#
# So this adds the two halves that were missing. Retry first, because the store
# is usually back within seconds and the writes are upserts keyed by memory id, so
# trying again is both safe and usually enough. Then a counter on the /metrics
# endpoint Prometheus already scrapes, because the failure needs somewhere to
# go that someone actually reads: the repo's older alarm marker is a log line
# no rule matches, and writing another one of those would repeat the mistake.
#
# The third half, repair, is scripts/vector_parity.py, which compares the two
# stores and can re-embed whatever fell through.

_TRANSIENT_VECTOR_ERRORS = (
    "connection refused",      # the store is down, e.g. restarting after an OOM kill
    "connection reset",
    "connection aborted",
    "timed out",               # what earlier partial failures recorded
    "timeout",
    "broken pipe",
    "temporarily unavailable",
    "service unavailable",
    "502 bad gateway",
    "503",
    "504",
    "max retries exceeded",
    "cannot connect",
    "server disconnected",
    "remote end closed",
)


def _vector_retry_delays() -> List[float]:
    """Backoff schedule in seconds, longest-first-failure to last attempt.

    The default spans 26 seconds because a typical store restart
    lasts about fifteen seconds. A schedule shorter than the thing it is meant to survive is
    decoration. Set SYNAP_VECTOR_WRITE_RETRY_DELAYS to a comma-separated list
    to change it, or to an empty string to disable retrying entirely.
    """
    raw = os.environ.get("SYNAP_VECTOR_WRITE_RETRY_DELAYS")
    if raw is None:
        return [2.0, 6.0, 18.0]
    if not raw.strip():
        return []
    out: List[float] = []
    for part in raw.split(","):
        try:
            value = float(part.strip())
        except ValueError:
            continue
        if value >= 0:
            out.append(value)
    return out


def _is_transient_vector_error(exc: BaseException) -> bool:
    """True when trying the same write again could plausibly work.

    Deliberately narrow. "Number of questions must match number of embeddings"
    is a real failure this pipeline has produced, and retrying it three times
    only spends 26 seconds arriving at the same answer.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError, OSError)):
        return True
    return any(marker in text for marker in _TRANSIENT_VECTOR_ERRORS)


async def _with_vector_retry(leg: str, call):
    """Run a vector-store write, retrying transient failures.

    ``call`` is a zero-argument coroutine factory rather than a coroutine,
    because a coroutine object cannot be awaited twice.

    Every write behind this is an upsert keyed by the memory id, so a retry
    after a partial write re-writes the same rows rather than duplicating them.
    That is what makes retrying safe here and it is worth checking again before
    this helper is pointed at anything else.
    """
    delays = _vector_retry_delays()
    last: Optional[BaseException] = None

    for attempt in range(len(delays) + 1):
        try:
            result = await call()
            if attempt:
                logger.info(
                    "vector_write_recovered leg=%s attempt=%d/%d",
                    leg, attempt + 1, len(delays) + 1,
                )
                inc_counter(
                    "synap_ingestion_vector_write_recovered_total",
                    labels={"leg": leg},
                    help_text="Vector writes that failed at least once and then succeeded on retry",
                )
            return result
        except Exception as exc:
            last = exc
            transient = _is_transient_vector_error(exc)
            if not transient or attempt >= len(delays):
                break
            delay = delays[attempt]
            logger.warning(
                "vector_write_retry leg=%s attempt=%d/%d delay=%.1fs error=%s",
                leg, attempt + 1, len(delays) + 1, delay, exc,
            )
            inc_counter(
                "synap_ingestion_vector_write_retries_total",
                labels={"leg": leg},
                help_text="Retried vector-store writes, by pipeline leg",
            )
            await asyncio.sleep(delay)

    # Out of attempts. The caller's gather turns this into a partial_failure
    # and the status tracker records it, but neither of those reaches a human,
    # so count it here where an alert rule can see it.
    inc_counter(
        "synap_ingestion_vector_write_failures_total",
        labels={"leg": leg, "transient": str(_is_transient_vector_error(last)).lower()},
        help_text="Vector-store writes that failed after every retry; these memories have no vector and no query can reach them",
    )
    logger.error(
        "vector_write_failed leg=%s attempts=%d error=%s -- memories persisted to "
        "Postgres have NO vector and are unreachable by retrieval; repair with "
        "scripts/vector_parity.py --repair",
        leg, len(delays) + 1, last,
        exc_info=last,
    )
    raise last  # type: ignore[misc]


class Stage4Organizer:
    """
    Organizes and persists extracted data to storage backends.

    Persistence order:
    1. RDBMS (sync) - generates authoritative IDs
    2. Graph (async) - uses RDBMS IDs
    3. Vector (async) - uses RDBMS IDs

    Graph and Vector happen in parallel after RDBMS completes.
    """

    def __init__(
        self,
        config: MemoryArchitectureConfig,
        storage: StorageCoordinator,
        entity_resolver: Optional["EntityResolutionService"] = None,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        knowledge_update_detector: Optional["KnowledgeUpdateDetector"] = None,
        hqe_generator: Optional["Stage5HQEGenerator"] = None,
    ):
        self._config = config
        self._storage = storage
        self._entity_resolver = entity_resolver
        self._session_factory = session_factory
        self._knowledge_update_detector = knowledge_update_detector
        # Optional HQE generator for re-running hypothetical-question
        # generation on merged memories. If absent, merge synthesis still works
        # but merged memories have no HQE entries (still retrievable via direct
        # vector similarity on the memory text).
        self._hqe_generator = hqe_generator

    async def _generate_and_persist_hqe_for_merged(
        self,
        merged_records: List[Dict],
        scope: Scope,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Generate hypothetical questions for merged memories and
        persist them to the questions vector collection.

        merged_records comes from execute_merges and contains:
          {memory_id, content, memory_type, source_document_ids}

        We build minimal ExtractedMemory wrappers, run the existing Stage 5
        HQE pipeline on them, then call the existing question-vector persist.
        Originals' HQE was already hard-deleted by delete_memory_vectors
        inside execute_merges, so no de-dup needed.
        """
        if not merged_records or self._hqe_generator is None:
            return

        from uuid import UUID
        from datetime import datetime, timezone
        from synap.cloud.agents.ingestion.models.enums import MemoryType

        # Build ExtractedMemory wrappers carrying the resolved metadata so
        # HQE generation has the same context any non-merged memory does:
        # the actual extracted_at (for temporal-aware questions), the
        # entities (for ERE-context lookups), and the source document(s).
        wrappers: List[ExtractedMemory] = []
        for rec in merged_records:
            try:
                mtype = MemoryType(rec.get("memory_type") or "facts")
            except ValueError:
                mtype = MemoryType.FACTS
            # source_document_id is a single legacy column; the merged
            # memory's authoritative source list is in memory_source_documents.
            # Pass the FIRST contributor doc so HQE code paths that read it
            # (e.g. for legacy provenance) still work, while the join table
            # stays the source of truth.
            source_docs = rec.get("source_document_ids") or []
            primary_source = source_docs[0] if source_docs else None
            wrappers.append(ExtractedMemory(
                memory_id=UUID(rec["memory_id"]),
                memory_type=mtype,
                content=rec["content"],
                confidence=0.9,
                destination=MemoryDestination.MEMORIES,
                source_document_id=primary_source,
                entities_mentioned=list(rec.get("entity_ids") or []),
                hypothetical_questions=[],
                extracted_at=rec.get("extracted_at") or datetime.now(timezone.utc),
                document_created_at=rec.get("document_created_at"),
            ))

        await self._hqe_generator.generate(
            memories=wrappers,
            request_scope=scope,
            correlation_id=correlation_id,
        )

        # Persist the populated hypothetical_questions. Skips memories with
        # an empty list (e.g. HQE batch failed for that one).
        #
        # Retried like the other four legs. This is the merge path rather than
        # the ingest path, but it writes to the same store over the same socket
        # and loses the same way when that store restarts.
        await _with_vector_retry(
            "questions_merge",
            lambda: self._storage.persist_question_vectors(
                memories=wrappers,
                scope=scope,
            ),
        )

    async def _attach_metadata_to_merges(self, merges: List) -> List:
        """Lossless-merge metadata resolution.

        For each MergeAction, two batched lookups against Postgres:

          1. memory_source_documents   → union of source_document_ids
                                          (with legacy fallback)
          2. memories                  → per-contributor metadata used by
                                          knowledge_update.resolve_merge_metadata
                                          to derive the resolved fields.
          3. memory_entity_links       → union of entity_ids per cluster
                                          (so the merged row inherits the
                                          contributors' graph reachability).

        Mutates each MergeAction in place; returns the same list for chaining.

        All three lookups happen on contributors that haven't been deprecated
        yet — Stage 4 calls this BEFORE execute_merges runs, which is the
        order that lets us read entity links before they cascade-delete.
        """
        if not merges:
            return merges

        from sqlalchemy import text
        from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
            resolve_merge_metadata,
        )
        from synap.cloud.database.connection import get_session_factory

        session_factory = get_session_factory()
        refused: List = []
        async with session_factory() as session:
            for merge in merges:
                contrib_ids = list(merge.contributor_memory_ids)

                # 1. Source documents (union via memory_source_documents).
                rows = await session.execute(
                    text("""SELECT DISTINCT source_document_id
                            FROM memory_source_documents
                            WHERE memory_id = ANY(CAST(:ids AS uuid[]))"""),
                    {"ids": contrib_ids},
                )
                sources = [r[0] for r in rows.fetchall() if r[0]]
                if not sources:
                    # Defensive fallback to legacy column.
                    rows = await session.execute(
                        text("""SELECT DISTINCT source_document_id
                                FROM memories
                                WHERE id = ANY(CAST(:ids AS uuid[]))
                                  AND source_document_id IS NOT NULL"""),
                        {"ids": contrib_ids},
                    )
                    sources = [r[0] for r in rows.fetchall() if r[0]]
                merge.contributor_source_documents = sources

                # 2. Per-contributor metadata for resolution.
                # event_date is not a real column — synthesize it for episodes
                # so resolve_merge_metadata can fall back when valid_from is
                # missing.
                #
                # ⚠ `created_at` used to be the last term of the COALESCE, as
                # "the closest timestamp we have". It is our extraction minute,
                # and the merged `valid_from` built from it is served as the
                # event date when there is no event window. The closest
                # timestamp we have is not the same as a date we may assert, so
                # the fallback now stops at the conversation's own date.
                from synap.cloud.shared import event_window_schema as ews
                _contrib_sql = """SELECT
                                memory_type, category, subcategory, confidence,
                                temporal_category, valid_from, valid_until,
                                temporal_confidence, is_episode, pii_obfuscated,
                                pii_detected,
                                document_created_at,
                                event_start, event_end, event_precision,
                                COALESCE(created_at, document_created_at) AS extracted_at,
                                CASE WHEN temporal_category = 'episode'
                                     THEN COALESCE(valid_from, document_created_at)
                                     ELSE NULL END AS event_date
                            FROM memories
                            WHERE id = ANY(CAST(:ids AS uuid[]))"""
                try:
                    rows = await session.execute(text(ews.adapt(_contrib_sql)), {"ids": contrib_ids})
                except Exception as exc:
                    # Database without migration 154: resolve the merge without windows.
                    if not (ews.available() and ews.is_missing_column_error(exc)):
                        raise
                    await session.rollback()
                    ews.mark_missing(exc)
                    rows = await session.execute(text(ews.adapt(_contrib_sql)), {"ids": contrib_ids})
                contributor_meta = [dict(r._mapping) for r in rows.fetchall()]
                resolved = resolve_merge_metadata(contributor_meta)
                for k, v in resolved.items():
                    setattr(merge, k, v)
                if merge.merge_refused_reason:
                    # The contributors' event windows cannot all describe one
                    # event, so they are not one fact and this merge is dropped.
                    # Previously it went ahead with all three window fields set
                    # to NULL, which silently replaced two real dates with none
                    # and left the row serving as though it happened on the day
                    # it was mentioned.
                    refused.append(merge)
                    logger.info(
                        "stage4_merge_refused reason=%s contributors=%s",
                        merge.merge_refused_reason,
                        ",".join(str(i) for i in contrib_ids))
                    continue

                # 3. Entity links union — used by execute_merges to write
                # memory_entity_links rows for the merged memory.
                rows = await session.execute(
                    text("""SELECT DISTINCT entity_id
                            FROM memory_entity_links
                            WHERE memory_id = ANY(CAST(:ids AS uuid[]))"""),
                    {"ids": contrib_ids},
                )
                merge.contributor_entity_ids = [r[0] for r in rows.fetchall() if r[0]]

        if refused:
            # Dropped, not flagged: execute_merges consumes whatever this
            # returns, so a refused merge has to leave the list or it still
            # runs. The contributors simply stay alive, each keeping its own
            # event window, which is the point.
            logger.info("stage4_merges_refused count=%d of %d",
                        len(refused), len(merges))
            merges = [m for m in merges if not m.merge_refused_reason]

        return merges

    async def organize_and_persist(
        self,
        stage3_output: Stage3Output,
        scope: Scope,
        document_id: str,
        s3_path: Optional[str] = None,
        content: Optional[str] = None,
        request_id: Optional[str] = None,
        http_correlation_id: Optional[str] = None,
        ingestion_id: Optional[str] = None,
    ) -> Stage4Output:
        """
        Organize and persist all extracted data.

        Args:
            stage3_output: Output from Stage 3 extraction
            scope: Tenant scope information
            document_id: Document identifier
            s3_path: S3 path where original content is stored (already persisted)
            content: Original text content (used for contextual entity resolution)

        Returns:
            Stage4Output with IDs of persisted records
        """
        # Entity resolution: resolve extracted entities to canonical names
        if self._entity_resolver and stage3_output.entities:
            # The LLM calls resolution makes are charged by request id; see
            # resolution/correlation.py for why it rides a ContextVar.
            from synap.cloud.agents.ingestion.resolution import correlation
            _bound = correlation.bind(request_id)
            try:
                stage3_output.entities = await self._entity_resolver.resolve_entities(
                    entities=stage3_output.entities,
                    scope=scope,
                    context=content,
                )
                hold_unsure = getattr(self._entity_resolver, "hold_unsure", None)
                if hold_unsure is not None:
                    # A mention resolution is not sure about is neither merged nor
                    # created: it and its relationships wait for review, and
                    # nothing below sees them.
                    stage3_output.entities, stage3_output.relationships = await hold_unsure(
                        stage3_output.entities, stage3_output.relationships, scope, document_id,
                        memories=stage3_output.memories,
                    )
            finally:
                correlation.unbind(_bound)
            resolved_count = sum(
                1 for e in stage3_output.entities if e.canonical_name
            )
            logger.info(
                "Stage 4: Entity resolution complete - %d/%d resolved",
                resolved_count,
                len(stage3_output.entities),
            )

        # Separate memories by destination. Stage 3 assigns destination based
        # on the request's scope_level, so for any given request exactly one
        # of (memories/candidates), customer_knowledge, or client_knowledge
        # will be populated — not a mix.
        memories_to_persist = self._filter_by_destination(
            stage3_output.memories,
            MemoryDestination.MEMORIES
        )
        candidates_to_persist = self._filter_by_destination(
            stage3_output.memories,
            MemoryDestination.CANDIDATES
        )
        client_knowledge = self._filter_by_destination(
            stage3_output.memories,
            MemoryDestination.CLIENT_KNOWLEDGE
        )
        customer_knowledge = self._filter_by_destination(
            stage3_output.memories,
            MemoryDestination.CUSTOMER_KNOWLEDGE
        )
        # Vector + graph persistence treats qualified memories uniformly — the
        # scope_level on the Scope object drives namespace + user_id metadata
        # via vector_wrapper. RDBMS routing (below) keeps them in separate
        # tables via the filtered lists above.
        vector_memories: List[ExtractedMemory] = (
            memories_to_persist + customer_knowledge + client_knowledge
        )

        # 1. RDBMS first (sync) - generates IDs
        async with start_span("ingestion.stage4.rdbms_persist", {
            "synap.memories_count": len(memories_to_persist),
            "synap.candidates_count": len(candidates_to_persist),
            "synap.entities_count": len(stage3_output.entities),
            "synap.client_knowledge_count": len(client_knowledge),
            "synap.customer_knowledge_count": len(customer_knowledge),
        }):
            start_rdbms = time.time()
            logger.info(f"Stage 4: Persisting to RDBMS memories_count={len(memories_to_persist)} candidates_count={len(candidates_to_persist)} entities_count={len(stage3_output.entities)} client_knowledge_count={len(client_knowledge)} customer_knowledge_count={len(customer_knowledge)} memories_preview={log_data_sample([{'content': truncate_string(m.content, 80), 'confidence': m.confidence} for m in memories_to_persist], sample_size=3, label='memories')} entities_preview={log_data_sample([{'name': e.name, 'type': e.entity_type} for e in stage3_output.entities], sample_size=3, label='entities')}")

            rdbms_result = await self._storage.persist_rdbms(
                memories=memories_to_persist,
                candidates=candidates_to_persist,
                entities=stage3_output.entities,
                scope=scope,
                document_id=document_id,
                client_knowledge=client_knowledge,
                customer_knowledge=customer_knowledge,
                http_correlation_id=http_correlation_id,
                ingestion_id=ingestion_id,
            )

            rdbms_duration = (time.time() - start_rdbms) * 1000
            set_current_span_attributes({
                "synap.memory_ids_count": len(rdbms_result.memory_ids),
                "synap.candidate_ids_count": len(rdbms_result.candidate_ids),
                "synap.entity_ids_count": len(rdbms_result.entity_ids),
                "synap.rdbms_latency_ms": round(rdbms_duration, 2),
            })
            logger.info(f"Stage 4: RDBMS persistence complete duration_ms={rdbms_duration:.2f} memory_ids_count={len(rdbms_result.memory_ids)} entity_ids_count={len(rdbms_result.entity_ids)}")

        # Chunk embedding is enabled by default — chunks back the provenance-based
        # chunk evidence retrieval (see ENABLE_CHUNK_EVIDENCE). Set
        # SKIP_CHUNK_EMBEDDING=true to explicitly disable chunk storage.
        skip_chunks = os.environ.get("SKIP_CHUNK_EMBEDDING", "false").lower() not in ("false", "0", "no")

        # 2. Compute all embeddings upfront (shared across graph + vector persistence)
        start_parallel = time.time()
        logger.info(f"Stage 4: Starting parallel persistence entities_count={len(stage3_output.entities)} relationships_count={len(stage3_output.relationships)} memories_to_embed={len(memories_to_persist)} relationships_preview={log_data_sample([{'source': r.source_entity_id, 'relation': r.relationship_type, 'target': r.target_entity_id} for r in stage3_output.relationships], sample_size=3, label='relationships')} entities_preview_detailed={log_data_sample([{'name': e.name, 'type': e.entity_type, 'canonical_name': getattr(e, 'canonical_name', None)} for e in stage3_output.entities], sample_size=5, label='entities_detailed')} scope=(instance_id={scope.instance_id}, user_id={scope.user_id}, customer_id={scope.customer_id})")

        # --- Compute all embeddings in a single batch ---
        entity_embeddings = []
        triple_embeddings = []
        memory_embeddings = []
        chunk_embeddings = []
        question_embeddings = []

        async with start_span("ingestion.stage4.compute_embeddings", {
            "synap.entity_count": len(stage3_output.entities),
            "synap.triple_count": len(stage3_output.relationships),
            "synap.memory_count": len(memories_to_persist),
        }):
            _t_embed = time.time()
            try:
                # Build all content lists
                entity_contents = []
                triple_contents = []
                memory_contents = []

                from synap.cloud.agents.ingestion.services.writer_entity import shown_name
                if stage3_output.entities:
                    entity_contents = [
                        f"{shown_name(e.properties, e.canonical_name or e.name)} ({e.entity_type})"
                        for e in stage3_output.entities
                    ]

                if stage3_output.relationships:
                    entity_name_map = {
                        e.entity_id: shown_name(e.properties, e.canonical_name or e.name)
                        for e in stage3_output.entities
                    }
                    for rel in stage3_output.relationships:
                        src = entity_name_map.get(rel.source_entity_id, rel.source_entity_id)
                        tgt = entity_name_map.get(rel.target_entity_id, rel.target_entity_id)
                        triple_contents.append(f"{src} {rel.relationship_type} {tgt}")

                if vector_memories:
                    memory_contents = [m.content for m in vector_memories]

                chunk_windows = []
                if not skip_chunks and content and content.strip():
                    chunk_windows = self._storage.split_into_windows(content)

                # Collect hypothetical questions for embedding
                question_texts = []
                skip_questions = os.environ.get("ENABLE_QUESTION_EMBEDDINGS", "true").lower() not in ("true", "1", "yes")
                if not skip_questions and vector_memories:
                    for m in vector_memories:
                        for q in (m.hypothetical_questions or []):
                            if q and q.strip():
                                question_texts.append(q.strip())

                # Single batch call for all embeddings (1 thread dispatch instead of N).
                # Chunks are excluded — chunk evidence uses metadata-filtered GET,
                # not similarity search, so chunk embeddings are never used.
                # We pass zero-vectors to ChromaDB instead (saves ~1-2s per batch on CPU).
                all_contents = entity_contents + triple_contents + memory_contents + question_texts
                if all_contents:
                    all_embeddings = await self._storage._embedding.embed_batch(all_contents)
                    # Split results back by section
                    idx = 0
                    entity_embeddings = all_embeddings[idx:idx + len(entity_contents)]
                    idx += len(entity_contents)
                    triple_embeddings = all_embeddings[idx:idx + len(triple_contents)]
                    idx += len(triple_contents)
                    memory_embeddings = all_embeddings[idx:idx + len(memory_contents)]
                    idx += len(memory_contents)
                    question_embeddings = all_embeddings[idx:idx + len(question_texts)]
                    # Chunk embeddings: zero-vectors (chunks are retrieved by metadata filter,
                    # not similarity search, so real embeddings are unnecessary)
                    dimension = len(all_embeddings[0]) if all_embeddings else 1024
                    chunk_embeddings = [[0.0] * dimension for _ in chunk_windows] if chunk_windows else []
                else:
                    chunk_embeddings = []
                    question_embeddings = []

                set_current_span_attributes({
                    "synap.entity_embeddings": len(entity_embeddings),
                    "synap.triple_embeddings": len(triple_embeddings),
                    "synap.memory_embeddings": len(memory_embeddings),
                    "synap.chunk_embeddings": len(chunk_embeddings),
                    "synap.question_embeddings": len(question_embeddings),
                    "synap.total_vectors": len(entity_embeddings) + len(triple_embeddings) + len(memory_embeddings) + len(chunk_embeddings) + len(question_embeddings),
                    "synap.embed_latency_ms": round((time.time() - _t_embed) * 1000, 2),
                })
                logger.info(f"Stage 4: Embeddings computed upfront entity_embeddings={len(entity_embeddings)} triple_embeddings={len(triple_embeddings)} memory_embeddings={len(memory_embeddings)} chunk_embeddings={len(chunk_embeddings)} question_embeddings={len(question_embeddings)}")
            except Exception as e:
                set_current_span_attributes({"synap.embed_error": True})
                logger.error(f"Stage 4: Embedding computation failed: {e}", exc_info=True)
                # Reset to None so downstream methods trigger their own recompute
                # (empty [] would bypass the `is None` check and cause ValueError)
                entity_embeddings = None
                triple_embeddings = None
                memory_embeddings = None
                chunk_embeddings = None
                question_embeddings = None

        # --- Knowledge-update detection (optional) ---
        # KU v2: classifier emits four outcomes per pair: SUPERSEDE / NONE /
        # UNRELATED / ENRICH. Each is routed to a different storage operation:
        #   SUPERSEDE -> execute_supersessions (deprecate old, link to new)
        #   NONE      -> execute_skips         (drop the new memory)
        #   ENRICH    -> execute_enrichments   (log only)
        #   UNRELATED -> no action
        # The cascade phase from v1 was removed: top-15 NEW-embedding sweeps
        # over-deleted by 5-14x with no equivalent in any comparable system.
        skipped_new_memory_ids: set[str] = set()
        if (
            self._knowledge_update_detector is not None
            and memories_to_persist
            and memory_embeddings is not None
            and len(memory_embeddings) > 0
            and os.environ.get("ENABLE_KNOWLEDGE_UPDATE_DETECTION", "true").lower() in ("true", "1", "yes")
        ):
            try:
                async with start_span("ingestion.stage4.knowledge_update_detection", {
                    "synap.memories_count": len(memories_to_persist),
                }):
                    ku_result = await self._knowledge_update_detector.detect_updates(
                        memories=memories_to_persist,
                        memory_embeddings=memory_embeddings,
                        scope=scope,
                        # request_id, NOT http_correlation_id: the ingestion
                        # charge sums usage rows keyed on the request id, so
                        # stamping these calls with anything else leaves them
                        # invisible to every per-request cost view.
                        correlation_id=request_id,
                        # Already held for contextual entity resolution; the Jev
                        # candidate screen reads it as context and nothing else does.
                        document_content=content,
                    )
                    superseded_count = await self._storage.execute_supersessions(
                        actions=ku_result.supersessions,
                        scope=scope,
                        http_correlation_id=http_correlation_id,
                        ingestion_id=ingestion_id,
                    )
                    skipped_count = await self._storage.execute_skips(
                        actions=ku_result.skips,
                        scope=scope,
                        http_correlation_id=http_correlation_id,
                        ingestion_id=ingestion_id,
                    )

                    # Consolidation D2 (SYNAP_KU_SHARED_BANDS, default off):
                    # defer-band verdicts become adjudication cases — both
                    # memories keep serving, the cycles decide. One DEFER
                    # trail row per case for the audit trail. Non-fatal: a
                    # failed case-open loses one deferral, never the ingest.
                    for defer in getattr(ku_result, "defers", None) or []:
                        try:
                            await edition.acall(
                                "consolidation_links",
                                "defer_case",
                                defer=defer,
                                scope=scope,
                                http_correlation_id=http_correlation_id,
                                ingestion_id=ingestion_id,
                            )
                        except Exception as defer_exc:  # noqa: BLE001
                            logger.warning(
                                "defer case-open failed (non-fatal): %s", defer_exc,
                            )

                    # Gap 6: explicit natural-language deletions. Flag-gated
                    # (ENABLE_KU_DELETE, default off) so the "never deletes"
                    # invariant holds by default; hasattr guards storage impls
                    # that predate execute_deletions. Soft-delete: deprecate +
                    # drop from retrieval, audit trail preserved.
                    deleted_count = 0
                    delete_enabled = os.environ.get(
                        "ENABLE_KU_DELETE", "false"
                    ).lower() in ("true", "1", "yes")
                    if (
                        delete_enabled
                        and getattr(ku_result, "deletions", None)
                        and hasattr(self._storage, "execute_deletions")
                    ):
                        deleted_count = await self._storage.execute_deletions(
                            actions=ku_result.deletions,
                            scope=scope,
                            http_correlation_id=http_correlation_id,
                            ingestion_id=ingestion_id,
                        )

                    # Retraction subject sweep (ENABLE_KU_DELETE_SUBJECT_GUARD):
                    # a confirmed retraction retires ALL current memories
                    # matching its subject, not just the one vector-paired row
                    # (this used to mis-target: "delete my home
                    # address" deleted "The user lives in Texas" and the
                    # address fact survived). Runs after execute_deletions so
                    # pair-verdict targets are already deprecated and skipped
                    # by the current-rows lookup; the just-persisted batch is
                    # excluded — pair verdicts own those. Flag check lives in
                    # the storage method; hasattr guards older impls.
                    if (
                        delete_enabled
                        and getattr(ku_result, "retraction_sweeps", None)
                        and hasattr(self._storage, "execute_retraction_sweeps")
                    ):
                        await self._storage.execute_retraction_sweeps(
                            sweeps=ku_result.retraction_sweeps,
                            scope=scope,
                            exclude_memory_ids={
                                str(m.memory_id) for m in memories_to_persist
                            },
                            http_correlation_id=http_correlation_id,
                            ingestion_id=ingestion_id,
                        )

                    # Echo resurrection guard (ENABLE_ECHO_GUARD, WS2 4b):
                    # assistant-sourced memories that re-assert a value the
                    # user already retired are stored pre-deprecated so they
                    # never re-enter retrieval. Runs after the pair-level KU
                    # actions so classifier verdicts win first; flag check
                    # lives inside the storage method. hasattr guards older
                    # storage impls.
                    if hasattr(self._storage, "suppress_retired_value_echoes"):
                        suppressed_echo_ids = await self._storage.suppress_retired_value_echoes(
                            memories=memories_to_persist,
                            scope=scope,
                            already_skipped_ids={
                                a.new_memory_id for a in ku_result.skips
                            },
                            http_correlation_id=http_correlation_id,
                            ingestion_id=ingestion_id,
                        )
                        # The memory-vector upsert (persist_vector) and HQE
                        # question vectors run AFTER this block, so the
                        # sweep's own vector delete is only a backstop —
                        # union the suppressed ids into the KU skip set so
                        # the echoes are never vectorized in the first place
                        # (otherwise a swept echo's vector
                        # was re-created by the later persist step and kept
                        # being served by the vector arm).
                        skipped_new_memory_ids |= set(suppressed_echo_ids or ())

                    # ENRICH merge synthesis. When the flag is on,
                    # build clusters from the ENRICH pairs, synthesize merged
                    # memories via LLM, deprecate the contributors. When off,
                    # fall back to the earlier behavior (just log ENRICH rows).
                    merge_synthesis = os.environ.get(
                        "ENABLE_KU_MERGE_SYNTHESIS", "false"
                    ).lower() in ("true", "1", "yes")

                    enriched_count = 0
                    merged_count = 0
                    merged_records: List[Dict] = []
                    if merge_synthesis and ku_result.enrichments:
                        merges = await self._knowledge_update_detector.synthesize_merges(
                            enrichments=ku_result.enrichments,
                            scope=scope,
                            correlation_id=request_id,
                        )
                        # Resolve all the metadata fields the merged memory
                        # should inherit from its contributors: source documents,
                        # the 13 lossless-merge fields (memory_type, category,
                        # confidence, temporal_*, DCT range, etc.), and the
                        # entity-link union. One helper, three batched SELECTs
                        # per cluster.
                        if merges:
                            merges = await self._attach_metadata_to_merges(
                                merges=merges,
                            )
                        merged_records = await self._storage.execute_merges(
                            merges=merges,
                            scope=scope,
                            embedding_client=self._storage._embedding,
                            http_correlation_id=http_correlation_id,
                            ingestion_id=ingestion_id,
                        )
                        merged_count = len(merged_records)
                        # Drop the merged-contributors from downstream vector
                        # persistence so the originals don't get re-upserted
                        # alongside the new merged memory. (Same lockstep
                        # filter we use for NONE.)
                        merged_contributor_ids = {
                            cid for m in merges for cid in m.contributor_memory_ids
                        }
                        skipped_new_memory_ids |= merged_contributor_ids
                    else:
                        enriched_count = await self._storage.execute_enrichments(
                            actions=ku_result.enrichments,
                            scope=scope,
                            http_correlation_id=http_correlation_id,
                            ingestion_id=ingestion_id,
                        )

                    # Re-run HQE on merged memories. The merged
                    # content is richer than any single contributor, so
                    # questions tailored to it are higher quality than just
                    # carrying the originals' (now-deleted) HQE forward.
                    # Originals' HQE was hard-deleted by execute_merges via
                    # delete_memory_vectors. If no HQE generator is wired or
                    # nothing was merged, this is a no-op.
                    if merged_records and self._hqe_generator is not None:
                        try:
                            await self._generate_and_persist_hqe_for_merged(
                                merged_records=merged_records,
                                scope=scope,
                                correlation_id=http_correlation_id,
                            )
                        except Exception as e:
                            logger.warning(
                                "Stage 4: HQE for merged memories failed (non-fatal): %s",
                                e, exc_info=True,
                            )

                    # Subject identity (migration 159). Runs after every KU
                    # action so a memory KU dropped as a duplicate is never
                    # grouped, and before the vector persist below only because
                    # that is where this block sits — the subject lives on the
                    # Postgres row and the read path fills it from there, so
                    # nothing here depends on Chroma.
                    skipped_new_memory_ids |= {a.new_memory_id for a in ku_result.skips}

                    subject_count = 0
                    if (getattr(ku_result, "subject_candidates", None)
                            and hasattr(self._storage, "execute_subject_links")):
                        # A retired memory is dropped from the shortlist rather
                        # than repointed at its survivor: the statement the judge
                        # will be shown would not be the statement that is left,
                        # and guessing the substitution is how a wrong group gets
                        # written. A shortlist emptied this way is dropped whole.
                        surviving = []
                        for l in ku_result.subject_candidates:
                            if l.new_memory_id in skipped_new_memory_ids:
                                continue
                            kept = [(mid, txt) for mid, txt in l.candidates
                                    if mid not in skipped_new_memory_ids]
                            if kept:
                                l.candidates = kept
                                surviving.append(l)
                        subject_count = await self._storage.execute_subject_links(
                            links=surviving,
                            scope=scope,
                            http_correlation_id=http_correlation_id,
                            ingestion_id=ingestion_id,
                        )
                    logger.info(
                        "Stage 4: Knowledge-update complete superseded=%d skipped=%d enriched=%d merged=%d deleted=%d subject_grouped=%d",
                        superseded_count, skipped_count, enriched_count, merged_count, deleted_count, subject_count,
                    )
            except Exception as e:
                logger.warning(
                    "Stage 4: Knowledge-update detection failed (non-fatal): %s", e,
                    exc_info=True,
                )

        # If KU classified any new memories as NONE, drop them from the
        # downstream vector/graph persistence so they never land in ChromaDB.
        # The RDBMS row for each was already soft-deleted by execute_skips.
        # `memory_embeddings` is aligned 1:1 to `vector_memories`, so filter
        # both in lockstep to preserve alignment for the persist step below.
        if skipped_new_memory_ids:
            keep_mask = [
                str(m.memory_id) not in skipped_new_memory_ids
                for m in vector_memories
            ]
            # `question_embeddings` is a FLAT list built by walking
            # vector_memories in order and emitting one entry per non-empty
            # hypothetical question (see question_texts construction above).
            # When a memory is dropped, its question embeddings must be dropped
            # in lockstep — otherwise the question-vector persist leg rebuilds
            # the question list from the (now-shorter) vector_memories and sees
            # fewer questions than embeddings, raising "Number of questions must
            # match number of embeddings" -> PARTIAL_SUCCESS. Compute this mask
            # against the pre-filter vector_memories, before it is reassigned.
            if question_embeddings:
                question_keep_mask = [
                    keep
                    for m, keep in zip(vector_memories, keep_mask)
                    for q in (m.hypothetical_questions or [])
                    if q and q.strip()
                ]
                question_embeddings = [
                    e for e, keep in zip(question_embeddings, question_keep_mask) if keep
                ]
            memories_to_persist = [
                m for m in memories_to_persist
                if str(m.memory_id) not in skipped_new_memory_ids
            ]
            vector_memories = [m for m, keep in zip(vector_memories, keep_mask) if keep]
            if memory_embeddings is not None:
                memory_embeddings = [
                    e for e, keep in zip(memory_embeddings, keep_mask) if keep
                ]

        # --- Persist in parallel (all with pre-computed embeddings) ---
        # vector_memories includes customer/client knowledge so they also land
        # in ChromaDB — vector_wrapper reads scope.scope_level to decide the
        # namespace and user_id metadata.
        graph_task = asyncio.create_task(
            self._persist_graph_safe(
                stage3_output=stage3_output,
                memories=vector_memories,
                scope=scope,
                entity_embeddings=entity_embeddings,
                triple_embeddings=triple_embeddings,
                memory_embeddings=memory_embeddings,
            )
        )

        vector_task = asyncio.create_task(
            self._persist_vector_safe(
                memories=vector_memories,
                entities=stage3_output.entities,
                scope=scope,
                embeddings=memory_embeddings,
            )
        )

        entity_vector_task = asyncio.create_task(
            self._persist_entity_vectors_safe(
                entities=stage3_output.entities,
                scope=scope,
                embeddings=entity_embeddings,
            )
        )

        triple_vector_task = asyncio.create_task(
            self._persist_triple_vectors_safe(
                relationships=stage3_output.relationships,
                entities=stage3_output.entities,
                scope=scope,
                embeddings=triple_embeddings,
            )
        )

        # Chunk vectors use document_id as source_document_id (matching memories)
        # so provenance-based chunk retrieval can find them.
        # For Stage 0 sub-chunks (__chunkN suffix in request_id), we pass
        # request_id as chunk_meta so the deterministic chunk ID includes
        # the sub-chunk suffix — preventing overwrites between sub-chunks.
        chunk_source_id = document_id
        chunk_extra_meta = {}
        if request_id and "__chunk" in request_id:
            chunk_extra_meta["sub_chunk_id"] = request_id

        if skip_chunks:
            async def _noop_chunk():
                from synap.cloud.agents.ingestion.storage.coordinator import ChunkVectorPersistResult
                return ChunkVectorPersistResult(chunk_vector_ids=[], chunk_count=0)
            chunk_vector_task = asyncio.create_task(_noop_chunk())
        else:
            chunk_vector_task = asyncio.create_task(
                self._persist_chunk_vectors_safe(
                    content=content or "",
                    scope=scope,
                    source_document_id=chunk_source_id,
                    embeddings=chunk_embeddings if chunk_embeddings else None,
                    chunk_meta=chunk_extra_meta if chunk_extra_meta else None,
                )
            )

        # HQE: Hypothetical Question Embeddings
        if skip_questions or not question_texts:
            async def _noop_questions():
                from synap.cloud.agents.ingestion.storage.coordinator import QuestionVectorPersistResult
                return QuestionVectorPersistResult(question_vector_ids=[], question_count=0)
            question_vector_task = asyncio.create_task(_noop_questions())
        else:
            question_vector_task = asyncio.create_task(
                self._persist_question_vectors_safe(
                    memories=vector_memories,
                    scope=scope,
                    embeddings=question_embeddings if question_embeddings else None,
                )
            )

        # Wait for all with exception handling
        graph_result, vector_result, entity_vector_result, triple_vector_result, chunk_vector_result, question_vector_result = (
            await asyncio.gather(
                graph_task,
                vector_task,
                entity_vector_task,
                triple_vector_task,
                chunk_vector_task,
                question_vector_task,
                return_exceptions=True,
            )
        )

        # Handle any exceptions from parallel tasks and track failures
        failures = []

        graph_entity_ids = []
        graph_relationship_ids = []
        if isinstance(graph_result, Exception):
            logger.error(f"Graph persistence failed: {graph_result}", exc_info=graph_result)
            failures.append(f"graph: {graph_result}")
        else:
            graph_entity_ids = graph_result.entity_ids
            graph_relationship_ids = graph_result.relationship_ids

        vector_ids = []
        if isinstance(vector_result, Exception):
            logger.error(f"Vector persistence failed: {vector_result}", exc_info=vector_result)
            failures.append(f"vector: {vector_result}")
        else:
            vector_ids = vector_result.vector_ids

        entity_vector_ids = []
        if isinstance(entity_vector_result, Exception):
            logger.error(f"Entity vector persistence failed: {entity_vector_result}", exc_info=entity_vector_result)
            failures.append(f"entity_vector: {entity_vector_result}")
        else:
            entity_vector_ids = entity_vector_result.entity_vector_ids

        triple_vector_ids = []
        if isinstance(triple_vector_result, Exception):
            logger.error(f"Triple vector persistence failed: {triple_vector_result}", exc_info=triple_vector_result)
            failures.append(f"triple_vector: {triple_vector_result}")
        else:
            triple_vector_ids = triple_vector_result.triple_vector_ids

        chunk_vector_ids = []
        chunk_count = 0
        if isinstance(chunk_vector_result, Exception):
            logger.error(f"Chunk vector persistence failed: {chunk_vector_result}", exc_info=chunk_vector_result)
            failures.append(f"chunk_vector: {chunk_vector_result}")
        else:
            chunk_vector_ids = chunk_vector_result.chunk_vector_ids
            chunk_count = chunk_vector_result.chunk_count

        question_vector_ids = []
        question_count_result = 0
        if isinstance(question_vector_result, Exception):
            logger.error(f"Question vector persistence failed: {question_vector_result}", exc_info=question_vector_result)
            failures.append(f"question_vector: {question_vector_result}")
        else:
            question_vector_ids = question_vector_result.question_vector_ids
            question_count_result = question_vector_result.question_count

        parallel_duration = (time.time() - start_parallel) * 1000
        total_duration = (time.time() - start_rdbms) * 1000

        # Cross-region replication used to be announced from here, by an
        # emit that also stamped memories.replicated_at. Nothing calls it now:
        # the rows this stage writes are carried by the WAL reader, which sees
        # every writer rather than this one path, and their vectors by the
        # adapter's write log. Sending from here as well meant every ingestion
        # crossed twice (the relay's read-back suppressed 1,276 of those a day
        # by checking the stamp). See synap/cloud/replication/wal_reader.py.

        logger.info(f"Stage 4: All persistence complete rdbms_memory_count={len(rdbms_result.memory_ids)} rdbms_candidate_count={len(rdbms_result.candidate_ids)} graph_entity_count={len(graph_entity_ids)} graph_relationship_count={len(graph_relationship_ids)} vector_count={len(vector_ids)} entity_vector_count={len(entity_vector_ids)} triple_vector_count={len(triple_vector_ids)} chunk_vector_count={len(chunk_vector_ids)} chunk_count={chunk_count} question_vector_count={len(question_vector_ids)} question_count={question_count_result} parallel_duration_ms={parallel_duration:.2f} total_duration_ms={total_duration:.2f} partial_failure={len(failures) > 0}")

        # Settle any date that was waiting on an event we now know about. This
        # runs AFTER persistence on purpose: the document's own memories are in,
        # so resolution works in both directions — this document can date an older
        # memory's anchor, and an older memory can date one of this document's.
        #
        # Deliberately here and not in a consolidation node: consolidation is not
        # run on benchmark data by choice, so a node there would be invisible to
        # every run we measure.
        # organize_and_persist has no `correlation_id`; the ingestion's own id is
        # request_id (see the knowledge-update call above). Referencing the bare
        # name raised NameError, and the except below referenced it again, so the
        # "never fails a persist" guard itself failed every persist it ran on.
        anchor_correlation_id = request_id or http_correlation_id or ""
        if anchor_resolver.enabled_for(getattr(scope, "instance_id", None)):
            try:
                counts = await anchor_resolver.resolve_open_anchors(
                    self._session_factory, scope,
                    vector=getattr(self._storage, "_vector", None),
                    correlation_id=anchor_correlation_id)
                if counts.get("open"):
                    logger.info(
                        "stage4_anchor_resolution open=%d resolved=%d refused=%d "
                        "vectors_updated=%d jev_screened=%d deferred=%d "
                        "correlation_id=%s",
                        counts["open"], counts["resolved"], counts["refused"],
                        counts["vectors_updated"], counts.get("screened", 0),
                        counts.get("deferred", 0), anchor_correlation_id)
            except Exception:  # noqa: BLE001 — never fails a persist
                logger.warning("stage4_anchor_resolution_failed correlation_id=%s",
                               anchor_correlation_id, exc_info=True)

        return Stage4Output(
            input_id=stage3_output.input_id,
            rdbms_memory_ids=[str(id) for id in rdbms_result.memory_ids],
            rdbms_knowledge_ids=[str(id) for id in rdbms_result.knowledge_ids],
            rdbms_candidate_ids=[str(id) for id in rdbms_result.candidate_ids],
            graph_entity_ids=graph_entity_ids,
            graph_relationship_ids=graph_relationship_ids,
            vector_ids=vector_ids,
            entity_vector_ids=entity_vector_ids,
            triple_vector_ids=triple_vector_ids,
            chunk_vector_ids=chunk_vector_ids,
            chunk_count=chunk_count,
            question_vector_ids=question_vector_ids,
            question_count=question_count_result,
            s3_original_path=s3_path,
            partial_failure=len(failures) > 0,
            failure_details=failures,
        )

    async def _persist_graph_safe(
        self,
        stage3_output: Stage3Output,
        memories: List[ExtractedMemory],
        scope: Scope,
        entity_embeddings: List = None,
        triple_embeddings: List = None,
        memory_embeddings: List = None,
    ):
        """Persist to graph with error handling."""
        logger.info(f"Stage 4: Persisting to graph store entities_to_persist={len(stage3_output.entities)} relationships_to_persist={len(stage3_output.relationships)} memories_to_attach={len(memories)} entity_types={list(set(e.entity_type for e in stage3_output.entities))} relationship_types={list(set(r.relationship_type for r in stage3_output.relationships))}")

        async with start_span("ingestion.stage4.graph_persist", {
            "synap.entities_count": len(stage3_output.entities),
            "synap.relationships_count": len(stage3_output.relationships),
            "synap.memories_count": len(memories),
        }):
            _t0 = time.time()
            result = await self._storage.persist_graph(
                entities=stage3_output.entities,
                relationships=stage3_output.relationships,
                memories=memories,
                scope=scope,
                entity_embeddings=entity_embeddings,
                triple_embeddings=triple_embeddings,
                memory_embeddings=memory_embeddings,
            )
            set_current_span_attributes({
                "synap.nodes_created": len(getattr(result, "entity_ids", [])),
                "synap.edges_created": len(getattr(result, "relationship_ids", [])),
                "synap.graph_latency_ms": round((time.time() - _t0) * 1000, 2),
            })

            # Register relationship types in the per-instance registry (non-blocking)
            if stage3_output.relationships and self._session_factory:
                try:
                    await self._register_relationship_types(
                        instance_id=scope.instance_id,
                        relationships=stage3_output.relationships,
                    )
                except Exception as e:
                    # Registration failure must never block ingestion
                    # The text of a database error here repeats the whole embedding vector: keep the first line.
                    logger.warning("Stage 4: Relationship type registration failed (non-fatal): %s",
                                   str(e).splitlines()[0][:300] if str(e) else type(e).__name__)

            return result

    async def _register_relationship_types(
        self,
        instance_id: str,
        relationships: List,
    ) -> None:
        """Register newly discovered relationship types in the per-instance registry.

        Embeds each unique type name and upserts into the registry table.
        Uses its own session to avoid coupling with the storage coordinator's transaction.
        """
        from synap.cloud.database.repositories.relationship_type_registry_repo import (
            RelationshipTypeRegistryRepository,
        )

        unique_types = set(r.relationship_type.upper() for r in relationships)
        if not unique_types:
            return

        async with start_span("ingestion.stage4.register_relationship_types", {
            "synap.unique_types": len(unique_types),
            "synap.instance_id": instance_id,
        }):
            # Embed type names: "DIAGNOSED_WITH" → embed "diagnosed with"
            type_names = list(unique_types)
            readable_names = [t.replace("_", " ").lower() for t in type_names]
            embeddings = await self._storage._embedding.embed_batch(readable_names)

            async with self._session_factory() as session:
                repo = RelationshipTypeRegistryRepository(session)
                types_with_embeddings = list(zip(type_names, embeddings))
                await repo.upsert_batch(instance_id, types_with_embeddings, source="ingestion")
                await session.commit()

            logger.info(
                f"Stage 4: Registered relationship types instance_id={instance_id} "
                f"types={type_names}"
            )

    async def _persist_vector_safe(
        self,
        memories: List[ExtractedMemory],
        entities: List,
        scope: Scope,
        embeddings: List = None,
    ):
        """Persist to vector with error handling."""
        logger.info(f"Stage 4: Persisting to vector store memories_to_persist={len(memories)} entities_to_persist={len(entities)} memory_types={list(set(m.memory_type for m in memories))} has_embeddings={len(embeddings) if embeddings else 0}")

        async with start_span("ingestion.stage4.vector_persist", {
            "synap.memories_count": len(memories),
        }):
            _t0 = time.time()
            result = await _with_vector_retry(
                "memories",
                lambda: self._storage.persist_vector(
                    memories=memories,
                    entities=entities,
                    scope=scope,
                    embeddings=embeddings,
                ),
            )
            set_current_span_attributes({
                "synap.vectors_stored": len(getattr(result, "vector_ids", [])),
                "synap.vector_latency_ms": round((time.time() - _t0) * 1000, 2),
            })
            return result

    async def _persist_entity_vectors_safe(
        self,
        entities: List,
        scope: Scope,
        embeddings: List = None,
    ):
        """Persist entity embeddings to entity vector collection."""
        logger.info(f"Stage 4: Persisting entity vectors entities_to_persist={len(entities)} entity_types={list(set(e.entity_type for e in entities))} has_embeddings={len(embeddings) if embeddings else 0}")

        async with start_span("ingestion.stage4.entity_vector_persist", {
            "synap.entities_count": len(entities),
        }):
            _t0 = time.time()
            result = await _with_vector_retry(
                "entities",
                lambda: self._storage.persist_entity_vectors(
                    entities=entities,
                    scope=scope,
                    embeddings=embeddings,
                ),
            )
            set_current_span_attributes({
                "synap.entity_vectors_stored": len(getattr(result, "entity_vector_ids", [])),
                "synap.entity_vector_latency_ms": round((time.time() - _t0) * 1000, 2),
            })
            return result

    async def _persist_triple_vectors_safe(
        self,
        relationships: List,
        entities: List,
        scope: Scope,
        embeddings: List = None,
    ):
        """Persist relationship triples to triples vector collection."""
        logger.info(f"Stage 4: Persisting triple vectors relationships_to_persist={len(relationships)} entities_context={len(entities)} relationship_types={list(set(r.relationship_type for r in relationships))} has_embeddings={len(embeddings) if embeddings else 0}")

        async with start_span("ingestion.stage4.triple_vector_persist", {
            "synap.relationships_count": len(relationships),
        }):
            _t0 = time.time()
            result = await _with_vector_retry(
                "triples",
                lambda: self._storage.persist_triple_vectors(
                    relationships=relationships,
                    entities=entities,
                    scope=scope,
                    embeddings=embeddings,
                ),
            )
            set_current_span_attributes({
                "synap.triple_vectors_stored": len(getattr(result, "triple_vector_ids", [])),
                "synap.triple_vector_latency_ms": round((time.time() - _t0) * 1000, 2),
            })
            return result

    async def _persist_chunk_vectors_safe(
        self,
        content: str,
        scope: Scope,
        source_document_id: str,
        embeddings: List = None,
        chunk_meta: dict = None,
    ):
        """Persist raw content chunks to chunks vector collection."""
        if not content or not content.strip():
            from synap.cloud.agents.ingestion.storage.coordinator import ChunkVectorPersistResult
            return ChunkVectorPersistResult(chunk_vector_ids=[], chunk_count=0)

        logger.info(f"Stage 4: Persisting chunk vectors source_doc={source_document_id} content_len={len(content)} has_embeddings={len(embeddings) if embeddings else 0}")

        return await _with_vector_retry(
            "chunks",
            lambda: self._storage.persist_chunk_vectors(
                content=content,
                scope=scope,
                source_document_id=source_document_id,
                embeddings=embeddings,
                chunk_meta=chunk_meta,
            ),
        )

    async def _persist_question_vectors_safe(
        self,
        memories: List[ExtractedMemory],
        scope: Scope,
        embeddings: List = None,
    ):
        """Persist hypothetical question vectors with error handling."""
        question_count = sum(len(m.hypothetical_questions or []) for m in memories)
        if question_count == 0:
            from synap.cloud.agents.ingestion.storage.coordinator import QuestionVectorPersistResult
            return QuestionVectorPersistResult(question_vector_ids=[], question_count=0)

        logger.info(f"Stage 4: Persisting question vectors questions_to_persist={question_count} memories_with_questions={sum(1 for m in memories if m.hypothetical_questions)}")

        async with start_span("ingestion.stage4.question_vector_persist", {
            "synap.question_count": question_count,
        }):
            _t0 = time.time()
            result = await _with_vector_retry(
                "questions",
                lambda: self._storage.persist_question_vectors(
                    memories=memories,
                    scope=scope,
                    embeddings=embeddings,
                ),
            )
            set_current_span_attributes({
                "synap.question_vectors_stored": len(getattr(result, "question_vector_ids", [])),
                "synap.question_vector_latency_ms": round((time.time() - _t0) * 1000, 2),
            })
            return result

    def _filter_by_destination(
        self,
        memories: List[ExtractedMemory],
        destination: MemoryDestination,
    ) -> List[ExtractedMemory]:
        """Filter memories by destination."""
        return [m for m in memories if m.destination == destination]

