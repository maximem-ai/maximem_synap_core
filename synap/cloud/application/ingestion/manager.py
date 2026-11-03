"""Ingestion Manager - Application layer for ingestion requests."""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Union
from uuid import UUID, uuid4

from synap.cloud.shared.step_logger import StepLogger, log_substep
from synap.cloud.shared import edition
from synap.cloud.shared.instance_flags import enabled_for
from synap.cloud.observability.tracing import start_span, set_current_span_attributes

from synap.cloud.agents.ingestion.models import (
    ConvertSessionRequest,
    DocumentSource,
    DocumentType,
    IngestBatchRequest,
    IngestBatchResponse,
    IngestDocumentRequest,
    IngestFromListeningAgentRequest,
    IngestPriority,
    IngestResponse,
    IngestStatus,
)
from synap.cloud.application.models.errors import ValidationError, RateLimitExceeded
from synap.cloud.agents.ingestion.pipeline.stage0_preprocessor import (
    Stage0Preprocessor,
    maca_chunking_enabled_for,
)
from .idempotency import IdempotencyChecker, normalized_content_hash
from .queue import IngestionQueue
from .rate_limiter import RateLimiter
from .status import StatusTracker
from .validators import RequestValidator

logger = logging.getLogger(__name__)


@dataclass
class StorageTreatment:
    """What `treat_for_storage` did to one text. Counts only, never values."""
    text: str
    failed_at: Optional[str] = None   # None = treated; else "policy" | "detector" | "gate"
    reason: Optional[str] = None
    detected: Optional[bool] = None
    floor_blocked: int = 0
    aliased: int = 0
    policy_version: Optional[int] = None
    applied: bool = True              # False = there is no PII step in this build, so nothing was treated


class PIIPolicyEnforcementError(RuntimeError):
    """An approved PII policy could not be applied to a document.

    Deliberately not caught alongside every other gate failure. Watch mode
    failing is a gap in a count; a policy failing is a value going somewhere a
    client told us it must not go. The first is logged and moved past, the
    second refuses the document.

    Carries the policy version and nothing about the document, so it can be
    logged and returned to a caller without leaking anything (invariant I1).
    """


class StorageProvider(Protocol):
    """Protocol for storage provider (S3 + short-term storage)."""
    
    async def persist_original(
        self,
        ingestion_id: UUID,
        content: Union[str, bytes],
        scope: Any,
    ) -> None:
        """Persist original content to S3."""
        ...
    
    async def persist_short_term(
        self,
        ingestion_id: UUID,
        content: str,
        scope: Any,
        session_id: str,
        qualification_result: Optional[Dict[str, Any]],
    ) -> None:
        """Persist directly to short-term storage."""
        ...
    
    async def get_short_term_by_session(
        self,
        session_id: str,
        scope: Any,
    ) -> List[Dict[str, Any]]:
        """Fetch short-term memories for a session."""
        ...


class IngestionManager:
    """
    Manages incoming ingestion requests and orchestrates the pipeline.

    This Class Handles:
    - Validate requests
    - Check rate limits
    - Ensure idempotency
    - Enqueue for pipeline processing
    - Track status
    - Handle direct storage (pre-qualified short-term from Listening Agent)

    Pipeline Workers (orchestrated via queues):
    - Stage 1: Type/Source/Nature qualification
    - Stage 2+3: Categorization + LLM-driven memory extraction
    - Stage 4: Organize and persist to Vector/Graph/RDBMS

    Together, the Manager + Workers form the complete Ingestion Manager system.
    """

    def __init__(
        self,
        queue: IngestionQueue,
        idempotency: IdempotencyChecker,
        rate_limiter: RateLimiter,
        status_tracker: StatusTracker,
        validator: RequestValidator,
        storage_provider: StorageProvider,
        preprocessor: Optional[Stage0Preprocessor] = None,
    ):
        """Initialize ingestion manager.

        Args:
            queue: Queue for pipeline processing
            idempotency: Idempotency checker
            rate_limiter: Rate limiter
            status_tracker: Status tracker
            validator: Request validator
            storage_provider: Storage provider for S3 and short-term storage
            preprocessor: Stage 0 preprocessor for smart document chunking
        """
        self._queue = queue
        self._idempotency = idempotency
        self._rate_limiter = rate_limiter
        self._status = status_tracker
        self._validator = validator
        self._storage = storage_provider
        self._preprocessor = preprocessor or Stage0Preprocessor()
        # None when no usage tracker is registered (Core): tracking is skipped.
        self._usage_tracker = edition.call("ingestion_usage_tracking")

    async def _chunking_for(self, request: IngestDocumentRequest):
        """The instance's ``chunking_config`` if it is allowed to steer Stage 0.

        None for everything else, which is the fixed rules and the default:
        see :func:`maca_chunking_enabled_for` for why this is opt-in. A failed
        lookup is None too; chunking must never be what stops an ingest.
        """
        instance_id = request.scope.instance_id
        if not maca_chunking_enabled_for(instance_id):
            return None
        try:
            from synap.cloud.memory_config.loader import get_config_loader
            config = await get_config_loader().get_config(str(instance_id))
            return getattr(config, "chunking_config", None)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "chunking config lookup failed for instance=%s (%s); using the fixed rules",
                instance_id, exc)
            return None

    async def _enforce_user_daily_cap(self, request: IngestDocumentRequest) -> None:
        """Refuse a submission once the user has used
        ``guardrails.rate_limits.max_memories_per_user_per_day`` today.

        Off unless SYNAP_GUARDRAIL_INGEST_CAP is on for the instance, and then
        the config is not even read when the request names no user. The unit is
        a request, not an extracted memory (see RateLimiter.check_user_day).
        Only a value written in the config counts: the schema's own default of
        100 is a placeholder nobody chose and must not cap every user.
        A limiter must never take ingestion down, so any failure allows the
        request and logs a warning. Raises RateLimitExceeded, the same error
        the per-client limiter raises.
        """
        instance_id = request.scope.instance_id
        user_id = getattr(request.scope, "user_id", None)
        if user_id is None or not enabled_for("SYNAP_GUARDRAIL_INGEST_CAP", instance_id):
            return
        try:
            from synap.cloud.memory_config.loader import get_config_loader
            config = await get_config_loader().get_config(str(instance_id))
            limits = config.guardrails.rate_limits
            cap = limits.max_memories_per_user_per_day
            if "max_memories_per_user_per_day" not in limits.model_fields_set \
                    or not cap or cap <= 0:
                return
            result = await self._rate_limiter.check_user_day(instance_id, user_id, cap)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ingestion_user_cap_unchecked instance_id=%s (%s); allowing the request",
                instance_id, exc)
            return
        if not result.allowed:
            logger.warning(
                "ingestion_user_cap_reached request_id=%s client_id=%s instance_id=%s retry_after=%s",
                request.request_id, request.scope.client_id, instance_id, result.retry_after)
            raise RateLimitExceeded(result.retry_after)

    async def _check_rate_limit(self, request: IngestDocumentRequest) -> None:
        """Count one request against the client's per-minute limit; raise RateLimitExceeded when over."""
        async with start_span("ingestion.entry.rate_limit", {
            "synap.request_id": request.request_id or "",
            "synap.client_id": str(request.scope.client_id),
            "synap.instance_id": str(request.scope.instance_id),
        }):
            rate_result = await self._rate_limiter.check(
                client_id=request.scope.client_id,
                instance_id=request.scope.instance_id,
            )
            set_current_span_attributes({
                "synap.allowed": rate_result.allowed,
                "synap.retry_after": rate_result.retry_after or 0,
            })
        if not rate_result.allowed:
            logger.warning(
                "ingestion_rate_limited request_id=%s client_id=%s instance_id=%s retry_after=%s",
                request.request_id,
                request.scope.client_id,
                request.scope.instance_id,
                rate_result.retry_after,
            )
            raise RateLimitExceeded(rate_result.retry_after)
        logger.info(
            "ingestion_rate_limit_passed request_id=%s client_id=%s instance_id=%s",
            request.request_id,
            request.scope.client_id,
            request.scope.instance_id,
        )

    async def ingest_document(
        self,
        request: IngestDocumentRequest,
        skip_per_minute_check: bool = False,
        _chunk_index: Optional[int] = None,
    ) -> IngestResponse:
        """
        Ingest a single document.

        Flow:
        1. Validate request
        2. Check rate limits
        3. Check idempotency (request ID + content hash)
        4. Persist original to S3 (async, immediate)
        5. Enqueue for pipeline processing
        6. Return response with ingestion_id

        Args:
            request: Document ingestion request

        Returns:
            IngestResponse with ingestion_id and status

        Raises:
            ValidationError: If validation fails
            RateLimitExceeded: If rate limit exceeded
        """
        import time as _time

        _span_name = "ingestion.entry" if _chunk_index is None else "ingestion.entry.chunk"
        _span_attrs: Dict[str, Any] = {
            "synap.request_id": request.request_id or "",
            "synap.priority": request.priority.value if hasattr(request.priority, "value") else str(request.priority),
            "synap.document_type": request.document_type.value if hasattr(request.document_type, "value") else str(request.document_type),
            "synap.source": request.source.value if hasattr(request.source, "value") else str(request.source),
            "synap.content_len": len(request.content) if request.content else 0,
            "synap.client_id": str(request.scope.client_id),
            "synap.instance_id": str(request.scope.instance_id),
        }
        if _chunk_index is not None:
            _span_attrs["synap.chunk_index"] = _chunk_index

        async with start_span(_span_name, _span_attrs):
            _t_entry = _time.monotonic()

            # Step logging for request
            step_logger = StepLogger("IngestionManager.ingest_document")
            step_logger.log_request(
                payload={
                    "request_id": request.request_id,
                    "document_id": request.document_id,
                    "document_type": request.document_type.value if hasattr(request.document_type, 'value') else str(request.document_type),
                    "source": request.source.value if hasattr(request.source, 'value') else str(request.source),
                    "priority": request.priority.value if hasattr(request.priority, 'value') else str(request.priority),
                    # No content preview. This log line carried the first
                    # 500 characters of whatever the client sent, which is
                    # raw personal data in a log sink (PII Protection,
                    # invariant I1). Length and type are enough to debug a
                    # malformed payload.
                    "content_kind": "text" if isinstance(request.content, str) else "binary",
                    "content_len": len(request.content) if request.content else 0,
                    "scope": {
                        "client_id": request.scope.client_id,
                        "instance_id": request.scope.instance_id,
                        "customer_id": request.scope.customer_id if hasattr(request.scope, 'customer_id') else None,
                        "user_id": request.scope.user_id if hasattr(request.scope, 'user_id') else None,
                    },
                    "metadata": request.metadata,
                },
                correlation_id=request.request_id,
            )

            # 1. Validate
            async with start_span("ingestion.entry.validate", {
                "synap.request_id": request.request_id or "",
            }):
                validation_result = await self._validator.validate(request)
                set_current_span_attributes({
                    "synap.is_valid": validation_result.is_valid,
                    "synap.error_count": len(validation_result.errors),
                })
            if not validation_result.is_valid:
                logger.warning(
                    "ingestion_validate_failed request_id=%s client_id=%s instance_id=%s errors=%s",
                    request.request_id,
                    request.scope.client_id,
                    request.scope.instance_id,
                    validation_result.errors,
                )
                raise ValidationError("; ".join(validation_result.errors))

            # Per-user daily cap. Before Stage 0 and outside the skip flag on
            # purpose: a split document returns below before step 2, and the
            # vision route takes its client slot itself and passes the skip
            # flag, so step 2 would miss both. `_chunk_index is None` is the
            # request as the caller sent it, counted once, not once per chunk.
            if _chunk_index is None:
                await self._enforce_user_daily_cap(request)

            # 1a. The ingress gate (PII Protection).
            #
            # Here, and not later, for three reasons. It is before Stage 0, so
            # the gate sees the whole document and the consistency pass can
            # match a value that appears with context in one chunk and bare in
            # another (feature spec 9.4). It is before the queue, so it is
            # before anything reaches a model (invariant I7). And it is inside
            # the `_chunk_index is None` guard because this method recurses
            # once per chunk, and running the gate again per chunk would both
            # double-count and defeat the whole-document view.
            #
            # In watch mode this changes nothing. It reads, it counts, it returns
            # the same characters. A failure here must not stop a client's
            # ingest, which is why `inspect` never raises and why the result
            # is only recorded.
            if _chunk_index is None and request.content:
                await self._run_ingress_gate(request)

            # 1b. Stage 0: Smart chunking — split large/multi-session documents
            sub_requests = self._preprocessor.preprocess(
                request, chunking=await self._chunking_for(request))
            if len(sub_requests) > 1:
                logger.info(
                    "ingestion_stage0_split request_id=%s chunks=%d",
                    request.request_id,
                    len(sub_requests),
                )
                # This used to return the first chunk's response, so the caller
                # polled one chunk of a document over the chunk budget (3000
                # tokens by default) and saw "completed" while the rest were
                # queued; with no document_id given, every chunk minted its own.
                # Now one document_id, and one ingestion_id whose status is the
                # chunks' combined status.
                # One count for the request, as for an unsplit one; the chunks
                # below are queued with the per-minute check skipped.
                if not skip_per_minute_check:
                    await self._check_rate_limit(request)
                content_hash = self._compute_content_hash(request.content)
                existing = await self._idempotency.check(
                    request_id=request.request_id,
                    content_hash=content_hash,
                    scope=request.scope,
                )
                if existing:
                    return existing
                document_id = request.document_id or str(uuid4())
                # The copy of the text is saved once, under the id the caller gets back,
                # which is also the id an uploaded file's original is saved under. The
                # chunks do not save a copy of their own.
                parent_id = uuid4()
                await self._persist_original_async(
                    ingestion_id=parent_id,
                    content=request.content,
                    scope=request.scope,
                    request_id=request.request_id,
                )
                children = []
                for i, sub_req in enumerate(sub_requests):
                    sub_req.document_id = document_id
                    children.append(await self.ingest_document(
                        sub_req, skip_per_minute_check=True, _chunk_index=i))
                response = IngestResponse(
                    request_id=request.request_id,
                    ingestion_id=parent_id,
                    status=IngestStatus.QUEUED,
                    document_id=document_id,
                    queued_at=datetime.now(timezone.utc),
                )
                await self._status.create_parent(
                    response.ingestion_id, response, [c.ingestion_id for c in children])
                await self._idempotency.store(
                    request.request_id, response,
                    content_hash=content_hash, scope=request.scope,
                )
                return response
            logger.info(
                "ingestion_validate_passed request_id=%s document_id=%s priority=%s client_id=%s instance_id=%s",
                request.request_id,
                request.document_id,
                request.priority.value,
                request.scope.client_id,
                request.scope.instance_id,
            )

            # 2. Rate limit check (skipped for batch-internal calls where daily quota is pre-validated)
            if not skip_per_minute_check:
                await self._check_rate_limit(request)

            # 3. Idempotency check + atomic reservation
            async with start_span("ingestion.entry.idempotency_check", {
                "synap.request_id": request.request_id or "",
            }):
                content_hash = self._compute_content_hash(request.content)
                existing = await self._idempotency.check(
                    request_id=request.request_id,
                    content_hash=content_hash,
                    scope=request.scope,
                )
                if existing:
                    set_current_span_attributes({
                        "synap.cache_hit": True,
                        "synap.content_hash_prefix": content_hash[:12],
                    })
                    logger.info(
                        "ingestion_idempotency_hit request_id=%s ingestion_id=%s document_id=%s",
                        request.request_id,
                        existing.ingestion_id,
                        existing.document_id,
                    )
                    return existing  # Return cached response

                reserved = await self._idempotency.try_reserve(content_hash, request.scope)
                set_current_span_attributes({
                    "synap.cache_hit": False,
                    "synap.reserved": reserved,
                    "synap.content_hash_prefix": content_hash[:12],
                })
                if not reserved:
                    existing = await self._idempotency.check(
                        request_id=request.request_id,
                        content_hash=content_hash,
                        scope=request.scope,
                    )
                    if existing:
                        return existing
                    # Reservation held but not yet complete — proceed (rare race, idempotent outcome)

            logger.info(
                "ingestion_idempotency_miss request_id=%s content_hash_prefix=%s",
                request.request_id,
                content_hash[:12],
            )

            # Generate IDs
            ingestion_id = uuid4()
            document_id = request.document_id or str(uuid4())
            set_current_span_attributes({"synap.ingestion_id": str(ingestion_id)})
            logger.info(
                "ingestion_ids_allocated request_id=%s ingestion_id=%s document_id=%s",
                request.request_id,
                ingestion_id,
                document_id,
            )

            # 4. Persist original to S3 (async fire-and-forget — no span, does not block)
            if _chunk_index is None:
                await self._persist_original_async(
                    ingestion_id=ingestion_id,
                    content=request.content,
                    scope=request.scope,
                    request_id=request.request_id,
                )

            # 5. Enqueue for processing
            async with start_span("ingestion.entry.enqueue", {
                "synap.ingestion_id": str(ingestion_id),
                "synap.priority": request.priority.value if hasattr(request.priority, "value") else str(request.priority),
            }):
                await self._queue.enqueue(
                    ingestion_id=ingestion_id,
                    request=request,
                    priority=request.priority,
                )
            logger.info(
                "ingestion_enqueued request_id=%s ingestion_id=%s priority=%s",
                request.request_id,
                ingestion_id,
                request.priority.value,
            )

            # 6. Create and track response
            response = IngestResponse(
                request_id=request.request_id,
                ingestion_id=ingestion_id,
                status=IngestStatus.QUEUED,
                document_id=document_id,
                queued_at=datetime.now(timezone.utc),
            )

            await self._status.create(ingestion_id, response)
            logger.info(
                "ingestion_status_created request_id=%s ingestion_id=%s status=%s",
                request.request_id,
                ingestion_id,
                response.status.value,
            )
            await self._idempotency.store(
                request.request_id,
                response,
                content_hash=content_hash,
                scope=request.scope,
            )
            logger.info(
                "ingestion_idempotency_stored request_id=%s ingestion_id=%s",
                request.request_id,
                ingestion_id,
            )

            set_current_span_attributes({
                "synap.total_entry_latency_ms": round((_time.monotonic() - _t_entry) * 1000, 2),
                "synap.final_status": response.status.value,
            })

            # Step logging for response
            step_logger.log_response(
                response={
                    "request_id": response.request_id,
                    "ingestion_id": str(response.ingestion_id),
                    "status": response.status.value if hasattr(response.status, 'value') else str(response.status),
                    "document_id": response.document_id,
                    "queued_at": response.queued_at.isoformat() if response.queued_at else None,
                },
                status_code=200,
                success=True,
                correlation_id=request.request_id,
            )

            return response

    async def treat_for_storage(
        self, text: str, scope: Any, *, request_id: str
    ) -> "StorageTreatment":
        """The text as an ingest would hold it after the PII gate, plus the verdict.

        For copies kept outside the pipeline (the whole-conversation backup).
        It runs the same gate and the same client policy as `ingest_document`.

        It never refuses. When treatment fails the original text comes back with
        `failed_at` saying where, and the caller stores both, so a copy that is
        less protected than the pipeline's is marked as such instead of being
        indistinguishable from one that is not. `failed_at` is "policy" (the
        client's approved policy could not be applied), "detector" (a recogniser
        threw, so some values may have been missed) or "gate" (anything else).
        """
        out = await edition.acall("backup_pii_step", self, text, scope, request_id=request_id)
        # No PII feature registered (Core): the text is kept as it came.
        return out if out is not None else StorageTreatment(text=text, applied=False)

    async def _run_ingress_gate(self, request: IngestDocumentRequest) -> None:
        """Look at the whole document before anything else touches it.

        Runs off the event loop because it is CPU-bound: regular expressions
        over a document that can be a hundred kilobytes. Blocking the loop for
        that would slow every other request on the worker.

        What it then does depends entirely on one row of data.

        With **no approved policy** it counts and changes nothing, except the
        floor if floor enforcement is on. That is watch mode and floor enforcement, and it is
        what every client gets until they have configured something.

        With **an approved policy** it applies that policy: values become
        aliases, and what continues down the pipeline is the model view. From
        here the rest of the system never sees the original characters, which is
        the point. This is aliasing and it is the first thing in the feature that
        a client turned on themselves.

        Swallows everything **except** a failure while a policy is in force.
        Watch mode has no right to break a client's ingest. Aliasing does: if a
        client asked us not to send a value to a model and we cannot honour
        that, continuing anyway is the one outcome the whole feature exists to
        prevent, so the document is refused and the caller decides.
        """
        await edition.acall("pii_ingestion", "ingress_gate", self, request)

    async def ingest_batch(
        self,
        request: IngestBatchRequest,
    ) -> IngestBatchResponse:
        """
        Ingest multiple documents.

        - Validates entire batch first
        - Enqueues each document with BOOTSTRAP priority
        - Returns batch response with per-document status
        
        Args:
            request: Batch ingestion request
            
        Returns:
            IngestBatchResponse with per-document results
            
        Raises:
            RateLimitExceeded: If rate limit exceeded for batch
        """
        # Validate all documents
        for doc in request.documents:
            doc.scope = request.scope  # Ensure shared scope
            doc.priority = request.priority

        # Rate limit for batch
        rate_result = await self._rate_limiter.check_batch(
            client_id=request.scope.client_id,
            count=len(request.documents),
        )
        if not rate_result.allowed:
            # The single-document path above logs this rejection; the batch path did not, so a
            # client rejected only on batches was invisible to the rate-limit signal, which reads
            # this line. Same event name and same fields, so one query covers both paths.
            logger.warning(
                "ingestion_rate_limited request_id=%s client_id=%s instance_id=%s retry_after=%s batch=%d",
                getattr(request, "request_id", "-"),
                request.scope.client_id,
                request.scope.instance_id,
                rate_result.retry_after,
                len(request.documents),
            )
            raise RateLimitExceeded(rate_result.retry_after)

        batch_id = uuid4()
        results: List[IngestResponse] = []

        for doc in request.documents:
            try:
                result = await self.ingest_document(doc)
                results.append(result)
            except Exception as e:
                if request.fail_fast:
                    raise
                # Record failure but continue
                results.append(IngestResponse(
                    request_id=doc.request_id,
                    ingestion_id=uuid4(),
                    status=IngestStatus.FAILED,
                    document_id=doc.document_id or "unknown",
                    queued_at=datetime.now(timezone.utc),
                    error_message=str(e),
                ))

        succeeded = sum(1 for r in results if r.status == IngestStatus.QUEUED)
        failed = len(results) - succeeded

        return IngestBatchResponse(
            request_id=request.request_id,
            batch_id=batch_id,
            status=IngestStatus.PARTIAL_SUCCESS if failed > 0 else IngestStatus.QUEUED,
            total_documents=len(results),
            succeeded=succeeded,
            failed=failed,
            document_results=results,
            queued_at=datetime.now(timezone.utc),
        )

    async def ingest_from_listening_agent(
        self,
        request: IngestFromListeningAgentRequest,
    ) -> IngestResponse:
        """
        Handle ingestion from Listening Agent.

        Two paths:
        - pre_qualified=True, destination="short_term": Direct storage (no pipeline)
        - pre_qualified=False OR destination="long_term": Full pipeline
        
        Args:
            request: Listening agent ingestion request
            
        Returns:
            IngestResponse
        """
        if request.pre_qualified and request.destination == "short_term":
            # Direct storage path - no LLM needed
            return await self._persist_short_term_direct(request)
        else:
            # Full pipeline
            doc_request = IngestDocumentRequest(
                request_id=request.request_id,
                content=request.content,
                document_type=DocumentType.CONVERSATION,
                source=DocumentSource.LISTENING_AGENT,
                scope=request.scope,
                priority=request.priority,
                metadata={
                    "session_id": request.session_id,
                    "conversation_id": request.conversation_id,
                    "content_type": request.content_type,
                },
            )
            return await self.ingest_document(doc_request)

    async def convert_session_to_longterm(
        self,
        request: ConvertSessionRequest,
    ) -> IngestResponse:
        """
        Convert short-term session memories to long-term.

        Called when a session ends.
        - Fetches all short-term memories for session
        - Enqueues for full pipeline processing
        - Marks short-term for deletion after conversion
        
        Args:
            request: Session conversion request
            
        Returns:
            IngestResponse
        """
        # Fetch short-term memories for session
        short_term_memories = await self._storage.get_short_term_by_session(
            session_id=request.session_id,
            scope=request.scope,
        )

        if not short_term_memories:
            return IngestResponse(
                request_id=request.request_id,
                ingestion_id=uuid4(),
                status=IngestStatus.COMPLETED,
                document_id=f"session:{request.session_id}",
                queued_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                memories_created=0,
            )

        # Create aggregated content for pipeline
        aggregated_content = self._aggregate_session_content(short_term_memories)

        doc_request = IngestDocumentRequest(
            request_id=request.request_id,
            content=aggregated_content,
            document_type=DocumentType.CONVERSATION,
            source=DocumentSource.LISTENING_AGENT,
            scope=request.scope,
            priority=request.priority,
            metadata={
                "session_id": request.session_id,
                "conversion": "short_term_to_long_term",
                "original_memory_count": len(short_term_memories),
            },
        )

        return await self.ingest_document(doc_request)

    async def get_status(self, ingestion_id: UUID) -> Optional[IngestResponse]:
        """Get current status of an ingestion job.
        
        Args:
            ingestion_id: Ingestion ID to query
            
        Returns:
            Current IngestResponse or None if not found
        """
        return await self._status.get(ingestion_id)

    async def _persist_short_term_direct(
        self,
        request: IngestFromListeningAgentRequest,
    ) -> IngestResponse:
        """
        Direct persistence for pre-qualified short-term content.

        Bypasses the full pipeline since Listening Agent already:
        - Qualified the content
        - Compacted it
        - Determined it's short-term worthy
        
        Args:
            request: Listening agent request
            
        Returns:
            IngestResponse with completed status
        """
        ingestion_id = uuid4()

        # Persist directly to short-term storage
        await self._storage.persist_short_term(
            ingestion_id=ingestion_id,
            content=request.content,
            scope=request.scope,
            session_id=request.session_id,
            qualification_result=request.qualification_result,
        )
        if self._usage_tracker is not None:
            try:
                await self._usage_tracker.track_memory_stored(
                    client_id=str(request.scope.client_id),
                    instance_id=str(request.scope.instance_id),
                    memory_count=1,
                    user_id=str(request.scope.user_id) if request.scope.user_id else None,
                    conversation_id=request.conversation_id,
                    endpoint="ingestion.short_term_direct",
                    correlation_id=request.request_id,
                )
                logger.info(
                    "cloud_usage_track_memories_stored source=ingestion.short_term_direct memory_count=1 client_id=%s instance_id=%s user_id=%s correlation_id=%s",
                    request.scope.client_id,
                    request.scope.instance_id,
                    request.scope.user_id,
                    request.request_id,
                )
            except Exception as e:
                logger.warning(
                    "cloud_usage_track_memories_stored_failed source=ingestion.short_term_direct client_id=%s instance_id=%s correlation_id=%s error=%s",
                    request.scope.client_id,
                    request.scope.instance_id,
                    request.request_id,
                    e,
                )

        # Activation analytics emit. This Listening-Agent path
        # BYPASSES Stage4, so without instrumenting it here, listening-driven
        # customers under-count memory_stored. Shares the same seam +
        # first_memory_stored milestone. persist_unit_id = ingestion_id (one
        # short-term persist per call). Cloud-only.
        try:
            await edition.acall(
                "ingestion_analytics",
                "memory_stored",
                client_id=str(request.scope.client_id),
                instance_id=str(request.scope.instance_id),
                scope_user_id=str(request.scope.user_id) if request.scope.user_id else None,
                conversation_id=request.conversation_id,
                correlation_id=request.request_id,
                persist_unit_id=str(ingestion_id),
                milestone_key="first_memory_stored",
                properties={"memory_count": 1, "mode": "short_term_direct"},
            )
        except Exception:  # noqa: BLE001 — analytics never fails a persist
            logger.debug("short_term_direct activation emit raised (non-fatal)", exc_info=True)

        return IngestResponse(
            request_id=request.request_id,
            ingestion_id=ingestion_id,
            status=IngestStatus.COMPLETED,
            document_id=f"shortterm:{request.session_id}:{ingestion_id}",
            queued_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            memories_created=1,  # Direct persist = 1 short-term memory
        )

    def _compute_content_hash(self, content: Union[str, bytes]) -> str:
        """SHA-256 of the content with whitespace normalised, for deduplication."""
        return normalized_content_hash(content)

    async def _persist_original_async(
        self,
        ingestion_id: UUID,
        content: Union[str, bytes],
        scope: Any,
        request_id: Optional[str] = None,
    ) -> None:
        """Persist original content to S3 (fire and forget, with retry).

        Args:
            ingestion_id: Ingestion ID
            content: Content to persist
            scope: Request scope
            request_id: Optional request ID for logging
        """
        asyncio.create_task(
            self._persist_original_with_retry(ingestion_id, content, scope, request_id)
        )
        logger.info(
            "ingestion_original_persist_scheduled request_id=%s ingestion_id=%s",
            request_id,
            ingestion_id,
        )

    async def _persist_original_with_retry(
        self,
        ingestion_id: UUID,
        content: Union[str, bytes],
        scope: Any,
        request_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        """Persist original content with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                await self._storage.persist_original(
                    ingestion_id=ingestion_id,
                    content=content,
                    scope=scope,
                )
                logger.info(
                    "ingestion_original_persist_completed request_id=%s ingestion_id=%s",
                    request_id,
                    ingestion_id,
                )
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning(
                        "ingestion_original_persist_retry request_id=%s ingestion_id=%s attempt=%d error=%s",
                        request_id,
                        ingestion_id,
                        attempt + 1,
                        e,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "ingestion_original_persist_failed request_id=%s ingestion_id=%s error=%s",
                        request_id,
                        ingestion_id,
                        e,
                    )

    def _aggregate_session_content(
        self,
        short_term_memories: List[Dict[str, Any]],
    ) -> str:
        """Aggregate short-term memories into content for pipeline.
        
        Args:
            short_term_memories: List of short-term memory dicts
            
        Returns:
            Aggregated content string
        """
        # Simple concatenation with separators
        parts = []
        for memory in short_term_memories:
            content = memory.get("content", "")
            if content:
                parts.append(content)
        
        return "\n\n---\n\n".join(parts)

    async def trigger_post_batch_consolidation(
        self,
        batch_id: UUID,
        instance_id: str,
        client_id: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post-batch deduplication. Currently unused — call explicitly after batch completion.

        TODO: Wire into batch_create response or provide SDK method to trigger.

        This should be called by the SDK or application layer after verifying
        that all documents in a batch have completed processing.

        Args:
            batch_id: Batch identifier
            instance_id: Instance to deduplicate
            client_id: Client ID for vector namespace resolution
            user_id: Optional - scope to single user
            customer_id: Optional - scope to customer

        Returns:
            Consolidation result
        """
        # RETIRED (audit 2026-09-07, decision log D-023): this hook constructed
        # the legacy ConsolidationManager, which hard-deleted memories
        # (DELETE FROM memories) outside the reversible executors and with no
        # trail. It never had a caller outside tests. Consolidation now runs in its
        # own worker (synap/cloud/application/consolidation); ingestion must not
        # route through this path. Fail loud rather than start the old engine.
        raise RuntimeError(
            "trigger_post_batch_consolidation is retired (D-023): it started a "
            "hard-deleting legacy engine. Use synap/cloud/application/consolidation.")
        return await edition.acall(
            "consolidation_links", "post_batch",
            batch_id=batch_id, instance_id=instance_id, client_id=client_id,
            user_id=user_id, customer_id=customer_id,
        )
