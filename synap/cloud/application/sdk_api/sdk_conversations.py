"""SDK conversation tracking routes.

Provides endpoints for the SDK to explicitly record conversation messages
when not using the gRPC streaming path. Messages are persisted to the
conversation / conversation_session / message Postgres tables, enabling
the compaction worker to discover and process them.

Also hosts the one-shot transcript ingest endpoint
(POST /v1/conversations/ingest — spec
Synap_Async_Integration_and_User_Profile_Spec_v1 §4): idempotent transcript
push + extraction enqueue + transcript-push summary compaction.
"""

import asyncio
import hashlib
import json
import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from synap.cloud.agents.ingestion.models import (
    DocumentSource,
    DocumentType as IngestionDocumentType,
    IngestDocumentRequest,
    IngestPriority,
)
from synap.cloud.agents.ingestion.pipeline.stage0_preprocessor import TURN_PATTERN
from synap.cloud.application.sdk_api.sdk_auth import (
    SDKAuthContext,
    get_sdk_auth,
)
from synap.cloud.application.sdk_api.readiness import require_instance_ready
# Reuse the memory routes' scope resolution (B2C collapse via MACA
# user_context_isolation) and their lazily-built core IngestionManager —
# the transcript push must land in exactly the same scope/queue as
# sdk.memories.create() would.
from synap.cloud.shared.scope import collapse_b2c_identity
from synap.cloud.application.sdk_api.sdk_memories import (
    AttachmentResult, _resolve_scope,
    get_ingestion_manager,
)
from synap.cloud.application.conversation.tracker import ConversationTracker
from synap.cloud.application.ingestion import IngestionManager as CoreIngestionManager
from synap.cloud.shared.hook_signals import log_rate_limited
from synap.cloud.application.models.errors import RateLimitExceeded, ValidationError
from synap.cloud.database.connection import get_session_factory
from synap.cloud.shared.conversation_id_norm import coerce_conversation_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/conversations", tags=["SDK Conversations"])

# Lazily-constructed singleton. Built on first use (after app startup) so it can
# be wired with the shared Redis client. record_message only buffers raw turns to
# Redis (`conv_messages:{conversation_id}`) — which powers
# `conversation_context.recent_turns` — when its Redis client is set. The
# gRPC/playground tracker is wired the same way in main.py; without this the SDK
# record path persists to Postgres only and recent_turns stays empty.
_tracker: Optional[ConversationTracker] = None


def _get_tracker() -> ConversationTracker:
    """Return the shared tracker, wiring the Redis client on first use.

    Self-heals: if a prior build happened before Redis was available, rebuild
    once Redis appears so the raw-turn buffer (and thus recent_turns) starts
    populating. Falls back to Postgres-only (no buffer) if Redis is absent.
    """
    global _tracker
    if _tracker is not None and getattr(_tracker, "_redis", None) is not None:
        return _tracker

    redis_client = None
    try:
        from synap.cloud.application.shared.providers import get_cached_providers
        redis_client = get_cached_providers().get("redis")
    except Exception:  # noqa: BLE001 — degrade to Postgres-only if providers unavailable
        logger.warning(
            "Conversation tracker: shared Redis provider unavailable; recent_turns "
            "buffering disabled (messages still persist to Postgres).",
            exc_info=True,
        )

    if _tracker is None or getattr(_tracker, "_redis", None) is not redis_client:
        _tracker = ConversationTracker(redis_client=redis_client)
    return _tracker


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageContentPart(BaseModel):
    type: Literal["image"]
    filename: Optional[str] = None
    media_type: Optional[str] = None
    # base64. Read by the vision reader and dropped; never stored.
    data: str


ContentPart = Annotated[
    Union[TextContentPart, ImageContentPart], Field(discriminator="type")
]


class RecordMessageRequest(BaseModel):
    conversation_id: str
    role: str = Field(..., pattern="^(user|assistant)$")
    # A string, as always, or an ordered list of text and image parts. A list
    # is rendered to one string (images read by the vision reader) before the
    # tracker sees it, so what is stored is still text.
    content: Union[str, List[ContentPart]]
    session_id: Optional[str] = None
    user_id: str
    # Optional: on B2C instances (user_context_isolation == "equals_customer")
    # the server collapses customer_id := user_id, matching /conversations/ingest,
    # memories.create, and the fetch routes. B2B instances must supply it.
    customer_id: Optional[str] = None
    # The rung this conversation belongs to, named in full.
    #
    # `/conversations/ingest` takes one already, but that is the one-shot
    # transcript path. This is the streaming path, and it is most real traffic.
    # Without it the conversation row can only say customer and user, so on a
    # ladder with a rung between them every memory extracted from a streamed
    # conversation landed at the tenant.
    #
    # Sent once per conversation is enough: `ensure_conversation` fills a
    # missing path and never overwrites one, so a caller can put it on the
    # first turn, on every turn, or on neither.
    scope: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Optional client-supplied message time (e.g. replayed/imported turns).
    # Absent -> server now(UTC), exactly the previous behavior.
    timestamp: Optional[datetime] = None

    @field_validator("conversation_id", "session_id")
    @classmethod
    def coerce_to_uuid(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            _uuid.UUID(v)  # Already a valid UUID, use as-is
        except (ValueError, AttributeError):
            # Convert non-UUID strings to deterministic UUIDv5
            v = str(_uuid.uuid5(_uuid.NAMESPACE_URL, v))
        return v


class RecordMessageResponse(BaseModel):
    message_id: str
    conversation_id: str
    session_id: str
    recorded_at: datetime
    # Messages the conversation holds after this write. The SDK compares it with
    # its local short-term cache: a match means the cache is complete and
    # get_context_for_prompt can answer without a round trip. None when the
    # count could not be read, which the SDK treats as "ask the server".
    conversation_message_count: Optional[int] = None
    # One result per image, in request order. A failed read still records the
    # surrounding turn, but callers need to know that the image was not read.
    attachments: List[AttachmentResult] = Field(default_factory=list)


class RecordBatchRequest(BaseModel):
    messages: List[RecordMessageRequest]


class BatchError(BaseModel):
    index: int
    conversation_id: str
    error: str


class RecordBatchResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: List[RecordMessageResponse]
    errors: List[BatchError] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

async def _render_content_parts(
    parts: List[Any], auth: SDKAuthContext,
) -> tuple[str, List[Any]]:
    """A list of content parts as the one string the tracker records.

    Every image is validated before any is read (400 / 413 / 415). An image the
    vision model cannot read is recorded as a placeholder rather than failing
    the turn: the conversation happened, and the text around it still counts.
    """
    from synap.cloud.application.ingestion import multimodal as mm

    try:
        wire = mm.parts_from_wire([p.model_dump() for p in parts])
    except mm.ImagePartError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    rendered, reads = await mm.read_and_render(
        wire, client_id=auth.client_id, instance_id=auth.instance_id,
        operation="conversation.record_message.vision",
    )
    return rendered, reads


@router.post("/messages", response_model=RecordMessageResponse, dependencies=[Depends(require_instance_ready)])
async def record_message(
    request: RecordMessageRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
):
    """Record a single conversation message.

    Creates the conversation and session rows if they don't exist,
    then inserts the message. The migration-011 trigger automatically
    updates conversation stats (last_message_at, message_count,
    approximate_token_count).
    """
    user_id = request.user_id
    tracker = _get_tracker()
    # Do this before an image read, which may take up to the vision timeout.
    # The client-supplied timestamp remains authoritative for replayed turns.
    received_at = request.timestamp or datetime.now(timezone.utc)

    await _b2c_check(auth, request.customer_id,
                     where="POST /v1/conversations/messages", user_id=user_id)

    # B2C collapses customer := user, and it does so whether or not the caller
    # ALSO sent a customer_id. Only filling in a missing one left the row
    # carrying the caller's customer while `_fetch_user_summary` looks the
    # conversation up under the user id, so a B2C caller sending both never got
    # their own summaries back.
    isolation = await _load_user_context_isolation(auth.instance_id)
    is_b2c = isolation == "equals_customer"
    _, customer_id = collapse_b2c_identity(
        user_id, request.customer_id, is_b2c=is_b2c,
        where="sdk_conversations.record_message",
    )
    if customer_id is None:
        raise HTTPException(
            status_code=400,
            detail="customer_id is required on B2B instances",
        )

    content = request.content
    reads = []
    if not isinstance(content, str):
        content, reads = await _render_content_parts(content, auth)

    try:
        await tracker.ensure_conversation(
            conversation_id=request.conversation_id,
            instance_id=auth.instance_id,
            user_id=user_id,
            customer_id=customer_id,
            scope_path=request.scope,
        )

        session_id = await tracker.ensure_session(
            conversation_id=request.conversation_id,
            user_id=user_id,
            session_id=request.session_id,
        )

        message_id = await tracker.record_message(
            conversation_id=request.conversation_id,
            session_id=session_id,
            role=request.role,
            content=content,
            metadata=request.metadata or None,
            timestamp=received_at,
            # Without these the tracker applies the floor and stops, so a
            # client's own settings governed the memory and not the turn the
            # memory came from. The parameters have always been there; nothing
            # passed them.
            client_id=auth.client_id,
            instance_id=auth.instance_id,
            # Scope for the alias key, so the turn and the memory agree on
            # which label a value has.
            customer_id=customer_id,
            user_id=user_id,
        )

        return RecordMessageResponse(
            message_id=message_id,
            conversation_id=request.conversation_id,
            session_id=session_id,
            recorded_at=datetime.now(timezone.utc),
            conversation_message_count=await tracker.count_messages(
                request.conversation_id
            ),
            attachments=[AttachmentResult(**read.attachment()) for read in reads],
        )

    except Exception as e:
        logger.exception("Failed to record conversation message")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record message: {str(e)}",
        )


@router.post("/messages/batch", response_model=RecordBatchResponse, dependencies=[Depends(require_instance_ready)])
async def record_messages_batch(
    request: RecordBatchRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
):
    """Record multiple conversation messages in a batch.

    Each message is processed independently; failures in one message
    do not block others.
    """
    results: List[RecordMessageResponse] = []
    errors: List[BatchError] = []
    succeeded = 0
    failed = 0

    for i, msg in enumerate(request.messages):
        try:
            resp = await record_message(msg, auth)
            results.append(resp)
            succeeded += 1
        except HTTPException as e:
            failed += 1
            errors.append(BatchError(index=i, conversation_id=msg.conversation_id, error=e.detail))
        except Exception as e:
            logger.exception("Batch message recording failed")
            failed += 1
            errors.append(BatchError(index=i, conversation_id=msg.conversation_id, error=str(e)))

    return RecordBatchResponse(
        total=len(request.messages),
        succeeded=succeeded,
        failed=failed,
        results=results,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# One-shot transcript ingest (spec §4)
# ---------------------------------------------------------------------------

MAX_TRANSCRIPT_TURNS = 2000
MAX_JSON_FIELD_BYTES = 64 * 1024  # analysis / metadata cap (spec §4.4 step 1)


class TranscriptTurn(BaseModel):
    """A single transcript turn (spec §4.1 — the preferred typed form)."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[datetime] = None  # turn time; server time if absent
    speaker: Optional[str] = None         # display label, e.g. "Agent Priya"
    metadata: Optional[Dict[str, Any]] = None  # persisted on the message row


class TranscriptIngestRequest(BaseModel):
    """Request body for POST /v1/conversations/ingest (spec §4.1/§4.2).

    ``conversation_id`` is the client's RAW id — it is UUID5-coerced in the
    route (NOT in a validator) because the raw form must be persisted as
    ``external_conversation_id`` so it round-trips back to the client.
    """
    conversation_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    # The rung this transcript's memories belong to, named in full.
    #
    # `memories.create` gained this first; this route did not, and it is the
    # surface most integrations actually write through. On a ladder with a rung
    # between the tenant and the identity, two ids cannot say which one a
    # transcript belongs to, so every memory extracted from it was refused.
    #
    # Optional. Omitting it behaves exactly as before.
    scope: Optional[Dict[str, str]] = None
    transcript: Union[str, List[TranscriptTurn]]
    customer_id: Optional[str] = None
    conversation_type: Optional[str] = Field(default=None, max_length=64)
    analysis: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    @field_validator("transcript")
    @classmethod
    def cap_turn_count(cls, v):
        if isinstance(v, list) and len(v) > MAX_TRANSCRIPT_TURNS:
            raise ValueError(
                f"transcript exceeds the {MAX_TRANSCRIPT_TURNS}-turn limit"
            )
        return v

    @field_validator("analysis", "metadata")
    @classmethod
    def cap_json_size(cls, v, info):
        if v is not None:
            size = len(json.dumps(v, ensure_ascii=True, default=str).encode("utf-8"))
            if size > MAX_JSON_FIELD_BYTES:
                raise ValueError(
                    f"{info.field_name} exceeds the {MAX_JSON_FIELD_BYTES // 1024}KB limit"
                )
        return v


class TranscriptIngestResponse(BaseModel):
    """Response for POST /v1/conversations/ingest (spec §4.1 + review
    finding 5: external_conversation_id echoed so the id round-trips)."""
    conversation_id: str            # coerced (UUID form)
    external_conversation_id: str   # the raw client-supplied id
    ingestion_id: str               # poll via sdk.memories.status()/wait
    status: Literal["queued", "duplicate"]
    turns_recorded: int
    summary_status: Literal["in_progress", "already_compacted", "skipped"]
    queued_at: datetime
    # Purely informational. Additive (defaults empty) so no existing client
    # parsing this response can break, and nothing about the ingest changes.
    # Populated when the transcript carries no event time at all (see
    # _TIME_BLIND_WARNING): those memories get document_created_at = the ingest
    # minute, so recency ranking flattens to a constant and consolidation has no
    # order to supersede along. The client cannot see that from a 200, and the
    # only downstream symptom is answers that feel stale — so we say it here, at
    # the one moment the integrator who can fix it is looking.
    warnings: List[str] = Field(default_factory=list)


# --- transcript helpers -----------------------------------------------------

_TIME_BLIND_WARNING = (
    "no event time supplied — memories from this conversation cannot be ordered "
    "by when things happened; recency ranking will treat them as all equally "
    "recent. Send `ended_at` (or per-turn timestamps) to enable it."
)


def _conversation_said_at(started_at, ended_at,
                          turns: List[Dict[str, Any]]) -> Optional[datetime]:
    """When the client says this conversation happened, or None if they never
    said. This is the document's own date (DCT): the day the words were said,
    and the anchor every relative reference in them resolves against.

    It is NOT our clock. It used to be: the call site read
    ``request.ended_at or queued_at``, so a client that sent no ``ended_at``
    had the ingest minute stored as the day they spoke. A fifth of
    stored memories carried such a value, and it leaves the system,
    because the read path serves this field as ``said_at``. The extraction time
    already has its own field, ``IngestDocumentRequest.ingested_at``, which is
    never served; it does not need a second home in this one.

    ``ended_at`` first because it is the end of the conversation and so the
    latest moment anything in it was said. The latest turn timestamp next, as
    the same thing derived rather than declared. ``started_at`` last: it is a
    true time the client gave us, just the furthest from the end. Turn
    timestamps are only ever client-supplied -- this route normalises their
    timezone and never fills one in -- so none of the three can be our clock.

    None rather than a substitute, because a memory honestly carrying no date
    can be recognised and handled, while one carrying today's date cannot."""
    if ended_at is not None:
        return ended_at
    stamps = [t["timestamp"] for t in turns if t.get("timestamp") is not None]
    if stamps:
        return max(stamps)
    return started_at


def _time_blind_warnings(said_at: Optional[datetime]) -> List[str]:
    """One warning iff the transcript carries no event time anywhere, which is
    exactly when `_conversation_said_at` came back None and the memories will
    be stored with no document date at all.

    Keyed on the resolved value on purpose. It used to re-test the three inputs
    itself, and its test was the AND of all three while the fallback it warned
    about fired on ``ended_at`` alone -- so a client sending ``started_at`` and
    per-turn times was silently stamped with the ingest minute and told
    nothing. One value, read by both, cannot drift like that again."""
    return [] if said_at is not None else [_TIME_BLIND_WARNING]


def _split_string_transcript(raw_text: str) -> List[Dict[str, Any]]:
    """Split a plain-string transcript into turn dicts.

    Uses the Stage-0 turn grammar (``TURN_PATTERN``: user|assistant|human|
    ai|system|bot labels). Labels outside the storage enum map like the
    tracker does: user/human -> user, everything else -> assistant. Text
    before the first label (if any) is preserved as a user turn; a string
    with no labels at all becomes a single user turn.
    """
    matches = list(TURN_PATTERN.finditer(raw_text))
    if not matches:
        return [{"role": "user", "content": raw_text.strip()}]

    turns: List[Dict[str, Any]] = []
    preamble = raw_text[: matches[0].start()].strip()
    if preamble:
        turns.append({"role": "user", "content": preamble})
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        content = raw_text[m.end():end].strip()
        if not content:
            continue
        label = m.group(1).lower()
        role = "user" if label in ("user", "human") else "assistant"
        turns.append({"role": role, "content": content})
    return turns


def _canonical_transcript_hash(turns: List[Dict[str, Any]]) -> str:
    """sha256 over the canonicalized ordered turns (role+content+timestamp).

    The idempotency gate (spec §4.4 step 2): same turns -> same hash, so a
    re-push is detected regardless of transport form. speaker/metadata are
    deliberately excluded — they are presentation details, not content.
    """
    canonical = [
        {
            "role": t.get("role"),
            "content": t.get("content"),
            "timestamp": t["timestamp"].isoformat() if t.get("timestamp") else None,
        }
        for t in turns
    ]
    payload = json.dumps(
        canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _transcript_document_id(coerced_cid: str, transcript_hash: str) -> str:
    """Deterministic document id (spec §4.4 step 4 — the compaction-promotion
    precedent): retries of the same push enqueue the same document."""
    digest = hashlib.sha256(
        f"transcript:{coerced_cid}:{transcript_hash}".encode("utf-8")
    ).hexdigest()
    return "transcript_" + digest[:24]


def _render_transcript_document(
    turns: List[Dict[str, Any]],
    coerced_cid: str,
    conversation_type: Optional[str],
    started_at: Optional[datetime],
    ended_at: Optional[datetime],
) -> str:
    """Render turns as the canonical extraction document.

    The header carries conversation_id ON PURPOSE: the ingestion manager's
    7-day content-hash dedupe is content-only, so byte-identical transcripts
    on DIFFERENT conversations (voicemails, wrong numbers) would otherwise
    be silently swallowed (adversarial review, backend finding 4).
    """
    header = (
        "[CONVERSATION TRANSCRIPT"
        f" conversation_id={coerced_cid}"
        f" type={conversation_type or 'unknown'}"
        f" started_at={started_at.isoformat() if started_at else 'unknown'}"
        f" ended_at={ended_at.isoformat() if ended_at else 'unknown'}]"
    )
    lines = [header, ""]
    lines.extend(f"{t['role']}: {t['content']}" for t in turns)
    return "\n".join(lines)


# --- DB lookups (module-level so tests can patch them) ----------------------

async def _load_conversation_gate_row(coerced_cid: str):
    """Row for the idempotency gate; None when the conversation is new."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text("""
                SELECT transcript_hash, message_count, last_ingestion_id,
                       external_conversation_id
                FROM conversation
                WHERE conversation_id = :cid
            """),
            {"cid": coerced_cid},
        )
        return result.first()


async def _has_compacted_context(coerced_cid: str) -> bool:
    """True when a compacted summary row already exists (any version)."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT 1 FROM compacted_contexts "
                "WHERE conversation_id = :cid LIMIT 1"
            ),
            {"cid": coerced_cid},
        )
        return result.first() is not None


async def _find_unsummarized_transcript_conversation(
    instance_id: str, user_id: str, exclude_cid: str
):
    """One prior transcript-push conversation of this user whose summary
    compaction never landed.

    Transcript rows are status='completed', which the idle sweeper skips,
    so a failed fire-and-forget summary is otherwise never retried
    (adversarial review, backend finding 16). Returns
    (conversation_id, user_id, customer_id) or None.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text("""
                SELECT c.conversation_id, c.user_id, c.customer_id
                FROM conversation c
                WHERE c.instance_id = :iid
                  AND c.user_id = :uid
                  AND c.transcript_hash IS NOT NULL
                  AND c.conversation_id != :exclude_cid
                  AND NOT EXISTS (
                      SELECT 1 FROM compacted_contexts cc
                      WHERE cc.conversation_id = c.conversation_id
                  )
                ORDER BY c.last_message_at DESC NULLS LAST
                LIMIT 1
            """),
            {"iid": instance_id, "uid": user_id, "exclude_cid": exclude_cid},
        )
        return result.first()


async def _b2c_check(auth: SDKAuthContext, customer_id, *, where: str, user_id=None) -> None:
    """B2C single-identifier contract. See shared.b2c_contract for the reasoning."""
    from synap.cloud.shared.b2c_contract import check_customer_id
    await check_customer_id(auth.instance_id, customer_id, where=where, user_id=user_id)


async def _load_user_context_isolation(instance_id: str) -> str:
    """MACA user_context_isolation, defaulting to strict (B2B) — the same
    fallback _resolve_scope uses."""
    try:
        from synap.cloud.application.client_api.retrieval_manager import _load_maca_config
        maca = await _load_maca_config(instance_id)
        if maca:
            return maca.user_context_isolation
    except Exception as e:
        logger.debug(f"MACA load failed during transcript scope resolution: {e}")
    return "strict"


# --- transcript-push summary compaction (spec §4.4 step 5) ------------------

# Strong references so fire-and-forget tasks aren't GC'd mid-flight.
_background_tasks: set = set()


def _spawn(coro) -> None:
    """asyncio.create_task wrapper (patchable in tests)."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_transcript_compaction(
    manager,
    storage,
    conversation_id: str,
    instance_id: str,
    user_id: Optional[str],
    customer_id: Optional[str],
    client_id: Optional[str],
) -> None:
    """Body of the fire-and-forget summary compaction task.

    Caller must have ALREADY acquired the ``compacting:{cid}`` Redis lock
    (manager.compact never touches it — every caller acquires/clears
    explicitly). Mirrors sdk_compaction.compact_conversation's
    ``_compact_async``: failure marker in Redis + finally-clear of the lock.
    """
    try:
        from synap.cloud.application.compaction import CompactionTriggerType

        messages = await storage.load_conversation_messages(conversation_id)
        if not messages:
            logger.warning(
                f"Transcript-push compaction: no messages found for "
                f"{conversation_id}, skipping"
            )
            return

        # force=True is LOAD-BEARING: with force=False and no prior
        # compaction, manager.compact overwrites the caller's trigger_type
        # with its own detection (manager.py should_compact fallback) — the
        # promotion guard's trigger_type signal would be lost.
        result = await manager.compact(
            conversation_id=conversation_id,
            instance_id=instance_id,
            messages=messages,
            trigger_type=CompactionTriggerType.TRANSCRIPT_PUSH,
            force=True,
            user_id=user_id,
            customer_id=customer_id,
            client_id=client_id,
        )
        logger.info(
            f"Transcript-push compaction completed for {conversation_id}: "
            f"compression_ratio={result.metadata.compression_ratio:.2f}"
        )
    except Exception as e:
        logger.error(
            f"Transcript-push compaction failed for {conversation_id}: {e}",
            exc_info=True,
        )
        # Store failure in Redis so get_compaction_status() can report it
        try:
            r = await storage._get_redis()
            failure_key = f"compaction_failed:{conversation_id}"
            await r.setex(failure_key, 3600, json.dumps({
                "error_message": str(e),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }))
        except Exception:
            pass  # Best-effort failure tracking
    finally:
        await storage.clear_compaction_in_progress(conversation_id)


async def _kickoff_transcript_summary(
    coerced_cid: str,
    instance_id: str,
    user_id: Optional[str],
    customer_id: Optional[str],
    client_id: Optional[str],
) -> str:
    """Kick off the summary compaction; returns the response summary_status.

    Acquires the ``compacting:{cid}`` lock atomically (SET NX); if it is
    already held another compaction is running -> "already_compacted".
    """
    from synap.cloud.application.compaction import get_compaction_manager

    manager = get_compaction_manager()
    storage = manager.storage
    if not await storage.set_compaction_in_progress(coerced_cid):
        logger.debug(
            f"Transcript-push compaction lock not acquired for {coerced_cid}, skipping"
        )
        return "already_compacted"
    _spawn(_run_transcript_compaction(
        manager, storage, coerced_cid, instance_id, user_id, customer_id, client_id,
    ))
    return "in_progress"


async def _retry_missing_summary_for_user(
    instance_id: str,
    user_id: str,
    client_id: Optional[str],
    exclude_cid: str,
) -> None:
    """Re-fire the summary compaction for at most ONE prior transcript-push
    conversation of this user whose summary never landed.

    Best-effort and fully non-fatal: transcript conversations are
    status='completed' (invisible to the idle sweeper), so this piggybacked
    retry is the only path that heals a failed fire-and-forget summary.
    """
    try:
        row = await _find_unsummarized_transcript_conversation(
            instance_id, user_id, exclude_cid
        )
        if not row:
            return
        other_cid, other_user, other_customer = row[0], row[1], row[2]

        from synap.cloud.application.compaction import get_compaction_manager
        manager = get_compaction_manager()
        storage = manager.storage
        if not await storage.set_compaction_in_progress(other_cid):
            return
        logger.info(
            f"Retrying missing transcript summary for {other_cid} "
            f"(piggybacked on ingest of {exclude_cid})"
        )
        _spawn(_run_transcript_compaction(
            manager, storage, other_cid, instance_id,
            other_user, other_customer, client_id,
        ))
    except Exception as e:
        logger.warning(f"Transcript summary retry sweep failed (non-fatal): {e}")


@router.post("/ingest", response_model=TranscriptIngestResponse, dependencies=[Depends(require_instance_ready)])
async def ingest_transcript(
    request: TranscriptIngestRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    ingestion_manager: CoreIngestionManager = Depends(get_ingestion_manager),
):
    """One-shot conversation transcript ingest (spec §4).

    Records the full transcript, enqueues extraction through the core
    ingestion pipeline, and fires a transcript-push summary compaction —
    all idempotently:

    - same conversation_id + same transcript -> 200 ``status="duplicate"``
      (and the extraction is re-enqueued iff the original enqueue never
      landed — crash-retry hole);
    - a conversation whose turns were STREAMED, then a transcript pushed at
      the end of the call -> 200 ``status="queued"`` with
      ``turns_recorded=0`` and a warning. The transcript becomes the
      extraction document; the streamed turns are left exactly as they are.

      This was a 409 telling the caller to use a new conversation_id, which
      produced two conversations holding the same call, each extracted
      separately. Accepting it costs a re-extraction of content we have
      already seen, and inline KU in stage 4 adjudicates that against what is
      stored rather than appending a second copy.
    """
    external_cid = request.conversation_id
    coerced_cid = coerce_conversation_id(external_cid)

    # ---- 1. Normalize transcript into turn dicts (spec §4.4 step 1) ----
    if isinstance(request.transcript, str):
        raw_text = request.transcript.strip()
        if not raw_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="transcript is empty",
            )
        turns = _split_string_transcript(raw_text)
        if len(turns) > MAX_TRANSCRIPT_TURNS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"transcript exceeds the {MAX_TRANSCRIPT_TURNS}-turn limit",
            )
    else:
        turns = [t.model_dump() for t in request.transcript]

    if not turns or all(not (t.get("content") or "").strip() for t in turns):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="transcript is empty",
        )

    # Naive client timestamps are taken as UTC (also keeps the canonical
    # hash and the tracker's timestamp sort consistent).
    for t in turns:
        ts = t.get("timestamp")
        if ts is not None and ts.tzinfo is None:
            t["timestamp"] = ts.replace(tzinfo=timezone.utc)

    transcript_hash = _canonical_transcript_hash(turns)
    # Informational: does this payload carry any event time at all? (§ clock)
    conversation_said_at = _conversation_said_at(
        request.started_at, request.ended_at, turns)
    ingest_warnings = _time_blind_warnings(conversation_said_at)

    # ---- 2. Resolve the effective external customer id ----
    # conversation.customer_id is NOT NULL: B2C (equals_customer) collapses
    # customer := user (mirrors sdk_memories._resolve_scope); B2B requires
    # an explicit customer_id.
    await _b2c_check(auth, request.customer_id,
                     where="POST /v1/conversations/ingest", user_id=request.user_id)

    isolation = await _load_user_context_isolation(auth.instance_id)
    _, effective_customer_id = collapse_b2c_identity(
        request.user_id, request.customer_id,
        is_b2c=isolation == "equals_customer",
        where="sdk_conversations.ingest_transcript",
    )
    if not effective_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "customer_id is required on B2B instances "
                "(user_context_isolation != equals_customer)"
            ),
        )

    document_id = _transcript_document_id(coerced_cid, transcript_hash)
    queued_at = datetime.now(timezone.utc)
    tracker = _get_tracker()

    # Resolved ONCE, here, rather than inside `_enqueue_extraction`.
    #
    # It used to be a local of that closure, which meant two things. The
    # conversation row could not see it, so a transcript ingested at a deep
    # rung filed its memories correctly and left the row with no path, and
    # everything that later reads the rung back off the conversation lost it.
    # And a scope the instance rejects raised its 400 at step 5, AFTER the
    # transcript had been written, leaving a stored conversation behind a
    # failed request. Resolving here fixes both: the row and the memories are
    # guaranteed to agree because they come from the same value, and a bad
    # scope is refused before anything is stored.
    try:
        scope = await _resolve_scope(
            auth, request.user_id, effective_customer_id, request.scope)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )

    async def _enqueue_extraction() -> str:
        """Build + enqueue the extraction document; returns ingestion_id.

        Deterministic document_id + the ingestion pipeline's server-side
        idempotency make re-enqueues safe.
        """
        metadata: Dict[str, Any] = {
            "conversation_id": coerced_cid,
            "external_conversation_id": external_cid,
            "conversation_type": request.conversation_type,
            "source": "transcript_push",
        }
        if request.analysis is not None:
            metadata["analysis"] = request.analysis
        # Mirror sdk_memories._to_ingest_document_request's trace injection.
        metadata["_trace"] = {
            "correlation_id": auth.correlation_id,
            "credential_id": str(auth.credential_id),
        }
        doc_request = IngestDocumentRequest(
            document_id=document_id,
            content=_render_transcript_document(
                turns, coerced_cid, request.conversation_type,
                request.started_at, request.ended_at,
            ),
            document_type=IngestionDocumentType.CONVERSATION,
            source=DocumentSource.API,
            document_created_at=conversation_said_at,
            scope=scope,
            priority=IngestPriority.REALTIME,
            metadata=metadata,
        )
        result = await ingestion_manager.ingest_document(doc_request)
        return str(result.ingestion_id)

    try:
        # ---- 3. Idempotency gate (spec §4.4 step 2) ----
        row = await _load_conversation_gate_row(coerced_cid)
        if row is not None:
            existing_hash = row[0]
            message_count = row[1] or 0
            last_ingestion_id = row[2]
            row_external_cid = row[3]

            if existing_hash == transcript_hash:
                # Duplicate re-push: free retry. Re-enqueue extraction iff
                # the original enqueue never landed (crash between record
                # and enqueue left last_ingestion_id NULL).
                if last_ingestion_id is None:
                    ingestion_id = await _enqueue_extraction()
                    await tracker.set_last_ingestion_id(coerced_cid, ingestion_id)
                else:
                    ingestion_id = str(last_ingestion_id)

                if await _has_compacted_context(coerced_cid):
                    summary_status = "already_compacted"
                else:
                    # The original summary never landed — heal it now.
                    summary_status = await _kickoff_transcript_summary(
                        coerced_cid, auth.instance_id, request.user_id,
                        effective_customer_id, auth.client_id,
                    )
                await _retry_missing_summary_for_user(
                    auth.instance_id, request.user_id, auth.client_id, coerced_cid
                )
                return TranscriptIngestResponse(
                    conversation_id=coerced_cid,
                    external_conversation_id=row_external_cid or external_cid,
                    ingestion_id=ingestion_id,
                    status="duplicate",
                    turns_recorded=message_count,
                    summary_status=summary_status,
                    queued_at=queued_at,
                    warnings=ingest_warnings,
                )

            if message_count > 0:
                # A transcript for a conversation that already has messages.
                #
                # This was a 409 whose message told the caller to use a new
                # conversation_id, and following that advice is the worst
                # available outcome: a second conversation holding the same
                # call, extracted twice, with nothing tying the two together.
                #
                # The shape that produces it is ordinary. A voice client
                # streams turns live so anticipation has something to work
                # with, and its speech system produces the corrected, diarised
                # transcript only after the call ends. Both are real, and the
                # second one is the better document: it also carries
                # `analysis`, `conversation_type` and explicit start and end
                # times, none of which the streaming path can express.
                #
                # So it is accepted. What it does NOT do is rewrite history:
                # the streamed turns stay exactly as they are. Re-extracting
                # content we have already seen is safe because inline KU in
                # stage 4 adjudicates against what is already stored rather
                # than appending a second copy (`stage4_organization.py`,
                # source="inline_ku").
                #
                # The turns are not inserted a second time either, and that is
                # the one judgement call here: the client is telling us what
                # the call WAS, not adding to it, so duplicating every turn in
                # the message table would corrupt the recency ordering and the
                # compaction input for nothing. The conversation row is
                # finalized, the transcript becomes the extraction document,
                # and `transcript_push_promotion_skip` stops a later compaction
                # from promoting the same content a third time.
                ingest_warnings = list(ingest_warnings) + [
                    "conversation already had streamed turns; the transcript "
                    "was accepted for extraction and the existing turns were "
                    "left in place"
                ]
                await tracker.finalize_transcript(
                    conversation_id=coerced_cid,
                    conversation_type=request.conversation_type,
                    analysis=request.analysis,
                    conversation_metadata=request.metadata,
                    transcript_hash=transcript_hash,
                    started_at=request.started_at,
                    ended_at=request.ended_at,
                )
                ingestion_id = await _enqueue_extraction()
                await tracker.set_last_ingestion_id(coerced_cid, ingestion_id)
                return TranscriptIngestResponse(
                    conversation_id=coerced_cid,
                    external_conversation_id=row_external_cid or external_cid,
                    ingestion_id=ingestion_id,
                    status="queued",
                    turns_recorded=0,
                    summary_status=(
                        "already_compacted"
                        if await _has_compacted_context(coerced_cid) else "skipped"
                    ),
                    queued_at=queued_at,
                    warnings=ingest_warnings,
                )
            # Row exists but has no messages (e.g. ensure-only) -> treat as
            # fresh: record the transcript and overwrite the stale hash.

        # ---- 4. Record conversation + messages (spec §4.4 step 3) ----
        record = await tracker.record_transcript(
            conversation_id=external_cid,
            instance_id=auth.instance_id,
            user_id=request.user_id,
            customer_id=effective_customer_id,
            turns=turns,
            conversation_type=request.conversation_type,
            analysis=request.analysis,
            conversation_metadata=request.metadata,
            transcript_hash=transcript_hash,
            started_at=request.started_at,
            ended_at=request.ended_at,
            external_conversation_id=external_cid,
            # See `record_message` above: the turn path needs the client to
            # apply anything beyond the floor, and this call omitted it.
            client_id=auth.client_id,
            # The caller's PATH (`request.scope`, level_key -> id), not the
            # resolved `Scope` object. The column holds a path: it is what
            # `record_message` and the gRPC path store, what
            # `_resolve_scope_from_conversation` hands back, and what
            # `resolve_identity` re-resolves on the read with the same B2C
            # collapse and MACA that `_resolve_scope` applied above, so the
            # row and the memories agree by construction. Handing the
            # resolved object here (a pydantic model) made
            # `ensure_conversation`'s json.dumps raise on every transcript
            # ingest: a 500 before the first row was written, for a day. The
            # path is still checked before anything is stored, because
            # `_resolve_scope` above refuses a bad one with a 400.
            scope_path=request.scope,
        )

        # ---- 5. Enqueue extraction, then persist the handle ----
        # Write order matters (crash-safety): commit txn (step 4) ->
        # enqueue -> UPDATE last_ingestion_id. A crash between 4 and 5 is
        # healed by the duplicate branch's NULL-handle re-enqueue.
        ingestion_id = await _enqueue_extraction()
        await tracker.set_last_ingestion_id(coerced_cid, ingestion_id)

        # ---- 6. Fire-and-forget summary compaction (spec §4.4 step 5) ----
        if record["turns_recorded"] == 0:
            summary_status = "skipped"
        else:
            summary_status = await _kickoff_transcript_summary(
                coerced_cid, auth.instance_id, request.user_id,
                effective_customer_id, auth.client_id,
            )
        await _retry_missing_summary_for_user(
            auth.instance_id, request.user_id, auth.client_id, coerced_cid
        )

        # ---- 7. Respond (spec §4.1) ----
        return TranscriptIngestResponse(
            conversation_id=coerced_cid,
            external_conversation_id=external_cid,
            ingestion_id=ingestion_id,
            status="queued",
            turns_recorded=record["turns_recorded"],
            summary_status=summary_status,
            queued_at=queued_at,
            warnings=ingest_warnings,
        )

    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=getattr(exc, "message", str(exc)),
        )
    except RateLimitExceeded as exc:
        log_rate_limited(auth, key_id=str(auth.credential_id),
                         limit="ingestion", client_id=str(auth.client_id))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=getattr(exc, "message", str(exc)),
            headers={"Retry-After": str(getattr(exc, "retry_after", 1))},
        )
    except Exception as e:
        logger.exception(
            "Transcript ingest failed for conversation %s", external_cid
        )
        # Not the exception text. That is ours, and for a day it put a
        # Python type name on a client's console. The correlation id is what
        # the caller needs to quote back, and it is on the logger.exception
        # line above.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Transcript ingest failed. Reference id: "
                f"{auth.correlation_id}"
            ),
        )
