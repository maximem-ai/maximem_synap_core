"""SDK memory CRUD routes.

The SDK ingestion endpoints are wired to the core IngestionManager pipeline.
"""

import json
import logging
import os
import time
import uuid as _uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.agents.ingestion.models import (
    DocumentSource,
    DocumentType as IngestionDocumentType,
    IngestDocumentRequest,
    IngestPriority,
    Scope,
    ScopeRung,
)
from synap.cloud.application.ingestion import (
    IdempotencyChecker,
    IngestionManager as CoreIngestionManager,
    IngestionQueue,
    RateLimiter,
    RequestValidator,
    StatusTracker,
)
from synap.cloud.application.sdk_api.document_extractor import (
    ACCEPTED_UPLOAD_EXTENSIONS,
    extract_text,
)
from synap.cloud.database.connection import get_async_session
from synap.cloud.shared.hook_signals import log_rate_limited
from synap.cloud.application.models.errors import RateLimitExceeded, ValidationError
from synap.cloud.application.sdk_api.sdk_auth import (
    SDKAuthContext,
    get_sdk_auth,
)
from synap.cloud.application.sdk_api.readiness import require_instance_ready
from synap.cloud.observability.tracing import start_span, set_current_span_attributes
from synap.cloud.application.ingestion.storage_provider import (
    FileStoreStorageProvider,
    backup_uploaded_file,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/memories", tags=["SDK Memories"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateMemoryRequest(BaseModel):
    document: Union[str, Dict[str, Any]]
    document_type: str = "ai-chat-conversation"
    document_id: Optional[str] = None
    document_created_at: Optional[datetime] = None
    # Scope IDs are optional — the effective scope level is derived from which
    # combination is passed and the instance's user_context_isolation (B2C/B2B).
    # See synap.cloud.shared.scope.derive_scope_level.
    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    # The rung this write belongs to, named in full, one entry per level down
    # to it: {"customer": "acme", "team": "payments", "user": "dana"}.
    #
    # Reads have carried this since 0.4.6 and writes could not, which meant a
    # client could define a level and never file anything on it. On a ladder
    # with a level between the tenant and the identity, `user_id` and
    # `customer_id` cannot say which one a write belongs to, so the write was
    # refused with a message asking for a path this model had no field for.
    #
    # Optional, and omitting it behaves exactly as before. Sending one on an
    # instance without nested scoping is an error rather than ignored: the
    # caller asked to be narrowed, and ignoring them widens.
    scope: Optional[Dict[str, str]] = None
    mode: str = "long-range"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Why the caller is writing this ("conversation", "bootstrap",
    # "knowledge-base", "record", "profile", or free text). Stored as
    # `metadata["_objective"]` and nothing more yet. Absent unless the caller
    # set it, so existing request bodies are unchanged.
    objective: Optional[str] = None


class CreateMemoryResponse(BaseModel):
    ingestion_id: str
    document_id: str
    status: str = "completed"
    queued_at: datetime
    error_message: Optional[str] = None


class AttachmentResult(BaseModel):
    filename: str
    media_type: str
    status: str  # "read" | "failed"
    error: Optional[str] = None


class CreateMemoryMultipartResponse(CreateMemoryResponse):
    # One per file part, in the order they appear in the manifest.
    attachments: List[AttachmentResult] = Field(default_factory=list)


class BatchCreateRequest(BaseModel):
    documents: List[CreateMemoryRequest]
    fail_fast: bool = False


class BatchCreateResponse(BaseModel):
    batch_id: str
    total: int
    succeeded: int
    failed: int
    results: List[CreateMemoryResponse]


class MemoryStatusResponse(BaseModel):
    ingestion_id: str
    document_id: str
    status: str
    queued_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    memories_created: int = 0
    error_message: Optional[str] = None
    memory_ids: List[str] = []


class MemoryResponse(BaseModel):
    memory_id: str
    memory_type: str
    content: str
    confidence: float = 0.0
    category: Optional[str] = None
    subcategory: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    source_document_id: Optional[str] = None


class UpdateMemoryRequest(BaseModel):
    document: Union[str, Dict[str, Any]]
    merge_strategy: str = "smart-merge"
    document_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DeleteMemoryResponse(BaseModel):
    memory_id: str
    deleted_at: datetime
    # Whether the record is gone from the SEARCH INDEX and not only from
    # Postgres. Those two could come apart and the caller was told the
    # good half: HTTP 200, Postgres row gone, vector
    # record still present, and the text still served on the next fetch.
    #
    # The cause is fixed below. This field is here because the cause was never
    # the only problem. The cleanup sat in a bare `except` that logged
    # "non-fatal" and returned success, so the NEXT failure of any kind would
    # be just as quiet. Someone deleting to meet a privacy obligation has to be
    # able to tell.
    #
    # Defaulted, so an SDK that does not know the field is unaffected.
    fully_removed: bool = True
    # Named for a person deciding whether they have met an obligation, not for
    # the stores. Empty when everything worked.
    still_present_in: List[str] = Field(default_factory=list)
    # True when the source document this memory came from was left in place
    # because other memories still rest on it. Not an error: the text is shared,
    # and removing it would strip the provenance from every other memory
    # extracted from the same document. False means either the source was
    # purged with the memory, or there was no source to purge.
    source_text_kept: bool = False
    # What the id named: "memory" (a saved memory), or a graph record served by
    # accurate mode: "relationship" (an edge, "the user works at Acme Corp") or
    # "entity" (a node). Defaulted so an SDK that does not know it is unaffected.
    kind: Literal["memory", "relationship", "entity"] = "memory"
    # For a graph record: the live memories that still say what the record said.
    # Deleting the record removes the graph line only; these memories are kept
    # and are still served (fast mode serves them directly). Delete them by id to
    # remove the fact everywhere. Empty for a memory.
    still_asserted_by: List[str] = Field(default_factory=list)
    # What the graph step changed. None when no graph step ran (no graph
    # configured, or the graph lifecycle is off). Counts are of records.
    graph_released: Optional["GraphReleased"] = None


class GraphReleased(BaseModel):
    edges_pruned: int = 0
    edges_retired: int = 0
    edges_deleted: int = 0
    nodes_pruned: int = 0
    nodes_retired: int = 0
    nodes_deleted: int = 0


DeleteMemoryResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from synap.cloud.shared.scope import parse_or_deterministic_uuid as _parse_or_deterministic_uuid  # noqa: E402


async def _b2c_check(auth: SDKAuthContext, customer_id, *, where: str, user_id=None) -> None:
    """B2C single-identifier contract. See shared.b2c_contract for the reasoning."""
    from synap.cloud.shared.b2c_contract import check_customer_id
    await check_customer_id(auth.instance_id, customer_id, where=where, user_id=user_id)


def parse_scope_path_form(raw: Optional[str]) -> Optional[Dict[str, str]]:
    """A scope path arriving as a multipart form field.

    The JSON routes take an object; this one is multipart and cannot, so the
    path comes as a JSON string. Refused loudly rather than ignored: a caller
    who asked to be narrowed and is ignored gets widened instead, which is the
    whole reason the JSON routes refuse a malformed path too.
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail='`scope` must be JSON, one entry per rung, for example {"customer":"acme"}.',
        )
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise HTTPException(
            status_code=400,
            detail="`scope` must be a flat object of level key to id, both strings.",
        )
    return {k: v for k, v in parsed.items() if v.strip()} or None


def _scope_of_stored_row(auth: SDKAuthContext, client_uuid, row) -> Scope:
    """The scope a STORED memory already has, built from its own ids.

    ⚠ Never `_resolve_scope` for this. That function maps EXTERNAL ids: it
    hands `customer_id` to `IdMapper.resolve_customer`, which returns
    `uuid5(client_uuid, external_id)`. A `memories` row holds the internal uuid
    already, so passing it there hashes a uuid a second time and produces a
    third uuid belonging to nobody.

    Two things went wrong every time that happened:

      * **The vector cleanup aimed at a collection that never existed.** The
        namespace is `synap_{customer_id}_v2`, so a wrong customer id is a
        wrong collection. The delete raised, the caller swallowed it as
        "non-fatal", and the client got a 200 while the record kept being
        served. The result was HTTP 200, the Postgres row gone, and the
        vector record still there.

      * **It invented a customer.** `resolve_customer` auto-creates on first
        sight, so each of these calls also wrote a `customer` row whose
        external id is a uuid string no client will ever send. Over time
        that left stray customer rows and memories filed under them.

    So this maps nothing at all. The row's ids ARE the internal ids.

    The level is read off which ids are present rather than passed through
    `derive_scope_level`, because that function answers a different question:
    where should a NEW write land, given what a caller sent and how the
    instance is configured. Here the record already exists and has an answer.
    """
    customer_uuid = getattr(row, "customer_id", None)
    user_uuid = getattr(row, "user_id", None)

    if customer_uuid is None:
        level = "client"
    elif user_uuid is None:
        level = "customer"
    else:
        level = "user"

    # `client_uuid` is handed in rather than derived again here. The caller
    # already used this exact value to FIND the row, so it provably matches
    # what the row carries, and deriving it a second time is a second chance to
    # derive it differently: some clients have a random client
    # uuid that derivation does not reproduce (N-14).
    #
    # `instance_id` is metadata on the record and plays no part in choosing a
    # collection, so the deterministic parse is right for it.
    return Scope(
        client_id=_uuid.UUID(str(client_uuid)),
        instance_id=_parse_or_deterministic_uuid(auth.instance_id),
        customer_id=_uuid.UUID(str(customer_uuid)) if customer_uuid else None,
        user_id=_uuid.UUID(str(user_uuid)) if user_uuid else None,
        scope_level=level,
    )


class ScopePathUnresolved(RuntimeError):
    """A write named a scope path that could not be resolved for a reason of
    ours (id mapping unavailable), so it was refused rather than widened."""


async def _resolve_scope(
    auth: SDKAuthContext,
    user_id: Optional[str],
    customer_id: Optional[str],
    scope_path: Optional[Dict[str, str]] = None,
) -> Scope:
    """Resolve SDK auth/request identifiers to ingestion Scope.

    Both ``user_id`` and ``customer_id`` are optional. The effective scope
    level is derived from which combination is passed and the instance's
    ``user_context_isolation`` (B2C/B2B) via ``derive_scope_level``:

    - B2C: either ID → user-scope (both ids collapse onto the ONE identity the
      request has, which is the user_id when one was sent; see
      `collapse_b2c_identity`); neither → client-scope.
    - B2B: both → user-scope; customer_id only → customer-scope; neither →
      client-scope; user_id without customer_id raises ValueError (400).
    """
    from synap.cloud.shared.scope import resolve_identity

    # Isolation is NOT resolved here any more. This block used to read the MACA
    # config itself and fall back to a hardcoded "strict" when there was none,
    # which is a second answer to a question `resolve_identity` already asks —
    # and the two disagreed for every instance whose MACA run had not finished.
    # Passing None means "I do not know", and the one authority decides. See
    # `scope.effective_user_context_isolation` for the trace.
    #
    # The SAME function the read path calls. This used to be a second
    # implementation that agreed with the read side by convention, and when it
    # stopped agreeing about B2C the write and the read addressed different
    # buckets while both returned 200.
    ident = await resolve_identity(
        auth.client_id, auth.instance_id, user_id, customer_id,
        None,
        # The write path has always degraded to derived ids rather than failing
        # the ingest outright. Preserved explicitly.
        derive_on_failure=True,
        # An explicit path is an instruction, not a hint: degrading three ids
        # to derived ones still files the record where the caller meant, while
        # degrading a path would file it somewhere they did not name.
        scope_path=scope_path,
    )
    # `derive_on_failure` above also skips the path: with no id mapping
    # `resolve_identity` resolves no chain at all, so a write to
    # `{"customer": "acme", "team": "payments"}` was filed at the customer,
    # readable by every user under it. Not the caller's input, so not a
    # ValueError (400): a RuntimeError is a 5xx the SDKs retry.
    if scope_path and not ident.mapped:
        raise ScopePathUnresolved(
            "The scope path could not be resolved right now, so the write was "
            "refused rather than filed at a wider scope. Retry it."
        )
    # `chain` is passed in the constructor, not attached afterwards. A Scope
    # assembled wholesale from one source is a copy, and a copy that omits a
    # field is not a copy: test_chain_call_sites.py fails the build for it,
    # because a dropped chain silently changes where the data lands.
    return Scope(
        client_id=ident.client_uuid,
        instance_id=ident.instance_uuid,
        customer_id=ident.customer_uuid,
        user_id=ident.user_uuid,
        scope_level=ident.scope_level,
        chain=[ScopeRung(**r) for r in ident.chain_rungs] if ident.chain_rungs else None,
    )


def _extract_ingestion_content(document: Union[str, Dict[str, Any]]) -> str:
    """Normalize SDK document payload into string content for ingestion."""
    if isinstance(document, str):
        return document
    if "text" in document and isinstance(document["text"], str):
        return document["text"]
    return json.dumps(document, ensure_ascii=True, sort_keys=True)


def _extract_content(document: Union[str, Dict[str, Any]]) -> dict:
    """Normalise document to JSONB content dict for CRUD handlers."""
    if isinstance(document, str):
        return {"text": document, "confidence": 0.8}
    if "text" not in document:
        document["text"] = str(document)
    if "confidence" not in document:
        document["confidence"] = 0.8
    return document


_DOC_TYPE_MAP: Dict[str, IngestionDocumentType] = {
    "ai-chat-conversation": IngestionDocumentType.CONVERSATION,
    "conversation": IngestionDocumentType.CONVERSATION,
    "document": IngestionDocumentType.DOCUMENT,
    "email": IngestionDocumentType.EMAIL,
    "pdf": IngestionDocumentType.PDF,
    "image": IngestionDocumentType.IMAGE,
    "audio": IngestionDocumentType.AUDIO,
    "meeting-transcript": IngestionDocumentType.MEETING_TRANSCRIPT,
    "meeting_transcript": IngestionDocumentType.MEETING_TRANSCRIPT,
    "internal-comm": IngestionDocumentType.INTERNAL_COMM,
    "internal_comm": IngestionDocumentType.INTERNAL_COMM,
}


MAX_OBJECTIVE_CHARS = 500


def _map_document_type(document_type: str) -> IngestionDocumentType:
    """Map SDK document types to ingestion enum values."""
    return _DOC_TYPE_MAP.get(document_type, IngestionDocumentType.DOCUMENT)


def _caller_metadata(metadata: Any) -> Dict[str, Any]:
    """The caller's metadata without the keys the pipeline writes for itself.

    The pipeline reads `_chunk` (Stage 0's chunk record; its
    `preceding_summary` is pasted into the extraction prompt) and `pii_*`
    (the ingress gate's verdict, stamped onto every memory) from this same
    dict, so a caller who sent them was believed. A non-object is dropped:
    the upload form used to 500 on a JSON list.
    """
    if not isinstance(metadata, dict):
        return {}
    return {
        k: v for k, v in metadata.items()
        if not (str(k).startswith("_") or str(k).startswith("pii_"))
    }


async def _to_ingest_document_request(
    request: CreateMemoryRequest,
    auth: SDKAuthContext,
    priority: IngestPriority,
    scope: Optional[Scope] = None,
) -> IngestDocumentRequest:
    """Map SDK request model to core ingestion document request.

    `scope` is for a caller that already resolved it from this same request.
    """
    metadata = _caller_metadata(request.metadata)
    metadata["_trace"] = {
        "correlation_id": auth.correlation_id,
        "credential_id": str(auth.credential_id),
    }
    if request.objective is not None:
        # ValueError is the 400 the create routes already map to.
        if len(request.objective) > MAX_OBJECTIVE_CHARS:
            raise ValueError(
                f"objective must be {MAX_OBJECTIVE_CHARS} characters or fewer."
            )
        metadata["_objective"] = request.objective
    return IngestDocumentRequest(
        document_id=request.document_id,
        content=_extract_ingestion_content(request.document),
        document_type=_map_document_type(request.document_type),
        source=DocumentSource.API,
        document_created_at=request.document_created_at,
        scope=scope if scope is not None else await _resolve_scope(
            auth, request.user_id, request.customer_id, request.scope),
        priority=priority,
        mode=request.mode,
        metadata=metadata,
    )


class _IngestionManagerContainer:
    """Lazy singleton container for ingestion dependencies."""

    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = Redis.from_url(redis_url, decode_responses=True)
        self.manager = CoreIngestionManager(
            queue=IngestionQueue(redis_client),
            idempotency=IdempotencyChecker(redis_client),
            rate_limiter=RateLimiter(redis_client),
            status_tracker=StatusTracker(redis_client),
            validator=RequestValidator(max_content_size_mb=10),
            storage_provider=FileStoreStorageProvider(),
        )


@lru_cache(maxsize=1)
def _get_manager_container() -> _IngestionManagerContainer:
    return _IngestionManagerContainer()


def get_ingestion_manager() -> CoreIngestionManager:
    """Dependency provider for the core ingestion manager."""
    return _get_manager_container().manager


@router.post("/upload", response_model=CreateMemoryResponse, dependencies=[Depends(require_instance_ready)])
async def sdk_upload_memory(
    user_id: str = Form(...),
    # Optional now. It was `Form(...)`, i.e. required, which on a B2C instance
    # DEMANDED the one field the contract forbids: a B2C client could not call
    # this route correctly at all.
    customer_id: Optional[str] = Form(None),
    relationship_type: str = Form(...),
    # The rung this document's memories belong to, as JSON, because this route
    # is multipart and cannot carry a nested object. Same contract as the JSON
    # routes: omit it and nothing changes.
    scope: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    document_type: Optional[str] = Form(None),
    mode: str = Form("long-range"),
    metadata: Optional[str] = Form(None),
    auth: SDKAuthContext = Depends(get_sdk_auth),
    ingestion_manager: CoreIngestionManager = Depends(get_ingestion_manager),
):
    """Upload a file or text and ingest it into the memory pipeline via SDK auth.

    Accepts the same file types as the dashboard upload endpoint. The client_id
    and instance_id are taken from the API key credential — developers do not
    pass them explicitly.
    """
    if relationship_type not in ("b2b", "b2c"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="relationship_type must be 'b2b' or 'b2c'",
        )

    if not file and not (text and text.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either a file upload or pasted text.",
        )

    MAX_UPLOAD_BYTES = 10 * 1024 * 1024

    raw: Optional[bytes] = None
    filename = ""
    if file:
        filename = file.filename or "upload"
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ACCEPTED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(ACCEPTED_UPLOAD_EXTENSIONS))}",
            )
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File exceeds the 10 MB limit.",
            )
        content_text = await extract_text(
            filename, raw, client_id=auth.client_id, instance_id=auth.instance_id)
    else:
        content_text = text.strip()

    if not content_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the uploaded content.",
        )

    await _b2c_check(auth, customer_id, where="POST /api/v1/memories/upload",
                     user_id=user_id)

    # The B2C collapse happens in _resolve_scope -> resolve_identity, from the
    # instance's own user_context_isolation, the same way /create and every
    # read do. It is NOT taken from `relationship_type`: both SDKs default that
    # to "b2c", so on a B2B instance an upload with the default had its
    # customer_id replaced with the user_id here and was filed under a customer
    # nobody reads. `relationship_type` is still validated above, and recorded
    # in the trace, but it no longer decides where a memory is filed.
    effective_user_id = user_id

    parsed_metadata: Dict[str, Any] = {}
    if metadata:
        try:
            import json as _json
            parsed_metadata = _caller_metadata(_json.loads(metadata))
        except Exception:
            pass

    parsed_metadata["_trace"] = {
        "correlation_id": auth.correlation_id,
        "credential_id": str(auth.credential_id),
        "relationship_type": relationship_type,
    }

    doc_type_str = document_type or "document"
    # A ValueError here is the caller's input: a B2B user with no customer, or
    # a scope path the ladder refuses (ScopePathRejected). Every other route
    # turns it into a 400; this one let it through as a 500, which the SDKs
    # read as a transient ServiceUnavailableError and retried.
    scope_path = parse_scope_path_form(scope)
    try:
        scope = await _resolve_scope(auth, effective_user_id, customer_id, scope_path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    ingest_req = IngestDocumentRequest(
        content=content_text,
        document_type=_map_document_type(doc_type_str),
        source=DocumentSource.UPLOADED,
        scope=scope,
        priority=IngestPriority.EXPLICIT,
        mode=mode,
        metadata=parsed_metadata,
    )

    logger.info(
        "sdk_upload client_id=%s user_id=%s customer_id=%s relationship_type=%s file=%s",
        auth.client_id, effective_user_id, customer_id, relationship_type,
        file.filename if file else "text",
    )

    try:
        result = await ingestion_manager.ingest_document(ingest_req)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message)
    except RateLimitExceeded as exc:
        log_rate_limited(auth, key_id=str(auth.credential_id),
                         limit="ingestion", client_id=str(auth.client_id))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=exc.message,
            headers={"Retry-After": str(exc.retry_after)},
        )
    except Exception as exc:
        logger.exception("sdk_upload_failed client_id=%s", auth.client_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {exc}",
        )

    # The manager's own backup holds the extracted text; keep the file as sent.
    await backup_uploaded_file(ingestion_manager, result.ingestion_id, raw, filename, scope)

    return CreateMemoryResponse(
        ingestion_id=str(result.ingestion_id),
        document_id=result.document_id,
        status=result.status.value,
        queued_at=result.queued_at,
    )


@router.post("/create", response_model=CreateMemoryResponse, dependencies=[Depends(require_instance_ready)])
async def create_memory(
    request: CreateMemoryRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    ingestion_manager: CoreIngestionManager = Depends(get_ingestion_manager),
):
    """Create a single memory from an SDK document via ingestion pipeline."""
    await _b2c_check(auth, request.customer_id, where="POST /api/v1/memories/create",
                     user_id=request.user_id)
    return await _create_memory(request, auth, ingestion_manager)


async def _create_memory(
    request: CreateMemoryRequest,
    auth: SDKAuthContext,
    ingestion_manager: CoreIngestionManager,
    *,
    scope: Optional[Scope] = None,
    rate_checked: bool = False,
) -> CreateMemoryResponse:
    """The `/create` path, shared with `/create-multipart`.

    One body so the two routes cannot drift on scope, the `_trace` metadata or
    how errors map to statuses. The B2C check stays in each route, where
    `test_b2c_contract` looks for it (and where the multipart route can run it
    before paying for vision reads).

    `scope` and `rate_checked` are for the multipart route, which resolves the
    scope and takes the rate-limit slot before reading images; the ingest then
    skips its own check so the request is counted once.
    """
    from synap.cloud.shared.logging_utils import truncate_string, truncate_dict

    document_preview = None
    if isinstance(request.document, str):
        document_preview = truncate_string(request.document, 300)
    elif isinstance(request.document, dict):
        document_preview = truncate_dict(request.document, max_depth=2, max_items=5, max_str_len=150)
    else:
        document_preview = str(type(request.document))

    logger.info(
        f"Creating memory client_id={auth.client_id} instance_id={auth.instance_id} user_id={request.user_id} customer_id={request.customer_id} document_type={request.document_type} mode={request.mode} document_preview={document_preview} metadata={truncate_dict(request.metadata) if request.metadata else None}",
    )

    async with start_span("sdk.memory.create", {
        "synap.client_id": auth.client_id,
        "synap.instance_id": auth.instance_id,
        "synap.correlation_id": auth.correlation_id or "",
        "synap.document_type": request.document_type,
        "synap.mode": request.mode,
        "synap.transport": "http",
    }):
        try:
            t0 = time.monotonic()
            logger.info(
                "sdk_ingestion_request_received correlation_id=%s client_id=%s instance_id=%s user_id=%s customer_id=%s document_type=%s mode=%s",
                auth.correlation_id,
                auth.client_id,
                auth.instance_id,
                request.user_id,
                request.customer_id,
                request.document_type,
                request.mode,
            )
            ingest_request = await _to_ingest_document_request(
                request=request,
                auth=auth,
                priority=IngestPriority.EXPLICIT,
                scope=scope,
            )
            logger.info(
                "sdk_ingestion_request_mapped correlation_id=%s request_id=%s priority=%s",
                auth.correlation_id,
                ingest_request.request_id,
                ingest_request.priority.value,
            )
            ingest_response = await ingestion_manager.ingest_document(
                ingest_request, skip_per_minute_check=rate_checked)
            logger.info(
                "sdk_ingestion_queued correlation_id=%s request_id=%s ingestion_id=%s document_id=%s status=%s",
                auth.correlation_id,
                ingest_request.request_id,
                ingest_response.ingestion_id,
                ingest_response.document_id,
                ingest_response.status.value,
            )

            # A memory that arrives without a conversation has no conversation end to
            # update the profile from. Where the edition updates the profile at the end
            # of a conversation, run the same update once this batch has been ingested.
            try:
                from synap.cloud.shared import edition

                if edition.profile_at_conversation_end:
                    from synap.cloud.application.profile.conversation_end import spawn_on_memory_created

                    spawn_on_memory_created(
                        scope=ingest_request.scope, ingestion=ingestion_manager,
                        ingestion_id=ingest_response.ingestion_id,
                    )
            except Exception:  # noqa: BLE001 -- the profile never fails an ingestion
                logger.warning("profile_memory_created_not_started", exc_info=True)

            set_current_span_attributes({
                "synap.ingestion_id": str(ingest_response.ingestion_id),
                "synap.document_id": ingest_response.document_id or "",
                "synap.ingest_status": ingest_response.status.value,
                "synap.queue_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })
            logger.info(
                f"Memory created successfully ingestion_id={ingest_response.ingestion_id} document_id={ingest_response.document_id}",
            )

            return CreateMemoryResponse(
                ingestion_id=str(ingest_response.ingestion_id),
                document_id=ingest_response.document_id,
                status=ingest_response.status.value,
                queued_at=ingest_response.queued_at,
            )
        except ValueError as e:
            set_current_span_attributes({"synap.error": True, "synap.error_type": "ValueError"})
            logger.warning(f"Memory creation scope validation failed error={e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except ValidationError as e:
            set_current_span_attributes({"synap.error": True, "synap.error_type": "ValidationError"})
            logger.warning(f"Memory creation validation failed error={e.message}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
        except RateLimitExceeded as e:
            set_current_span_attributes({"synap.error": True, "synap.error_type": "RateLimitExceeded"})
            # Was an f-string naming only retry_after, so the one question this signal
            # exists to answer, WHICH key or address, had no field to read.
            log_rate_limited(auth, key_id=str(auth.credential_id),
                             limit="ingestion", client_id=str(auth.client_id))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=e.message,
                headers={"Retry-After": str(e.retry_after)},
            )
        except Exception as e:
            set_current_span_attributes({"synap.error": True, "synap.error_type": type(e).__name__})
            logger.exception(
                f"Failed to create memory via ingestion pipeline error_type={type(e).__name__} correlation_id={auth.correlation_id} client_id={auth.client_id} instance_id={auth.instance_id}",
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create memory: {str(e)}",
            )


class ManifestTextPart(BaseModel):
    type: Literal["text"]
    text: str


class ManifestFilePart(BaseModel):
    type: Literal["file"]
    # The form field carrying this file's bytes: file_0, file_1, ...
    field: str = Field(..., min_length=1)
    filename: Optional[str] = None
    media_type: Optional[str] = None


ManifestPart = Annotated[
    Union[ManifestTextPart, ManifestFilePart], Field(discriminator="type")
]


class CreateMultipartManifest(BaseModel):
    """The `manifest` form field of `/create-multipart`, as JSON.

    `CreateMemoryRequest` with `document` as an ordered list of text and file
    parts. Typed, so a malformed part is a 400 here and never an error after
    the document has been ingested.
    """

    document: List[ManifestPart]
    document_type: Optional[str] = None
    document_id: Optional[str] = None
    document_created_at: Optional[datetime] = None
    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    # Same contract as `CreateMemoryRequest.scope`: omit it and nothing changes.
    scope: Optional[Dict[str, str]] = None
    mode: str = "long-range"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    objective: Optional[str] = None


# The manifest carries the text parts, so it may be as large as a `/create`
# document. Starlette caps a non-file form field at 1 MB by default.
_MANIFEST_MAX_BYTES = 10 * 1024 * 1024 + 64 * 1024


async def _read_multipart_form(http_request: Request):
    from synap.cloud.application.ingestion.multimodal import MAX_IMAGES_PER_CALL

    try:
        return await http_request.form(
            # One over the limit, so an eleventh file reaches the clear 400
            # below rather than the parser's own message.
            max_files=MAX_IMAGES_PER_CALL + 1,
            max_fields=8,
            max_part_size=_MANIFEST_MAX_BYTES,
        )
    except TypeError:
        # Starlette < 0.40 takes no limits, and has no per-field cap to lift.
        return await http_request.form()


def _parse_manifest(raw: Any) -> CreateMultipartManifest:
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="`manifest` must be a JSON form field.")
    try:
        return CreateMultipartManifest.model_validate_json(raw)
    except Exception as exc:  # pydantic ValidationError or bad JSON
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"`manifest` must be a JSON object describing the document: {exc}",
        )


@router.post("/create-multipart", response_model=CreateMemoryMultipartResponse)
async def create_memory_multipart(
    http_request: Request,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    ingestion_manager: CoreIngestionManager = Depends(get_ingestion_manager),
):
    """`/create` for a document that has images in it.

    Form fields: `manifest` (JSON, see `CreateMultipartManifest`) and one file
    field per file part. The images are read by the vision reader, the
    document is rendered to one string (see `ingestion.multimodal`) and
    ingested exactly as `/create` would ingest that string. An image that
    cannot be read is still ingested, as a placeholder, and reported in
    `attachments`. The image bytes are not kept.

    The form is read here rather than declared, because the file fields are
    named by the manifest and the manifest needs a larger cap than the default.
    """
    from starlette.datastructures import UploadFile as _FormFile

    from synap.cloud.application.ingestion import multimodal as mm

    where = "POST /api/v1/memories/create-multipart"
    form = await _read_multipart_form(http_request)
    manifests = form.getlist("manifest")
    if len(manifests) != 1:
        raise HTTPException(status_code=400, detail="Send exactly one `manifest` field.")
    # Checked here as well as by the parser: a form sent urlencoded (no files)
    # goes through a parser with no per-field cap at all.
    if isinstance(manifests[0], str) and len(manifests[0]) > _MANIFEST_MAX_BYTES:
        raise HTTPException(status_code=413, detail="`manifest` exceeds the 10 MB limit.")
    m = _parse_manifest(manifests[0])
    if not m.document:
        raise HTTPException(status_code=400, detail="`document` must have at least one part.")
    if m.objective is not None and len(m.objective) > MAX_OBJECTIVE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"objective must be {MAX_OBJECTIVE_CHARS} characters or fewer.",
        )

    fields = [p.field for p in m.document if isinstance(p, ManifestFilePart)]
    if len(fields) > mm.MAX_IMAGES_PER_CALL:
        raise HTTPException(
            status_code=400,
            detail=f"At most {mm.MAX_IMAGES_PER_CALL} files per call.",
        )
    if len(set(fields)) != len(fields):
        raise HTTPException(
            status_code=400,
            detail="Each file part needs its own `field` naming its form field.",
        )
    uploads: Dict[str, List[Any]] = {}
    for key, value in form.multi_items():
        if isinstance(value, _FormFile):
            uploads.setdefault(key, []).append(value)
    if any(len(v) > 1 for v in uploads.values()):
        raise HTTPException(status_code=400, detail="Each form field may carry one file.")
    if set(fields) != set(uploads):
        raise HTTPException(
            status_code=400,
            detail=(
                "The manifest's file parts and the uploaded files do not match: "
                f"manifest names {sorted(fields)}, uploaded {sorted(uploads)}."
            ),
        )

    # Validate every file before reading any, so a bad tenth file does not
    # cost nine vision calls.
    parts: List[Any] = []
    for part in m.document:
        if isinstance(part, ManifestTextPart):
            parts.append(part.text)
            continue
        upload = uploads[part.field][0]
        if upload.size is not None and upload.size > mm.MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Each image must be 10 MB or smaller.")
        data = await upload.read()
        try:
            media_type = mm.validate_image(data, part.media_type or upload.content_type)
        except mm.ImagePartError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)
        parts.append(mm.ImageInput(
            data=data, media_type=media_type,
            filename=part.filename or upload.filename or None,
        ))

    # Everything that can still refuse the request runs before the vision
    # reads, so a refused request is never paid for: the B2C contract, the
    # scope, and the rate limit. The slot taken here is the request's one
    # slot; the ingest below skips its own check.
    await _b2c_check(auth, m.customer_id, where=where, user_id=m.user_id)
    try:
        scope = await _resolve_scope(auth, m.user_id, m.customer_id, m.scope)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    rate = await ingestion_manager._rate_limiter.check(
        client_id=scope.client_id, instance_id=scope.instance_id,
    )
    if not rate.allowed:
        log_rate_limited(auth, key_id=str(auth.credential_id),
                         limit="ingestion", client_id=str(auth.client_id))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=RateLimitExceeded(rate.retry_after).message,
            headers={"Retry-After": str(rate.retry_after)},
        )

    rendered, reads = await mm.read_and_render(
        parts, client_id=auth.client_id, instance_id=auth.instance_id,
        operation="memories.create.vision",
    )
    has_text = any(isinstance(p, str) and p.strip() for p in parts)
    if not has_text and reads and not any(r.ok for r in reads):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "None of the images could be read and there is no text to ingest.",
                "attachments": [r.attachment() for r in reads],
            },
        )

    request = CreateMemoryRequest(
        document=rendered,
        # Part 1 has only the conversation pipeline for images.
        document_type=m.document_type or "ai-chat-conversation",
        document_id=m.document_id,
        document_created_at=m.document_created_at,
        user_id=m.user_id,
        customer_id=m.customer_id,
        scope=m.scope,
        mode=m.mode,
        metadata=m.metadata,
        objective=m.objective,
    )
    response = await _create_memory(
        request, auth, ingestion_manager, scope=scope, rate_checked=True,
    )
    return CreateMemoryMultipartResponse(
        **response.model_dump(),
        attachments=[AttachmentResult(**r.attachment()) for r in reads],
    )


@router.post("/batch", response_model=BatchCreateResponse, dependencies=[Depends(require_instance_ready)])
async def batch_create_memories(
    request: BatchCreateRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    ingestion_manager: CoreIngestionManager = Depends(get_ingestion_manager),
):
    """Create multiple memories in a batch via ingestion pipeline."""
    # Checked for EVERY document before any of them is ingested. Checking as we
    # go would leave a batch half-written when the fifth document is the one
    # carrying a customer_id, and a partially applied batch is worse than a
    # refused one.
    for _i, _doc in enumerate(request.documents):
        await _b2c_check(auth, _doc.customer_id,
                         where=f"POST /api/v1/memories/batch[{_i}]",
                         user_id=_doc.user_id)
    logger.info(
        "sdk_ingestion_batch_received correlation_id=%s client_id=%s instance_id=%s total_documents=%s fail_fast=%s",
        auth.correlation_id,
        auth.client_id,
        auth.instance_id,
        len(request.documents),
        request.fail_fast,
    )
    batch_id = str(_uuid.uuid4())
    succeeded = 0
    failed = 0
    results: List[CreateMemoryResponse] = []

    # Pre-validate daily quota for entire batch (only needs client_id)
    client_uuid = _parse_or_deterministic_uuid(auth.client_id)
    batch_rate = await ingestion_manager._rate_limiter.check_batch(
        client_id=client_uuid,
        count=len(request.documents),
    )
    if not batch_rate.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Daily quota exceeded. Remaining: {batch_rate.remaining_day}, requested: {len(request.documents)}",
            headers={"Retry-After": str(batch_rate.retry_after)},
        )

    for doc in request.documents:
        try:
            ingest_request = await _to_ingest_document_request(
                request=doc,
                auth=auth,
                priority=IngestPriority.BOOTSTRAP,
            )
            ingest_response = await ingestion_manager.ingest_document(
                ingest_request, skip_per_minute_check=True
            )
            logger.info(
                "sdk_ingestion_batch_document_queued correlation_id=%s batch_id=%s request_id=%s ingestion_id=%s status=%s",
                auth.correlation_id,
                batch_id,
                ingest_request.request_id,
                ingest_response.ingestion_id,
                ingest_response.status.value,
            )
            results.append(
                CreateMemoryResponse(
                    ingestion_id=str(ingest_response.ingestion_id),
                    document_id=ingest_response.document_id,
                    status=ingest_response.status.value,
                    queued_at=ingest_response.queued_at,
                )
            )
            succeeded += 1
        except ValueError as e:
            failed += 1
            results.append(
                CreateMemoryResponse(
                    ingestion_id=str(_uuid.uuid4()),
                    document_id=doc.document_id or "unknown",
                    status="failed",
                    queued_at=datetime.now(timezone.utc),
                    error_message=f"Scope error: {e}",
                )
            )
            if request.fail_fast:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )
        except ValidationError as e:
            failed += 1
            results.append(
                CreateMemoryResponse(
                    ingestion_id=str(_uuid.uuid4()),
                    document_id=doc.document_id or "unknown",
                    status="failed",
                    queued_at=datetime.now(timezone.utc),
                    error_message=f"Validation error: {e.message}",
                )
            )
            if request.fail_fast:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=e.message,
                )
        except RateLimitExceeded as e:
            failed += 1
            results.append(
                CreateMemoryResponse(
                    ingestion_id=str(_uuid.uuid4()),
                    document_id=doc.document_id or "unknown",
                    status="failed",
                    queued_at=datetime.now(timezone.utc),
                    error_message=f"Rate limit exceeded (retry after {e.retry_after}s)",
                )
            )
            if request.fail_fast:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=e.message,
                    headers={"Retry-After": str(e.retry_after)},
                )
        except Exception as e:
            failed += 1
            logger.exception(
                "sdk_ingestion_batch_document_failed correlation_id=%s batch_id=%s",
                auth.correlation_id,
                batch_id,
            )
            results.append(
                CreateMemoryResponse(
                    ingestion_id=str(_uuid.uuid4()),
                    document_id=doc.document_id or "unknown",
                    status="failed",
                    queued_at=datetime.now(timezone.utc),
                    error_message=str(e),
                )
            )
            if request.fail_fast:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to batch create memories: {str(e)}",
                )

    return BatchCreateResponse(
        batch_id=batch_id,
        total=len(request.documents),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@router.get("/status/{ingestion_id}", response_model=MemoryStatusResponse)
async def get_memory_status(
    ingestion_id: str,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    ingestion_manager: CoreIngestionManager = Depends(get_ingestion_manager),
    session: AsyncSession = Depends(get_async_session),
):
    """Get ingestion status from the core pipeline status tracker."""
    logger.info(
        "sdk_ingestion_status_requested correlation_id=%s ingestion_id=%s client_id=%s instance_id=%s",
        auth.correlation_id,
        ingestion_id,
        auth.client_id,
        auth.instance_id,
    )
    try:
        parsed_id = _uuid.UUID(ingestion_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ingestion_id format",
        )

    status_response = await ingestion_manager.get_status(parsed_id)
    if status_response is None:
        logger.warning(
            "sdk_ingestion_status_not_found correlation_id=%s ingestion_id=%s",
            auth.correlation_id,
            ingestion_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job not found",
        )
    logger.info(
        "sdk_ingestion_status_resolved correlation_id=%s ingestion_id=%s request_id=%s status=%s started_at=%s completed_at=%s",
        auth.correlation_id,
        ingestion_id,
        status_response.request_id,
        status_response.status.value,
        status_response.started_at,
        status_response.completed_at,
    )

    # Verify memory_ids still exist in DB (they may have been deleted by previous runs).
    # Customer- and client-scope facts live in their own tables, so an id that is not in
    # memories is still real if it is in one of those; checking memories alone dropped every
    # one of them and the response listed no ids for an ingest that stored thousands.
    raw_ids = [str(mid) for mid in (status_response.memory_ids or [])]
    if raw_ids:
        client_uuid = str(_parse_or_deterministic_uuid(auth.client_id))
        verify_result = await session.execute(
            text("""
                SELECT id::text FROM memories
                WHERE id = ANY(CAST(:ids AS uuid[]))
                  AND client_id = CAST(:client_id AS uuid)
                  AND deleted_at IS NULL
                UNION
                SELECT id::text FROM customer_knowledge
                WHERE id = ANY(CAST(:ids AS uuid[]))
                  AND client_id = CAST(:client_id AS uuid)
                UNION
                SELECT id::text FROM client_knowledge
                WHERE id = ANY(CAST(:ids AS uuid[]))
                  AND client_id = CAST(:client_id AS uuid)
            """),
            {"ids": raw_ids, "client_id": client_uuid},
        )
        valid_ids = [row[0] for row in verify_result.fetchall()]
    else:
        valid_ids = []

    return MemoryStatusResponse(
        ingestion_id=str(status_response.ingestion_id),
        document_id=status_response.document_id,
        status=status_response.status.value,
        queued_at=status_response.queued_at,
        started_at=status_response.started_at,
        completed_at=status_response.completed_at,
        memories_created=status_response.memories_created,
        error_message=status_response.error_message,
        memory_ids=valid_ids,
    )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
):
    """Get a single memory by ID."""
    try:
        _uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory_id format: must be a valid UUID",
        )

    client_uuid = str(_parse_or_deterministic_uuid(auth.client_id))
    result = await session.execute(
        text("""
            SELECT id, memory_type, content, confidence,
                   category, subcategory, source_document_id, created_at, updated_at
            FROM memories
            WHERE id = CAST(:memory_id AS uuid)
              AND client_id = CAST(:client_id AS uuid)
              AND deleted_at IS NULL
        """),
        {"memory_id": memory_id, "client_id": client_uuid},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )

    return MemoryResponse(
        memory_id=str(row.id),
        memory_type=row.memory_type,
        content=row.content,
        confidence=float(row.confidence) if row.confidence is not None else 0.0,
        category=row.category,
        subcategory=row.subcategory,
        created_at=row.created_at,
        updated_at=row.updated_at,
        source_document_id=row.source_document_id,
    )


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    request: UpdateMemoryRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
):
    """Update an existing memory."""
    try:
        _uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory_id format: must be a valid UUID",
        )

    client_uuid = str(_parse_or_deterministic_uuid(auth.client_id))

    # Verify ownership with client_id scope
    check = await session.execute(
        text("""
            SELECT content, customer_id, user_id, memory_type, confidence, category, subcategory
            FROM memories
            WHERE id = CAST(:memory_id AS uuid)
              AND client_id = CAST(:client_id AS uuid)
              AND deleted_at IS NULL
        """),
        {"memory_id": memory_id, "client_id": client_uuid},
    )
    existing_row = check.fetchone()
    if not existing_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )

    new_content = _extract_content(request.document)
    content_text = new_content.get("text", str(new_content)) if isinstance(new_content, dict) else str(new_content)

    # Apply merge strategy
    merge_strategy = request.merge_strategy or "replace"
    if merge_strategy == "append":
        existing_content = existing_row.content or ""
        content_text = f"{existing_content}\n{content_text}"
    elif merge_strategy == "smart-merge":
        # The SDK's default. It is not built, so the new text replaces the old: say so, but it is not a
        # fault of this call, so it is not a warning on every default update.
        logger.info(
            "merge_strategy 'smart-merge' is not built yet, so the update replaces memory %s",
            memory_id,
        )
    elif merge_strategy != "replace":
        logger.warning(
            "merge_strategy '%s' not yet implemented for memory %s; falling back to 'replace'",
            merge_strategy,
            memory_id,
        )
    # else: "replace" — use content_text as-is

    # Build SET clause dynamically
    set_parts = ["content = :content", "updated_at = NOW()"]
    params = {
        "memory_id": memory_id,
        "client_id": client_uuid,
        "content": content_text,
    }

    if request.document_type:
        set_parts.append("memory_type = :memory_type")
        params["memory_type"] = request.document_type

    set_clause = ", ".join(set_parts)
    await session.execute(
        text(f"""
            UPDATE memories
            SET {set_clause}
            WHERE id = CAST(:memory_id AS uuid)
              AND client_id = CAST(:client_id AS uuid)
        """),
        params,
    )

    # The old graph support is owed a release (reason "updated"): its outbox
    # row goes in the same transaction as the new text, then commit, so the
    # graph step runs only on a committed edit and a failed one is retried.
    graph_rows = await _enqueue_graph(
        session, _scope_of_stored_row(auth, client_uuid, existing_row), [memory_id],
        op="retire", reason="updated")
    await session.commit()

    # Vector re-embedding and graph update, from the row's OWN ids.
    #
    # The same defect the delete path had, and quieter. `_resolve_scope` maps
    # external ids, so feeding it the row's internal uuids hashed them again
    # and sent the new embedding to a collection nobody reads. The OLD text
    # stayed searchable while the caller was told the update had succeeded.
    # See `_scope_of_stored_row`.
    from synap.cloud.application.shared.providers import get_cached_providers
    providers = get_cached_providers()
    scope = _scope_of_stored_row(auth, client_uuid, existing_row)
    resolved_memory_type = request.document_type or existing_row.memory_type

    try:
        import asyncio as _asyncio
        from synap.providers.vector.models import VectorDocument
        from datetime import datetime as _dt, timezone as _tz

        embedding_provider = providers.get("embedding")
        vector_provider = providers.get("vector")  # Already AsyncVectorWrapper
        if embedding_provider and vector_provider:
            new_embedding = await embedding_provider.embed(content_text)
            namespace = vector_provider._get_namespace(scope)
            doc = VectorDocument(
                id=memory_id,
                embedding=new_embedding,
                metadata={
                    "client_id": str(scope.client_id),
                    "instance_id": str(scope.instance_id),
                    "customer_id": str(scope.customer_id),
                    "user_id": str(scope.user_id) if scope.user_id else None,
                    "memory_type": resolved_memory_type,
                    "confidence": existing_row.confidence,
                    "category": existing_row.category,
                    "subcategory": existing_row.subcategory,
                    "updated_at": _dt.now(_tz.utc).isoformat(),
                },
                content=content_text,
            )
            await _asyncio.to_thread(vector_provider._provider.upsert, namespace, [doc])
    except Exception as e:
        logger.warning("Vector update failed for memory %s (non-fatal): %s", memory_id, e)

    try:
        graph_provider = providers.get("graph")  # Already AsyncGraphWrapper
        if graph_provider and _graph_lifecycle_on(scope):
            # The graph lines were extracted from the OLD text, so they no
            # longer have this memory's support: retire it (owner decision 3:
            # close the old lines now; re-extracting from the new text is a
            # later, flagged step). The new text is served through the vector
            # index at once.
            await _graph_lifecycle(graph_provider).release(
                scope, [memory_id], mode="retire", reason="updated")
            await _complete_graph_rows(graph_rows)
        elif graph_provider:
            await graph_provider.update_memory_on_edges(
                memory_id=memory_id,
                new_content=content_text,
                new_memory_type=resolved_memory_type,
                new_confidence=existing_row.confidence,
                scope=scope,
            )
    except Exception as e:
        logger.warning("Graph update failed for memory %s (non-fatal): %s", memory_id, e)

    # Cached bundles still hold the old text.
    await _invalidate_query_cache(client_uuid)

    # Record the UPDATE in memory_actions so the Requests page can render a diff.
    # existing_row.content was captured before the UPDATE above ran, so
    # previous_content is safe. Best-effort: failures here must not fail the UPDATE.
    if auth.correlation_id:
        try:
            from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import SQLAlchemyRDBMSWrapper
            await SQLAlchemyRDBMSWrapper().insert_memory_action(
                correlation_id=auth.correlation_id,
                ingestion_id=None,
                scope=scope,
                action="UPDATE",
                memory_id=memory_id,
                content=content_text,
                previous_content=existing_row.content,
                memory_type=resolved_memory_type,
            )
        except Exception as e:
            logger.warning("memory_action UPDATE audit failed memory_id=%s (non-fatal): %s", memory_id, e)

    # Re-fetch updated row
    return await get_memory(memory_id, auth, session)


@router.delete("/{memory_id}", response_model=DeleteMemoryResponse)
async def delete_memory(
    memory_id: str,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
    customer_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """Delete a memory by ID, or a graph record by its graph id.

    ``customer_id`` / ``user_id`` (optional query parameters) say who is
    deleting. They matter only for a graph id: such a record is erased only
    from the graph the caller WRITES to, never from a shared customer- or
    client-level graph it can merely read.
    """
    # The B2C single-identifier contract, as every other route that accepts a
    # customer_id applies it (shared.b2c_contract).
    await _b2c_check(auth, customer_id, where="DELETE /api/v1/memories/{memory_id}",
                     user_id=user_id)
    try:
        _uuid.UUID(memory_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory_id format: must be a valid UUID",
        )

    client_uuid = str(_parse_or_deterministic_uuid(auth.client_id))
    # Capture the row's content and scope BEFORE the hard-delete runs; once
    # DELETE FROM memories commits, the row is gone and we can't reconstruct
    # previous_content for the memory_actions changelog. This SELECT replaces
    # the former existence check (which only returned 1) for that reason.
    check = await session.execute(
        text("""
            SELECT content, memory_type, customer_id, user_id, source_document_id
            FROM memories
            WHERE id = CAST(:memory_id AS uuid)
              AND client_id = CAST(:client_id AS uuid)
        """),
        {"memory_id": memory_id, "client_id": client_uuid},
    )
    existing_row = check.fetchone()
    if not existing_row:
        # Accurate mode serves graph records (edges and entity nodes) with their
        # graph ids. Such an id is not a memory: look it up in the graph and
        # erase that record only (owner decision: the memories behind it are
        # kept and reported, not deleted).
        answer = await _delete_graph_record(memory_id, auth, client_uuid, session,
                                            customer_id=customer_id, user_id=user_id)
        if answer is not None:
            return answer
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )

    # Consolidation F12: cascade BEFORE the row disappears — the propagation
    # needs the provenance links to classify derived artifacts as wholly vs
    # mixed. Wholly-derived artifacts die with their source; mixed ones are
    # suppressed and marked for regeneration. Non-fatal: the memory delete
    # proceeds regardless.
    try:
        from synap.cloud.database.connection import get_session_factory
        from synap.cloud.shared import edition
        # Cloud registers the consolidation cascade; Core has no derived
        # artifacts, registers nothing, and this does nothing.
        await edition.acall(
            "memory_deletion_propagation", [memory_id],
            session_factory=get_session_factory())
    except Exception as e:  # noqa: BLE001
        logger.warning("deletion propagation failed memory_id=%s (non-fatal): %s",
                       memory_id, e)

    # Deleting a newer fact restores the older fact it replaced (owner
    # decision): the foreign key clears `superseded_by` on every memory this one
    # superseded. Read which ones NOW, in this transaction, before the delete
    # erases the evidence; the graph puts them back on the records that
    # released them (GraphLifecycle.release(restore_ids=...)).
    scope_for_graph = _scope_of_stored_row(auth, client_uuid, existing_row)
    restore_map: Dict[str, List[str]] = {}
    if _graph_lifecycle_on(scope_for_graph):
        from synap.cloud.agents.ingestion.storage.graph_sync_outbox import superseded_by_any
        restore_map = await superseded_by_any(session, [memory_id])

    # Clean up related data before deleting the memory
    await session.execute(
        text("DELETE FROM memory_entity_links WHERE memory_id = CAST(:memory_id AS uuid)"),
        {"memory_id": memory_id},
    )
    await session.execute(
        text("""
            DELETE FROM memories
            WHERE id = CAST(:memory_id AS uuid)
              AND client_id = CAST(:client_id AS uuid)
        """),
        {"memory_id": memory_id, "client_id": client_uuid},
    )
    # The graph erase is owed from this moment: its outbox row is written in
    # the same transaction as the delete, so a failed or interrupted graph step
    # is retried by the drain worker. Then commit before touching other
    # stores: the graph erase cannot be undone, and a Postgres rollback after
    # it would leave a live memory whose graph support is gone.
    graph_rows = await _enqueue_graph(session, scope_for_graph, [memory_id],
                                      op="erase", reason="deleted", restore_ids=restore_map)
    await session.commit()

    # Vector and graph cleanup, from the row's OWN ids.
    #
    # This used to hand the row's internal uuids to `_resolve_scope`, which
    # maps external ids, so they were hashed a second time and the delete aimed
    # at a collection that never existed. See `_scope_of_stored_row`.
    from synap.cloud.application.shared.providers import get_cached_providers
    providers = get_cached_providers()
    scope = _scope_of_stored_row(auth, client_uuid, existing_row)

    # Record the DELETE in memory_actions with the content we captured above;
    # the memories row is gone now, so this is our only chance.
    if auth.correlation_id:
        try:
            from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import SQLAlchemyRDBMSWrapper
            await SQLAlchemyRDBMSWrapper().insert_memory_action(
                correlation_id=auth.correlation_id,
                ingestion_id=None,
                scope=scope,
                action="DELETE",
                memory_id=memory_id,
                content=None,
                previous_content=existing_row.content,
                memory_type=existing_row.memory_type,
            )
        except Exception as e:
            logger.warning("memory_action DELETE audit failed memory_id=%s (non-fatal): %s", memory_id, e)

    # Which stores still hold this record after the attempt. The Postgres row
    # is already gone, so failing the request now would be worse than
    # answering it: the caller is told instead, in the body.
    still_present_in: List[str] = []
    # The source document is shared, so "kept" is an ordinary outcome and not a
    # failure. It is reported separately from `still_present_in`, which means
    # something went wrong.
    source_still_held = False

    try:
        vector_provider = providers.get("vector")  # Already AsyncVectorWrapper
        if vector_provider:
            # `delete_memory_vectors`, not `delete`. `delete` clears the
            # memories collection alone, and a memory's hypothetical questions
            # carry a copy of its text in `memory_content`. Leaving those
            # behind means the deleted sentence keeps being retrieved and keeps
            # reaching the model, which is the symptom that started this: a
            # patient's allergy still coming back after the memory was deleted.
            await vector_provider.delete_memory_vectors(
                memory_ids=[memory_id], scope=scope
            )
    except Exception as e:
        still_present_in.append("search index")
        # Error, not warning. This is the line that said "non-fatal" 49 times
        # while the record stayed searchable, and warning is the level people
        # filter out.
        logger.error(
            "Vector cleanup FAILED for memory %s: the row is deleted but the "
            "record is still in the search index and can still be served: %s",
            memory_id, e,
        )

    # The source text the memory was extracted from.
    #
    # Deleting a memory used to leave the sentences it came from in the chunks
    # collection, still searchable and still holding the original wording. A
    # client deleting an allergy for a privacy reason had the allergy removed
    # and the sentence stating it left behind.
    #
    # ⚠ ONLY WHEN THIS WAS THE LAST MEMORY FROM THAT DOCUMENT. One document
    # backs many memories: a busy document can back
    # hundreds. Purging on every delete would strip the provenance from all the
    # others, which is worse than the problem being fixed. So the source goes when
    # nothing is left that came from it, and stays otherwise. The caller is
    # told which happened rather than left to assume.
    source_doc = getattr(existing_row, "source_document_id", None)
    if source_doc:
        try:
            still = await session.execute(
                text("""
                    SELECT count(*) FROM memories
                    WHERE source_document_id = :doc
                      AND client_id = CAST(:client_id AS uuid)
                """),
                {"doc": source_doc, "client_id": client_uuid},
            )
            remaining = still.scalar() or 0
            if remaining == 0:
                vp = providers.get("vector")
                if vp:
                    n = await vp.delete_source_document_chunks(scope, str(source_doc))
                    logger.info(
                        "delete_purged_source memory_id=%s document=%s chunks=%d",
                        memory_id, source_doc, n,
                    )
            else:
                # Not a failure and not silence: the text is still here because
                # other memories still rest on it.
                source_still_held = True
                logger.info(
                    "delete_kept_source memory_id=%s document=%s still_backing=%d",
                    memory_id, source_doc, remaining,
                )
        except Exception as e:
            still_present_in.append("source text")
            logger.error(
                "Source cleanup FAILED for memory %s document %s: the memory is "
                "deleted but the text it came from may still be searchable: %s",
                memory_id, source_doc, e,
            )

    # Graph: erase this memory's support. Edges and nodes it alone supported
    # are deleted (an edge's sentence and embedding are a copy of the fact);
    # records other live memories support only lose the id. The old helper
    # (`soft_delete_memory_from_edges`) only understood copied text and found
    # nothing in id mode, and the route then reported fully_removed=True.
    graph_released: Optional[GraphReleased] = None
    graph_provider = providers.get("graph")  # Already AsyncGraphWrapper
    if graph_provider:
        try:
            if _graph_lifecycle_on(scope):
                result = await _graph_lifecycle(graph_provider).release(
                    scope, [memory_id], mode="erase", reason="deleted",
                    restore_ids=restore_map.get(str(memory_id)))
                graph_released = _graph_released(result)
                await _complete_graph_rows(graph_rows)
            else:
                await graph_provider.soft_delete_memory_from_edges(
                    memory_id=memory_id,
                    scope=scope,
                )
        except Exception as e:
            still_present_in.append("graph")
            logger.error(
                "Graph cleanup FAILED for memory %s: the row is deleted but its "
                "edges remain until the graph outbox worker retries (pending rows: %d): %s",
                memory_id, len(graph_rows), e,
            )

    # A cached bundle may still hold the deleted text for up to the cache TTL.
    if not await _invalidate_query_cache(client_uuid):
        still_present_in.append("cache")

    return DeleteMemoryResponse(
        memory_id=memory_id,
        deleted_at=datetime.now(timezone.utc),
        fully_removed=not still_present_in,
        still_present_in=still_present_in,
        source_text_kept=source_still_held,
        kind="memory",
        graph_released=graph_released,
    )


# ---------------------------------------------------------------------------
# Graph lifecycle helpers (delete, update, delete by graph id)
# ---------------------------------------------------------------------------

def _graph_lifecycle_on(scope) -> bool:
    from synap.cloud.agents.ingestion.storage.coordinator import graph_lifecycle_enabled
    return graph_lifecycle_enabled(scope)


def _graph_lifecycle(graph_provider):
    """The graph lifecycle service over the cached graph wrapper. A module
    function so tests can replace it with a fake."""
    from synap.cloud.agents.ingestion.storage.graph_lifecycle import GraphLifecycle
    from synap.cloud.application.shared.providers import get_cached_providers
    # The vector store too: erasing an edge or node also deletes its triple
    # and entity vectors (a copy of the fact).
    return GraphLifecycle(graph_provider, vector=get_cached_providers().get("vector"))


async def _enqueue_graph(session, scope, memory_ids, *, op: str, reason: str,
                         restore_ids: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Write the graph outbox rows for *memory_ids* with *session*, without
    committing (the caller commits them with its memory change). Every caller
    here applies the rows itself after the commit, so they are written leased
    to it (``inline=True``) and the drain worker leaves them alone unless that
    step fails. [] when the
    lifecycle is off or the outbox table does not exist (graph_reconcile is
    then the backstop)."""
    if not _graph_lifecycle_on(scope):
        return []
    from synap.cloud.agents.ingestion.storage import graph_sync_outbox
    return await graph_sync_outbox.enqueue(session, memory_ids=memory_ids, scope=scope,
                                           op=op, reason=reason, restore_ids=restore_ids,
                                           inline=True)


async def _complete_graph_rows(row_ids: List[str]) -> None:
    """Mark outbox rows done after the inline graph step succeeded. A failure
    here only means the worker applies the (idempotent) release again."""
    if not row_ids:
        return
    try:
        from synap.cloud.agents.ingestion.storage.graph_sync_outbox import PostgresOutboxStore
        await PostgresOutboxStore().complete(list(row_ids))
    except Exception as e:  # noqa: BLE001
        logger.warning("graph outbox rows not marked done (the worker re-applies them): %s", e)


def _graph_released(result) -> Optional[GraphReleased]:
    if result is None:
        return None
    return GraphReleased(**{
        f: int(getattr(result, f, 0) or 0) for f in GraphReleased.model_fields
    })


async def _invalidate_query_cache(client_id: str) -> bool:
    """Drop the client's cached query results. False when that failed."""
    try:
        from synap.cloud.application.client_api.retrieval_manager import (
            invalidate_client_query_cache)
        await invalidate_client_query_cache(str(client_id))
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("query cache invalidation FAILED client_id=%s: a cached bundle "
                     "may serve the deleted text until its TTL: %s", client_id, e)
        return False


def _graph_write_namespaces(scope) -> List[str]:
    """The graph a caller at *scope* writes to: the namespace its own writes
    land in, by the ingestion writer's rule (`scope_key.write_scope_id`).
    Never a level above the caller. The pre-ladder (legacy) graph is NOT
    added for a user: it is the customer's shared graph, and a user cannot be
    told apart from the customer there."""
    from synap.cloud.shared import scope_key
    return [scope_key.write_scope_id(scope)]


async def _delete_graph_record(record_id: str, auth: SDKAuthContext, client_uuid: str,
                               session: AsyncSession, *, customer_id: Optional[str] = None,
                               user_id: Optional[str] = None) -> Optional[DeleteMemoryResponse]:
    """Delete one graph record (edge or entity node) by its graph id.

    Looked up only in the namespaces the caller can read (the lifecycle
    service resolves them from the scope). Erases that record only; the
    memories that backed it are left alone and the live ones are returned in
    `still_asserted_by`, so the caller can delete them too. `fully_removed` is
    true only when no live memory still asserts it. None when the id is not a
    graph record either (the caller answers 404).
    """
    from types import SimpleNamespace

    from synap.cloud.application.shared.providers import get_cached_providers

    graph_provider = get_cached_providers().get("graph")
    if not graph_provider:
        return None
    if customer_id or user_id:
        scope = await _resolve_scope(auth, user_id, customer_id)
    else:
        scope = _scope_of_stored_row(auth, client_uuid,
                                     SimpleNamespace(customer_id=None, user_id=None))
    if not _graph_lifecycle_on(scope):
        return None
    lifecycle = _graph_lifecycle(graph_provider)
    # Only where the caller writes: a user must not erase a record of the
    # customer's or the client's shared graph, which it can only read.
    import inspect
    if "namespaces" not in inspect.signature(lifecycle.erase_record).parameters:
        logger.error("delete by graph id refused: GraphLifecycle.erase_record cannot be limited "
                     "to the caller's write namespaces")
        return None
    erased = await lifecycle.erase_record(scope, record_id,
                                          namespaces=_graph_write_namespaces(scope))
    if erased is None:
        return None

    backing = [str(m) for m in (getattr(erased, "memory_ids", None) or [])]
    still_asserted_by: List[str] = []
    still_present_in: List[str] = []
    if backing:
        try:
            from synap.cloud.shared.memory_liveness import is_alive, load_supporter_rows
            rows = await load_supporter_rows(backing, session=session)
            still_asserted_by = [m for m in backing if is_alive(rows.get(m))]
        except Exception as e:  # noqa: BLE001
            # Unknown is not "none": report every backing id rather than
            # telling the caller nothing still says it.
            logger.error("liveness check FAILED for graph record %s: %s", record_id, e)
            still_asserted_by = backing
    if not await _invalidate_query_cache(client_uuid):
        still_present_in.append("cache")
    logger.info("delete_graph_record id=%s kind=%s backing=%d still_asserted_by=%d",
                record_id, getattr(erased, "kind", "?"), len(backing), len(still_asserted_by))
    return DeleteMemoryResponse(
        memory_id=record_id,
        deleted_at=datetime.now(timezone.utc),
        kind=getattr(erased, "kind", "relationship"),
        still_asserted_by=still_asserted_by,
        still_present_in=still_present_in,
        fully_removed=not still_asserted_by and not still_present_in,
        graph_released=GraphReleased(
            edges_deleted=int(getattr(erased, "edges_deleted", 0) or 0),
            nodes_deleted=int(getattr(erased, "nodes_deleted", 0) or 0),
        ),
    )
