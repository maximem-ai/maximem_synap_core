"""Ingestion input models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .enums import DocumentType, DocumentSource, IngestPriority


@dataclass
class UserContext:
    """User identity context for prompt injection.

    Provides the speaker/user identity so extraction prompts can:
    - Attribute "I/me/my" to a named entity instead of "the user"
    - Ground coreference resolution with a known speaker identity
    """
    user_label: str = "the user"


class ScopeRung(BaseModel):
    """One node on the caller's resolved scope chain, root to leaf.

    Produced by ``ScopeLadder.resolve``. Carried on the Scope so the write side
    can stamp ``node_id`` and the level-keyed ancestors onto every record
    without resolving the ladder a second time, which would be a second chance
    to resolve it differently.
    """
    node_id: UUID
    level_key: str
    role: Optional[str] = None      # root | tenant_of_record | identity | None
    # Every role this rung holds. Usually the same as ``[role]``; two in B2C,
    # where the tenant node IS the identity node. Reading only ``role`` there
    # concludes the request has no identity, which made the write side stamp
    # nothing at all for every B2C client. Optional so a chain built before
    # this field existed still validates.
    roles: Optional[List[str]] = None
    # Whether the CLIENT's ladder (not this chain) has a rung that the
    # customer_id/user_id pair cannot name. Set by `chain_to_rungs`; None for a
    # chain built before this field existed, which reads as "assume not" and so
    # keeps the three-rung behaviour every existing caller has.
    ladder_has_intermediate_rungs: Optional[bool] = None

    def has_role(self, name: str) -> bool:
        return name in (self.roles or ([self.role] if self.role else []))


class Scope(BaseModel):
    """Tenant scope for the ingestion.

    ``scope_level`` is derived from (user_id, customer_id, user_context_isolation)
    via ``synap.cloud.shared.scope.derive_scope_level`` and carries the
    narrowest level at which this request's memories should land:
    - "user"    → per-user memories (synap_{customer_id}, user_id = real UUID)
    - "customer"→ org-shared memories (synap_{customer_id}, user_id = sentinel)
    - "client"  → tenant-wide memories (synap_{client_id}, no customer/user)

    ``chain`` is the nested-scoping addition and is optional on purpose. Every
    caller today leaves it None and gets byte-identical metadata; a caller that
    supplies it also gets ``node_id`` and one ``anc_<level_key>`` per ancestor.
    Additive is what makes this safe to deploy before anything reads it.
    """
    client_id: UUID
    instance_id: UUID
    customer_id: Optional[UUID] = None  # None for client-scope ingestion
    user_id: Optional[UUID] = None      # None for customer/client-scope ingestion
    scope_level: Optional[str] = None   # "user" | "customer" | "client"
    chain: Optional[List[ScopeRung]] = None   # root -> leaf, from ScopeLadder


class IngestDocumentRequest(BaseModel):
    """Request to ingest a single document."""
    # Identity
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: Optional[str] = None  # Client-provided, auto-generated if not

    # Content
    content: Union[str, bytes]  # Raw content or file bytes
    content_encoding: Optional[str] = None  # "base64" if bytes are encoded

    # Classification hints (may be overridden by Stage 1)
    document_type: Optional[DocumentType] = None
    source: Optional[DocumentSource] = DocumentSource.API

    # Temporal
    document_created_at: Optional[datetime] = None  # DCT from spec
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Scope
    scope: Scope

    # Processing hints
    priority: IngestPriority = IngestPriority.EXPLICIT
    mode: str = "long-range"  # "fast" or "long-range"

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    security_info: Optional[Dict[str, Any]] = None


class IngestBatchRequest(BaseModel):
    """Request to ingest multiple documents."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    documents: List[IngestDocumentRequest]
    scope: Scope  # Shared scope for batch
    priority: IngestPriority = IngestPriority.BOOTSTRAP
    fail_fast: bool = False  # Stop on first error?


class IngestFromListeningAgentRequest(BaseModel):
    """Request from Listening Agent for realtime ingestion."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    conversation_id: str

    # Content
    content: str
    content_type: str  # "turn" | "compacted_context" | "session_summary"

    # Pre-qualification (Listening Agent already did LLM work)
    pre_qualified: bool = False
    qualification_result: Optional[Dict[str, Any]] = None

    # Scope
    scope: Scope

    # Destination
    destination: str  # "short_term" | "long_term"
    priority: IngestPriority = IngestPriority.REALTIME


class ConvertSessionRequest(BaseModel):
    """Request to convert short-term session to long-term memories."""
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    scope: Scope
    priority: IngestPriority = IngestPriority.REALTIME
