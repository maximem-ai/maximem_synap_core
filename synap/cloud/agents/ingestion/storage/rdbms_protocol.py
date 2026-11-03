"""Protocol defining the RDBMS storage interface for ingestion.

Both SQLAlchemyRDBMSWrapper and AsyncRDBMSWrapper conform to this protocol.
Use this as the type hint when either wrapper is acceptable.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class RDBMSStorageProtocol(Protocol):
    """Structural interface for RDBMS persistence in the ingestion pipeline.

    The StorageCoordinator and other consumers should type-hint against
    this protocol rather than a concrete wrapper class.

    Two implementations exist:
    - SQLAlchemyRDBMSWrapper: uses the app's async SQLAlchemy session pool
      (preferred inside the cloud application)
    - AsyncRDBMSWrapper: wraps a sync RDBMSInterface via asyncio.to_thread
      (used by consolidation manager and short-term storage)
    """

    async def insert_memory(self, memory, scope, document_id: str) -> UUID: ...

    async def insert_candidate(self, memory, scope, document_id: str) -> UUID: ...

    async def insert_entity_reference(self, memory_id, entity_id: str, scope) -> None: ...

    async def insert_client_knowledge(self, memory, scope, document_id: str) -> UUID: ...

    async def insert_customer_knowledge(self, memory, scope, document_id: str) -> UUID: ...

    async def soft_delete_memory(self, memory_id: str, superseded_by: str) -> bool: ...

    async def set_deprecated_at(self, memory_id: str) -> bool:
        """Mark a memory as deprecated (KU v2 lifecycle marker).

        Sets `memories.deprecated_at = NOW()` if the row exists and is
        currently active (deprecated_at IS NULL). Returns True if a row was
        updated, False otherwise (already deprecated or not found).

        This is the KU v2 replacement for `soft_delete_memory`. The legacy
        function continues to work for backward compatibility but new
        knowledge-update paths use this + `insert_memory_link` instead.
        """
        ...

    async def value_retired_for_user(
        self,
        scope,
        value_token: str,
        exclude_memory_ids: list,
    ) -> bool:
        """Echo resurrection guard lookup (ENABLE_ECHO_GUARD): True iff
        ``value_token`` appears in a retired memory (superseded/deprecated)
        for this user and in no current memory outside
        ``exclude_memory_ids`` (the just-persisted batch). Implementations
        without the KU lifecycle return False (never an echo)."""
        ...

    async def current_memories_matching(
        self,
        scope,
        content_tokens: list,
        exclude_memory_ids: list = None,
        limit: int = 50,
    ) -> list:
        """Retraction subject-sweep lookup (ENABLE_KU_DELETE_SUBJECT_GUARD):
        current memories of this user whose content contains ALL of
        ``content_tokens``. Implementations without the KU lifecycle return
        an empty list (nothing to sweep)."""
        ...

    async def insert_memory_link(
        self,
        *,
        from_memory_id: str,
        to_memory_id: str,
        kind: str,
        reason: Optional[str] = None,
    ) -> None:
        """Insert a row in the memory_links lineage table.

        `kind` must be one of DEPRECATES / ENRICHED / DISAMBIGUATES / CANCELS.
        Idempotent on (from_memory_id, to_memory_id, kind) — duplicate inserts
        are silently dropped via ON CONFLICT.
        """
        ...

    async def insert_memory_source_documents(
        self,
        memory_id: str,
        source_document_ids: "list[str]",
    ) -> None:
        """Insert M:N rows linking a memory to one or more source documents.

        For new memories during ingestion, called once with one source_document_id.
        For merged memories, called once with the union of all
        contributors' source_document_ids.
        Idempotent on (memory_id, source_document_id).
        """
        ...

    async def retire_memory_with_audit(
        self,
        *,
        memory_id: str,
        action: str,
        scope,
        previous_content,
        content=None,
        memory_type=None,
        superseded_by: Optional[str] = None,
        replaced_by_memory_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        ingestion_id=None,
        reason: Optional[str] = None,
        deprecate_with_link: bool = False,
        link_kind: str = "DEPRECATES",
        link_to_memory_id: Optional[str] = None,
        require_retire: bool = True,
    ) -> bool:
        """Retire a memory and write its trail row in ONE transaction.

        The trail row carries the before-state, which is the only thing a
        reversal can restore a memory from. Split across two transactions, a
        failure between them left memories retired and un-undoable; behind an
        `if correlation_id`, the audit went missing exactly where the caller
        was least ordinary. So implementations write both or neither, and no
        implementation may quietly no-op: a silent skip here is a memory that
        can never be brought back.

        Returns False when the row was already retired (nothing written).
        """
        ...

    async def insert_memory_action(
        self,
        *,
        correlation_id: str,
        ingestion_id,
        scope,
        action: str,
        memory_id: str,
        content,
        previous_content,
        memory_type,
        replaced_by_memory_id=None,
        reason: Optional[str] = None,
    ) -> None:
        """Write one row to the memory_actions changelog.

        Called from ingestion (CREATE, SUPERSEDE, NONE, ENRICH) and SDK routes
        (UPDATE, DELETE). Implementations that don't participate in
        HTTP-originated writes (e.g. the consolidation path) may no-op.

        `reason` is the LLM-generated rationale from the knowledge-update
        classifier; surfaced to clients in the Requests UI when present.
        """
        ...
