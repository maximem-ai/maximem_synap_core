"""Every change to a registry entry goes through here.

Moved from synap-er, which no longer owns storage. The order is what makes a
worker's in-memory index trustworthy: the database first, then this worker's
index, then a notification so the other workers update theirs.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from synap_er.memory.embedding import compute_embedding
from synap_er.memory.index_manager import RegistryIndexManager, ScopeKey

from synap.cloud.agents.ingestion.resolution.registry.models import RegistryEntry
from synap.cloud.agents.ingestion.resolution.registry.types import Scope

logger = logging.getLogger(__name__)

NOTIFY_CHANNEL = "registry_changes"


class WriteThroughManager:
    """Ensures write-through consistency between PostgreSQL and in-memory indices.

    Every mutation follows the sequence:

    1. Compute embedding if absent.
    2. DB commit (within transaction).
    3. Synchronous in-memory index update.
    4. ``pg_notify`` broadcast to other workers.

    Only returns success after all steps complete.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        index_manager: RegistryIndexManager,
    ) -> None:
        self._session_factory = session_factory
        self._index = index_manager

    # ------------------------------------------------------------------
    # Public write operations
    # ------------------------------------------------------------------

    async def create_entry(self, entry_data: dict[str, Any]) -> RegistryEntry:
        """Create a registry entry with write-through consistency.

        If ``entry_data`` does not contain an ``embedding``, one is
        computed from ``canonical_name`` via sentence-transformers.
        """
        # Step 1: pre-compute embedding if absent
        if not entry_data.get("embedding"):
            entry_data["embedding"] = compute_embedding(
                entry_data["canonical_name"]
            ).tolist()

        async with self._session_factory() as session:
            async with session.begin():
                # Step 2: DB write
                entry = RegistryEntry(**entry_data)
                session.add(entry)
                await session.flush()  # assigns canonical_id if defaulted

                # Step 3: in-memory update
                await self._safe_memory_update("add", entry)

                # Step 4: broadcast
                await self._notify(session, "add", entry)

        return entry

    async def update_entry(
        self, canonical_id: uuid.UUID, updates: dict[str, Any]
    ) -> RegistryEntry:
        """Update a registry entry with write-through consistency.

        Re-computes the embedding when ``canonical_name`` or ``aliases``
        change.
        """
        async with self._session_factory() as session:
            async with session.begin():
                entry = await session.get(RegistryEntry, canonical_id)
                if entry is None:
                    raise ValueError(
                        f"RegistryEntry {canonical_id} not found"
                    )

                name_changed = False
                for key, value in updates.items():
                    if key in ("canonical_name", "aliases"):
                        name_changed = True
                    setattr(entry, key, value)

                if name_changed:
                    entry.embedding = compute_embedding(
                        entry.canonical_name
                    ).tolist()

                await session.flush()

                await self._safe_memory_update("update", entry)
                await self._notify(session, "update", entry)

        return entry

    async def add_alias(self, canonical_id: uuid.UUID, alias: str) -> RegistryEntry:
        """Record *alias* as another spelling of an entry.

        A no-op when the entry already has it (case-insensitive) or it equals
        the canonical name. The next exact lookup of the spelling then finds
        the entry without any scoring.
        """
        async with self._session_factory() as session:
            entry = await session.get(RegistryEntry, canonical_id)
            if entry is None:
                raise ValueError(f"RegistryEntry {canonical_id} not found")
            known = {entry.canonical_name.lower()} | {a.lower() for a in entry.aliases or []}
            aliases = list(entry.aliases or [])
        if alias.lower() in known:
            return entry
        return await self.update_entry(canonical_id, {"aliases": aliases + [alias]})

    async def delete_entry(self, canonical_id: uuid.UUID) -> None:
        """Soft-delete a registry entry with write-through consistency.

        Sets ``is_deleted=True`` and ``deleted_at`` instead of removing
        the row, but still removes the entry from in-memory indices so
        it is no longer resolvable.
        """
        async with self._session_factory() as session:
            async with session.begin():
                entry = await session.get(RegistryEntry, canonical_id)
                if entry is None:
                    raise ValueError(
                        f"RegistryEntry {canonical_id} not found"
                    )

                entry.is_deleted = True
                entry.deleted_at = datetime.now(timezone.utc)
                await session.flush()

                # Remove from in-memory indices (soft-deleted = not resolvable)
                await self._safe_memory_update("remove", entry)
                await self._notify(session, "remove", entry)

    async def rebuild_scope(
        self, scope: Scope | str, scope_id: str | None
    ) -> None:
        """Full rebuild of in-memory indices for a scope from DB."""
        scope_val = scope.value if hasattr(scope, "value") else scope
        scope_key: ScopeKey = (scope_val, scope_id)
        self._index.clear_scope(scope_key)

        async with self._session_factory() as session:
            stmt = select(RegistryEntry).where(
                RegistryEntry.scope == scope_val,
                RegistryEntry.scope_id == scope_id,
                RegistryEntry.is_deleted == False,  # noqa: E712
            )
            result = await session.execute(stmt)
            entries = list(result.scalars().all())

        if entries:
            self._index.load_scope(scope_key, entries)
        logger.info(
            "Rebuilt scope (%s, %s) with %d entries",
            scope_val, scope_id, len(entries),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _notify(
        self,
        session: AsyncSession,
        action: str,
        entry: RegistryEntry,
    ) -> None:
        scope_val = (
            entry.scope.value if hasattr(entry.scope, "value") else entry.scope
        )
        payload = json.dumps({
            "action": action,
            "canonical_id": str(entry.canonical_id),
            "scope": scope_val,
            "scope_id": entry.scope_id,
        })
        await session.execute(
            text("SELECT pg_notify(:channel, :payload)"),
            {"channel": NOTIFY_CHANNEL, "payload": payload},
        )

    async def _safe_memory_update(
        self, operation: str, entry: RegistryEntry
    ) -> None:
        """Apply in-memory update; on failure trigger scope rebuild."""
        try:
            if operation == "add":
                self._index.add_entry(entry)
            elif operation == "update":
                self._index.update_entry(entry)
            elif operation == "remove":
                scope_key = self._scope_key(entry)
                self._index.remove_entry(entry.canonical_id, scope_key)
        except Exception:
            logger.exception(
                "Memory update failed (%s) for entry %s; rebuilding scope",
                operation,
                entry.canonical_id,
            )
            scope = entry.scope
            scope_id = entry.scope_id
            await self.rebuild_scope(scope, scope_id)

    @staticmethod
    def _scope_key(entry: RegistryEntry) -> ScopeKey:
        scope_val = (
            entry.scope.value if hasattr(entry.scope, "value") else entry.scope
        )
        return (scope_val, entry.scope_id)
