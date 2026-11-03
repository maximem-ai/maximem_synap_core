"""Merges that can be taken back.

A merge made on certainty (the same spelling, or one judge saying "same" with
no rival) is not logged. A merge made on a second look, or approved by a
reviewer, is: one ``er_merge_log`` row naming the spelling that was folded
into an entity, and the same id stamped on what that mention produced:

* the registry entry gains the spelling as an alias (listed in the row);
* every graph edge created from that mention carries the id in
  ``er_merge_ids`` (see StorageCoordinator), so the edges that came from the
  merged spelling can be told apart from the entity's own.

Undo reads the row back: the aliases are removed from the entry, the spelling
gets a registry entry of its own, and the row is marked undone. The next
mention of the spelling then matches its own entry.

⚠ Moving the stamped graph edges onto a node for the new entry is NOT done
here yet. Until it is, an undo fixes the registry (so the mistake stops
repeating) and the row tells an operator which edges to move.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MERGE_ID_PROPERTY = "er_merge_id"       # on an ExtractedEntity, for the coordinator
EDGE_MERGE_IDS_PROPERTY = "er_merge_ids"  # on a graph edge

DECIDED_BY_SECOND_LOOK = "second_look"
DECIDED_BY_HUMAN = "human"


def new_merge_id() -> str:
    return str(uuid.uuid4())


class InMemoryMergeLog:
    """Same contract as PostgresMergeLog. Tests and the harness."""

    def __init__(self) -> None:
        self.rows: Dict[str, dict] = {}

    async def record(self, **row: Any) -> None:
        self.rows[row["merge_id"]] = {**row, "undone_at": None, "undone_into_id": None}

    async def get(self, merge_id: str) -> Optional[dict]:
        return self.rows.get(merge_id)

    async def mark_undone(self, merge_id: str, undone_into_id: str) -> None:
        self.rows[merge_id].update(undone_at="now", undone_into_id=undone_into_id)


class PostgresMergeLog:
    """``er_merge_log`` (migration 163) through the shared session factory."""

    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    async def record(
        self,
        *,
        merge_id: str,
        scope: str,
        scope_id: Optional[str],
        into_canonical_id: str,
        into_canonical_name: str,
        surface: str,
        entity_type: Optional[str],
        aliases_added: List[str],
        decided_by: str,
        evidence: Optional[dict] = None,
    ) -> None:
        from sqlalchemy import text

        async with self._session_factory() as session:
            await session.execute(
                text(
                    "INSERT INTO er_merge_log (merge_id, scope, scope_id, into_canonical_id, "
                    "into_canonical_name, surface, entity_type, aliases_added, decided_by, evidence) "
                    "VALUES (:merge_id, :scope, :scope_id, :into_canonical_id, :into_canonical_name, "
                    ":surface, :entity_type, CAST(:aliases_added AS JSONB), :decided_by, "
                    "CAST(:evidence AS JSONB)) ON CONFLICT (merge_id) DO NOTHING"
                ),
                {
                    "merge_id": merge_id, "scope": scope, "scope_id": scope_id,
                    "into_canonical_id": into_canonical_id,
                    "into_canonical_name": into_canonical_name,
                    "surface": surface, "entity_type": entity_type,
                    "aliases_added": json.dumps(aliases_added),
                    "decided_by": decided_by,
                    "evidence": json.dumps(evidence) if evidence is not None else None,
                },
            )
            await session.commit()

    async def get(self, merge_id: str) -> Optional[dict]:
        from sqlalchemy import text

        async with self._session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM er_merge_log WHERE merge_id = :merge_id"),
                {"merge_id": merge_id},
            )
            row = result.mappings().first()
            return dict(row) if row is not None else None

    async def mark_undone(self, merge_id: str, undone_into_id: str) -> None:
        from sqlalchemy import text

        async with self._session_factory() as session:
            await session.execute(
                text(
                    "UPDATE er_merge_log SET undone_at = now(), undone_into_id = :into "
                    "WHERE merge_id = :merge_id AND undone_at IS NULL"
                ),
                {"merge_id": merge_id, "into": undone_into_id},
            )
            await session.commit()


async def undo_merge(merge_id: str, merge_log: Any, write_through: Any, index_manager: Any) -> Optional[Any]:
    """Take a logged merge back in the registry. Returns the entry the
    spelling now has, or None when there was nothing to undo.

    Idempotent: a merge already undone is left alone.
    """
    row = await merge_log.get(merge_id)
    if row is None or row.get("undone_at"):
        return None

    added = row.get("aliases_added") or []
    if isinstance(added, str):
        added = json.loads(added)
    scope_key = (row["scope"], row["scope_id"])
    into_id = uuid.UUID(str(row["into_canonical_id"]))

    entry = index_manager.get_entry(into_id, scope_key)
    if entry is not None:
        dropped = {a.lower() for a in added}
        kept = [a for a in (entry.aliases or []) if a.lower() not in dropped]
        if kept != list(entry.aliases or []):
            await write_through.update_entry(into_id, {"aliases": kept})

    surface = row["surface"]
    split = await write_through.create_entry({
        "canonical_name": surface,
        "entity_type": getattr(entry.entity_type, "value", entry.entity_type) if entry else "DOMAIN_ENTITY",
        "scope": row["scope"],
        "scope_id": row["scope_id"],
        "source": "RESOLVED",
        "confidence": 1.0,
        "aliases": [a for a in added if a.lower() != surface.lower()],
        "metadata_": {
            "split_from": str(into_id),
            "undone_merge_id": merge_id,
            **({"entity_subtype": row["entity_type"]} if row.get("entity_type") else {}),
        },
    })
    await merge_log.mark_undone(merge_id, str(split.canonical_id))
    logger.info(
        "Entity merge undone: merge_id=%s surface=%r split from %r",
        merge_id, surface, row["into_canonical_name"],
    )
    return split
