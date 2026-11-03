"""Mentions that wait for review, and what is kept for them meanwhile.

The rule: an entity is merged into another, or created, only when resolution
is sure. A mention it is not sure about is HELD:

* Nothing is written to the graph for it. No node, and no edge that touches
  it. The note's memories are stored as always.
* One ``er_held_mentions`` row keeps the mention as extracted, the candidates
  it was weighed against, every relationship of the note that touches it (its
  *links*), and the note's memories that mention it, so that everything a
  settled mention would have got can be written once this one is settled. See
  review_apply.ReviewApplier.
* The name is put in the registry as a PENDING entry. It is never something
  to merge into. It is there so later notes can see it: a mention that would
  have matched it joins the same review group instead of being created as a
  new entity, or matched to nothing.

Without this, an unsure mention got a placeholder node of its own, and the
entity it really was looked absent to every later note.

On unless SYNAP_ER_HOLD_UNSURE=false, inside adjudicated mode, when a hold
store is configured.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HOLD_ID_PROPERTY = "er_hold_id"                  # on a held ExtractedEntity
PENDING_ENTRY_PROPERTY = "er_pending_entry_id"   # the review group it belongs to
PENDING_KEY = "pending"                          # in a registry entry's metadata_
FOUNDING_HOLD_KEY = "hold_id"                    # the hold whose name the pending entry carries
JOINED_BY_PROPERTY = "er_joined_by"
JOINED_BY_NAME = "name"                          # joined its group by writing the very same name

STATUS_PENDING = "pending"
STATUS_APPLYING = "applying"     # claimed by one reviewer's request; not listed, not claimable
STATUS_APPLIED = "applied"

# A claim older than this is taken to belong to a request that died, and may be
# taken over. A decision is a few registry writes and one graph write.
CLAIM_TTL_SECONDS = 300


def hold_unsure_enabled() -> bool:
    return os.environ.get("SYNAP_ER_HOLD_UNSURE", "true").strip().lower() in ("true", "1", "yes")


def is_missing_table(exc: BaseException) -> bool:
    """Whether a database error means a table does not exist (Postgres 42P01).

    Migrations here are applied by hand, so a server can run this code before
    its database has the er_* tables.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, "sqlstate", None) == "42P01" or type(exc).__name__ == "UndefinedTableError":
            return True
        exc = getattr(exc, "orig", None) or exc.__cause__
    return False


def new_hold_id() -> str:
    return str(uuid.uuid4())


def is_pending_entry(entry: Any) -> bool:
    return entry is not None and bool((getattr(entry, "metadata_", None) or {}).get(PENDING_KEY))


def pending_checker(index_manager: Any) -> Callable[[str], bool]:
    """candidate key -> whether it is a pending registry entry."""

    def is_pending(key: Any) -> bool:
        try:
            return is_pending_entry(index_manager.get_entry(uuid.UUID(str(key))))
        except (ValueError, AttributeError):
            return False

    return is_pending


def split_held(entities: List[Any], relationships: List[Any]) -> Tuple[List[Any], List[Any], Dict[str, List[dict]]]:
    """Take held entities, and every relationship touching one, out of a note.

    Returns (settled entities, settled relationships, hold id -> links). A
    link is the relationship as extracted plus its other end: the settled
    entity as it was resolved in this note, or the hold id when the other end
    is held too.
    """
    held = {e.entity_id: e for e in entities if e.properties.get(HOLD_ID_PROPERTY)}
    links: Dict[str, List[dict]] = {e.properties[HOLD_ID_PROPERTY]: [] for e in held.values()}
    if not held:
        return entities, relationships, links
    by_id = {e.entity_id: e for e in entities}
    kept = []
    for rel in relationships:
        ends = (rel.source_entity_id, rel.target_entity_id)
        if ends[0] not in held and ends[1] not in held:
            kept.append(rel)
            continue
        if ends[0] == ends[1]:
            continue
        for mine, theirs in (ends, ends[::-1]):
            if mine not in held:
                continue
            other = by_id.get(theirs)
            if other is None:
                continue      # the extractor named an entity it did not return
            links[held[mine].properties[HOLD_ID_PROPERTY]].append({
                "relationship": rel.model_dump(mode="json"),
                "other_hold_id": other.properties.get(HOLD_ID_PROPERTY) if theirs in held else None,
                "other_entity": None if theirs in held else other.model_dump(mode="json"),
            })
    return [e for e in entities if e.entity_id not in held], kept, links


class PendingAwareAdjudicator:
    """Keeps a pending name from standing in the way of a sure match.

    A pending entry is offered as a candidate like any other. When the judge
    is sure the mention is one real entry, its doubt about a pending name is
    dropped: "Bob Smith" is Robert Smith whether or not an "R. Smith" is
    waiting. In every other case the pending verdicts stand, and a SAME or
    UNCLEAR on one sends the mention to that review group.
    """

    def __init__(self, inner: Any, is_pending: Callable[[str], bool]) -> None:
        self._inner = inner
        self._is_pending = is_pending

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def adjudicate(self, items: List[Any], context: Optional[str]) -> Dict[str, Dict[str, Any]]:
        from synap_er.pipeline.adjudication import Decision, Verdict, decide

        verdicts = await self._inner.adjudicate(items, context)
        for item in items:
            mine = verdicts.get(item.key)
            if not mine:
                continue
            pending = {c.key for c in item.candidates if not c.in_document and self._is_pending(c.key)}
            real = {k: v for k, v in mine.items() if k not in pending}
            if pending and real and decide(real)[0] is Decision.MATCH:
                for key in pending & set(mine):
                    mine[key] = Verdict.DIFFERENT
        return verdicts


class InMemoryHoldStore:
    """Same contract as PostgresHoldStore. Tests and the harness."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.rows: Dict[str, dict] = {}
        self._clock = clock
        self._claimed_at: Dict[str, float] = {}

    async def add(self, record: dict) -> None:
        self.rows[record["hold_id"]] = {
            "status": STATUS_PENDING, "links": [], "memories": [], "document_id": None,
            "joined_by": None,
            "resolved_as": None, "resolved_canonical_id": None,
            "resolved_canonical_name": None, "decided_by": None, **record,
        }

    async def set_links(self, hold_id: str, links: List[dict], document_id: Optional[str],
                        memories: Optional[List[dict]] = None) -> None:
        if hold_id in self.rows:
            self.rows[hold_id].update(links=links, document_id=document_id, memories=memories or [])

    async def get(self, hold_id: str) -> Optional[dict]:
        return self.rows.get(hold_id)

    async def pending(self, scope_id: Optional[str] = None) -> List[dict]:
        return [
            r for r in self.rows.values()
            if r["status"] == STATUS_PENDING and (
                scope_id is None or r["scope_id"] == scope_id
                # a hold kept at a user's scope still belongs to that customer
                or (r.get("tenant") or {}).get("customer_id") == scope_id)
        ]

    async def pending_in_group(self, pending_entry_id: str) -> List[dict]:
        return [
            r for r in self.rows.values()
            if r["status"] == STATUS_PENDING and r["pending_entry_id"] == pending_entry_id
        ]

    async def for_client(self, client_uuid: str, status: Optional[str] = STATUS_PENDING,
                         limit: int = 200) -> List[dict]:
        """A client's held mentions, across all its customers, oldest first."""
        rows = [
            r for r in self.rows.values()
            if str((r.get("tenant") or {}).get("client_id")) == str(client_uuid)
            and (status is None or r["status"] == status)
        ]
        return rows[:limit]

    async def claim(self, hold_id: str, decided_by: str) -> Optional[dict]:
        """Take the hold for one request, or None when it is missing, applied,
        or claimed by a request that is still alive. Atomic."""
        row = self.rows.get(hold_id)
        if row is None:
            return None
        stale = (row["status"] == STATUS_APPLYING
                 and self._clock() - self._claimed_at.get(hold_id, 0) > CLAIM_TTL_SECONDS)
        if row["status"] != STATUS_PENDING and not stale:
            return None
        row.update(status=STATUS_APPLYING, decided_by=decided_by)
        self._claimed_at[hold_id] = self._clock()
        return dict(row)

    async def save_decision(self, hold_id: str, *, resolved_as: str, canonical_id: str,
                            canonical_name: str, entity: dict) -> None:
        """Keep what was decided while the graph write is still to come, so a
        retry finishes the same decision instead of making a second entry."""
        row = self.rows[hold_id]
        if row["status"] == STATUS_APPLYING:
            row.update(resolved_as=resolved_as, resolved_canonical_id=canonical_id,
                       resolved_canonical_name=canonical_name, entity=entity)

    async def release(self, hold_id: str) -> None:
        """Give a claim back after a failure; the hold waits for review again."""
        row = self.rows.get(hold_id)
        if row is not None and row["status"] == STATUS_APPLYING:
            row.update(status=STATUS_PENDING, decided_by=None)

    async def mark_applied(self, hold_id: str, *, resolved_as: str, canonical_id: str,
                           canonical_name: str, decided_by: str, entity: dict) -> bool:
        if self.rows[hold_id]["status"] == STATUS_APPLIED:
            return False
        self.rows[hold_id].update(
            status=STATUS_APPLIED, resolved_as=resolved_as, resolved_canonical_id=canonical_id,
            resolved_canonical_name=canonical_name, decided_by=decided_by, entity=entity,
        )
        return True


_COLUMNS = (
    "hold_id, scope, scope_id, pending_entry_id, mention, entity_type, joined_by, document_id, snippet, "
    "candidates, tenant, entity, links, memories, status, resolved_as, resolved_canonical_id, "
    "resolved_canonical_name, decided_by, created_at, resolved_at"
)
_JSON = ("candidates", "tenant", "entity", "links", "memories")


def _row(mapping: Any) -> dict:
    row = dict(mapping)
    for key in _JSON:
        if isinstance(row.get(key), str):
            row[key] = json.loads(row[key])
    for key in ("hold_id", "pending_entry_id", "resolved_canonical_id"):
        if row.get(key) is not None:
            row[key] = str(row[key])
    for key in ("created_at", "resolved_at"):
        if row.get(key) is not None and hasattr(row[key], "isoformat"):
            row[key] = row[key].isoformat()
    return row


class PostgresHoldStore:
    """``er_held_mentions`` (migration 164) through the shared session factory."""

    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    async def add(self, record: dict) -> None:
        from sqlalchemy import text

        async with self._session_factory() as session:
            await session.execute(
                text(
                    "INSERT INTO er_held_mentions (hold_id, scope, scope_id, pending_entry_id, "
                    "mention, entity_type, joined_by, snippet, candidates, tenant, entity) "
                    "VALUES (:hold_id, :scope, :scope_id, :pending_entry_id, :mention, "
                    ":entity_type, :joined_by, :snippet, CAST(:candidates AS JSONB), CAST(:tenant AS JSONB), "
                    "CAST(:entity AS JSONB))"
                ),
                {
                    **{k: record.get(k) for k in (
                        "hold_id", "scope", "scope_id", "pending_entry_id", "mention",
                        "entity_type", "joined_by", "snippet")},
                    **{k: json.dumps(record.get(k)) for k in ("candidates", "tenant", "entity")},
                },
            )
            await session.commit()

    async def set_links(self, hold_id: str, links: List[dict], document_id: Optional[str],
                        memories: Optional[List[dict]] = None) -> None:
        from sqlalchemy import text

        async with self._session_factory() as session:
            await session.execute(
                text(
                    "UPDATE er_held_mentions SET links = CAST(:links AS JSONB), "
                    "memories = CAST(:memories AS JSONB), document_id = :document_id "
                    "WHERE hold_id = :hold_id"
                ),
                {"hold_id": hold_id, "links": json.dumps(links),
                 "memories": json.dumps(memories or []), "document_id": document_id},
            )
            await session.commit()

    async def _select(self, where: str, params: dict, limit: bool = False) -> List[dict]:
        from sqlalchemy import text

        async with self._session_factory() as session:
            result = await session.execute(
                text(f"SELECT {_COLUMNS} FROM er_held_mentions WHERE {where} ORDER BY created_at"
                     + (" LIMIT :limit" if limit else "")),
                params,
            )
            return [_row(m) for m in result.mappings().all()]

    async def get(self, hold_id: str) -> Optional[dict]:
        rows = await self._select("hold_id = :hold_id", {"hold_id": hold_id})
        return rows[0] if rows else None

    async def pending(self, scope_id: Optional[str] = None) -> List[dict]:
        if scope_id is None:
            return await self._select("status = 'pending'", {})
        # A hold kept at a user's scope (SYNAP_ER_SCOPED_REGISTRY) still belongs
        # to that customer; tenant carries the ingestion scope it came from.
        return await self._select(
            "status = 'pending' AND (scope_id = :scope_id OR tenant->>'customer_id' = :scope_id)",
            {"scope_id": scope_id})

    async def pending_in_group(self, pending_entry_id: str) -> List[dict]:
        return await self._select(
            "status = 'pending' AND pending_entry_id = :entry", {"entry": pending_entry_id}
        )

    async def for_client(self, client_uuid: str, status: Optional[str] = STATUS_PENDING,
                         limit: int = 200) -> List[dict]:
        """A client's held mentions, across all its customers, oldest first."""
        where = "tenant->>'client_id' = :client"
        params: Dict[str, Any] = {"client": str(client_uuid), "limit": int(limit)}
        if status is not None:
            where += " AND status = :status"
            params["status"] = status
        return await self._select(where, params, limit=True)

    async def claim(self, hold_id: str, decided_by: str) -> Optional[dict]:
        """Take the hold for one request, or None when it is missing, applied,
        or claimed by a request that is still alive. One UPDATE, so two
        reviewers cannot both get it. resolved_at doubles as the claim time
        until the hold is applied."""
        from sqlalchemy import text

        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "UPDATE er_held_mentions SET status = 'applying', decided_by = :decided_by, "
                    "resolved_at = now() WHERE hold_id = :hold_id AND (status = 'pending' OR "
                    "(status = 'applying' AND resolved_at < now() - make_interval(secs => :ttl))) "
                    f"RETURNING {_COLUMNS}"
                ),
                {"hold_id": hold_id, "decided_by": decided_by, "ttl": CLAIM_TTL_SECONDS},
            )
            row = result.mappings().first()
            await session.commit()
            return _row(row) if row is not None else None

    async def save_decision(self, hold_id: str, *, resolved_as: str, canonical_id: str,
                            canonical_name: str, entity: dict) -> None:
        """Keep what was decided while the graph write is still to come, so a
        retry finishes the same decision instead of making a second entry."""
        from sqlalchemy import text

        async with self._session_factory() as session:
            await session.execute(
                text(
                    "UPDATE er_held_mentions SET resolved_as = :resolved_as, "
                    "resolved_canonical_id = :canonical_id, resolved_canonical_name = :canonical_name, "
                    "entity = CAST(:entity AS JSONB) WHERE hold_id = :hold_id AND status = 'applying'"
                ),
                {"hold_id": hold_id, "resolved_as": resolved_as, "canonical_id": canonical_id,
                 "canonical_name": canonical_name, "entity": json.dumps(entity)},
            )
            await session.commit()

    async def release(self, hold_id: str) -> None:
        """Give a claim back after a failure; the hold waits for review again."""
        from sqlalchemy import text

        async with self._session_factory() as session:
            await session.execute(
                text(
                    "UPDATE er_held_mentions SET status = 'pending', decided_by = NULL, "
                    "resolved_at = NULL WHERE hold_id = :hold_id AND status = 'applying'"
                ),
                {"hold_id": hold_id},
            )
            await session.commit()

    async def mark_applied(self, hold_id: str, *, resolved_as: str, canonical_id: str,
                           canonical_name: str, decided_by: str, entity: dict) -> bool:
        from sqlalchemy import text

        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "UPDATE er_held_mentions SET status = 'applied', resolved_as = :resolved_as, "
                    "resolved_canonical_id = :canonical_id, resolved_canonical_name = :canonical_name, "
                    "decided_by = :decided_by, entity = CAST(:entity AS JSONB), resolved_at = now() "
                    "WHERE hold_id = :hold_id AND status IN ('pending', 'applying')"
                ),
                {
                    "hold_id": hold_id, "resolved_as": resolved_as, "canonical_id": canonical_id,
                    "canonical_name": canonical_name, "decided_by": decided_by,
                    "entity": json.dumps(entity),
                },
            )
            await session.commit()
            return bool(result.rowcount)
