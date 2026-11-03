"""Soft-delete memories and remove every copy that could still serve their text.

Marking ``deleted_at`` hides the row from reads that filter on it. It does not by
itself stop the text being served, because retrieval returns memory text straight
from vector metadata and from Redis, not from the Postgres row. So the same step
also removes the memory and question vectors, marks the graph edges, and clears the
caches. The delete-by-ID route does the first three and never the last; this one
does all four, because "not surfacing anywhere" is the point of it.

Each store is reported separately in ``still_present_in``. A failure in one does not
undo another: the row is already marked, and telling the caller which store still
holds the text is more useful than pretending all of it worked.

The copies that are not keyed by memory id (audit log text, loss manifests, saved
payloads, candidates, profile text, and the chunk, sentence, entity and triple
vectors) are cleared by ``erase_copies``, called right after this.
The memory_actions row this writes carries no text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from sqlalchemy import text

logger = logging.getLogger(__name__)

InvalidateFn = Callable[[], Awaitable[Any]]


@dataclass
class SoftDeleteResult:
    deleted_ids: List[str] = field(default_factory=list)
    # memory id -> its text at the moment it was marked, so the other copies can be
    # found by what they say. Held in memory for this call only; never stored.
    contents: Dict[str, str] = field(default_factory=dict)
    # Asked for but not marked: wrong user, already deleted, or not found.
    skipped_ids: List[str] = field(default_factory=list)
    still_present_in: List[str] = field(default_factory=list)


async def soft_delete_memories(
    session,
    *,
    scope,
    memory_ids: Sequence[str],
    reason: str,
    correlation_id: Optional[str] = None,
    vector=None,
    graph=None,
    invalidate: Optional[InvalidateFn] = None,
    audit=None,
) -> SoftDeleteResult:
    """Mark these memories deleted and clear their copies.

    ``scope`` must be user-level; the UPDATE is bound to that client, customer and
    user, so an ID that belongs to anyone else is skipped, not deleted.
    """
    result = SoftDeleteResult()
    ids = list(dict.fromkeys(str(i) for i in memory_ids))
    if not ids:
        return result
    if scope.user_id is None or scope.customer_id is None:
        raise ValueError("forget by description is scoped to one user; the scope has no user")

    rows = await session.execute(
        text("""
            UPDATE memories
               SET deleted_at = NOW(), deleted_reason = :reason
             WHERE id = ANY(CAST(:ids AS uuid[]))
               AND client_id = CAST(:client_id AS uuid)
               AND customer_id = CAST(:customer_id AS uuid)
               AND user_id = CAST(:user_id AS uuid)
               AND deleted_at IS NULL
         RETURNING id, content
        """),
        {"ids": ids, "reason": reason, "client_id": str(scope.client_id),
         "customer_id": str(scope.customer_id), "user_id": str(scope.user_id)},
    )
    returned = rows.fetchall()
    marked = [str(r[0]) for r in returned]
    result.contents = {str(r[0]): (r[1] or "") for r in returned}
    result.deleted_ids = marked
    result.skipped_ids = [i for i in ids if i not in set(marked)]
    if not marked:
        return result

    # Commit the mark before touching other stores, so a reader that races the
    # cache clear below re-reads a row that is already hidden and not one that
    # is about to be.
    await session.commit()

    if vector is not None:
        try:
            await vector.delete_memory_vectors(memory_ids=marked, scope=scope)
        except Exception as exc:  # noqa: BLE001
            result.still_present_in.append("search index")
            logger.error("forget_vector_cleanup_failed ids=%d: the rows are marked but "
                         "their vectors may still be searchable: %s", len(marked), exc)

    if graph is not None:
        try:
            for mid in marked:
                await graph.soft_delete_memory_from_edges(memory_id=mid, scope=scope,
                                                          redact=True)
        except Exception as exc:  # noqa: BLE001
            result.still_present_in.append("graph")
            logger.error("forget_graph_cleanup_failed: the rows are marked but their "
                         "edges remain: %s", exc)

    if invalidate is not None:
        try:
            await invalidate()
        except Exception as exc:  # noqa: BLE001
            result.still_present_in.append("cache")
            logger.error("forget_cache_invalidation_failed: a cached bundle may serve "
                         "the text until its TTL: %s", exc)

    if audit is not None and correlation_id:
        for mid in marked:
            try:
                # No text in either field: the audit row must not become another
                # copy of what the user asked to forget.
                await audit(correlation_id=correlation_id, scope=scope, memory_id=mid,
                            reason=reason)
            except Exception as exc:  # noqa: BLE001
                logger.warning("forget_audit_failed memory_id=%s (non-fatal): %s", mid, exc)

    return result
