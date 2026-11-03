"""The retry queue behind forget (table ``forget_retries``, migration 179).

Only a direct forget that matched nothing files a row. Rows are claimed with a lease
(``next_attempt_at`` pushed forward) under ``FOR UPDATE SKIP LOCKED``, so two workers
never run the same row and a crashed worker's row comes back after the lease.
"""

from __future__ import annotations

import json
import uuid as _uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text

from synap.cloud.application.memory_forget.models import Candidate

WAITING, DONE, GAVE_UP = "waiting", "done", "gave_up"

# Where the timed retries are counted from. ``to_jsonb(...)`` instead of the bare column so the
# statement also runs in a region that has not applied migration 180 yet (no ``anchor_at``).
_ANCHOR_SQL = "COALESCE((to_jsonb(forget_retries)->>'anchor_at')::timestamptz, requested_at)"

# Notes read per attempt. A user rarely has more than a handful of new notes inside the
# window; the cap only keeps one judge call bounded.
NOTE_LIMIT = 100


async def enqueue(session, *, scope, query: str, requested_by: Optional[str],
                  first_attempt_in_seconds: int,
                  conversation_id: Optional[str] = None,
                  unsent_wait_seconds: int = 1800) -> str:
    """File a forget that found nothing.

    With a ``conversation_id`` the row also records the conversation's message count right
    now (``request_position``), which is what lets a batch stored later be matched to it
    (migration 180). If that conversation still has messages that were never sent for saving,
    the first timed attempt is parked ``unsent_wait_seconds`` away and moved to
    ``first_attempt_in_seconds`` after the batch is sent (``on_batch_sent``); otherwise it is
    ``first_attempt_in_seconds`` from now. Without a conversation id the statement names only
    the migration 179 columns, so a region that has not applied 180 can still queue.
    """
    retry_id = str(_uuid.uuid4())
    params = {"id": retry_id, "client_id": str(scope.client_id),
              "instance_id": str(scope.instance_id) if scope.instance_id else None,
              "customer_id": str(scope.customer_id) if scope.customer_id else None,
              "user_id": str(scope.user_id), "scope_level": scope.scope_level,
              "query": query, "requested_by": requested_by,
              "first": int(first_attempt_in_seconds)}
    if conversation_id:
        params["conversation_id"] = conversation_id
        params["unsent_wait"] = int(unsent_wait_seconds)
        await session.execute(
            text("""
                INSERT INTO forget_retries
                    (id, client_id, instance_id, customer_id, user_id, scope_level, query,
                     requested_by, requested_at, next_attempt_at, last_checked_at,
                     conversation_id, request_position, anchor_at)
                VALUES (CAST(:id AS uuid), CAST(:client_id AS uuid), CAST(:instance_id AS uuid),
                        CAST(:customer_id AS uuid), CAST(:user_id AS uuid), :scope_level, :query,
                        :requested_by, NOW(),
                        CASE WHEN COALESCE((SELECT message_count > promoted_message_count
                                              FROM conversation
                                             WHERE conversation_id = CAST(:conversation_id AS text)),
                                           false)
                             THEN NOW() + (:unsent_wait * INTERVAL '1 second')
                             ELSE NOW() + (:first * INTERVAL '1 second') END,
                        NOW(),
                        CAST(:conversation_id AS text),
                        (SELECT message_count FROM conversation
                          WHERE conversation_id = CAST(:conversation_id AS text)),
                        NOW())
            """), params)
    else:
        await session.execute(
            text("""
                INSERT INTO forget_retries
                    (id, client_id, instance_id, customer_id, user_id, scope_level, query,
                     requested_by, requested_at, next_attempt_at, last_checked_at)
                VALUES (CAST(:id AS uuid), CAST(:client_id AS uuid), CAST(:instance_id AS uuid),
                        CAST(:customer_id AS uuid), CAST(:user_id AS uuid), :scope_level, :query,
                        :requested_by, NOW(), NOW() + (:first * INTERVAL '1 second'), NOW())
            """), params)
    return retry_id


async def claim_due(session, *, limit: int, lease_seconds: int) -> List[Dict[str, Any]]:
    """Rows that are waiting and due, each pushed ``lease_seconds`` ahead so nobody
    else takes it while this worker is on it."""
    rows = (await session.execute(
        text("""
            UPDATE forget_retries r
               SET next_attempt_at = NOW() + (:lease * INTERVAL '1 second')
             WHERE r.id IN (
                   SELECT id FROM forget_retries
                    WHERE status = 'waiting' AND next_attempt_at <= NOW()
                    ORDER BY next_attempt_at
                    LIMIT :limit
                      FOR UPDATE SKIP LOCKED)
         RETURNING r.id, r.client_id, r.instance_id, r.customer_id, r.user_id, r.scope_level,
                   r.query, r.requested_by, r.requested_at, r.attempt, r.errors,
                   r.last_checked_at, NOW() AS claimed_at,
                   (to_jsonb(r)->>'anchor_at')::timestamptz AS anchor_at,
                   (to_jsonb(r)->>'covered_at')::timestamptz AS covered_at
        """),
        {"limit": int(limit), "lease": int(lease_seconds)},
    )).mappings().all()
    return [_shape(r) for r in rows]


async def reschedule(session, *, retry_id: str, attempt: int, next_in_seconds_from_request: int,
                     checked_at: datetime) -> None:
    """An attempt ran and found nothing: the next one is counted from the anchor, which is
    when the batch was sent (or the request, when no batch was waiting)."""
    await session.execute(
        text("""
            UPDATE forget_retries
               SET attempt = :attempt, last_checked_at = :checked_at,
                   next_attempt_at = """ + _ANCHOR_SQL + """ + (:offset * INTERVAL '1 second')
             WHERE id = CAST(:id AS uuid)
        """),
        {"id": retry_id, "attempt": int(attempt), "checked_at": checked_at,
         "offset": int(next_in_seconds_from_request)},
    )


async def record_error(session, *, retry_id: str, retry_in_seconds: int) -> int:
    """An attempt could not run. Returns how many have failed so far."""
    row = await session.execute(
        text("""
            UPDATE forget_retries
               SET errors = errors + 1,
                   next_attempt_at = NOW() + (:wait * INTERVAL '1 second')
             WHERE id = CAST(:id AS uuid)
         RETURNING errors
        """),
        {"id": retry_id, "wait": int(retry_in_seconds)},
    )
    return int(row.scalar() or 0)


async def finish(session, *, retry_id: str, status: str, attempt: int,
                 result: Dict[str, Any]) -> None:
    """Close the row and blank the user's words."""
    await session.execute(
        text("""
            UPDATE forget_retries
               SET status = :status, attempt = :attempt, query = '',
                   result = CAST(:result AS jsonb), finished_at = NOW()
             WHERE id = CAST(:id AS uuid)
        """),
        {"id": retry_id, "status": status, "attempt": int(attempt),
         "result": json.dumps(result)},
    )


async def load_new_notes(session, *, scope, since: datetime, until: datetime) -> List[Candidate]:
    """The user's live notes created in (since, until], newest first.

    Read from Postgres, not the vector index, so a note that is saved but not yet
    indexed is still seen. Same filters as the first search: this one user, not
    superseded, deprecated or deleted.
    """
    rows = await session.execute(
        text("""
            SELECT id, content, memory_type FROM memories
             WHERE client_id = CAST(:client_id AS uuid)
               AND customer_id = CAST(:customer_id AS uuid)
               AND user_id = CAST(:user_id AS uuid)
               AND created_at > :since AND created_at <= :until
               AND superseded_by IS NULL AND deprecated_at IS NULL
               AND deleted_at IS NULL
             ORDER BY created_at DESC
             LIMIT :limit
        """),
        {"client_id": str(scope.client_id), "customer_id": str(scope.customer_id),
         "user_id": str(scope.user_id), "since": since, "until": until, "limit": NOTE_LIMIT},
    )
    return [Candidate(str(r[0]), r[1] or "", r[2]) for r in rows.fetchall()]


async def mark_covered(session, *, retry_ids: Sequence[str]) -> None:
    """The batch holding these requests has been stored and checked. Timed retries read only
    notes created up to now, so a note from a later batch is never mistaken for it."""
    ids = [str(i) for i in retry_ids if i]
    if not ids:
        return
    await session.execute(
        text("""
            UPDATE forget_retries SET covered_at = NOW()
             WHERE id = ANY(CAST(:ids AS uuid[])) AND status = 'waiting' AND covered_at IS NULL
        """),
        {"ids": ids},
    )


async def on_batch_sent(session, *, scope, conversation_id: str, from_position: int,
                        first_attempt_in_seconds: int) -> int:
    """A batch of this conversation was just sent for saving. Waiting requests made at or
    after its first message start their timed attempts from now. Returns rows moved."""
    row = await session.execute(
        text("""
            UPDATE forget_retries
               SET anchor_at = NOW(), attempt = 0,
                   next_attempt_at = NOW() + (:first * INTERVAL '1 second')
             WHERE user_id = CAST(:user_id AS uuid)
               AND client_id = CAST(:client_id AS uuid)
               AND conversation_id = :conversation_id
               AND status = 'waiting'
               AND request_position IS NOT NULL
               AND request_position >= :from_position
         RETURNING id
        """),
        {"user_id": str(scope.user_id), "client_id": str(scope.client_id),
         "conversation_id": conversation_id, "from_position": int(from_position),
         "first": int(first_attempt_in_seconds)},
    )
    return len(row.fetchall())


async def recent_conversations(session, *, instance_uuid, user_id: str,
                               customer_id: Optional[str], window_seconds: int,
                               limit: int = 5) -> List[str]:
    """This user's conversations that were active recently, newest first.

    Lets a forget that was not told its conversation find it on the server: a batch of chat
    that is still being saved belongs to a conversation that had messages in the last few
    minutes. ``conversation.user_id`` / ``customer_id`` hold the external ids the caller used,
    and ``conversation.instance_id`` the external instance id, so the instance is matched
    through its UUID. Only ids are returned; the request position is read when the row is
    filed (``enqueue``).
    """
    rows = await session.execute(
        text("""
            SELECT c.conversation_id
              FROM conversation c
              JOIN instance i ON i.instance_id = c.instance_id
             WHERE i.uuid = CAST(:instance_uuid AS uuid)
               AND c.user_id = :user_id
               AND (CAST(:customer_id AS text) IS NULL OR c.customer_id = :customer_id)
               AND c.last_message_at > NOW() - (:window * INTERVAL '1 second')
             ORDER BY c.last_message_at DESC
             LIMIT :limit
        """),
        {"instance_uuid": str(instance_uuid), "user_id": user_id, "customer_id": customer_id,
         "window": int(window_seconds), "limit": int(limit)},
    )
    return [str(r[0]) for r in rows.fetchall()]


async def markers_for_batch(session, *, scope, conversation_id: str, first_position: int,
                            window_seconds: int) -> List[Dict[str, Any]]:
    """Waiting forgets from this conversation that were made at or after the batch's first
    message, so the batch could hold what they ask about.

    A batch that starts AFTER the request (``first_position > request_position``) was
    written after it and is never matched, which is what stops a later "my new number is X"
    from being removed. Rows older than the window are left to the timed retries.
    """
    rows = (await session.execute(
        text("""
            SELECT id, query, requested_by, requested_at, request_position, attempt
              FROM forget_retries
             WHERE user_id = CAST(:user_id AS uuid)
               AND client_id = CAST(:client_id AS uuid)
               AND conversation_id = :conversation_id
               AND status = 'waiting'
               AND request_position IS NOT NULL
               AND request_position >= :first_position
               AND requested_at > NOW() - (:window * INTERVAL '1 second')
             ORDER BY requested_at
        """),
        {"user_id": str(scope.user_id), "client_id": str(scope.client_id),
         "conversation_id": conversation_id, "first_position": int(first_position),
         "window": int(window_seconds)},
    )).mappings().all()
    return [_shape(r) for r in rows]


async def load_notes_by_id(session, *, scope, memory_ids: Sequence[str]) -> List[Candidate]:
    """The user's live notes with these ids (what a batch just stored), same filters as the
    first search: this one user, not superseded, deprecated or deleted."""
    ids = [str(i) for i in memory_ids if i]
    if not ids:
        return []
    rows = await session.execute(
        text("""
            SELECT id, content, memory_type FROM memories
             WHERE id = ANY(CAST(:ids AS uuid[]))
               AND client_id = CAST(:client_id AS uuid)
               AND customer_id = CAST(:customer_id AS uuid)
               AND user_id = CAST(:user_id AS uuid)
               AND superseded_by IS NULL AND deprecated_at IS NULL
               AND deleted_at IS NULL
        """),
        {"ids": ids, "client_id": str(scope.client_id),
         "customer_id": str(scope.customer_id), "user_id": str(scope.user_id)},
    )
    return [Candidate(str(r[0]), r[1] or "", r[2]) for r in rows.fetchall()]


async def close_if_waiting(session, *, retry_id: str, status: str, attempt: int,
                           result: Dict[str, Any]) -> bool:
    """Close a row once. False when someone else closed it first (two batches, or the timed
    retry, matched the same request), so only one of them records the outcome."""
    row = await session.execute(
        text("""
            UPDATE forget_retries
               SET status = :status, attempt = :attempt, query = '',
                   result = CAST(:result AS jsonb), finished_at = NOW()
             WHERE id = CAST(:id AS uuid) AND status = 'waiting'
         RETURNING id
        """),
        {"id": retry_id, "status": status, "attempt": int(attempt),
         "result": json.dumps(result)},
    )
    return row.first() is not None


def _shape(row) -> Dict[str, Any]:
    d = dict(row)
    for key in ("id", "client_id", "instance_id", "customer_id", "user_id"):
        if d.get(key) is not None:
            d[key] = str(d[key])
    return d
