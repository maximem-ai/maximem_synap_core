"""Search again later for a forget that found nothing, when it was made inside a conversation.

A person who says "remove my phone number" a few seconds after saying it gets "nothing
matches": the note is still in a batch of chat waiting to be sent for saving (5 messages or
about 5 minutes of silence), or is being extracted, saved and indexed (about a minute). The
note lands afterwards and nothing removes it.

Only a forget that names its conversation is queued. Without a conversation id the caller
gets the plain answer ("nothing found") and nothing waits.

Two layers act on a queued request (see ``marker.py`` for the first):

  1. The moment a batch from that conversation is stored, its new notes are checked against
     the request. This is exact: it runs when the note exists, not on a guess.
  2. Timed attempts, counted from when the batch was SENT (``anchor_at``): 2 minutes,
     30 minutes and 5 hours. They exist for the cases layer 1 misses (the judge was down, a
     step failed after the note was saved). Each reads only the notes created since the last
     attempt, and only up to the moment the batch holding the request was checked
     (``covered_at``), so a note said after the request is never mistaken for it.

A request stops at the first removal. A note the user writes about the same subject later is
safe from layer 1 (a batch that starts after the request is never matched) and, once
``covered_at`` is set, from layer 2. Before that the timed window is
``SYNAP_FORGET_RETRY_NOTE_WINDOW_SECONDS`` (default 5 hours).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import timedelta
from typing import Any, Dict, Optional, Sequence, Tuple

from sqlalchemy.exc import ProgrammingError

from synap.cloud.application.memory_forget import retry_store as rstore
from synap.cloud.application.memory_forget import service
from synap.cloud.application.memory_forget.judge import JudgeUnavailable
from synap.cloud.shared import edition

logger = logging.getLogger(__name__)

REASON_RETRY = "forgotten on request, found on a later search"

# Seconds after the request: 2 minutes, 30 minutes, 5 hours.
DEFAULT_SCHEDULE: Tuple[int, ...] = (120, 1800, 18000)

# A note can be saved a little before the first search could see it (saved to Postgres,
# not yet in the vector index), so each search looks slightly further back than the
# time it last ran. Looking at a note twice is harmless: the judge says the same thing.
INDEX_LAG_SECONDS = 120

# A claimed row is invisible to other workers for this long.
LEASE_SECONDS = 600
# An attempt that could not run (judge unavailable) is tried again after this, up to
# MAX_ERRORS times, without using up one of the attempts.
ERROR_RETRY_SECONDS = 300
MAX_ERRORS = 5

# A request made while the conversation still holds messages that were never sent for saving
# parks its first timed attempt this far away; sending the batch moves it to
# ``schedule()[0]`` after the send. It covers a chat that is never flushed.
WAIT_FOR_BATCH_SECONDS = 1800

TICK_SECONDS = 30
CLAIM_BATCH = 20


def enabled() -> bool:
    return os.environ.get("SYNAP_FORGET_RETRY", "true").strip().lower() not in (
        "0", "false", "no", "off")


def schedule() -> Tuple[int, ...]:
    raw = os.environ.get("SYNAP_FORGET_RETRY_SCHEDULE_SECONDS", "").strip()
    if raw:
        try:
            values = tuple(int(p) for p in raw.split(",") if p.strip())
            if values and all(v > 0 for v in values) and list(values) == sorted(values):
                return values
        except ValueError:
            pass
        logger.warning("forget_retry: ignoring SYNAP_FORGET_RETRY_SCHEDULE_SECONDS=%r", raw)
    return DEFAULT_SCHEDULE


def recent_conversation_seconds() -> int:
    """How far back a conversation counts as "active" when the caller names none. A batch that
    could still be in flight comes from a chat with messages inside the idle flush (5 minutes)
    plus the saving time (about 2 minutes); the default leaves a wide margin."""
    raw = os.environ.get("SYNAP_FORGET_RECENT_CONVERSATION_SECONDS", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else 900


async def conversations_to_watch(session, *, scope, user_id: Optional[str],
                                 customer_id: Optional[str],
                                 conversation_id: Optional[str]) -> list:
    """The conversations a request should stay open for.

    The one the caller named, if any. Otherwise the user's recently active conversations,
    found on the server, so the agent does not have to send anything. Never raises; an empty
    list means there is nothing that could still be in flight and nothing will wait.
    """
    if conversation_id:
        return [conversation_id]
    if session is None or not enabled() or not user_id or scope is None \
            or getattr(scope, "instance_id", None) is None:
        return []
    try:
        return await rstore.recent_conversations(
            session, instance_uuid=scope.instance_id, user_id=user_id,
            customer_id=customer_id, window_seconds=recent_conversation_seconds())
    except Exception:
        logger.exception("forget_retry_recent_conversations_failed")
        return []


def note_window_seconds() -> int:
    raw = os.environ.get("SYNAP_FORGET_RETRY_NOTE_WINDOW_SECONDS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return schedule()[-1]


async def schedule_retry(session, *, scope, description: str,
                         requested_by: Optional[str],
                         conversation_id: Optional[str] = None) -> Optional[str]:
    """File a forget that found nothing. Returns the queue row's id, or None when nothing was
    queued (no session, switched off, no conversation id, or the write failed).

    Only a request that names its conversation is queued: the conversation is what lets the
    stored batch be matched to it. Never raises: a failure to queue must not turn a normal
    "nothing matches" answer into an error, and must not claim a retry that does not exist.
    """
    if session is None or not enabled() or not conversation_id:
        return None
    # With the conversation columns first, then without them: a region that has not applied
    # migration 180 cannot store them, but must still queue the timed retries. Each try is in
    # its own savepoint so a failure cannot poison the caller's transaction.
    for conv in (conversation_id, None):
        try:
            async with session.begin_nested():
                return await rstore.enqueue(session, scope=scope, query=description,
                                            requested_by=requested_by,
                                            first_attempt_in_seconds=schedule()[0],
                                            conversation_id=conv,
                                            unsent_wait_seconds=WAIT_FOR_BATCH_SECONDS)
        except Exception:
            logger.exception("forget_retry_enqueue_failed with_conversation=%s", bool(conv))
    return None


async def cancel_retry(session, retry_id: Optional[str]) -> None:
    """The request was answered by the search itself, so the waiting row is not needed.
    Never raises."""
    if not retry_id or session is None:
        return
    try:
        async with session.begin_nested():
            await rstore.close_if_waiting(session, retry_id=retry_id, status=rstore.DONE,
                                          attempt=0, result={"mode": "answered_at_request",
                                                             "deleted": []})
    except Exception:
        logger.exception("forget_retry_cancel_failed id=%s", retry_id)


async def on_batch_sent(*, scope, conversation_id: str, from_position: int) -> int:
    """A batch of this conversation was just sent for saving: start the timed attempts of the
    waiting requests that batch could hold from now. Never raises; returns rows moved."""
    if not enabled() or not conversation_id:
        return 0
    try:
        from synap.cloud.database.connection import get_db_session

        async with get_db_session() as session:
            return await rstore.on_batch_sent(
                session, scope=scope, conversation_id=conversation_id,
                from_position=from_position, first_attempt_in_seconds=schedule()[0])
    except Exception:
        # Before migration 180 there are no conversation rows to move; nothing to report.
        logger.debug("forget_retry_batch_sent_failed", exc_info=True)
        return 0


async def run_attempt(row: Dict[str, Any], *, session, deps: service.ForgetDeps,
                      scope) -> str:
    """One search for one claimed row. Returns the row's new status.

    ``waiting`` (try again later), ``done`` (removed something) or ``gave_up``.
    Raises ``JudgeUnavailable`` when the judge could not decide; the caller records it.
    """
    offsets = schedule()
    attempt = int(row["attempt"])
    checked_at = row["claimed_at"]
    anchor = row.get("anchor_at") or row["requested_at"]
    since = row["last_checked_at"] - timedelta(seconds=INDEX_LAG_SECONDS)
    # Once the batch holding the request was stored and checked, later notes come from later
    # batches (said after the request), so stop there; before that, the plain window.
    until = row.get("covered_at") or (anchor + timedelta(seconds=note_window_seconds()))

    notes = await rstore.load_new_notes(session, scope=scope, since=since, until=until)
    picked = await deps.judge(row["query"], notes) if notes else []

    if picked:
        res = await deps.delete(scope, picked, REASON_RETRY)
        result = {"deleted": res.deleted_ids, "skipped": res.skipped_ids,
                  "still_present_in": res.still_present_in, "attempt": attempt + 1,
                  "mode": "retry"}
        await rstore.finish(session, retry_id=row["id"], status=rstore.DONE,
                            attempt=attempt + 1, result=result)
        await _record_for_dashboard(session, scope=scope, row=row, picked=picked,
                                    status=service._decided_status(picked, res),
                                    result=result)
        logger.info("forget_retry_done id=%s attempt=%d deleted=%d notes_read=%d",
                    row["id"], attempt + 1, len(res.deleted_ids), len(notes))
        return rstore.DONE

    attempt += 1
    if attempt >= len(offsets):
        await rstore.finish(session, retry_id=row["id"], status=rstore.GAVE_UP,
                            attempt=attempt,
                            result={"attempt": attempt, "mode": "retry", "deleted": []})
        logger.info("forget_retry_gave_up id=%s attempts=%d notes_read=%d",
                    row["id"], attempt, len(notes))
        return rstore.GAVE_UP

    await rstore.reschedule(session, retry_id=row["id"], attempt=attempt,
                            next_in_seconds_from_request=offsets[attempt],
                            checked_at=checked_at)
    logger.info("forget_retry_nothing id=%s attempt=%d next_offset_s=%d notes_read=%d",
                row["id"], attempt, offsets[attempt], len(notes))
    return rstore.WAITING


async def _record_for_dashboard(session, *, scope, row, picked: Sequence, status: str,
                                result: Dict[str, Any]) -> None:
    """File what the retry removed beside the direct deletes in the dashboard queue.
    Best effort: the notes are already marked deleted."""
    try:
        async with session.begin_nested():
            await edition.acall(
                "forget_dashboard_queue_write",
                session, scope=scope, query=row["query"], candidates=picked,
                requested_by=row.get("requested_by") or "forget-retry",
                status=status, result=result)
    except Exception:
        logger.exception("forget_retry_record_failed id=%s", row.get("id"))


class ForgetRetryWorker:
    """Polls for due rows. Started from ``application/main.py`` on app startup."""

    def __init__(self, tick_seconds: int = TICK_SECONDS):
        self.tick_seconds = tick_seconds
        self._stopped = asyncio.Event()
        self._missing_table_logged = False

    def stop(self) -> None:
        self._stopped.set()

    async def run_forever(self) -> None:
        logger.info("forget_retry_worker_starting tick_seconds=%s enabled=%s schedule=%s",
                    self.tick_seconds, enabled(), schedule())
        while not self._stopped.is_set():
            try:
                if enabled():
                    await self.tick_once()
            except Exception:
                logger.exception("forget_retry_tick_failed")
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self.tick_seconds)
            except asyncio.TimeoutError:
                continue

    async def tick_once(self) -> int:
        """Claim the due rows and run each. Returns how many were run."""
        from synap.cloud.database.connection import get_db_session

        try:
            async with get_db_session() as session:
                due = await rstore.claim_due(session, limit=CLAIM_BATCH,
                                             lease_seconds=LEASE_SECONDS)
        except ProgrammingError:
            # Migration 179 is not applied in this region. Say so once, not every tick.
            if not self._missing_table_logged:
                logger.error("forget_retry_table_missing: apply migration 179; retries are off")
                self._missing_table_logged = True
            return 0
        for row in due:
            await self._run_one(row)
        return len(due)

    async def _run_one(self, row: Dict[str, Any]) -> None:
        from synap.cloud.database.connection import get_db_session

        scope = service.scope_from_request(row)
        try:
            async with get_db_session() as session:
                deps = service.build_deps(session, scope=scope,
                                          correlation_id=f"forget-retry-{row['id']}")
                await run_attempt(row, session=session, deps=deps, scope=scope)
        except Exception as exc:
            # Judge down, a store down, a bad row: none of these may use up an
            # attempt or stop the other rows. The row is tried again shortly.
            kind = "judge_unavailable" if isinstance(exc, JudgeUnavailable) else "error"
            logger.warning("forget_retry_attempt_failed id=%s kind=%s", row["id"], kind,
                           exc_info=kind == "error")
            await self._note_error(row)

    async def _note_error(self, row: Dict[str, Any]) -> None:
        from synap.cloud.database.connection import get_db_session

        try:
            async with get_db_session() as session:
                errors = await rstore.record_error(session, retry_id=row["id"],
                                                   retry_in_seconds=ERROR_RETRY_SECONDS)
                if errors >= MAX_ERRORS:
                    await rstore.finish(session, retry_id=row["id"], status=rstore.GAVE_UP,
                                        attempt=int(row["attempt"]),
                                        result={"mode": "retry", "deleted": [],
                                                "error": "attempts could not run"})
                    logger.error("forget_retry_gave_up_on_errors id=%s errors=%d",
                                 row["id"], errors)
        except Exception:
            logger.exception("forget_retry_record_error_failed id=%s", row.get("id"))
