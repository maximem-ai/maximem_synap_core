"""Run a waiting forget the moment the batch it was about has been stored.

A forget that finds nothing is usually early: the note is still in a batch of chat that
has not been sent for saving (a batch goes out after 5 messages or about 5 minutes of
silence), or is being saved. Searching again on a clock cannot know which. So the request
is also remembered with the conversation it came from and how many messages that
conversation had (``forget_retries.conversation_id`` and ``request_position``).

When a batch from that conversation has been fully stored, this runs once for it:

  * find this user's waiting requests from this conversation made at or after the batch's
    first message (a batch that starts after the request cannot hold what it asked about);
  * read the notes that batch just stored;
  * ask the same judge whether any of them is what the request describes;
  * remove those with the normal forget delete, which also clears the other copies.

It costs one indexed lookup per stored batch, and a judge call only when a matching
request is waiting. It never sends a batch early, so it adds no extraction cost.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from sqlalchemy.exc import ProgrammingError

from synap.cloud.application.memory_forget import retry
from synap.cloud.application.memory_forget import retry_store as rstore
from synap.cloud.application.memory_forget import service
from synap.cloud.application.memory_forget.judge import JudgeUnavailable

logger = logging.getLogger(__name__)
_columns_missing_logged = False

REASON_MARKER = "forgotten on request, removed when its chat was saved"


async def after_persist(*, scope, metadata: Optional[Dict[str, Any]],
                        memory_ids: Sequence[str]) -> int:
    """Check waiting forgets against a batch that was just stored. Returns how many notes
    were removed. Never raises: a stored batch must never fail because of this."""
    global _columns_missing_logged
    try:
        return await _after_persist(scope=scope, metadata=metadata, memory_ids=memory_ids)
    except ProgrammingError:
        # Migration 180 is not applied in this region. Say so once, not on every batch.
        if not _columns_missing_logged:
            logger.error("forget_marker_columns_missing: apply migration 180; "
                         "requests are matched by the timed retries only")
            _columns_missing_logged = True
        return 0
    except Exception:
        logger.exception("forget_marker_failed")
        return 0


async def _after_persist(*, scope, metadata, memory_ids) -> int:
    if not retry.enabled():
        return 0
    meta = metadata if isinstance(metadata, dict) else {}
    conversation_id = meta.get("conversation_id")
    source_range = meta.get("source_message_range") or {}
    first = source_range.get("from_position")
    # Only a conversation batch (see conversation/promotion.py) carries both.
    if not conversation_id or first is None:
        return 0
    if getattr(scope, "user_id", None) is None or getattr(scope, "customer_id", None) is None:
        return 0

    from synap.cloud.database.connection import get_db_session
    from synap.cloud.shared.conversation_id_norm import coerce_conversation_id

    removed = 0
    async with get_db_session() as session:
        markers = await rstore.markers_for_batch(
            session, scope=scope, conversation_id=coerce_conversation_id(str(conversation_id)),
            first_position=int(first), window_seconds=retry.note_window_seconds())
        if not markers:
            return 0
        to_position = source_range.get("to_position")
        # A batch that stored nothing (or whose notes an earlier request already used) is
        # still the batch that held these requests: it must be marked covered either way.
        closed = set()
        notes = await rstore.load_notes_by_id(session, scope=scope, memory_ids=memory_ids)
        if notes:
            deps = service.build_deps(session, scope=scope,
                                      correlation_id=f"forget-marker-{markers[0]['id']}")
            for marker in markers:
                if not notes:
                    break
                try:
                    picked = await deps.judge(marker["query"], notes)
                except JudgeUnavailable:
                    # The timed retries are still queued for this request.
                    logger.warning("forget_marker_judge_unavailable id=%s", marker["id"])
                    continue
                if not picked:
                    logger.info("forget_marker_no_match id=%s notes_read=%d",
                                marker["id"], len(notes))
                    continue
                res = await deps.delete(scope, picked, REASON_MARKER)
                result = {"deleted": res.deleted_ids, "skipped": res.skipped_ids,
                          "still_present_in": res.still_present_in, "mode": "marker"}
                closed.add(marker["id"])
                if await rstore.close_if_waiting(session, retry_id=marker["id"],
                                                 status=rstore.DONE,
                                                 attempt=int(marker.get("attempt") or 0),
                                                 result=result):
                    await retry._record_for_dashboard(
                        session, scope=scope, row=marker, picked=picked,
                        status=service._decided_status(picked, res), result=result)
                removed += len(res.deleted_ids)
                gone = set(res.deleted_ids)
                notes = [n for n in notes if n.memory_id not in gone]
                logger.info("forget_marker_done id=%s deleted=%d", marker["id"],
                            len(res.deleted_ids))
        # Requests this batch contained and did not remove: notes created after this come
        # from later batches, so the timed retries stop reading there.
        await rstore.mark_covered(session, retry_ids=[
            m["id"] for m in markers
            if m["id"] not in closed and _holds_the_request(m, to_position)])
    return removed


def _holds_the_request(marker: Dict[str, Any], to_position: Any) -> bool:
    """True when the batch runs up to or past the request's own message."""
    try:
        return int(to_position) >= int(marker["request_position"])
    except (TypeError, ValueError):
        return False
