"""Settle open anchors at the end of a document's ingestion.

An ``event_anchor`` with no ``event_start`` is an open question: we know the event
happened a stated distance from another event, and we do not yet know when that
other event was. The answer often arrives later. In the Airbnb scope, session 1
says the stay was "booked three months in advance" and session 2 dates the wedding,
and session 1 was ingested first — so nothing running only at session 1's ingest
could ever have resolved it.

This runs after each document persists, for that document's scope. That is the
right trigger on its own merits: new information has just landed, which is exactly
when a previously unresolvable anchor can become resolvable, in either direction —
this document may date an older memory's anchor, or an older memory may date this
one's.

It is deliberately NOT a consolidation node. Consolidation is not run on benchmark
data by choice, so anything living there would be invisible to every run we measure.

Two things bound the damage a mistake here can do:

- the UPDATE carries ``AND event_start IS NULL``, so this can only ever FILL a
  date, never change one. A bug in the caller cannot overwrite a correct
  resolution, and the worst case is that an open question stays open.
- every guardrail lives in ``event_anchor.resolve_anchor`` and prefers refusing.

Which stored memory IS the event an anchor names is decided by the Jev screen, not
by comparing words. Word containment used to do it and was DELETED: it missed "a
wedding for his best friend" against "best friend's wedding" over one apostrophe,
and could never see "the user's best friend got married". There is no word-matching
fallback for when the judge is down either, because the same document ingested
twice would then get different dates depending on whether Jev happened to be up.
A judge that is down costs a deferral, not a date: the anchor stays open and the
next document's persist asks again. See the comment on the screen call below.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from synap.cloud.agents.ingestion.services.event_anchor import (
    AnchorCandidate,
    resolve_anchor,
)

logger = logging.getLogger(__name__)

_FLAG_ENV = "SYNAP_RESOLVE_OPEN_ANCHORS"
_BY_INSTANCE_ENV = "SYNAP_RESOLVE_OPEN_ANCHORS_BY_INSTANCE"
_TRUTHY = {"1", "true", "on", "yes"}

# A scope's dated memories are the candidate pool. Capped so one enormous scope
# cannot turn a per-document hook into a long scan.
CANDIDATE_CAP = 400
OPEN_ANCHOR_CAP = 50

_STOPWORDS = frozenset("""
the a an and or of to in on at for with their they it is was were this that
be been has have had do does did will would can could should user users about
from by as not but my our your his her its
""".split())


def enabled_for(instance_id: Optional[object]) -> bool:
    """Whether the open-anchor resolver is on for this instance.

    One switch for the whole set, ``SYNAP_TEMPORAL_V2``; this feature's own pair
    of variables still overrides it. See services/temporal_flags for precedence.
    """
    from synap.cloud.agents.ingestion.services import temporal_flags
    return temporal_flags.enabled(_FLAG_ENV, _BY_INSTANCE_ENV, instance_id)


def _loads(value: Any) -> Optional[Dict[str, Any]]:
    if value is None or isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


async def resolve_open_anchors(session_factory, scope, *, vector=None,
                               correlation_id: Optional[str] = None,
                               screen_fn=None) -> Dict[str, int]:
    """Fill what can now be filled for this scope. Never raises.

    Returns counts, which are worth logging: a resolver that silently declines
    everything looks exactly like one that never ran.
    """
    counts = {"open": 0, "resolved": 0, "refused": 0, "vectors_updated": 0,
              "screened": 0, "deferred": 0}
    # ⚠ Which stored memory IS the event an anchor names is now decided by the
    # judge and by nothing else, so this resolver DEPENDS on the screen being on
    # for the instance. With it off, open anchors are deferred rather than settled
    # by the word matcher that used to live here. That couples two switches that
    # were independent, and it is the honest version: the matching was the part
    # that made the feature work, and the matcher it used was rejected.
    #
    # Injected like coalesce's two LLM calls, so the behaviour is testable without
    # a network and without turning the flag on.
    if screen_fn is None:
        from synap.cloud.agents.ingestion.services import relatedness
        if relatedness.enabled_for(getattr(scope, "instance_id", None)):
            screen_fn = relatedness.screen_anchor
    screen_on = screen_fn is not None
    user_id = getattr(scope, "user_id", None)
    if session_factory is None or not user_id:
        return counts

    from sqlalchemy import text
    from synap.cloud.shared import event_window_schema as ews
    if not ews.anchor_available() or not ews.available():
        return counts

    # Bound before the try: the block below can raise before assigning it, and the
    # `if filled` after the try would then be a NameError instead of a logged miss.
    filled: Dict[str, Any] = {}
    try:
        async with session_factory() as session:
            open_rows = (await session.execute(text(
                "SELECT id::text, content, event_anchor FROM memories "
                "WHERE user_id = CAST(:uid AS uuid) "
                "  AND event_start IS NULL AND event_anchor IS NOT NULL "
                "  AND superseded_by IS NULL AND deprecated_at IS NULL AND deleted_at IS NULL "
                "LIMIT :lim"),
                {"uid": str(user_id), "lim": OPEN_ANCHOR_CAP})).fetchall()
            counts["open"] = len(open_rows)
            if not open_rows:
                return counts

            pool_rows = (await session.execute(text(
                "SELECT id::text, content, event_start, event_end, event_precision "
                "FROM memories "
                "WHERE user_id = CAST(:uid AS uuid) AND event_start IS NOT NULL "
                "  AND superseded_by IS NULL AND deprecated_at IS NULL AND deleted_at IS NULL "
                "ORDER BY event_start DESC LIMIT :lim"),
                {"uid": str(user_id), "lim": CANDIDATE_CAP})).fetchall()
            pool = [AnchorCandidate(memory_id=r[0], content=r[1], event_start=r[2],
                                    event_end=r[3], event_precision=r[4])
                    for r in pool_rows]

            for memory_id, _content, raw_anchor in open_rows:
                anchor = _loads(raw_anchor)
                if not anchor:
                    counts["refused"] += 1
                    continue
                # Which stored memory IS the event this anchor names.
                #
                # ⚠ There is no fallback, on purpose, and there used to be. Word
                # containment required every significant word of the phrase to
                # appear in the candidate, which missed "a wedding for his best
                # friend" against "best friend's wedding" over one apostrophe and
                # could never see "the user's best friend got married". Keeping it
                # for when the judge is unavailable sounded prudent and was not:
                # the same document ingested twice would get different dates
                # depending on whether Jev happened to be up, and nobody could see
                # which path had run.
                #
                # What makes deleting it cheap is that this work is RETRYABLE. This
                # runs after EVERY document of the scope persists, and the UPDATE
                # below carries `AND event_start IS NULL`. So a judge that is down
                # now costs a deferral, not a date: the anchor stays open and the
                # next document's persist asks again. Deferring is a much smaller
                # price than a silent, invisible second answer.
                own = [c for c in pool if c.memory_id != memory_id]
                admitted = await screen_fn(
                    anchor.get("anchor_text") or "", own,
                    correlation_id=correlation_id) if screen_on else None
                if admitted is None:
                    counts["deferred"] += 1
                    logger.info(
                        "anchor_deferred memory_id=%s reason=%s correlation_id=%s",
                        memory_id,
                        "screen_off" if not screen_on else "judge_unavailable",
                        correlation_id or "")
                    continue
                counts["screened"] += 1
                matched = [c for i, c in enumerate(own) if i in admitted]
                outcome = resolve_anchor(anchor, matched)
                if not outcome.resolved:
                    counts["refused"] += 1
                    logger.info("anchor_unresolved memory_id=%s reason=%s "
                                "candidates=%d correlation_id=%s",
                                memory_id, outcome.reason, len(matched),
                                correlation_id or "")
                    continue
                # Fill-only, enforced here and not in Python: a resolver that can
                # overwrite a date it did not set is the one mistake in this whole
                # change that would be hard to undo.
                result = await session.execute(text(
                    "UPDATE memories SET event_start = :s, event_end = :e, "
                    "       event_precision = :p "
                    " WHERE id = CAST(:id AS uuid) AND event_start IS NULL"),
                    {"s": outcome.event_start, "e": outcome.event_end,
                     "p": outcome.event_precision, "id": memory_id})
                if (result.rowcount or 0) == 0:
                    # Someone dated it between the SELECT and now. Their value wins.
                    counts["refused"] += 1
                    continue
                counts["resolved"] += 1
                filled[memory_id] = outcome
                logger.info("anchor_resolved memory_id=%s anchor=%s start=%s "
                            "correlation_id=%s", memory_id, outcome.anchor_memory_id,
                            outcome.event_start, correlation_id or "")
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        if ews.is_missing_column_error(exc):
            ews.mark_missing(exc)
            return counts
        logger.warning("resolve_open_anchors_failed user_id=%s correlation_id=%s",
                       user_id, correlation_id or "", exc_info=True)
        return counts

    if filled and vector is not None:
        counts["vectors_updated"] = await _refresh_vector_metadata(
            vector, scope, filled, correlation_id)
    return counts


async def _refresh_vector_metadata(vector, scope, filled: Dict[str, Any],
                                   correlation_id: Optional[str]) -> int:
    """Push the new window into the vector metadata.

    ⚠ Not optional. Retrieval reads a memory's date off its VECTOR metadata, not
    off Postgres, so a Postgres-only update leaves the whole fix invisible — the
    exact shape of bug that has caught this work before. ``update_metadata`` merges
    keys without touching the embedding, so nothing has to be re-embedded.
    """
    from synap.cloud.agents.ingestion.services.event_window import event_window_metadata
    try:
        namespace = vector._get_namespace(scope)
        updates = {
            memory_id: event_window_metadata(o.event_start, o.event_end,
                                             o.event_precision)
            for memory_id, o in filled.items()
        }
        store = getattr(vector, "_provider", None)
        if store is None or not hasattr(store, "update_metadata"):
            logger.warning(
                "anchor_vector_metadata_skipped reason=no_update_metadata "
                "memories=%d correlation_id=%s — the dates are in Postgres but "
                "retrieval reads them off the vector metadata, so they will not be "
                "served until those vectors are rewritten",
                len(updates), correlation_id or "")
            return 0
        changed = store.update_metadata(namespace, updates)
        logger.info("anchor_vector_metadata_updated namespace=%s changed=%d "
                    "correlation_id=%s", namespace, changed, correlation_id or "")
        return int(changed or 0)
    except Exception:  # noqa: BLE001 — Postgres already holds the truth
        logger.warning("anchor_vector_metadata_failed memories=%d correlation_id=%s",
                       len(filled), correlation_id or "", exc_info=True)
        return 0
