"""
Retrieval Manager - Core business logic for context retrieval.

This manager is the "hands" that executes retrieval based on ContextDecision.
The Agent (brain) decides WHAT to retrieve, the Manager decides HOW.
"""

from __future__ import annotations

from typing import Dict, Any, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
from contextvars import ContextVar
from dataclasses import replace as _dc_replace
from dataclasses import dataclass, field
from enum import Enum
import math
import os
import re
from datetime import datetime, timezone
import asyncio
import base64
import hashlib
import json
import uuid
import time
import logging
import numpy as np

from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType
from synap.cloud.agents.models.bundles import (
    ContextBundle,
    ContextItem,
    ConversationContext,
    DegradationLevel,
)
from synap.cloud.shared import graph_boundary as _graph_boundary
from synap.cloud.shared import scope_filter as _scope_filter
from synap.cloud.shared.scope import ScopeResolver, ScopeHierarchy, derive_item_scope
from synap.cloud.shared.cache_keys import CacheKeyGenerator
from synap.cloud.shared.scope_rung import rung_key_from_rungs as _rung_key_from_rungs
from synap.cloud.shared.confidence import meets_confidence_threshold, ConfidenceConfig
from synap.cloud.shared.tokens import TokenCounter
from synap.cloud.shared.logging_utils import log_data_sample, truncate_string, truncate_dict
from synap.cloud.shared.tokenizer import tokenize as _tokenize_stemmed
from synap.cloud.shared.step_logger import StepLogger, log_substep
from synap.cloud.observability.tracing import start_span, set_current_span_attributes
from synap.cloud.shared import edition
from synap.cloud.agents.ingestion.services import custom_categories
from synap.providers.rdbms.models import SQLQuery
from synap.providers.graph.models import TraversalPattern

if TYPE_CHECKING:
    from synap.cloud.application.compaction.models import CompactedContext

logger = logging.getLogger(__name__)


# Detached fire-and-forget tasks (metering/telemetry) are held here so they are
# not garbage-collected before completion (CPython asyncio keeps only a weak ref
# to running tasks). Used to keep DB metering off a request's response path.
_DETACHED_TASKS: set = set()

# Same GC-safety concern, dedicated to the consolidation tripwire producer
# (`publish_query_event`, scheduled from `fetch_long_term_context`): a bare
# ``asyncio.create_task(...)`` with no reference held can be collected
# mid-flight, silently swallowing whatever it was doing. Held here until its
# done-callback discards it (see the hoisted publish call below).
_TRIPWIRE_PUBLISH_TASKS: set = set()


def _run_detached(coro) -> None:
    """Schedule ``coro`` off the caller's critical path.

    For metering/telemetry writes that MUST happen but MUST NOT add DB
    round-trips to a request's response latency. A strong reference is retained
    until the task finishes so it is not GC'd mid-flight. The coroutine is
    responsible for swallowing its own exceptions (metering never fails a read).
    """
    try:
        task = asyncio.create_task(coro)
    except RuntimeError:
        # No running loop (e.g. called from sync context/tests) — run inline.
        # This preserves behaviour rather than silently dropping the metering.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()
        return
    _DETACHED_TASKS.add(task)
    task.add_done_callback(_DETACHED_TASKS.discard)


async def drain_detached_tasks(timeout: float = 4.0) -> int:
    """Await in-flight detached metering/analytics tasks on shutdown.

    Returns the number of tasks awaited. Must run BEFORE the analytics hook
    shuts down in the app lifespan so nothing enqueues an event after the
    delivery queue has been flushed and torn down.
    Best-effort: bounded by ``timeout`` and never raises.
    """
    pending = [t for t in list(_DETACHED_TASKS) if not t.done()]
    if not pending:
        return 0
    try:
        await asyncio.wait(pending, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — shutdown drain must never raise
        logger.warning("drain_detached_tasks raised (non-fatal): %s", exc)
    return len(pending)


def _latency_bucket(latency_ms: int) -> str:
    """Coarse latency bucket for the context_fetched event (allowlisted prop).

    Bucketed rather than raw so the property stays low-cardinality for analytics
    breakdowns and never leaks a fingerprintable exact timing.
    """
    if latency_ms < 50:
        return "0-50ms"
    if latency_ms < 200:
        return "50-200ms"
    if latency_ms < 1000:
        return "200-1000ms"
    return "1000ms+"


def _ensure_aware(dt) -> datetime:
    """Return dt as a timezone-aware UTC datetime, converting naive datetimes or ISO strings."""
    if dt is None:
        return datetime.now(timezone.utc)
    if isinstance(dt, str):
        from dateutil.parser import isoparse
        dt = isoparse(dt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _safe_parse_datetime(val) -> Optional[datetime]:
    """Safely parse an ISO datetime string, returning None on failure."""
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val).replace('Z', '+00:00')).replace(tzinfo=None)
    except (ValueError, AttributeError):
        return None


def _event_fields(get) -> Dict[str, Any]:
    """event_date / event_end / event_precision for a ContextItem, from any
    row-like getter (a Postgres row, Chroma metadata, a graph node).

    event_date is the event window's start (migration 154) and nothing else.
    It used to fall back to valid_from and then to DCT, and both of those
    fallbacks are gone: the second one because undated
    time-bound facts were being served as "happened" on the day they were
    merely mentioned. A row with no stored window now has no event date, which
    is the honest answer, and `said_at` still says when it was spoken."""
    # When the user SAID it. The source document's own date, and carried
    # regardless of whether a window exists, because a row WITH a window used to
    # drop it: "(2022) ... five sessions" and "(said May 11, 2023) ... three
    # sessions" are two statements about one event, and only the speaking dates
    # order them. The first was spoken on Oct 30, 2023 (f9e8c073) and the
    # renderer had no way to say so.
    #
    # It does NOT fall back to `created_at`. That line used to, on the reasoning
    # that "on the vector path created_at carries the document date too", and
    # that reasoning is wrong: every Chroma writer sets
    # `metadata["created_at"] = memory.extracted_at` (vector_wrapper.py:395 and
    # :487, and :849 for questions), so `created_at` is the minute WE ran
    # extraction, on both paths. Serving it here answered "when did I say this"
    # with a date from our own pipeline that the client has no way to recognise
    # as ours.
    #
    # Empty when the document carries no date, which is what ingestion now
    # stores when the client never said: see `stage23_worker` and
    # `DCT_SOURCE_SERVER`.
    said = _safe_parse_datetime(get("document_created_at"))
    # WHO said it. Set at extraction on most rows and dropped on
    # the way out: of the ten places a ContextItem is built, only three copied
    # it, so 67% of served items arrived with no speaker. Measured twice — 71%
    # unset across the lme16 serve, 67.1% (2,922 of 4,352) across lme17 — and
    # the cost is concrete: on "how many citrus fruits have I used", 32 of 40
    # items carried no speaker and the reader counted a fruit that appears only
    # in the assistant's suggestions. Carried here rather than at each site for
    # the same reason the dates are: ten sites cannot drift from one helper.
    asserted = get("assertion_source") or None
    # The custom category it was extracted under, when it is one this request knows (A12e).
    custom = _custom_category(get)
    # An event date, or none. Never the document's own date standing in for one.
    #
    # This used to fall through to `document_created_at` whenever ingestion had
    # stored no window and no validity start -- which is the documented, correct
    # thing for ingestion to store when the turn gives no time ("the user likes
    # jazz", "recently", "the other day"). The result was that `event_date` was
    # never empty: on the 500-question run, 5,601 of 10,443 served items (53.6%)
    # carried the conversation's own clock in it, and a client had no way to tell
    # those from the 35.4% that were real. The substitution was then flagged
    # `event_date_is_mention` so four separate readers downstream could each
    # remember to distrust it, which is a mask over a value we should not have
    # invented. The day the words were said has its own field, `said_at`, right
    # above; a reader that wants it can read it by name.
    # ⚠ `valid_from` used to stand in here when there was no event window, and
    # that was the second door the conversation's own date walked through. On a
    # 25-question run against the fixed build, 140 of 140 time-bound facts with
    # no window carried a `valid_from` equal to the day of the conversation, and
    # all 140 were served to the reader as "happened <that day>": habits the
    # speaker said they "have been" doing, things they did "once" or "recently",
    # and plans nobody has carried out yet.
    #
    # A validity start is when a fact BEGAN BEING TRUE. An event date is when
    # something HAPPENED. They are different questions and the second one is
    # often honestly unanswerable. `knowledge_update` already names this as the
    # defect: "the row falls back to valid_from / DCT and starts serving as
    # though the thing happened on the day it was mentioned".
    #
    # Checked before removing: zero rows in that ingest have a `valid_from`
    # worth keeping that no event window already holds. All 22 rows carrying a
    # `valid_until`, the festivals and deadlines, have an event window too. So
    # this loses no date that anybody stated.
    start = _safe_parse_datetime(get("event_start"))
    if start is None:
        return {"event_date": None,
                "event_end": None, "event_precision": None,
                "said_at": said,
                "assertion_source": asserted,
                "custom_category": custom}
    return {"event_date": start,
            "event_end": _safe_parse_datetime(get("event_end")),
            "event_precision": get("event_precision") or None,
            "said_at": said,
            "assertion_source": asserted,
            "custom_category": custom}


def _custom_category(get) -> Optional[str]:
    """The stored `category` of a row or Chroma metadata, only when it names one of this
    request's enabled custom categories. Rows written by the retired
    categoriser hold other non-empty values; those are never served as a custom name."""
    settings = _REQUEST_MACA.get()
    names = settings.custom_category_names if settings is not None else ()
    if not names:
        return None
    stored = get("category")
    return stored if stored in names else None


def _format_temporal_tag(item: ContextItem) -> str:
    """Format temporal metadata as inline tag for context display."""
    if not item.event_date:
        return ""
    if getattr(item, "event_precision", None):
        from synap.cloud.agents.ingestion.services.event_window import format_event_window
        label = format_event_window(item.event_date, item.event_end, item.event_precision)
        if label:
            return f" | {label}"
    try:
        date_str = item.event_date.strftime("%b %-d, %Y")
    except ValueError:
        date_str = item.event_date.strftime("%b %d, %Y")
    if item.valid_until:
        try:
            end_str = item.valid_until.strftime("%b %-d, %Y")
        except ValueError:
            end_str = item.valid_until.strftime("%b %d, %Y")
        return f" | {date_str} – {end_str}"
    elif item.temporal_category == "temporal_fact":
        return f" | since {date_str}"
    return f" | {date_str}"


def _reserve_per_type(
    items_sorted: List["ContextItem"], budget: int, floor: int
) -> List["ContextItem"]:
    """Trim ``items_sorted`` (already global-relevance-desc) to ``budget`` items
    while guaranteeing at least ``floor`` items of each ``context_type`` survive.

    Workstream B lever 2. Pass 1 walks the globally-sorted list and reserves the
    top ``floor`` items of each type; pass 2 fills the remaining budget by global
    relevance. ``floor <= 0`` is never routed here (the caller guards), so today's
    single global cut is preserved exactly.

    ⚠ The re-sort at the end MUST use ``_ranking_key``, the same key the caller
    sorted with. It used ``_final_score`` alone, which drops Jev's verdict and
    hands back a different order from the one it was given. That was survivable
    only while a later per-category sort re-applied the verdict; that sort is gone
    (see ``assemble_context_bundle``), so this is now the order the reader reads.
    """
    if budget <= 0:
        return []
    reserved: List["ContextItem"] = []
    picked: set = set()
    per_type: Dict[str, int] = {}
    # Pass 1 — floor per type, highest-scored first (list is pre-sorted).
    for it in items_sorted:
        if len(reserved) >= budget:
            break
        t = getattr(it, "context_type", None)
        if per_type.get(t, 0) < floor:
            reserved.append(it)
            picked.add(id(it))
            per_type[t] = per_type.get(t, 0) + 1
    # Pass 2 — fill the remaining budget by global relevance.
    for it in items_sorted:
        if len(reserved) >= budget:
            break
        if id(it) not in picked:
            reserved.append(it)
            picked.add(id(it))
    reserved.sort(key=_ranking_key, reverse=True)
    return reserved


def _reserve_per_query(
    items_sorted: List["ContextItem"],
    budget: int,
    per_query: int,
    queries: List[str],
) -> List["ContextItem"]:
    """Trim ``items_sorted`` (best first) to ``budget``, giving every query a turn.

    One cut by overall rank lets a query with many strong matches take every
    place, and a query whose single answer ranks lower gets none. Here the
    queries take turns, in the order the caller sent them, each taking its own
    best item not yet taken, until a query has ``per_query`` items or nothing
    left. ``budget`` is ``max_results`` and is never exceeded: when the shares
    add up to more than it, the turns simply stop early, so three queries at
    five each under a budget of ten end 4, 3 and 3.

    An item several queries found counts against each of them, and occupies one
    place. A query that is already full does not stop another from taking an
    item they share.

    An item no query accounts for belongs to no share. Those fill whatever room
    is left under the budget after the turns, best first.

    The result is in the order the items came in.
    """
    if budget <= 0:
        return []
    rank = {id(it): i for i, it in enumerate(items_sorted)}
    by_query = {q: [it for it in items_sorted if q in it.matched_queries] for q in queries}
    cursor = {q: 0 for q in queries}
    taken = {q: 0 for q in queries}
    picked: List["ContextItem"] = []
    picked_ids: set = set()

    progressed = True
    while len(picked) < budget and progressed:
        progressed = False
        for q in queries:
            if len(picked) >= budget:
                break
            if taken[q] >= per_query:
                continue
            candidates = by_query[q]
            i = cursor[q]
            while i < len(candidates) and id(candidates[i]) in picked_ids:
                i += 1
            if i >= len(candidates):
                cursor[q] = i
                continue
            item = candidates[i]
            cursor[q] = i + 1
            picked.append(item)
            picked_ids.add(id(item))
            for owner in item.matched_queries:
                if owner in taken:
                    taken[owner] += 1
            progressed = True

    for item in items_sorted:
        if len(picked) >= budget:
            break
        if id(item) in picked_ids:
            continue
        if not any(owner in taken for owner in item.matched_queries):
            picked.append(item)
            picked_ids.add(id(item))

    picked.sort(key=lambda it: rank[id(it)])
    return picked


def _note_query(item: "ContextItem", owner: Optional[str]) -> None:
    """Record that ``owner``, one of the caller's search queries, found ``item``."""
    if owner and owner not in item.matched_queries:
        item.matched_queries.append(owner)


def _merge_matched_queries(into: "ContextItem", other: "ContextItem") -> None:
    """Keep every query that found a memory when two copies of it collapse.

    Each search arm returns its own copy, and so does each query inside an arm.
    The dedup keeps one. The queries that found the others must not go with them.
    """
    for owner in getattr(other, "matched_queries", None) or ():
        _note_query(into, owner)


def _owner_at(query_owners: Optional[List[str]], idx: int) -> Optional[str]:
    if not query_owners or idx < 0 or idx >= len(query_owners):
        return None
    return query_owners[idx]


def _provenance_discount() -> Optional[float]:
    """Multiplier applied to the final score of assistant-sourced items
    (SYNAP_RETRIEVAL_PROVENANCE_DISCOUNT). Unset/invalid/out-of-range = None
    = feature off, ranking byte-identical. Sane range (0, 1]: 1.0 is a legal
    no-op for A/B, values <=0 or >1 are treated as misconfiguration and
    ignored. Read per assembly call (not module-load) so it can be toggled via
    env without a restart of long-lived workers being strictly required."""
    raw = os.environ.get("SYNAP_RETRIEVAL_PROVENANCE_DISCOUNT")
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not (0.0 < value <= 1.0):
        return None
    return value


# The reading call bounds itself, like the Jev call beside it, because it runs
# outside the retrieval mode's own asyncio.timeout block.
_TEMPORAL_READING_TIMEOUT_SECS = float(os.environ.get("SYNAP_TEMPORAL_READING_TIMEOUT_SECS") or "3.0")


def dated_evidence_enabled() -> bool:
    """Whether a chunk of source text is labelled with the date it was said.

    The memory LINE carries a date prefix; the evidence block under it does not,
    so a relative word inside the raw text has no date beside it. On 71017276 a
    reader quoted "I also got a stunning crystal chandelier from my aunt today"
    out of a chunk and answered "0 weeks ago" to a question asked four weeks
    later. The memory line above it said "(Mar 4, 2023)" and the chunk said
    nothing.

    Off by default and enabled per deployment with
    ``SYNAP_RETRIEVAL_DATED_EVIDENCE=on``, because it changes the text every
    SDK caller receives, so it does not flip for the whole fleet on a merge.
    """
    return os.environ.get("SYNAP_RETRIEVAL_DATED_EVIDENCE", "").strip().lower() in {"1", "true", "on", "yes"}


def evidence_said_from_document_enabled() -> bool:
    """Whether EVERY excerpt is dated, from the conversation it was cut out of.

    ``dated_evidence_enabled`` dates an excerpt only when the fact it hangs
    under came from a single document AND that fact's ``event_date`` already is
    the day it was said. That is 55.5% of the 4,250 excerpts served on the 293;
    the other 1,890 go out with no date at all, and a reader takes an undated
    passage as current. On 830ce83f the sentence "User: She moved to Chicago."
    was attached, undated, to a weather fact, and the answer was Chicago --
    five days after the user said the friend had moved back to the suburbs.

    The date used here is the one the excerpt actually has: the date of the
    document the chunk came from, which is known for every excerpt. Filling the
    gap with the fact's ``event_date`` instead would print a WRONG date on
    55.8% of those 1,890, because an event_date is when the thing happened and
    not when it was said -- 127 of them are out by more than six months.

    Off by default, per deployment, like the flag above:
    ``SYNAP_RETRIEVAL_EVIDENCE_SAID_FROM_DOCUMENT=on``.
    """
    return os.environ.get("SYNAP_RETRIEVAL_EVIDENCE_SAID_FROM_DOCUMENT", "").strip().lower() in {"1", "true", "on", "yes"}


def evidence_speaker_label_enabled() -> bool:
    """Whether an excerpt that starts mid-turn says who is speaking.

    A chunk is a 500-token window, not a turn, so 3,152 of the 4,250 excerpts
    served for the 293-question cohort (74.2%) begin in the middle
    of somebody's turn with no ``User:`` or ``Assistant:`` above them. Half the
    characters of those excerpts (50.3%) sit in that opening turn, and 481 of
    them carry no marker anywhere, so the whole passage is unattributed. Every
    later turn inside a window still carries its own marker, so the only
    speaker ever missing is the first one.

    What this is NOT, measured rather than assumed: it is not the fix for
    c4a1ceb8 ("how many different types of citrus fruits have I used", gold 3,
    answered 4). Of the eight served excerpts there that mention grapefruit --
    a fruit only the assistant ever brings up -- seven already show
    "Assistant:" before the mention and the reader counted it anyway. One gains
    the missing speaker. Counting only what the user said is a different fix.

    Off by default: ``SYNAP_EVIDENCE_SPEAKER_LABEL=on``.
    """
    return os.environ.get("SYNAP_EVIDENCE_SPEAKER_LABEL", "").strip().lower() in {"1", "true", "on", "yes"}


# Only these two. A transcript is full of lines like "Pros:", "Ingredients:"
# and "Instructions:" -- 93, 51 and 47 of them in the excerpts served on the
# 293 -- and every one of those would otherwise read as a speaker.
_SPEAKER_MARKER = re.compile(r"\b(User|Assistant)\s*:")


def _opening_speaker(chunk_texts: List[str], index: int) -> Optional[str]:
    """Who is talking at the START of ``chunk_texts[index]``.

    ``None`` when the passage already opens on a marker, so there is nothing to
    add, or when the document has no transcript shape at all.

    The rule is alternation, not the previous window. A transcript alternates,
    so whoever speaks FIRST inside the passage tells us the partial turn above
    it belongs to the other one. The previous window is not usable on its own
    because windows overlap by 75 tokens: its last marker is often inside the
    overlap, i.e. after this window has already started.

    Measured against ground truth on 575 windows cut from 60 real LongMemEval
    sessions by the real ingest windowing: alternation 379 right and **0
    wrong**, silent on 54; the previous window's last marker 68 wrong (15.7%).
    Falling back to the previous window only for those 54 -- a window with no
    marker at all is one long turn, so the speaker has not changed -- answers
    all 433 mid-turn windows with nothing wrong.
    """
    if not (0 <= index < len(chunk_texts)):
        return None
    text = chunk_texts[index] or ""
    head = text.lstrip()
    if head.startswith("User:") or head.startswith("Assistant:"):
        return None
    first = _SPEAKER_MARKER.search(text)
    if first:
        return "Assistant" if first.group(1) == "User" else "User"
    # No marker in the passage: one long turn, so whoever spoke last before it
    # is still speaking. Walk back until a window has one.
    for prev in range(index - 1, -1, -1):
        last = None
        for m in _SPEAKER_MARKER.finditer(chunk_texts[prev] or ""):
            last = m.group(1)
        if last:
            return last
    return None


def _said_prefix(when: datetime) -> str:
    """"(said Mar 4, 2023) " -- the same wording the memory line above it uses."""
    try:
        return "(said %s) " % when.strftime("%b %-d, %Y")
    except ValueError:  # platforms without %-d
        return "(said %s) " % when.strftime("%b %d, %Y")


def _excerpt_label(
    text: str,
    home: Optional[Tuple[str, int]],
    doc_chunks: Dict[str, List[str]],
    doc_said: Dict[str, datetime],
    fallback_said: Optional[datetime],
    with_date: bool,
    with_speaker: bool,
) -> str:
    """Put the date and the speaker in front of one raw excerpt.

    `home` is the document the chunk came from and its position in that
    document's chunk list -- both known at serve time, and both thrown away
    before this existed because the chunks were unioned into one list of
    strings.
    """
    prefix = ""
    doc_id, idx = home if home else (None, -1)
    if with_date:
        said = doc_said.get(doc_id) if doc_id else None
        if said is None:
            # The document's date is the right one. The fact's own date is the
            # old behaviour, kept for an excerpt whose document we could not
            # date, so this can only ever add a label.
            said = fallback_said
        if said is not None:
            prefix += _said_prefix(said)
    if with_speaker and doc_id is not None:
        who = _opening_speaker(doc_chunks.get(doc_id) or [], idx)
        if who:
            prefix += "[%s] " % who
    return prefix + text if prefix else text


_RECENT_TURNS_FLOOR = 20


def _uncompacted_tail(messages: List[Any]) -> List[Any]:
    """The turns after the stored compaction that are served as recent turns.

    The last 20, as before, and further back while the whole tail stays within
    the compaction minimum (``MIN_SPAN_TOKENS``, 2,000). Compaction does not
    run on fewer new tokens than that, and reads no longer start one, so a long
    exchange of short messages can sit uncompacted past 20 turns. Cutting at
    20 would then lose its opening, with no summary holding it.
    """
    if len(messages) <= _RECENT_TURNS_FLOOR:
        return list(messages)
    from synap.cloud.application.compaction.manager import manager_min_span

    def _tokens(m: Any) -> int:
        try:
            return int(m.token_count)
        except Exception:
            return len(str(getattr(m, "content", "") or "")) // 4

    keep = _RECENT_TURNS_FLOOR
    total = sum(_tokens(m) for m in messages[-keep:])
    budget = manager_min_span()
    while keep < len(messages):
        nxt = _tokens(messages[-keep - 1])
        if total + nxt > budget:
            break
        total += nxt
        keep += 1
    return list(messages[-keep:])


def temporal_window_filter_enabled() -> bool:
    """Whether a question that names a period may remove out-of-window items.

    Off by default, and a request without ``query_date`` cannot run it whatever
    the flag says. ``SYNAP_RETRIEVAL_TEMPORAL_WINDOW_FILTER=on``.
    """
    return os.environ.get("SYNAP_RETRIEVAL_TEMPORAL_WINDOW_FILTER", "").strip().lower() in {"1", "true", "on", "yes"}


def accurate_searches_all_queries() -> bool:
    """Whether ACCURATE runs its long-term search on every query the caller sent.

    From 2026-02-21 (498babb8) to 2026-10-08 it searched only the first:
    ``_fetch_accurate`` took ``search_queries[0]`` as its "main query", wrote
    LLM subqueries from it, and nothing picked up ``search_queries[1:]``. Their
    embeddings were computed and they reached the short-term search, so the
    request looked served. FAST never had the gap.

    On by default. ``SYNAP_RETRIEVAL_ACCURATE_ALL_QUERIES=off`` restores the
    first-query-only search exactly, as a way back if a multi-query caller
    regresses. A request with one query is identical either way.
    """
    return os.environ.get("SYNAP_RETRIEVAL_ACCURATE_ALL_QUERIES", "").strip().lower() not in {"0", "false", "off", "no"}


def _failure_code(exc: BaseException) -> str:
    """Why a retrieval failed, in words a caller can branch on.

    Closed vocabulary: "timeout", "overloaded", "unavailable", "internal". It
    says what the caller can do about it (wait, back off, retry, or report),
    never which of our services broke.
    """
    name = type(exc).__name__
    text = f"{name} {exc}".lower()
    if "timeout" in text or "timedout" in text:
        return "timeout"
    if "429" in text or "overloaded" in text or "rate" in text and "limit" in text:
        return "overloaded"
    if any(k in text for k in ("circuit open", "connect", "unavailable",
                               "refused", "embeddingserviceerror", "502",
                               "503", "504")):
        return "unavailable"
    return "internal"


async def jev_rerank_items(items, query, correlation_id=None, on_fault=None):
    """``client_api.jev_rerank.rerank_items`` in Cloud (the ``jev_rerank`` hook); in Core, ``items`` as they came."""
    reranked = await edition.acall(
        "jev_rerank", "items", items, query,
        correlation_id=correlation_id, on_fault=on_fault,
    )
    return items if reranked is None else reranked


def jev_is_primary_ranker() -> bool:
    """Whether Jev's verdict, not the computed relevance score, decides the order.

    Deliberately reads the SAME switch as the re-rank itself
    (``SYNAP_RETRIEVAL_JEV_RERANK``) rather than adding a second one. The two
    steps are mutually exclusive by construction: with the switch off no item
    carries a verdict and the sort below is byte-identical to what it has always
    been; with it on, Jev ranks and the computed score becomes the tiebreak.
    There is no weight to tune and no way to end up half-configured.
    """
    return bool(edition.call("jev_rerank", "enabled"))


def _stable_tiebreak(item) -> str:
    """The last element of every ordering key, so no tie is left to chance.

    Python's sort is stable, so two items with an equal key come out in the order
    they arrived -- and they arrive from several concurrent search arms whose
    completion order is not fixed. The benchmark team measured the consequence on
    two runs against an untouched store: the retrieved id SET was identical on
    500 of 500 questions and the id SEQUENCE differed on 406 of them, which put
    their run-to-run noise floor at 1.86pp, about nine questions.

    Ties are not rare. Measured on the 10,661 items served across the 470 scored
    questions, 49.2% sat in a tie group, in 1,785 groups with a median size of 2
    and a longest of 24, so roughly half the served sequence was decided by
    nothing. That was measured against the per-category key of the day (relevance
    banded to 0.1, then event date), deleted on 2026-10-08; the key in force now
    is ``_ranking_key``, whose ties are narrower but not gone, because Jev's
    verdict reaches 0.0 and 1.0 often.

    Order is not only cosmetic. ``_apply_token_budget`` walks one list ranked on
    this key and stops at the first item that does not fit, so where the budget
    bites, the sequence decides WHICH memories the reader sees.

    The id, not the content: the id is already unique per row, and sorting on
    content would make the order depend on the words, which reads as a ranking
    signal and is not one. Descending, like every other element of these keys --
    the direction is arbitrary, being the SAME every time is the whole point.
    """
    for attr in ("item_id", "id", "memory_id"):
        value = getattr(item, attr, None)
        if value:
            return str(value)
    return ""


def _ranking_key(item: ContextItem) -> tuple:
    """Sort key for the final ranking. Higher is better, descending sort.

    WHY A TUPLE, AND WHY JEV FIRST

    Jev already judges semantic relevance over the memory text and its date, so
    running it and then re-sorting on the computed relevance score threw its
    verdict away: measured on a benchmark run, the served order
    followed the computed score in 500 of 500 questions and the served SET was
    100% identical to a run with no re-rank at all. The re-rank could only
    permute ties, mean displacement 0.5 places. So when Jev is ranking, Jev's
    verdict has to BE the key, not an input to it.

    The computed score stays as the SECOND element, which is not a blend and not
    redundant. Three signals feed it that Jev cannot see:

      * the per-client MACA config (``boost_categories``, ``ranking_signals``,
        ``category_priorities``) — a product feature clients configure, which
        would otherwise silently stop applying the moment the re-rank came on;
      * the ingestion ``confidence``;
      * today's date, so recency DECAY. Jev is given each memory's date but is
        never told what today is, so it can order dates against each other and
        nothing more.

    Jev's verdict is also coarse (a probability over four rubric levels) and
    reaches 0.0 or 1.0 often, so without a second element those large ties would
    resolve arbitrarily. This is exactly today's failure mode seen from the other
    side: the score the reader is ordered by currently takes 11 distinct values
    across 14,934 served items, 56% of them the same 0.9.

    An item with NO verdict sorts on ``(0.0, computed score)``. That is
    deliberate and it is why chunks and the over-limit tail keep their existing
    relative order among themselves. It does mean an unscored item sorts below a
    Jev-scored one; chunks are separately score-gated in ``_search_sentences``
    and appended for evidence rather than ranked against memories, so that is
    the intended relationship, not an accident.
    """
    jev = getattr(item, "_jev_relevance", None)
    return (float(jev) if jev is not None else 0.0, item._final_score,
            _stable_tiebreak(item))


# A chunk's confidence, for scoring only.
#
# A chunk is a quote from the source conversation, not an extracted memory, so it
# has no ingestion confidence. `_search_sentences` makes one up from the cosine
# (0.5 + (cosine - 0.55)), which puts a 0.60 chunk at 0.55 while memories sit at
# 0.85 to 0.95. Scored on that, every chunk would lose about 0.07 of importance
# against a memory that matches the question equally well, for a reason that has
# nothing to do with importance. So the score uses the typical memory value and
# the item keeps its own: the served `confidence` is unchanged, which matters
# because it is in every default-shape response.
_CHUNK_SCORING_CONFIDENCE = 0.9

# How much the computed score adds to the published rank score when Jev ranks.
#
# Jev's verdict decides the order and the computed score breaks its ties (see
# `_ranking_key`). `rank_score` promises ONE number that reproduces that order,
# so it is the verdict plus a small share of the score: small enough that the
# verdict decides whenever two verdicts differ by more than 0.001, which a
# probability over four rubric levels almost always does. With Jev off the
# published number is the computed score, as it always was.
_JEV_RANK_TIEBREAK_WEIGHT = 0.001


def _published_rank_score(item: ContextItem, jev_ranks: bool) -> float:
    """The `rank_score` a caller sees: higher is better, and sorting on it
    descending reproduces the served order. An item Jev did not judge (a chunk)
    counts as a verdict of 0.0, exactly as `_ranking_key` treats it."""
    final = float(item._final_score)
    if not jev_ranks:
        return final
    jev = getattr(item, "_jev_relevance", None)
    return (float(jev) if jev is not None else 0.0) + _JEV_RANK_TIEBREAK_WEIGHT * final


def _budget_rank_key(item: ContextItem) -> tuple:
    """`_ranking_key`, safe for an item that never went through assembly.

    The token budget has to put items from different types on one ranked list,
    and the only honest way is the key assembly already ranked them with. An
    item built outside assembly (a cached bundle, a hand-made one in a test) has
    no ``_final_score``; it ranks on its relevance, then similarity, instead of
    raising.
    """
    score = getattr(item, "_final_score", None)
    if score is None:
        score = getattr(item, "relevance_score", None)
    if score is None:
        score = getattr(item, "similarity_score", 0.0)
    jev = getattr(item, "_jev_relevance", None)
    return (float(jev) if jev is not None else 0.0, float(score or 0.0),
            _stable_tiebreak(item))


def _ranked_across_types(
    items_by_type: Dict[str, List[ContextItem]],
    tier_of,
) -> List[Tuple[int, ContextItem]]:
    """Every item of every type as ONE list, best first, as (tier, item) pairs.

    Each type's own served order is kept exactly (so a chunk placed at position
    2 of its type stays there), and the types are merged by comparing the next
    item of each on the ranking key. That is the global ranking, rebuilt
    without a second sort that could disagree with the one assembly made.

    ``tier_of(type)`` is 0 for every type unless the client configured
    ``category_priorities``, in which case a lower tier is served before a
    higher one and ranking decides only within a tier.
    """
    queues = {t: list(items) for t, items in items_by_type.items() if items}
    order = {t: i for i, t in enumerate(queues)}
    merged: List[Tuple[int, ContextItem]] = []
    while queues:
        best = max(
            queues,
            key=lambda t: (-tier_of(t), _budget_rank_key(queues[t][0]), -order[t]),
        )
        merged.append((tier_of(best), queues[best].pop(0)))
        if not queues[best]:
            del queues[best]
    return merged


def _compute_relevance_score(
    item: ContextItem,
    maca: "MACARetrievalSettings" = None,
    confidence: Optional[float] = None,
) -> float:
    """
    Compute a unified relevance score that is comparable across sources.

    Without MACA config (backward compat):
    1. relevance_score (from reranking) - most authoritative
    2. similarity_score (from vector search) - actual query relevance
    3. Fallback baseline weighted by confidence

    With MACA config:
    - Weighted combination of relevance, recency (temporal decay), and importance
    - Category boosts from MACA ranking config
    """
    # `confidence` replaces the item's own for this calculation only. It is how
    # a chunk is scored without changing the confidence it is served with.
    conf = item.confidence if confidence is None else confidence

    if item.relevance_score > 0.0:
        base_score = item.relevance_score
    elif item.similarity_score > 0.0:
        base_score = item.similarity_score
    else:
        UNSCORED_BASELINE = 0.10
        base_score = UNSCORED_BASELINE * conf

    if maca is None:
        return base_score

    signals = maca.ranking_signals
    relevance_w = signals.get("relevance", 0.5)
    recency_w = signals.get("recency", 0.3)
    importance_w = signals.get("importance", 0.2)

    # Temporal decay. When it happened, else when it was said, else when we
    # wrote the row. `said_at` sits in the middle because it is a real date the
    # client gave us, while `created_at` is our own ingest clock: reading the
    # corpus by ingest time mis-ranked a large share of it, and
    # before event_date stopped standing in for the document date this line
    # never reached that fallback. Ranking only; none of these is served from
    # here, so nothing downstream can read said_at as an event date.
    recency_factor = 1.0
    temporal_anchor = (item.event_date
                       or getattr(item, "said_at", None)
                       or item.created_at)
    if temporal_anchor is not None and recency_w > 0:
        age_days = max(0, (datetime.now(timezone.utc) - _ensure_aware(temporal_anchor)).total_seconds() / 86400)
        if maca.decay_function == "exponential" and maca.decay_half_life_days > 0:
            recency_factor = math.exp(-0.693 * age_days / maca.decay_half_life_days)
        elif maca.decay_function == "linear":
            recency_factor = max(0.0, 1.0 - age_days / (maca.decay_half_life_days * 2))

    # Category boost
    category_boost = maca.boost_categories.get(item.context_type, 0.0)

    # Weighted combination
    score = (
        relevance_w * base_score
        + recency_w * recency_factor
        + importance_w * conf
        + category_boost
    )

    return min(score, 1.0)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Calculate Jaccard similarity between two texts using stemmed, stop-word-free tokens."""
    set_a = set(_tokenize_stemmed(text_a))
    set_b = set(_tokenize_stemmed(text_b))

    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


# A store read that fails part-way is the failure mode that hides best. The
# request still returns 200, with fewer items, and nothing in
# the response used to say so. A small fraction of retrievals can lose a vector
# search arm to a Chroma internal error ("Error
# finding id", "Error creating hnsw segment reader: Nothing found on disk"), the
# arm's memories were simply not searched, and neither the caller nor the
# benchmark harness could tell. The harness reported every one of those requests
# as a clean success.
#
# The code that swallows those errors sits several frames below the bundle, so
# the faults are collected in a ContextVar that fetch_context opens per request
# and folds into ``bundle.warnings`` before returning. A None collector means
# "not inside a fetch", and recording is then a no-op, so nothing here can raise
# into a read path or leak between requests.
_RETRIEVAL_FAULTS: ContextVar[Optional[List[str]]] = ContextVar(
    "synap_retrieval_faults", default=None
)


def record_retrieval_fault(where: str, detail: object, outcome: str = "failed") -> None:
    """Note that one leg of a retrieval failed, for the caller's benefit.

    ``where`` names the leg in words an operator can act on ("vector scope
    chain"), not a function name. ``outcome`` is the verb ("failed" by default;
    "not applied" for a leg that was deliberately not run). Duplicates are
    dropped: four arms failing the same way is one fact, not four.
    """
    faults = _RETRIEVAL_FAULTS.get()
    if faults is None:
        return
    if isinstance(detail, BaseException):
        msg = f"{where} {outcome}: {type(detail).__name__}: {detail}"
    else:
        msg = f"{where} {outcome}: {detail}"
    if msg not in faults:
        faults.append(msg)


# --- Graph liveness (accurate-mode graph arm) --------------------------------
#
# A graph edge or node only REFERENCES the memories that support it
# (``memory_ids``, or ``memory_contents`` entries in the legacy copies mode).
# Postgres says whether each of those memories is still alive. The graph arm
# asks it, in one query, and serves an entity line, a relationship sentence or
# a copied memory only while the shared rule (graph_lifecycle_rule) says the
# record still states something current. Without this, a statement whose only
# support was superseded or deleted kept being served ("the user works at
# Acme Corp" after the user resigned).
#
# A fault whose text starts with this prefix marks the response degraded and
# keeps it out of the L1 cache (see _cache_bundle).
_GRAPH_LIVENESS_FAULT = "graph liveness check"
_GRAPH_LIVENESS_FLAG = "SYNAP_GRAPH_SERVE_CHECKS_LIVENESS"


def _graph_serve_checks_liveness(instance_id: Optional[Any] = None) -> bool:
    """On by default in Synap Core, off by default elsewhere; the variable
    (or its ``_INSTANCES`` list) decides when set."""
    raw = os.environ.get(_GRAPH_LIVENESS_FLAG)
    if raw is not None and raw.strip():
        return raw.strip().lower() not in ("0", "false", "no", "off")
    if os.environ.get(f"{_GRAPH_LIVENESS_FLAG}_INSTANCES", "").strip():
        from synap.cloud.shared.instance_flags import enabled_for
        return enabled_for(_GRAPH_LIVENESS_FLAG, instance_id)
    from synap.cloud.shared import edition
    return bool(edition.is_core)


def _graph_record_memory_ids(props: Optional[Dict[str, Any]]) -> List[str]:
    """Every memory id a graph record references: ``memory_ids`` plus the
    ids of legacy ``memory_contents`` entries."""
    props = props or {}
    out: List[str] = []
    raw = props.get("memory_ids")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raw = [raw]
    for mid in raw or []:
        if mid:
            out.append(str(mid))
    legacy = props.get("memory_contents")
    if legacy:
        try:
            entries = json.loads(legacy) if isinstance(legacy, str) else legacy
        except (json.JSONDecodeError, TypeError):
            entries = []
        for entry in entries or []:
            if isinstance(entry, dict) and entry.get("memory_id"):
                out.append(str(entry["memory_id"]))
    return out


class _GraphLiveness:
    """The answer to "may this graph record be served?" for one discovery.

    ``checked`` False: the flag is off and every record passes, as before.
    ``failed`` True: Postgres could not be asked; no entity line, relationship
    sentence or copied memory passes (their truth is unknown).
    """

    KINDS = ("entity", "relationship", "memory")

    def __init__(self, checked: bool, rows: Optional[Dict[str, Any]] = None,
                 failed: bool = False, query_ms: float = 0.0, ids_checked: int = 0):
        self.checked = checked
        self.failed = failed
        self.rows = rows or {}
        self.query_ms = query_ms
        self.ids_checked = ids_checked
        self.dropped_not_alive = {k: 0 for k in self.KINDS}
        self.dropped_retired = {k: 0 for k in self.KINDS}
        self.dropped_unchecked = {k: 0 for k in self.KINDS}

    @classmethod
    async def load(cls, instance_id: Any, nodes: Sequence[Any], edges: Sequence[Any]) -> "_GraphLiveness":
        if not _graph_serve_checks_liveness(instance_id):
            return cls(checked=False)
        ids: List[str] = []
        for record in list(nodes) + list(edges):
            ids.extend(_graph_record_memory_ids(getattr(record, "properties", None)))
        ids = list(dict.fromkeys(ids))
        if not ids:
            return cls(checked=True)
        from synap.cloud.shared import memory_liveness
        _t0 = time.time()
        try:
            rows = await memory_liveness.load_supporter_rows(ids)
        except Exception as e:  # noqa: BLE001 - reported as a fault, never swallowed silently
            logger.warning(
                "graph_liveness_query_failed: instance_id=%s ids=%d error=%s: %s",
                instance_id, len(ids), type(e).__name__, e,
            )
            record_retrieval_fault(
                _GRAPH_LIVENESS_FAULT,
                "memory states could not be read, so no graph statements were returned",
            )
            return cls(checked=True, failed=True, ids_checked=len(ids),
                       query_ms=(time.time() - _t0) * 1000)
        return cls(checked=True, rows=rows, ids_checked=len(ids),
                   query_ms=(time.time() - _t0) * 1000)

    def memory_alive(self, memory_id: str) -> bool:
        """Kind C/D: one memory. True when unchecked."""
        if not self.checked:
            return True
        if self.failed:
            return False
        from synap.cloud.shared.memory_liveness import is_alive
        return is_alive(self.rows.get(str(memory_id)))

    def collect_dead(self, ids: Iterable[str]) -> set:
        if not self.checked or self.failed:
            return set()
        from synap.cloud.shared.memory_liveness import is_alive
        return {i for i in ids if not is_alive(self.rows.get(str(i)))}

    def record_ok(self, kind: str, props: Optional[Dict[str, Any]],
                  fixed: bool = False) -> Tuple[bool, Optional[List[str]]]:
        """Kind A/B: (serve?, alive backing memory ids or None when unchecked)."""
        if not self.checked:
            return True, None
        if self.failed:
            self.dropped_unchecked[kind] += 1
            return False, None
        props = props or {}
        if props.get("retired_at") and not fixed:
            self.dropped_retired[kind] += 1
            return False, None
        from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import (
            GraphRecord, graph_record_state,
        )
        released = props.get("released_memory_ids") or []
        state = graph_record_state(
            GraphRecord(
                memory_ids=_graph_record_memory_ids(props),
                released_memory_ids=[str(x) for x in released if x],
                is_fixed_entity=fixed,
            ),
            self.rows,
        )
        if not state.is_open:
            self.dropped_not_alive[kind] += 1
            return False, None
        return True, list(state.alive_memory_ids)

    def log(self, instance_id: Any) -> None:
        if not self.checked:
            return
        logger.info(
            "graph_liveness: instance_id=%s ids_checked=%d failed=%s graph_liveness_query_ms=%.1f "
            "graph_items_dropped_not_alive=%s graph_items_dropped_retired=%s "
            "graph_items_dropped_unchecked=%s",
            instance_id, self.ids_checked, self.failed, self.query_ms,
            self.dropped_not_alive, self.dropped_retired, self.dropped_unchecked,
        )


def _jaccard_similarity_sets(set_a: frozenset, set_b: frozenset) -> float:
    """Jaccard similarity between pre-computed token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# The near-duplicate rule's threshold, and the switch that turns it off.
#
# Measured on the LongMemEval lme2_ cohort, over its 500 retrievals:
# this rule removed 4,180 of the 11,264 candidates
# that survived exact dedup. In 17 of the 22 retrieval misses where the
# answer-bearing memory cleared the similarity floor, what it deleted was NOT a
# duplicate. Our memory sentences nearly all open "The user ...", so two
# memories that differ only in a number, a date, a frequency or one item of a
# list still share most of their token set and clear 0.45. Real pairs it
# collapsed, with the threshold this tokenizer computes:
#
#   0.545  "watched 12 films in the last 3 months, including 5 MCU films"
#     vs   "watched 4 MCU films in the last 3 months"
#   0.455  "Transcriptionist is a work-from-home job idea for seniors"
#     vs   "Remote IT support specialist is a work-from-home job idea"   <- one list
#   0.636  "started reading 'The Nightingale' on 2022-01-01"
#     vs   "finished reading 'The Nightingale' on 2022-01-15"
#
# Handing the reader back only what this rule removed moved the same reader and
# the same judge from 81.70% to 85.74% on 470 questions, against a measured
# +/-1.3 point noise floor: it fixed 28 questions and broke 9.
#
# So the default is OFF. Exact dedup still runs before this (7,616 items in the
# same run), and ResponseFilter's 0.7 Jaccard still runs after it, so turning
# this off does not mean duplicates reach the caller -- it means two memories
# that differ in a value are no longer treated as one.
_DEFAULT_DEDUP_JACCARD = 0.0


def dedup_jaccard_threshold() -> float:
    """The near-duplicate threshold, overridable with SYNAP_RETRIEVAL_DEDUP_JACCARD.

    ``0`` or ``off`` turns the rule off. A float in
    (0, 1] restores it at that threshold (0.45 was the historical value).
    Garbage or out-of-range values fall back to the default rather than
    silently picking a threshold nobody chose.
    """
    raw = os.environ.get("SYNAP_RETRIEVAL_DEDUP_JACCARD", "").strip()
    if not raw:
        return _DEFAULT_DEDUP_JACCARD
    # Only the explicit word. "none" is deliberately NOT accepted: a
    # stringified Python None reaching an env var is a plausible typo, and it
    # must not silently open the gate fleet-wide (see the bad-value test).
    if raw.lower() == "off":
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "SYNAP_RETRIEVAL_DEDUP_JACCARD=%r is not a number or an off-word; "
            "using %.2f", raw, _DEFAULT_DEDUP_JACCARD)
        return _DEFAULT_DEDUP_JACCARD
    if not 0.0 <= value <= 1.0:
        logger.warning(
            "SYNAP_RETRIEVAL_DEDUP_JACCARD=%s is outside 0..1; using %.2f",
            value, _DEFAULT_DEDUP_JACCARD)
        return _DEFAULT_DEDUP_JACCARD
    return value


def _deduplicate_jaccard(
    items: List[ContextItem],
    threshold: float = 0.45,
) -> List[ContextItem]:
    """
    Deduplicate using Jaccard similarity.

    When two items are duplicates, prefer the one with an actual
    query-relevance score (similarity_score > 0) over one without,
    since it was retrieved via vector search against the query.
    """
    if not items:
        return []

    # Off is off, and it has to be an early return rather than a threshold of 0:
    # ``similarity > 0`` would collapse every pair that shares a single token,
    # which is the opposite of disabling the rule.
    if threshold <= 0.0:
        return items

    # Visit in a deterministic, quality-ordered sequence rather than in arrival
    # order. Two things were wrong with arrival order.
    #
    # It is not stable. `all_items` is assembled from concurrently gathered
    # sources, so the same request can present the same items in a different
    # sequence between runs. This loop keeps the FIRST of a duplicate pair, so
    # an unstable order silently produced a different answer. For example, the same
    # client-scope fact, identical content and
    # identical score 0.398602, appeared in one run and not the next, for two
    # different users.
    #
    # It also picked the wrong winner. The tiebreak below only replaces an
    # existing item when the new one has a score and the existing one does not,
    # so between two SCORED duplicates the first to arrive won even when the
    # other scored higher.
    #
    # Sorting by score descending fixes both: the outcome no longer depends on
    # arrival, and the better of a duplicate pair is the one kept. `item_id` is
    # the tiebreak so equal scores still resolve the same way every time.
    #
    # The RESULT is emitted in the caller's original order (see the end of this
    # function), so this changes which items survive and never the sequence
    # they are returned in.
    order = sorted(
        range(len(items)),
        key=lambda i: (
            -max(items[i].similarity_score or 0.0, items[i].relevance_score or 0.0),
            str(getattr(items[i], "item_id", "") or ""),
        ),
    )
    original_position = {id(items[i]): i for i in range(len(items))}
    items = [items[i] for i in order]

    # Pre-compute token sets and their lengths once
    item_token_sets = [frozenset(_tokenize_stemmed(item.content or "")) for item in items]

    # For Jaccard > threshold, the smaller set must be at least
    # threshold/(1+threshold) of the larger set's size.
    min_size_ratio = threshold / (1.0 + threshold)

    kept: List[ContextItem] = []
    kept_token_sets: List[frozenset] = []
    kept_lens: List[int] = []

    for i, item in enumerate(items):
        item_tokens = item_token_sets[i]
        item_len = len(item_tokens)
        duplicate_idx = None

        for idx, kept_tokens in enumerate(kept_token_sets):
            # Size-based pruning: skip pairs that can't meet the threshold
            kept_len = kept_lens[idx]
            if item_len > kept_len:
                if kept_len < item_len * min_size_ratio:
                    continue
            else:
                if item_len < kept_len * min_size_ratio:
                    continue

            similarity = _jaccard_similarity_sets(item_tokens, kept_tokens)
            if similarity > threshold:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            kept.append(item)
            kept_token_sets.append(item_tokens)
            kept_lens.append(item_len)
        else:
            existing = kept[duplicate_idx]
            existing_has_score = existing.similarity_score > 0.0 or existing.relevance_score > 0.0
            new_has_score = item.similarity_score > 0.0 or item.relevance_score > 0.0

            if new_has_score and not existing_has_score:
                kept[duplicate_idx] = item
                kept_token_sets[duplicate_idx] = item_tokens
                kept_lens[duplicate_idx] = item_len

    # Back to the caller's order. Selection is score-ordered above; the
    # sequence handed back is the one the caller gave us.
    kept.sort(key=lambda it: original_position.get(id(it), 0))
    return kept


# Core subqueries always used in FAST template-based decomposition
_CORE_RETRIEVAL_TEMPLATES = [
    "What are the user's preferences related to {topic}?",
    "What relevant facts exist about this user regarding {topic}?",
    "What past interactions are relevant to {topic}?",
]

# B2B subqueries: only added when the user's graph contains B2B entity types
_B2B_RETRIEVAL_TEMPLATES = [
    "What customer or organization context applies to {topic}?",
    "What project or engagement history exists for {topic}?",
    "What team relationships or escalation paths relate to {topic}?",
]

# Entity types whose presence signals B2B data in the graph
_B2B_ENTITY_TYPES = frozenset({
    "organization", "company", "client", "project",
    "team", "department", "account",
})

# Combined list for backwards compatibility (MACA fallback_queries, etc.)
STANDARD_RETRIEVAL_FALLBACKS = _CORE_RETRIEVAL_TEMPLATES + _B2B_RETRIEVAL_TEMPLATES

@dataclass
class RetrievalConfig:
    """Configuration for retrieval behavior."""
    # Timeouts
    fast_timeout_ms: int = 8000
    accurate_timeout_ms: int = 30000
    subquery_timeout_ms: int = 5000

    # Limits
    max_vector_results: int = 25
    max_graph_hops: int = 2
    max_subqueries: int = 4

    # Max concurrent LLM re-rank batches per request. The filter splits candidates
    # into batches of 30 and fans them out; this bounds the burst so a single large
    # candidate set can't open ceil(N/30) simultaneous Groq calls. Output-neutral:
    # batches are filtered independently and merged by set-union, so the cap only
    # changes scheduling, never which items are kept.
    llm_filter_max_concurrency: int = 8

    # Confidence
    confidence_threshold: float = 0.7
    min_results_for_confidence: int = 5

    # Default token budget
    default_token_budget: int = 2000

    # FAST-mode LLM re-rank filter gate (opt-in; OFF by default).
    # When enabled, FAST mode skips the Groq relevance filter if the candidate
    # count already fits the caller's max_results — there is nothing to *select*,
    # only relevance-drop, so it saves a ~0.8-1.3s Groq round-trip. OFF by
    # default because the filter drops irrelevant items regardless of count, so
    # skipping is NOT output-identical and trades a little precision for latency;
    # validate with a recall eval before enabling.
    fast_skip_filter_when_scarce: bool = False


# The Groq relevance filter is INERT, and has been for as long as the logs go
# back. It asks for a KEEP line under a 200-token cap, and Groq's default model
# is now a reasoning model (openai/gpt-oss-120b), which spends that entire budget
# on reasoning tokens and is truncated before it emits a single visible
# character. The call returns "", the KEEP parse finds nothing, and the fail-open
# path keeps every candidate.
#
# In testing, seven calls out of seven came back with
# out_toks=200 (exactly the cap), response_preview='' and items_removed=0, at
# 566-611ms and roughly 800 tokens each. It was the single largest component of a
# ~770ms retrieval, and it decided nothing.
#
# So it is off by default, because off is what it has actually been doing, only
# without the round trip. This is a LATENCY change, not an output change: keeping
# every candidate is exactly what the fail-open path was already producing.
#
# One caveat before flipping it back on: a deployment that routes this call to a
# NON-reasoning model has a filter that really works, and for that one, off IS a
# behaviour change.
#
# _LLM_FILTER_MAX_TOKENS exists so that "on" means something again. The old cap
# could not survive a reasoning preamble; this one can.
_LLM_FILTER_ENABLED = os.environ.get(
    "SYNAP_RETRIEVAL_LLM_FILTER", "off"
).strip().lower() in {"1", "true", "on", "yes"}
_LLM_FILTER_MAX_TOKENS = int(
    os.environ.get("SYNAP_RETRIEVAL_LLM_FILTER_MAX_TOKENS") or "1200"
)


class FilterReason(Enum):
    """Reason an item was removed by ResponseFilter."""
    TOO_SHORT = "too_short"
    ECHO_OF_QUERY = "echo_of_query"
    ACTION_ITEM = "action_item"
    DUPLICATE = "duplicate"


class ResponseFilter:
    """Post-retrieval quality filter applied before returning a context bundle.

    Runs three lightweight checks in order:
    1. Utility: drop items whose content is under 10 characters.
    2. Echo: drop items that essentially restate the query verbatim.
    3. Dedup: drop near-duplicates via Jaccard token similarity.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    def filter_items(
        self, items: List["ContextItem"], query: str
    ) -> List["ContextItem"]:
        """Filter *items* against *query* and return the surviving items."""
        query_tokens = set(query.lower().split())
        kept: List["ContextItem"] = []

        for item in items:
            content = (item.content or "").strip()

            # 1. Utility check — too short to be useful
            if len(content) < 10:
                continue

            # 2. Echo check — content largely repeats the query
            content_tokens = set(content.lower().split())
            if query_tokens and content_tokens:
                overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
                if overlap >= 0.85 and len(content_tokens) <= len(query_tokens) + 3:
                    continue

            kept.append(item)

        # 3. Jaccard deduplication — remove near-duplicates (keep highest similarity score)
        kept = self._deduplicate(kept)
        return kept

    def _deduplicate(self, items: List["ContextItem"]) -> List["ContextItem"]:
        """Remove near-duplicate items using Jaccard on stemmed, stop-word-free tokens."""
        result: List["ContextItem"] = []
        token_sets: List[set] = []

        for item in items:
            tokens = set(_tokenize_stemmed(item.content or ""))
            is_dup = False
            for existing_tokens in token_sets:
                union = existing_tokens | tokens
                if not union:
                    continue
                jaccard = len(existing_tokens & tokens) / len(union)
                if jaccard >= self.similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                result.append(item)
                token_sets.append(tokens)

        return result


@dataclass
class MACARetrievalSettings:
    """Retrieval-relevant subset of MACA config, loaded per-instance.

    Extracted from MemoryArchitectureConfig JSONB for fast access at retrieval time.
    All defaults match current hardcoded behavior for zero-regression when no MACA config exists.
    """
    # Mode
    default_mode: str = "fast"
    modes: Dict[str, dict] = field(default_factory=dict)

    # Ranking
    ranking_signals: Dict[str, float] = field(default_factory=lambda: {"recency": 0.10, "relevance": 0.70, "importance": 0.20})
    boost_categories: Dict[str, float] = field(default_factory=dict)
    decay_function: str = "exponential"
    decay_half_life_days: int = 30

    # Context budget
    max_tokens: int = 4000
    reserve_for_memories: int = 1500

    # Scoping
    user_context_isolation: str = "strict"  # "strict" | "maca_suggests" | "equals_customer"
    allow_cross_user_retrieval: bool = False
    allow_cross_customer_retrieval: bool = False
    roll_up_to_customer: bool = True
    # R-03. How much score a record loses per rung of distance above the
    # requester. None means use the system default. Read on the retrieval side
    # only and deliberately NOT offered on the dashboard yet: the default is
    # unmeasured (Part 10 item 5) and a knob whose value nobody has measured is
    # worse than no knob, per Z-05. It becomes a client-facing setting when
    # SHIP-07 has a number behind it.
    scope_closeness_per_level_cost: Optional[float] = None

    # Enabled categories
    enabled_categories: List[str] = field(default_factory=lambda: ["facts", "preferences", "episodes", "emotions", "temporal_events"])
    category_priorities: Dict[str, str] = field(default_factory=dict)

    # Fallback queries (MACA-provided templates)
    fallback_queries: List[str] = field(default_factory=list)

    # Graph schema edge types (for Neo4j relationship type hints)
    graph_edge_types: List[str] = field(default_factory=list)

    # Confidence threshold
    confidence_threshold: float = 0.7

    # Per-type floor (Workstream B lever 2). Guarantees at least this many
    # items of EACH context_type survive the final trim before the rest of the
    # budget is filled by global relevance — so a single low-frequency type
    # (e.g. one email fact) isn't crowded out of the top-N by a cluster of
    # high-frequency policy facts. Default 0 = today's single global cut, so
    # instances without this MACA key are byte-for-byte unchanged. Playground
    # instances set it to 1–2 in their retrieval_config.
    per_type_floor: int = 0

    # Names of the enabled custom categories (canonical form). Emptied for a request whose
    # instance has SYNAP_CUSTOM_CATEGORIES off, in _fetch_context_inner.
    custom_category_names: Tuple[str, ...] = ()

    @classmethod
    def from_maca_config(cls, config_dict: dict) -> "MACARetrievalSettings":
        """Parse from raw MACA config JSONB dict."""
        rc = config_dict.get("retrieval_config", {})
        sa = config_dict.get("storage_architecture", {})
        mc = config_dict.get("memory_categories", {})
        me = config_dict.get("memory_extraction", {})

        # Enabled categories
        standard_cats = ["facts", "preferences", "episodes", "emotions", "temporal_events"]
        enabled = [c for c in standard_cats if mc.get(c, {}).get("enabled", True)]

        # Category priorities
        priorities = {}
        for cat in standard_cats:
            p = mc.get(cat, {}).get("priority")
            if p:
                priorities[cat] = p

        # Graph edge types from schemas
        edge_types = []
        for schema in sa.get("graph_schemas", []):
            if schema.get("enabled", True):
                edge_types.extend(schema.get("edge_types", []))

        scoping = sa.get("scoping", {})
        ranking = rc.get("ranking", {})
        budget = rc.get("context_budget", {})
        modes = {n: m for n, m in rc.get("modes", {}).items() if isinstance(m, dict)}

        return cls(
            default_mode=rc.get("default_mode", "fast"),
            modes=modes,
            ranking_signals=ranking.get("primary_signals", {"recency": 0.10, "relevance": 0.70, "importance": 0.20}),
            boost_categories=ranking.get("boost_categories", {}),
            decay_function=ranking.get("decay_function", "exponential"),
            decay_half_life_days=ranking.get("decay_half_life_days", 30),
            max_tokens=budget.get("max_tokens", 4000),
            reserve_for_memories=budget.get("reserve_for_memories", 1500),
            user_context_isolation=scoping.get("user_context_isolation", "strict"),
            allow_cross_user_retrieval=scoping.get("allow_cross_user_retrieval", False),
            allow_cross_customer_retrieval=scoping.get("allow_cross_customer_retrieval", False),
            roll_up_to_customer=scoping.get("roll_up_to_customer", True),
            scope_closeness_per_level_cost=scoping.get("scope_closeness_per_level_cost"),
            enabled_categories=enabled,
            category_priorities=priorities,
            fallback_queries=rc.get("fallback_queries", []),
            graph_edge_types=list(set(edge_types)),
            confidence_threshold=me.get("filters", {}).get("min_confidence_threshold", 0.7),
            per_type_floor=int(rc.get("per_type_floor", 0) or 0),
            custom_category_names=tuple(
                c.name for c in custom_categories.enabled_categories(mc.get("custom_categories"))),
        )


# --- MACA Config Cache ---
_maca_config_cache: Dict[str, tuple] = {}  # {instance_id: (MACARetrievalSettings|None, timestamp)}
_MACA_CACHE_TTL_SECONDS = 300  # 5 minutes
# "No active config" is cached far more briefly than a real config. A config
# can appear at any time (the build finishes minutes after the instance is
# created), and a cached None keeps every request on the DEFAULT settings until
# it expires. 5 minutes of that after a build completes is a wrong result nobody
# sees; 15 seconds is not.
_MACA_NONE_CACHE_TTL_SECONDS = 15


def _none_cache_age_offset(instance_id: str) -> float:
    """How far to back-date a cached "no config" so it expires at the right time.

    The 15 s TTL applies ONLY while the instance has a supervised build running
    or just finished (the config can appear any second). Every other instance
    (legacy, playground, review-gated, failed) keeps the 5 minute TTL, so they
    cost the database nothing extra.
    """
    try:
        from synap.cloud.application.lifecycle.instance_readiness import is_supervised_building

        if is_supervised_building(instance_id):
            return _MACA_CACHE_TTL_SECONDS - _MACA_NONE_CACHE_TTL_SECONDS
    except Exception:  # noqa: BLE001 - a hint only
        pass
    return 0


async def _apply_cross_user_grant(
    session, instance_id: str, settings: "MACARetrievalSettings"
) -> "MACARetrievalSettings":
    """A person's recorded decision decides cross-user sharing. Nothing else.

    The stored config is not the authority here, and deliberately so. It is
    rewritten by every MACA run, so a decision living in it would be undone by
    the next design pass without anybody choosing that. The grant table outlives
    a redesign and says who decided.

    Absence is the default and the default is OFF, which is what every instance
    has today. So this can only ever narrow what the config said,
    never widen it, unless a person has explicitly granted it.

    Never raises. A missing table means migration 105 has not run in this
    environment, and "we cannot read the grant" must read as "no grant". The
    alternative, treating an unreadable record as permission, would open a
    privacy boundary because a query failed.
    """
    import dataclasses

    from sqlalchemy import text

    try:
        row = (await session.execute(
            text("SELECT granted FROM instance_cross_user_grant WHERE instance_id = :i"),
            {"i": instance_id},
        )).scalar_one_or_none()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "cross-user grant not readable for %s (%s); treating as not granted",
            instance_id, type(exc).__name__,
        )
        row = None

    granted = bool(row)
    if granted == settings.allow_cross_user_retrieval:
        return settings

    # Said out loud in both directions. A boundary being opened is worth a log
    # line naming the instance, and a config asking for something it is not
    # getting is worth one too, because that is the state a stale MACA proposal
    # leaves behind.
    logger.warning(
        "cross_user_retrieval for %s is %s by human decision; the stored config "
        "says %s",
        instance_id, "ON" if granted else "OFF", settings.allow_cross_user_retrieval,
    )
    return dataclasses.replace(settings, allow_cross_user_retrieval=granted)


async def _load_maca_config(instance_id: str) -> Optional[MACARetrievalSettings]:
    """Load MACA config from Postgres with 5min TTL cache.

    Returns cached MACARetrievalSettings if within TTL, otherwise queries
    maca_config table for the active (approved) config for this instance.
    """
    if edition.config_source == "file":
        # Core: one YAML file for the install. The raw view is
        # what the developer wrote, so a missing ``ranking`` block lands on the
        # retrieval defaults. The loader re-reads the file when it changes, so
        # there is no TTL cache here, and no cross-user grant (Core has none).
        # A failure here is not swallowed: Core must not fall back to defaults silently.
        from synap.cloud.memory_config.loader import get_config_loader

        raw = await get_config_loader().get_raw_config(instance_id)
        return MACARetrievalSettings.from_maca_config(raw)

    now = time.time()
    cached = _maca_config_cache.get(instance_id)
    if cached and (now - cached[1]) < _MACA_CACHE_TTL_SECONDS:
        return cached[0]

    try:
        from synap.cloud.database.connection import get_session_factory
        from synap.cloud.database.repositories.maca_config_repo import MACAConfigRepository

        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = MACAConfigRepository(session)
            config_orm = await repo.get_active_for_instance(instance_id)

            if config_orm and config_orm.config_content:
                settings = MACARetrievalSettings.from_maca_config(config_orm.config_content)
                settings = await _apply_cross_user_grant(session, instance_id, settings)
                _maca_config_cache[instance_id] = (settings, now)
                logger.info(f"MACA config loaded for instance {instance_id}")
                return settings
            else:
                # Back-dated (supervised-building instances only) so the shared
                # 300s check expires it after _MACA_NONE_CACHE_TTL_SECONDS.
                _maca_config_cache[instance_id] = (None, now - _none_cache_age_offset(instance_id))
                return None
    except Exception as e:
        logger.warning(f"Failed to load MACA config for {instance_id}: {e}")
        return None


def invalidate_maca_cache(instance_id: str) -> None:
    """Drop the in-process MACA cache entry for an instance so a refreshed config
    (e.g. after data-driven schema refinement updates domain_hints) takes effect
    immediately instead of after the TTL. In-process only; other workers fall back
    to the TTL. Safe to call for an instance that isn't cached.
    """
    _maca_config_cache.pop(instance_id, None)


# Decision-critical relationship-type keywords. Graph-discovered memories reached
# via an edge whose type contains one of these bypass the similarity-based top-70%
# cut in _neo4j_graph_discovery: the relationship itself signals decision-relevance
# (e.g. a failing test that BLOCKS / a dependency a task DEPENDS_ON), so the memory
# must surface even when its text isn't similar to the query. This is what lets an
# accurate-mode graph hop bring an in-progress task's blocker into the bundle.
_CRITICAL_EDGE_KEYWORDS = (
    "BLOCK", "RISK", "FAIL", "SUPERSED", "SUPERCED", "INVALID",
    "PREVENT", "THREAT", "JEOPARD", "DEPEND", "TEST", "DEPRECAT",
)


def _is_critical_edge_type(edge_type: Optional[str]) -> bool:
    et = (edge_type or "").upper()
    return any(kw in et for kw in _CRITICAL_EDGE_KEYWORDS)


def _graph_traversal_instance_allowed(instance_id: Any) -> bool:
    """Per-instance gate for graph traversal (blast-radius control for rollout).

    When ``GRAPH_TRAVERSAL_INSTANCE_ALLOWLIST`` is set (comma-separated instance ids,
    raw ``inst_...`` or UUID form), graph traversal runs ONLY for those instances —
    so the feature can be enabled for one instance (e.g. a single customer)
    without turning it on for everyone, even while ``ENABLE_GRAPH_TRAVERSAL`` is on
    globally. When the allowlist is empty/unset, all instances are allowed (the
    global flag remains the only gate — backward compatible).
    """
    allow = os.environ.get("GRAPH_TRAVERSAL_INSTANCE_ALLOWLIST", "").strip()
    if not allow:
        return True
    iid = str(instance_id)
    allowed = set()
    for x in allow.split(","):
        x = x.strip()
        if not x:
            continue
        allowed.add(x)
        try:
            from synap.cloud.shared.scope import parse_or_deterministic_uuid
            allowed.add(str(parse_or_deterministic_uuid(x)))
        except Exception:
            pass
    return iid in allowed


# --- Cross-Instance Visibility Policy (read-side) ---
#
# A first-class, CLIENT-LEVEL policy controlling which instances' memories are
# visible to which other instances. Enforced as an opt-in `instance_id` filter
# on the customer-scoped collection. `instance_id` is already stamped on every
# vector (vector_wrapper._scope_metadata), so this is read-side only — no data
# migration. Default-open: a client with no policy row (or default_mode=
# 'shared_all') resolves to VISIBILITY_ALL and adds NO predicate, leaving the
# hot path byte-identical to today.


class _AllInstances:
    """Sentinel: no instance-visibility restriction (default-open)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "VISIBILITY_ALL"


# Module-level singleton sentinel. `is` comparisons identify the default-open case.
VISIBILITY_ALL = _AllInstances()


# These values describe the *caller*, not the RetrievalManager.  The manager is
# an application singleton, so keeping them on ``self`` lets one overlapping
# request replace another request's authorization boundary after an ``await``.
# ContextVar values are task-local and are copied into child tasks, which is
# exactly the lifetime needed by the parallel retrieval arms below.
_REQUEST_MACA: ContextVar[Optional[MACARetrievalSettings]] = ContextVar(
    "synap_request_maca", default=None
)
_REQUEST_VISIBILITY: ContextVar[Optional[Any]] = ContextVar(
    "synap_request_visibility", default=None
)


def _request_maca(manager: "RetrievalManager") -> MACARetrievalSettings:
    """Return this task's MACA settings, preserving the test-only fallback."""
    return _REQUEST_MACA.get() or getattr(manager, "_current_maca", None) or MACARetrievalSettings()


def _request_maca_or_none(manager: "RetrievalManager") -> Optional[MACARetrievalSettings]:
    """This task's MACA settings, or None when none were ever set.

    For the two readers that treat "no MACA config" as its own case: the ranking
    function returns the raw relevance score (no weights, no recency) and the
    token budget skips the configured priorities. Substituting default settings
    there would switch the weighted formula on for any caller that has no MACA,
    which is a different ranking. Inside a real request MACA is always set.
    """
    return _REQUEST_MACA.get() or getattr(manager, "_current_maca", None)


def _request_visibility(manager: "RetrievalManager"):
    """Return this task's visibility set, preserving the test-only fallback."""
    value = _REQUEST_VISIBILITY.get()
    if value is not None:
        return value
    return getattr(manager, "_current_visibility", VISIBILITY_ALL)


@dataclass(frozen=True)
class _VisibilityBundle:
    """Per-client policy inputs, cached and resolved per-viewer at request time.

    Only built when default_mode == 'isolated'; the 'shared_all'/no-row case is
    represented by the VISIBILITY_ALL sentinel instead of a bundle.
    """

    grants_by_viewer: Dict[str, frozenset]   # viewer_instance_id -> {source_instance_id}
    children_by_parent: Dict[str, tuple]     # parent_instance_id -> (child_instance_id, ...)
    shareable: Dict[str, bool]               # instance_id -> shareable (absent == True)


# {client_id: (VISIBILITY_ALL | _VisibilityBundle, timestamp)}
_visibility_bundle_cache: Dict[str, tuple] = {}
_VISIBILITY_CACHE_TTL_SECONDS = 60  # short: a privacy tightening must propagate fast
# How long "the policy could not be loaded" is remembered before asking again.
_VISIBILITY_RETRY_SECONDS = 5


class _PolicyUnavailable:
    """The visibility policy could not be loaded and none was ever loaded."""

    def __repr__(self) -> str:
        return "VISIBILITY_UNAVAILABLE"


_VISIBILITY_UNAVAILABLE = _PolicyUnavailable()
# {instance_id: client_id} — stable mapping, used only when the caller didn't
# pass a raw client_id. Never changes for an instance, so no TTL needed.
_instance_client_cache: Dict[str, str] = {}


def _resolve_allowed_from_bundle(bundle: "_VisibilityBundle", viewer_instance_id_raw: str) -> frozenset:
    """Pure resolution: (isolated bundle, viewer) -> allowed source UUID set.

    An instance always sees itself; plus explicitly-granted sources and derived
    children, each gated by the source's shareable flag (absent == True). Raw
    ids are converted to the deterministic-UUID form ChromaDB metadata stores.
    Extracted as a module function so it is unit-testable without a DB.
    """
    from synap.cloud.shared.scope import parse_or_deterministic_uuid

    allowed_raw = {viewer_instance_id_raw}  # an instance always sees itself
    for source in bundle.grants_by_viewer.get(viewer_instance_id_raw, ()):  # explicit grants
        if bundle.shareable.get(source, True):
            allowed_raw.add(source)
    for child in bundle.children_by_parent.get(viewer_instance_id_raw, ()):  # derived parent->child
        if bundle.shareable.get(child, True):
            allowed_raw.add(child)
    return frozenset(str(parse_or_deterministic_uuid(i)) for i in allowed_raw)


def invalidate_visibility_cache(client_id: str) -> None:
    """Drop the cached visibility bundle for a client.

    Called from the admin write path so a dashboard edit takes effect
    immediately *in this process*. NOTE: the cache is in-process; in a
    multi-process deployment other workers still honour the TTL. Full
    cross-process invalidation is a documented fast-follow.
    """
    _visibility_bundle_cache.pop(client_id, None)


async def invalidate_client_query_cache(client_id: str) -> int:
    """Bust every cached L1 query result for a client (Redis, cross-process).

    A visibility-policy change rewrites which instances' memories a fetch may
    return, but the L1 query-result cache key (``cache_keys.l1_query_key``) does
    NOT include the visibility policy — so a stale, pre-change bundle would be
    served for up to the 300s TTL. Dropping the in-process bundle cache
    (:func:`invalidate_visibility_cache`) is not enough on its own because the
    *result* of a fetch is memoised separately in Redis. We therefore delete the
    whole ``query:{client_uuid}:*`` namespace (all customers under the client),
    forcing the next fetch to recompute under the new policy.

    Keys are written with the *resolved* (UUID-form) client id
    (``retrieval_manager._build_cache_key`` -> ``str(scope.client_id)``), so we
    convert the raw ``cli_`` id the dashboard passes via
    ``parse_or_deterministic_uuid`` before matching. Fail-open: a Redis hiccup
    must never fail the policy write (the 300s TTL is the backstop).

    Returns the number of keys deleted.
    """
    try:
        from synap.cloud.application.shared.providers import get_cached_providers
        from synap.cloud.shared.scope import parse_or_deterministic_uuid

        cache = get_cached_providers().get("redis")
        if cache is None:
            logger.debug("query cache invalidation skipped: no redis provider")
            return 0

        client_uuid = str(parse_or_deterministic_uuid(client_id))
        pattern = f"query:{client_uuid}:*"
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await cache.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                await cache.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info(
            "visibility_l1_query_cache_invalidated client_id=%s client_uuid=%s keys_deleted=%s",
            client_id, client_uuid, deleted,
        )
        return deleted
    except Exception as e:  # never fail the policy write on a cache hiccup
        logger.warning(
            "L1 query cache invalidation failed (non-fatal) for client %s: %s", client_id, e
        )
        return 0


async def _resolve_client_id_for_instance(instance_id_raw: str) -> Optional[str]:
    """Look up an instance's raw client_id (cached indefinitely; it never changes)."""
    if not instance_id_raw:
        return None
    cached = _instance_client_cache.get(instance_id_raw)
    if cached is not None:
        return cached
    try:
        from synap.cloud.database.connection import get_session_factory
        from synap.cloud.database.repositories.instance_repo import InstanceRepository

        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = InstanceRepository(session)
            inst = await repo.get_by_instance_id(instance_id_raw)
            if inst is not None:
                _instance_client_cache[instance_id_raw] = inst.client_id
                return inst.client_id
    except Exception as e:
        logger.warning(f"Failed to resolve client_id for instance {instance_id_raw}: {e}")
    return None


def _is_missing_policy_table(exc: BaseException) -> bool:
    """True when a database error says the relation does not exist (SQLSTATE 42P01)."""
    cur: Optional[BaseException] = exc
    for _ in range(4):
        if cur is None:
            return False
        if getattr(cur, "sqlstate", None) == "42P01" or type(cur).__name__ == "UndefinedTableError":
            return True
        cur = getattr(cur, "orig", None) or cur.__cause__
    return False


async def _load_visibility_bundle(client_id_raw: str):
    """Load (and cache) the visibility bundle for a client.

    Returns VISIBILITY_ALL when the client has no policy or default_mode=
    'shared_all' (default-open). On a DB error it keeps serving the last value
    it loaded, restrictive or open (a lookup that fails says nothing new about
    the policy). With NO last value it returns ``_VISIBILITY_UNAVAILABLE`` and
    the caller reads nothing: not knowing whether a client is isolated is not a
    reason to assume it is not. That answer is cached only briefly, so the next
    request after the store recovers asks again.
    """
    now = time.time()
    cached = _visibility_bundle_cache.get(client_id_raw)
    if cached and (now - cached[1]) < _VISIBILITY_CACHE_TTL_SECONDS:
        return cached[0]

    try:
        # The policy and grants live in tables Core does not have. With no hook
        # registered this reads as "no policy": default-open, as a client with
        # no policy row has always been.
        loaded = await edition.acall("visibility_and_grants", client_id_raw)
        if loaded is None:
            _visibility_bundle_cache[client_id_raw] = (VISIBILITY_ALL, now)
            return VISIBILITY_ALL

        grants_by_viewer = loaded["grants_by_viewer"]
        children_by_parent = loaded["children_by_parent"]
        shareable = loaded["shareable"]
        bundle = _VisibilityBundle(
            grants_by_viewer={v: frozenset(s) for v, s in grants_by_viewer.items()},
            children_by_parent={p: tuple(c) for p, c in children_by_parent.items()},
            shareable=dict(shareable),
        )
        _visibility_bundle_cache[client_id_raw] = (bundle, now)
        logger.info(
            f"Visibility policy loaded for client {client_id_raw} "
            f"(isolated; {loaded['grant_count']} grants, {len(shareable)} shareable flags)"
        )
        return bundle
    except Exception as e:
        if _is_missing_policy_table(e):
            # Not an outage: the table does not exist (migration not applied),
            # so no client can have a policy. Default-open is the truth here,
            # and it is what an unmigrated environment has always done.
            logger.warning(
                f"Visibility policy table missing; treating {client_id_raw} as default-open: {e}"
            )
            _visibility_bundle_cache[client_id_raw] = (VISIBILITY_ALL, now)
            return VISIBILITY_ALL
        logger.warning(f"Failed to load visibility policy for {client_id_raw}: {e}")
        if cached and cached[0] is not _VISIBILITY_UNAVAILABLE:
            # Last known state, held for the TTL window so a missing table or an
            # outage does not re-query Postgres on every request.
            _visibility_bundle_cache[client_id_raw] = (cached[0], now)
            return cached[0]
        # Never widen a read on an authorization dependency failure. Remember
        # the failure for seconds, not the full TTL.
        _visibility_bundle_cache[client_id_raw] = (
            _VISIBILITY_UNAVAILABLE,
            now - _VISIBILITY_CACHE_TTL_SECONDS + _VISIBILITY_RETRY_SECONDS,
        )
        return _VISIBILITY_UNAVAILABLE


# ---------------------------------------------------------------------------
# Scope filter builder (shared across vector search sites)
# ---------------------------------------------------------------------------

def _legacy_read_predicate(scope, maca: "MACARetrievalSettings", visibility_allowed=VISIBILITY_ALL):
    """The scope rule for a request with no resolved chain, as a predicate tree.

    ONE definition. `_build_customer_scope_filter` (Chroma) and
    `_scope_filter_predicate` (Python post-filter) are both compiled from it,
    and so is the nested path's legacy arm, so no condition can be added to one
    form and forgotten in another. Add a condition in `scope_filter`, not here.
    """
    from synap.cloud.shared import scope_filter as _sf

    return _sf.legacy_read_predicate(
        customer_id=str(scope.customer_id) if scope.customer_id else None,
        user_id=str(scope.user_id) if scope.user_id else None,
        roll_up_to_customer=maca.roll_up_to_customer,
        cross_user=maca.allow_cross_user_retrieval,
        cross_customer=maca.allow_cross_customer_retrieval,
        visibility_allowed=(
            None
            if visibility_allowed is None or visibility_allowed is VISIBILITY_ALL
            else list(visibility_allowed)
        ),
    )


def _visibility_where(visibility_allowed=VISIBILITY_ALL) -> Dict[str, Any]:
    """The cross-instance clause alone, as a where-fragment ({} when unrestricted).

    For reads that build the rest of their filter themselves (short-term turns,
    the starvation repair, the anchor search). Spelled in `scope_filter`, once.
    """
    if visibility_allowed is None or visibility_allowed is VISIBILITY_ALL:
        return {}
    from synap.cloud.shared import scope_filter as _sf

    return _sf.to_flat_where(_sf.instance_visibility_predicate(list(visibility_allowed)))


def _memory_type_where(context_types) -> Dict[str, Any]:
    """The type condition as a where-fragment. Spelled in `scope_filter`, once."""
    from synap.cloud.shared import scope_filter as _sf

    return _sf.to_flat_where(_sf.memory_type_predicate(list(context_types)))


def _instance_lists_where(where):
    """Widen an instance clause so it also reads the LIST of instances.

    Entity and triple vectors are shared by every instance of a customer, so
    their metadata carries `instance_ids` (every writer) next to the scalar
    `instance_id` (the last writer; the only thing a document written before the
    list existed has). A restricted viewer sees a document when EITHER names an
    instance they may read. Where there is no instance clause (default-open,
    which is nearly every client) the filter is returned untouched, so the
    default query is exactly what it was.
    """
    if not where:
        return where
    if isinstance(where, dict) and "instance_id" not in where and not any(k in where for k in ("$and", "$or")):
        return where

    def widen(node):
        if not isinstance(node, dict):
            return node
        rest: Dict[str, Any] = {}
        widened: List[Dict[str, Any]] = []
        for key, value in node.items():
            if key in ("$and", "$or"):
                rest[key] = [widen(child) for child in value]
            elif key == "instance_id" and isinstance(value, dict) and "$in" in value:
                # The list clause is the shared predicate, compiled; this only ORs
                # it with the scalar so a document written before lists existed is
                # still read by its last writer.
                listed = _scope_filter.to_chroma_where(
                    _scope_filter.graph_instance_predicate(list(value["$in"])))
                widened.append({"$or": [{"instance_id": value}]
                                + (listed["$or"] if "$or" in listed else [listed])})
            else:
                rest[key] = value
        if not widened:
            return rest
        parts = [{k: v} for k, v in rest.items()] + widened
        return parts[0] if len(parts) == 1 else {"$and": parts}

    return widen(where)


def _build_customer_scope_filter(
    scope, maca: "MACARetrievalSettings", visibility_allowed=VISIBILITY_ALL
) -> Dict[str, Any]:
    """Build ChromaDB where-clause for the customer-level collection.

    Narrow-to-broad: a user-scope caller's filter also matches customer-scope
    docs (user_id = CUSTOMER_SCOPE_SENTINEL) when roll_up_to_customer is on.
    A customer-scope caller (no user_id passed) only sees customer-shared
    docs. A client-scope caller (no customer_id passed) skips this collection
    entirely — the client-level collection is queried separately.

    Escape hatches:
    - ``allow_cross_customer_retrieval`` drops the customer_id filter.
    - ``allow_cross_user_retrieval`` drops the user_id filter (user sees
      every user's personal docs within the customer).
    - ``roll_up_to_customer=False`` disables the user→customer rollup —
      user-scope caller sees only their own docs, not customer-shared ones.

    Cross-instance visibility (``visibility_allowed``):
    - VISIBILITY_ALL (default) → no instance predicate (default-open; hot path
      unchanged).
    - a frozenset of UUID-form source instance_ids → restrict to docs whose
      ``instance_id`` metadata is in the set. Applied ONLY to the customer-scope
      collection; client-scope shared knowledge is never instance-filtered.
      Compiled from the same predicate as ``_scope_filter_predicate``.

    Compiled from `_legacy_read_predicate`; see it. Raises ``NothingReadable``
    for an empty visibility set (callers return before asking, because an empty
    `$in` is not a clause the store accepts).
    """
    from synap.cloud.shared import scope_filter as _sf

    return _sf.to_flat_where(_legacy_read_predicate(scope, maca, visibility_allowed))


def _and_where(*clauses) -> Dict[str, Any]:
    """Combine Chroma where-clauses without losing an `$or`.

    Flat clauses are merged into one dict, which is what the call sites did
    before and is what keeps the emitted document byte-identical for a request
    with no resolved chain. A clause carrying a top-level operator cannot be
    merged that way: `{**a, **b}` keeps only one `$or` key, and the one it drops
    is the scope predicate, so the request reads the whole collection filtered
    on nothing but confidence. That is a leak produced by a spread operator, and
    it is the exact trap `_direct_similarity_search` documents next to its own
    nested branch.
    """
    kept = [c for c in clauses if c]
    if not kept:
        return {}
    if len(kept) == 1:
        return dict(kept[0])
    if any(any(str(k).startswith("$") for k in c) for c in kept):
        return {"$and": [dict(c) for c in kept]}
    merged: Dict[str, Any] = {}
    for c in kept:
        merged.update(c)
    return merged


# ---------------------------------------------------------------------------
# Python-side post-filter (faster equivalent of Chroma's `where` clause)
# ---------------------------------------------------------------------------
# Chroma 1.0's HNSW + metadata filter implementation does a metadata fetch +
# bitmap intersect per visited candidate during the graph traversal. For our
# real filter shape ($and of memory_type$in[5], customer$eq, user$in[2]) the
# per-call cost is ~840ms on a 3k-vec collection. With 2-4 parallel filtered
# batch_search calls per fetch_context this serializes at the server and
# dominates fetch_context latency (~7-8s, capped by the 8s timeout).
#
# Measured alternative: query Chroma WITHOUT a where-clause with n_results =
# top_k × overfetch_mult, then apply the same predicate in Python on the
# returned metadata. Empirically ~30ms per call, 22-31× speedup per single
# call, 30-33× per parallel cohort. ID-level overlap with the Chroma-filtered
# result is 100% across three test collections (synap_81a44e8b… 768d,
# synap_0770e50f… 768d, synap_e37b38da…_v2 1024d).
#
# Why post-filter is correct here: filter pass-rate ~1.0 in our actual usage
# (context_types is the full memory_type alphabet; collection name pins
# customer; in B2C user_id == customer_id so the user filter is a no-op).
# At F ≈ 1.0 the in-search filter does no semantic work but pays the full
# per-candidate cost. Post-filter is provably equivalent in result and
# cheaper. Verified by id-overlap probe. Full evidence in PR description.
#
# The env vars below are escape hatches. If a future deployment has highly
# selective filters (F → 0) post-filter will need a larger overfetch_mult,
# or fall back to Chroma's filter by setting SYNAP_VECTOR_POST_FILTER=false.

_USE_PYTHON_POST_FILTER: bool = (
    # Stays ON. Store-side filtering costs ~840ms per call against ~30ms here
    # (measured above), and a fetch times out at 8s. The one case where a
    # bounded unfiltered pool loses results, a caller whose readable records
    # are sparse, is handled by `_refetch_starved` below, which pays the
    # store-side cost only for the query that needs it.
    os.environ.get("SYNAP_VECTOR_POST_FILTER", "true").lower()
    in ("1", "true", "yes", "on")
)

# Lever for the starvation repair below. On by default; off restores the plain
# post-filter result (fewer rows for a caller with a small readable share, no
# extra store-side call).
_STARVED_REFETCH: bool = (
    os.environ.get("SYNAP_VECTOR_STARVED_REFETCH", "true").lower()
    in ("1", "true", "yes", "on")
)

_POST_FILTER_OVERFETCH_MULT: int = max(
    1, int(os.environ.get("SYNAP_VECTOR_OVERFETCH_MULT", "3"))
)

# Canonical memory_type alphabet. Used to detect whether the caller's
# context_types covers all possible memory_type values. When it does, the
# memory_type filter has pass-rate 1.0 (every doc has one of these values),
# so we can skip it entirely at Chroma and let the cheaper Python post-filter
# run instead. When the caller asks for a subset (e.g. ["facts"]), the
# memory_type filter is genuinely selective (F < 1.0): post-filter would
# need a huge overfetch and could miss high-similarity items past the
# raw HNSW result window. In that case we keep memory_type in Chroma's
# `where` clause for exactness, and only post-filter the scope predicate
# (which is always F ≈ 1.0 by construction — collection-name pins
# customer, in B2C user_id == customer_id).
_ALL_MEMORY_TYPES: frozenset = frozenset(
    ("facts", "preferences", "episodes", "emotions", "temporal_events")
)


def _is_full_memory_type_alphabet(context_types: Optional[List[str]]) -> bool:
    """True if context_types covers every memory_type value defined in the
    `MemoryCategoryName` enum. When True, the memory_type filter is
    effectively a no-op (filter pass-rate = 1.0) and can be moved Python-side
    without quality risk."""
    if not context_types:
        return False
    return _ALL_MEMORY_TYPES.issubset(frozenset(context_types))


# ---------------------------------------------------------------------------
# Nested scoping read path (S-07, R-11, R-12)
# ---------------------------------------------------------------------------
# A request that has resolved a scope chain compiles to `node_id IN chain` at
# the store, and its scope predicate is NOT post-filtered in Python.
#
# R-11 measured why. Post-filtering scope fails by pool starvation rather than
# by ranking: at overfetch 3 the realised pool fill equals recall exactly, 0.621
# at 20% readable falling to 0.021 at 1%, and the multiplier is not the lever
# (at 1% readable even 100x reaches only 0.277). Today that is survivable
# because the collection name already pins the customer and in B2C the user
# filter is a no-op, so the pass rate is about 1.0. A nested ladder is the
# opposite: the deeper a caller sits, the smaller their readable share, which is
# precisely where post-filtering collapses.
#
# The nested where-clause is built whole rather than merged into the flat dicts
# the call sites assemble, because it can contain `$or` (the transitional arm
# for records written before `node_id` existed) and a `$or` does not survive
# `{**a, **b}`.

# Returned instead of a where-clause when the requester can read nothing at all.
# The caller returns an empty result without touching the store, which is both
# correct and the cheapest thing it can do. A sentinel rather than an empty dict
# because an empty where-clause means "match everything", and confusing the two
# would be a cross-tenant leak rather than a slow query.
_NOTHING_READABLE = object()


def _roles_by_node(rungs) -> Dict[str, str]:
    """role -> node id, for every role each rung holds.

    A rung can hold two. In B2C the tenant node IS the identity node, and
    reading only `role` there yields no identity at all, which turns the legacy
    arm into "customer-shared only" and makes every unstamped B2C user record
    unreadable. `roles` carries the complete answer; `role` is the fallback for
    a chain built before this existed.
    """
    out: Dict[str, str] = {}
    for r in rungs or ():
        node = str(r["node_id"])
        names = r.get("roles") or ([r.get("role")] if r.get("role") else [])
        for name in names:
            if name:
                out.setdefault(name, node)
    return out


def _legacy_arm_ids(scope, rungs) -> tuple:
    """B-15. The ids the LEGACY read arm must filter on: the data-plane uuids.

    The legacy arm exists to match records written before nested scoping, and
    those records carry `customer_id` and `user_id` straight from the data
    plane. This used to take those VALUES off the resolved chain instead, and
    `scope_filter`'s own docstring states the assumption it rested on: "a legacy
    record's `user_id` *is* its node id".

    That is true only because `resolve_legacy_chain` passes the data-plane uuid
    as a node id hint (N-14). It is an accident of one code path, not a
    guarantee, and it is false for any node that a `scope` path created first,
    because those were minted with `gen_random_uuid()`.

    The consequence is B-15, and it is severe out of all proportion to its
    cause. One correct path-carrying request creates a person's registry node.
    From then on their chain resolves, the nested arm looks for a `node_id`
    their older records do not carry, and the legacy arm looks for a surrogate
    that matches nothing either. Both arms miss, the store is queried, and the
    person's entire history stops being readable. HTTP 200, empty bundle.

    For example, for one user on such a ladder:

    chain identity node   <uuid A>   (surrogate, made by a path)
    stored user_id        <uuid B>   (data-plane uuid, on every record)

    A switched-off instance reads those records without difficulty, which
    is the control proving the data is intact and indexed.

    So the arm is fed `scope.customer_id` and `scope.user_id`, which are the
    uuids the records actually carry, resolved for this very request. Where a
    node id already equals the data-plane uuid, which is every node the flat-id
    path created, this changes nothing at all.

    It does not widen. The data-plane customer uuid IS the requester's own
    tenant, and the user clause still pins their own identity or the customer
    sentinel, so the arm returns exactly what the same request would read with
    the feature switched off. That is the legacy arm's whole purpose.

    Falls back to the chain node when a data-plane id is absent, and that
    direction matters: passing None would DROP the clause and widen the read,
    while the chain node at worst matches nothing. Never widen on a fallback.
    """
    by_role = _roles_by_node(rungs)
    tenant = getattr(scope, "customer_id", None) or by_role.get("tenant_of_record")
    identity = getattr(scope, "user_id", None) or by_role.get("identity")
    # A chain that reached no identity rung is an ordinary customer-scope read,
    # not a truncated one, and it must stay that way: handing it the request's
    # user would narrow a customer read to one person.
    if by_role.get("identity") is None:
        identity = None
    return (str(tenant) if tenant else None, str(identity) if identity else None)


def _nested_scope_predicate(
    scope,
    maca: "MACARetrievalSettings",
    visibility_allowed=VISIBILITY_ALL,
    context_types: Optional[List[str]] = None,
):
    """The read predicate for a request with a resolved chain, as a tree.

    Split out of `_nested_scope_where` so the predicate can be compiled to
    something other than Chroma, which is the whole point of the tree in
    `scope_filter`. The immediate caller that needs it is the adversarial
    isolation test: it has to ask "would this request read that record", and
    the only honest way to answer is to compile THIS predicate with
    `to_python_predicate`. Rebuilding the same arms inside the test would be a
    second implementation that agrees by convention, which is the failure mode
    this repo keeps hitting.

    Returns None when there is no chain to compile.
    """
    rungs = getattr(scope, "chain_rungs", None)
    if not rungs:
        return None
    from synap.cloud.shared import scope_filter as _sf

    # One source of truth for "what is this request's chain", so a rung
    # carrying a null node id is refused in exactly one place. This used to
    # build its own list, and `str(None)` is the truthy string "None", so
    # hardening `_nested_chain_ids` alone left the literal "None" being sent
    # to the store from here.
    chain = _nested_chain_ids(scope)
    if not chain:
        return None
    tenant_id, identity_id = _legacy_arm_ids(scope, rungs)
    return _sf.build_scope_filter(
        chain,
        tenant_node_id=tenant_id,
        identity_node_id=identity_id,
        roll_up_to_customer=maca.roll_up_to_customer,
        cross_user=maca.allow_cross_user_retrieval,
        cross_customer=maca.allow_cross_customer_retrieval,
        visibility_allowed=(
            None if visibility_allowed is VISIBILITY_ALL else list(visibility_allowed)
        ),
        memory_types=context_types,
        memory_type_universe=_ALL_MEMORY_TYPES,
    )


def _nested_scope_where(
    scope,
    maca: "MACARetrievalSettings",
    visibility_allowed=VISIBILITY_ALL,
    context_types: Optional[List[str]] = None,
):
    """The complete store-side where-clause for a request with a resolved chain.

    Returns None when there is no chain, which is every request until
    SYNAP_NESTED_SCOPING is switched on. Callers keep their existing path in
    that case, byte for byte.
    """
    rungs = getattr(scope, "chain_rungs", None)
    if not rungs:
        return None
    try:
        from synap.cloud.shared import scope_filter as _sf

        pred = _nested_scope_predicate(scope, maca, visibility_allowed, context_types)
        if pred is None:
            return None
        return _sf.to_chroma_where(pred)
    except Exception as exc:  # noqa: BLE001
        from synap.cloud.shared.scope_filter import NothingReadable

        if isinstance(exc, NothingReadable):
            return _NOTHING_READABLE
        # Anything else means we could not compile the nested predicate. Falling
        # back to the legacy path is correct rather than merely convenient: it
        # is narrower than the nested one for every record that exists today,
        # so the failure mode is a missing new-style record, never a leak.
        logger.warning("nested scope filter not compiled, using legacy path: %s", exc)
        return None


def _use_python_post_filter_for(nested_where) -> bool:
    """R-11. A compiled nested clause means the scope predicate goes to the store.

    Keyed on the compiled clause, not on whether the scope has a chain, and the
    difference is not cosmetic. `_nested_scope_where` returns None when it could
    not compile, which is a full fall back to the legacy path. Deciding this
    from the presence of a chain instead would leave post-filtering switched off
    while the legacy where-clause went to the store: not a leak, since that
    clause is exact, but a silent third path with neither shape's latency
    profile, reached only when something has already gone wrong.
    """
    if nested_where is not None:
        return False
    return _USE_PYTHON_POST_FILTER


# ---------------------------------------------------------------------------
# Nested scoping ranking (R-02, R-09, R-10, R-14, R-15, R-17)
# ---------------------------------------------------------------------------
#
# The filter above decides WHAT a request may read. These decide what ORDER it
# comes back in, and whether the requester's own material was fetched at all.
#
# Both halves are needed and only one of them is visible when it is missing. A
# request with the filter and without these returns exactly the right records
# ranked by raw similarity, which is the dilution R-02 exists to prevent: a
# requester deep in a ladder can read everything above them, and everything
# above them is a far larger pool than their own node holds, so a generic fact
# true of the whole client outranks the specific fact about this person. The
# answer is well ordered and wrong, and nothing about it looks broken.


def _nested_chain_ids(scope) -> Optional[List[str]]:
    """This request's chain as plain node ids, or None when there is no chain.

    None is the ordinary case and stays the ordinary case: it is every request
    until SYNAP_NESTED_SCOPING is switched on for the instance, and every
    caller below treats it as "do nothing at all".
    """
    rungs = getattr(scope, "chain_rungs", None)
    if not rungs:
        return None
    try:
        # ⚠ `str(r["node_id"])` on a None yields the literal "None", which is
        # truthy, so the `all(ids)` guard below passed it through and the string
        # "None" was sent to the store as a node id. Exactly the defect D-05
        # records finding in the SDK cache key, in a function written to be
        # defensive about malformed chains. Found by audit 2026-08-23.
        raw = [r["node_id"] for r in rungs]
        if any(n is None for n in raw):
            logger.warning(
                "scope chain has a rung with no node id, ranking falls back"
            )
            return None
        ids = [str(n) for n in raw]
    except (KeyError, TypeError, AttributeError) as exc:
        # A rung missing its node id is a chain that never resolved properly.
        # Raising here would take down a request the legacy path can serve
        # perfectly well, and `_nested_scope_where` already treats an
        # uncompilable chain as "use the legacy path". This is the same
        # decision, made in the same direction, for the same reason.
        logger.warning("scope chain is malformed, ranking falls back: %s", exc)
        return None
    return ids if all(ids) else None


def _nested_leaf_reserve(
    scope,
    maca: "MACARetrievalSettings",
    visibility_allowed=VISIBILITY_ALL,
    context_types: Optional[List[str]] = None,
    top_k: int = 25,
):
    """R-10. The reserve fetch: (where_clause, top_k), or (None, 0) to skip it.

    Ranking decides the order of what was fetched and can do nothing about what
    was never fetched. A requester several rungs down competes for top_k against
    everything above them, and one similarity-ordered fetch can come back with
    nothing of their own in it, not because their material is irrelevant but
    because there is a thousand times more of it one rung up. So the store is
    asked a second, narrow question, and both answers are ranked together once.

    Skipped for a chain of one, where the requester IS the root and the main
    fetch already reads only their node.

    Failing here skips the reserve rather than failing the request. The reserve
    only ever ADDS candidates to a pool that is ranked once afterwards, so its
    absence is a worse answer and never a wrong one.
    """
    chain = _nested_chain_ids(scope)
    if not chain:
        return None, 0
    try:
        from synap.cloud.shared import scope_filter as _sf
        from synap.cloud.shared import scope_pool as _sp

        rungs = scope.chain_rungs
        # Same ids as the main predicate, from the same helper. Two callers
        # deriving the legacy arm separately is how they come to disagree, and
        # a reserve that scopes differently from the fetch it supplements would
        # add candidates the main filter would have excluded.
        tenant_id, identity_id = _legacy_arm_ids(scope, rungs)
        # The CHAIN's tenant node, which is a different question from the ids
        # above and must not be confused with them. Those are data-plane uuids
        # for filtering stored metadata; this one is compared against `chain[-1]`
        # to ask "is the leaf the tenant", and both sides of that comparison
        # have to be node ids. They agree only for a node the flat-id path
        # created, so using the data-plane uuid here would silently stop the
        # skip below from ever firing.
        by_role = _roles_by_node(rungs)
        plan = _sp.plan_pool(
            chain,
            top_k,
            tenant_node_id=tenant_id,
            identity_node_id=identity_id,
            roll_up_to_customer=maca.roll_up_to_customer,
            cross_user=maca.allow_cross_user_retrieval,
            cross_customer=maca.allow_cross_customer_retrieval,
            visibility_allowed=(
                None if visibility_allowed is VISIBILITY_ALL else list(visibility_allowed)
            ),
            memory_types=context_types,
            memory_type_universe=_ALL_MEMORY_TYPES,
        )
        if plan.leaf is None or plan.leaf_top_k <= 0:
            return None, 0

        # ⚠ The reserve only helps when the main fetch can return records from
        # a rung the reserve excludes AND those records live in the SAME
        # collection. Rungs at or above the tenant of record do not: the root's
        # material is in the client-level collection, read by its own task.
        #
        # So when the leaf IS the tenant, which is every B2C request and every
        # customer-scope caller, the reserve is a third store call that narrows
        # nothing reachable. Measured at up to +19.9% p95 at 16 concurrent
        # (SHIP-15) for no candidates at all. `plan_pool` cannot know this,
        # because it is store-agnostic and this is a fact about how the read
        # path splits collections. Found by audit 2026-08-23.
        tenant = by_role.get("tenant_of_record")
        if tenant is not None and chain[-1] == tenant:
            return None, 0

        return _sf.to_chroma_where(plan.leaf), plan.leaf_top_k
    except Exception as exc:  # noqa: BLE001
        logger.warning("leaf reserve not compiled, skipping the reserve fetch: %s", exc)
        return None, 0


def _closeness_scored(
    raw_score: float,
    meta: Optional[Dict[str, Any]],
    chain_ids: Optional[List[str]],
    per_level_cost: float,
    *,
    from_client_scope: bool = False,
    widened: bool = False,
) -> Optional[float]:
    """One record's relevance after distance. None means drop it entirely.

    Optional rather than a sentinel because the caller's next statement is
    `continue`, and every case below is enumerated so there is no fallthrough:

      no chain                -> unchanged. The feature is off (SHIP-04).
      from the client task    -> the root rung's distance (R-15).
      no `node_id` on the row -> distance 0, the requester's own node (R-14).
      `node_id` on the chain  -> its distance from the leaf.
      `node_id` off the chain -> None, drop it (R-07, R-08).

    R-14 and R-15 are the two that would be quietly destructive if guessed.

    A record with no `node_id` is one written before the metadata backfill, and
    the only reason it is in this result set at all is the transitional `$or`'s
    legacy arm, which is exactly the requester's own customer and user scope. So
    own-node is what it actually is. Reading the absence as "not on the chain"
    would score every pre-backfill record 0 and drop it, which presents as data
    loss rather than as a reordering.

    A record from the client-level collection carries no customer id, no user id
    and no node id, because the collection name is its entire scope. The
    metadata cannot answer the distance question so the caller does, by saying
    which fetch it came from. Reading the absence would delete all shared
    knowledge; calling it distance 0 would let generic client-wide material
    compete level with the requester's own.

    Nothing here takes a permission, a grant or a predicate, and that is R-07:
    once "far away" and "forbidden" differ only by degree, a high enough
    similarity score defeats isolation. What may be read was settled by the
    filter, before any of this ran.
    """
    if not chain_ids:
        return raw_score

    from synap.cloud.shared import scope_closeness as _sc

    if from_client_scope:
        distance: Optional[int] = len(chain_ids) - 1
    else:
        node_id = (meta or {}).get("node_id")
        if not node_id:
            # R-14, corrected. An unstamped record used to be scored at distance
            # 0, the requester's own node, on the reasoning that the legacy
            # filter arm "is exactly the requester's own customer and user
            # scope". That holds for the `user_id = identity` half and fails for
            # the other half: a record carrying the CUSTOMER_SCOPE sentinel is
            # tenant-shared material one rung up, not the requester's own.
            #
            # The consequence was that an identical shared record ranked ~30%
            # differently depending only on whether the backfill had reached it,
            # in the same pool, on the same query:
            #
            #   customer-shared, stamped    -> 0.56
            #   customer-shared, UNSTAMPED  -> 0.80
            #
            # Found by an audit 2026-08-24. The metadata already carries the
            # sentinel that says which case this is, so no guess is needed.
            #
            # Direction matters: the old behaviour ranked shared material too
            # HIGH, which is what happened before this feature existed, so this
            # is a precision fix rather than a leak. It also means the 1,440
            # bundle sweep that moved the per-level cost to 0.30 ran on a mix
            # where many shared records were silently exempt from the term.
            from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL

            if (meta or {}).get("user_id") == CUSTOMER_SCOPE_SENTINEL:
                # Tenant-shared: one rung above the leaf on a chain whose leaf
                # is the identity. `max(0, ...)` keeps a B2C collapsed chain,
                # where the tenant IS the leaf, at distance 0 rather than -1.
                distance = max(0, len(chain_ids) - 2) if len(chain_ids) >= 2 else 0
            else:
                distance = 0
        else:
            distance = _sc.distance_from_leaf(chain_ids, str(node_id))
            if distance is None:
                if widened:
                    # ⚠ The caller asked for this. With
                    # `allow_cross_user_retrieval` or
                    # `allow_cross_customer_retrieval` on, the legacy arm drops
                    # its user clause and the store deliberately returns other
                    # people's records, which are off the chain by definition.
                    #
                    # Dropping them here made the result set "whichever of the
                    # other users' records happen not to be stamped yet", which
                    # changed as the backfill progressed: a stamped record was
                    # dropped and an unstamped one kept, for the same user, on
                    # the same query. Found by audit 2026-08-23.
                    #
                    # Scored at the furthest rung rather than not demoted at
                    # all: the caller widened on purpose, so the record is
                    # legitimate, and it is still further away than anything on
                    # the requester's own chain.
                    distance = max(0, len(chain_ids) - 1)
                else:
                    return None
    return raw_score * _sc.closeness_multiplier(distance, per_level_cost)


def _per_level_cost(maca: "MACARetrievalSettings") -> float:
    """R-03 and R-07. A plain mapping reaches the scorer, never the MACA object.

    One key, one number. The scorer must depend on nothing that could carry a
    permission into it, and passing the settings object would hand it
    `allow_cross_customer_retrieval` and everything else beside it.
    """
    from synap.cloud.shared.scope_closeness import per_level_cost_for

    return per_level_cost_for(
        {"scope_closeness_per_level_cost": getattr(
            maca, "scope_closeness_per_level_cost", None
        )}
    )


def _scope_filter_predicate(scope, maca: "MACARetrievalSettings", visibility_allowed=VISIBILITY_ALL):
    """The Python form of `_build_customer_scope_filter`, for the post-filter.

    Returns ``predicate(metadata) -> bool``; True means the record would have
    been kept by the Chroma where-clause. It is compiled from the SAME
    `_legacy_read_predicate`, so the two cannot disagree about semantics.
    """
    from synap.cloud.shared import scope_filter as _sf

    try:
        pred = _legacy_read_predicate(scope, maca, visibility_allowed)
    except _sf.NothingReadable:
        return lambda metadata: False
    if pred is None:
        return lambda metadata: True
    return _sf.to_python_predicate(pred)


def _is_true(value) -> bool:
    """A flag that survived a round trip through a property store.

    ⚠ Not `bool(value)`. Neo4j stores the memory list as a JSON string and the
    booleans in it come back as the STRINGS "True" and "False" in practice,
    so a truthiness test reads "False" as true and would
    drop every memory rather than the deleted ones. The same shape has caused
    defects in this repo before: `str(None)` is the truthy string "None".
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _graph_scope_gate(
    scope,
    maca: "MACARetrievalSettings",
    namespace_customer_id: Optional[str] = None,
    visibility=VISIBILITY_ALL,
):
    """The read filter's own answer, asked of a graph record.

    The graph arm has always scoped by ids: a namespace pinned to the customer,
    then `customer_id` and `user_id` compared by hand. The read filter stopped
    being that in S-07. It is `node_id IN chain` first, with the id pair only as
    a transitional second arm, so a request whose chain runs
    client -> customer -> practice -> care_team -> user cannot be answered by
    the pair at all: the ladder has rungs the pair has no field for.

    Measured with a positive control in the same run
    because "nothing leaked" and "nothing came back" look identical. Given
    anchors belonging to one user, the traversal returned entity
    nodes belonging to OTHER users of the same customer in every live namespace
    probed: 2 refused of 26 returned, 5 of 15, 2 of 31, against 24, 10 and 29
    that the read filter admits. Compiling this predicate over those same
    records is what refuses them. The Cypher filters relationships by the id
    pair and never filters nodes at all, which is why the nodes came back.

    The answer is not rebuilt here. `_nested_scope_predicate` IS the read
    filter; this compiles it to Python and hands it the graph's spelling. A
    second implementation that agrees by convention is how the two halves
    drifted apart in the first place.

    Two clauses are deliberately left out, and neither of them is scope:

    - `memory_type`, because the graph arm has never filtered on it. Adding it
      here would be a recall change wearing a boundary fix's clothes.

    Cross-instance visibility is NOT part of that scope predicate, because the
    predicate's instance clause is a scalar `In` and a graph node is shared by
    every instance of a customer, so it carries a LIST (`instance_ids`, see
    shared/graph_boundary). It is applied here, after the scope rule, with the
    graph's own spelling: a record is admitted when any of its instances is
    visible, and a record that names none (written before the property
    existed) is refused. `visibility` is VISIBILITY_ALL for default-open, which
    adds nothing, so the default gate is what it was.
    """
    from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL as _SENT

    inner = None
    try:
        pred = _nested_scope_predicate(scope, maca, VISIBILITY_ALL, None)
        if pred is not None:
            from synap.cloud.shared import scope_filter as _sf

            inner = _sf.to_python_predicate(pred)
    except Exception as exc:  # noqa: BLE001
        # Same decision `_nested_scope_where` makes for the same reason: the
        # legacy predicate is the nested one's own second arm, so falling back
        # to it is strictly narrower. The failure mode is a missing new-style
        # record, never a leak.
        logger.warning("graph scope gate could not compile the chain: %s", exc)
        inner = None
    if inner is None:
        inner = _scope_filter_predicate(scope, maca, VISIBILITY_ALL)

    ns_customer = str(namespace_customer_id) if namespace_customer_id else None

    # None is default-open. Anything else restricts: a resolved set of instance
    # ids, or (a policy that could not be loaded) nothing at all.
    if visibility is VISIBILITY_ALL:
        allowed_instances = None
    elif isinstance(visibility, (set, frozenset)):
        allowed_instances = frozenset(str(v) for v in visibility)
    else:
        allowed_instances = frozenset()

    def gate(properties: Optional[Mapping[str, Any]]) -> bool:
        meta = dict(properties or {})
        # The graph OMITS a scope key rather than writing a value for it:
        # `upsert_entity_node` sets `user_id` only when the write had one, and
        # the traversal Cypher reads that back as `rel.user_id IS NULL`. The
        # vector store spells the same state as the sentinel, and this
        # predicate is written in the vector store's spelling, so translate
        # rather than widen. An absent user means "not filed under a person",
        # which is exactly what the sentinel means, and `roll_up_to_customer`
        # then decides whether it is readable, in one place, for both stores.
        if meta.get("user_id") is None:
            meta["user_id"] = _SENT
        # Same for the customer. The namespace already pins it, so a legacy row
        # inside it belongs to that customer whether or not it says so, and the
        # Cypher already lets it through on `rel.customer_id IS NULL`.
        if meta.get("customer_id") is None and ns_customer:
            meta["customer_id"] = ns_customer
        if not inner(meta):
            return False
        if allowed_instances is None:
            return True
        return _graph_boundary.record_visible(properties or {}, allowed_instances)

    return gate


def _apply_post_filter(
    batch_results: List[List[Dict[str, Any]]],
    scope_predicate=None,
    context_types: Optional[List[str]] = None,
) -> List[List[Dict[str, Any]]]:
    """Apply scope + memory_type filter to a batch_search response in Python.

    Mirrors the semantics of Chroma's ``where`` clause that `vector_search`,
    `_search_long_term_questions` and `_search_long_term_sentences` would
    have sent. Same input/output shape as `batch_search`: a list-of-lists
    where the outer list is per-query and the inner list is the per-query
    top-K results.

    Args:
        batch_results: List-of-lists from `self.vector.batch_search(...)`.
        scope_predicate: Optional callable from `_scope_filter_predicate()`.
            Pass `None` to skip scope filtering (e.g. for client-level
            collections where customer/user scope does not apply).
        context_types: Optional list of allowed `memory_type` values. Pass
            `None` to skip memory_type filtering (e.g. for questions or
            sentence collections where this field is absent from metadata).

    Returns:
        New list-of-lists with the same outer length, inner lists filtered.
    """
    from synap.cloud.shared import scope_filter as _sf

    filtered: List[List[Dict[str, Any]]] = []
    # Compiled from the same predicate `_memory_type_where` sends to the store.
    type_ok = (
        _sf.to_python_predicate(_sf.memory_type_predicate(context_types))
        if context_types else None
    )
    for results_for_query in batch_results:
        kept: List[Dict[str, Any]] = []
        for r in results_for_query:
            meta = r.get("metadata") or {}
            if type_ok is not None:
                if not type_ok(meta):
                    continue
            if scope_predicate is not None:
                if not scope_predicate(meta):
                    continue
            kept.append(r)
        filtered.append(kept)
    return filtered


async def _refetch_starved(
    raw: List[List[Dict[str, Any]]],
    kept: List[List[Dict[str, Any]]],
    *,
    pool_k: int,
    want: int,
    refetch,
    label: str,
) -> List[List[Dict[str, Any]]]:
    """Recover results a post-filter lost because the readable share was small.

    The post-filter asks the store for ``pool_k`` unfiltered candidates and
    keeps the ones the caller may read. That is exact while most candidates are
    readable, which is why it is the default. A caller who may read only a small
    share of a collection (a user inside a customer with many users) can have
    every readable record sit beyond the pool, and gets fewer results than exist
    without any error. R-11 measured the shape: the fill falls with the
    readable share and a larger multiplier barely moves it.

    A query is starved when the pool came back full (so more candidates may
    exist past it) and fewer than ``want`` survived. Only those queries are asked
    again, with the store-side filter, which is exact. Queries that are not
    starved cost nothing, so the common path keeps its latency.

    ``refetch(indices)`` returns one result list per index, in order. A failure
    there leaves the post-filtered result in place: this is a recall repair, and
    it must never turn a partial answer into an error.
    """
    if not _STARVED_REFETCH:
        return kept
    starved = [
        i for i, (r, k) in enumerate(zip(raw, kept))
        if isinstance(r, list) and isinstance(k, list)
        and len(r) >= pool_k and len(k) < want
    ]
    if not starved:
        return kept
    try:
        exact = await refetch(starved)
    except Exception as e:  # noqa: BLE001 — recall repair only, never a new failure
        logger.warning(
            "post_filter_starved_refetch_failed: label=%s queries=%d error=%s",
            label, len(starved), e,
        )
        return kept
    if not isinstance(exact, list) or len(exact) != len(starved):
        return kept
    repaired = list(kept)
    for i, res in zip(starved, exact):
        repaired[i] = res
    logger.info(
        "post_filter_starved_refetch: label=%s queries=%d of %d pool_k=%d want=%d",
        label, len(starved), len(kept), pool_k, want,
    )
    return repaired


# ---------------------------------------------------------------------------
# L2 in-memory adjacency cache
# ---------------------------------------------------------------------------
# Module-level dirty-key set: stage4 (throwaway RM) adds keys here so the
# singleton RM knows to skip its L2 entry on the next read.
_l2_dirty_keys: set = set()
_l2_flush_all: bool = False  # D-007 flush-on-resubscribe: set by InvalidationSubscriber; consumed once per RM read


@dataclass
class _L2CacheEntry:
    adjacency: Dict
    created_at: float          # time.monotonic()
    approx_bytes: int
    access_time: float         # time.monotonic(), updated on get()


class _AdjacencyL2Cache:
    """Process-local LRU cache for adjacency matrices.

    Safe without locks — single async event loop, no threading.
    """

    def __init__(self, max_entries: int = 50, max_bytes: int = 200_000_000, ttl_seconds: int = 600):
        self._store: Dict[str, _L2CacheEntry] = {}
        self._total_bytes: int = 0
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _estimate_bytes(adjacency: Dict) -> int:
        """Estimate memory footprint without serializing large embedding lists."""
        def _size(obj) -> int:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            if isinstance(obj, dict):
                return sum(_size(val) for val in obj.values())
            if isinstance(obj, list):
                if obj and isinstance(obj[0], float):
                    # Embedding vector: each float64 is 8 bytes
                    return len(obj) * 8
                return sum(_size(item) for item in obj)
            if isinstance(obj, str):
                return len(obj)
            return 8  # int, float, bool, None
        total = sum(_size(v) for v in adjacency.values())
        # 20% overhead for Python dicts / strings / object headers
        return int(total * 1.2)

    def _is_expired(self, entry: _L2CacheEntry) -> bool:
        return (time.monotonic() - entry.created_at) > self._ttl_seconds

    def _sweep_expired(self) -> None:
        """Remove all TTL-expired entries."""
        expired = [k for k, e in self._store.items() if self._is_expired(e)]
        for k in expired:
            self._evict(k, reason="ttl")

    def _evict(self, key: str, reason: str = "lru") -> None:
        entry = self._store.pop(key, None)
        if entry is not None:
            self._total_bytes -= entry.approx_bytes
            self._evictions += 1
            logger.info(
                "adjacency_l2_cache_eviction: key=%s, reason=%s, approx_mb=%.1f",
                key, reason, entry.approx_bytes / 1_000_000,
            )

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry."""
        if not self._store:
            return
        lru_key = min(self._store, key=lambda k: self._store[k].access_time)
        self._evict(lru_key, reason="lru")

    # -- public API -------------------------------------------------------

    def peek(self, key: str) -> Optional[Dict]:
        """Read-only cache peek: no logging, no counter mutation, no LRU touch."""
        entry = self._store.get(key)
        if entry is None or self._is_expired(entry):
            return None
        return entry.adjacency

    def get(self, key: str) -> Optional[Dict]:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            logger.info("adjacency_l2_cache_miss: key=%s", key)
            return None
        if self._is_expired(entry):
            self._evict(key, reason="ttl")
            self._misses += 1
            logger.info("adjacency_l2_cache_miss: key=%s (expired)", key)
            return None
        entry.access_time = time.monotonic()
        self._hits += 1
        age_s = time.monotonic() - entry.created_at
        logger.info(
            "adjacency_l2_cache_hit: key=%s, age_s=%.1f, approx_mb=%.1f",
            key, age_s, entry.approx_bytes / 1_000_000,
        )
        return entry.adjacency

    def put(self, key: str, adjacency: Dict) -> None:
        # If key already present, remove old entry first
        if key in self._store:
            self._total_bytes -= self._store[key].approx_bytes
            del self._store[key]

        approx = self._estimate_bytes(adjacency)

        # Evict expired entries first
        self._sweep_expired()

        # Evict LRU until within bounds
        while (len(self._store) >= self._max_entries
               or self._total_bytes + approx > self._max_bytes) and self._store:
            self._evict_lru()

        now = time.monotonic()
        self._store[key] = _L2CacheEntry(
            adjacency=adjacency,
            created_at=now,
            approx_bytes=approx,
            access_time=now,
        )
        self._total_bytes += approx
        logger.info(
            "adjacency_l2_cache_put: key=%s, approx_mb=%.1f, total_entries=%d, total_mb=%.1f",
            key, approx / 1_000_000, len(self._store), self._total_bytes / 1_000_000,
        )

    def delete(self, key: str) -> bool:
        entry = self._store.pop(key, None)
        if entry is not None:
            self._total_bytes -= entry.approx_bytes
            return True
        return False

    def delete_scope(self, scope_key: str) -> int:
        """Drop the entry for a scope and every boundary variant of it.

        An entry is keyed by its scope (the shared, standard-boundary graph) or
        by its scope plus the viewer's read boundary. An invalidation event
        names only the scope, so it must reach all of them and none of another
        scope's. Ids are UUIDs, so ``scope_key + ":"`` cannot match a different
        scope.
        """
        prefix = scope_key + ":"
        doomed = [k for k in self._store if k == scope_key or k.startswith(prefix)]
        for k in doomed:
            self.delete(k)
        return len(doomed)

    def clear(self) -> None:
        self._store.clear()
        self._total_bytes = 0

    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._store),
            "total_mb": round(self._total_bytes / 1_000_000, 1),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }


def _revealable_items(bundle) -> list:
    """Every object in a bundle that can hold a label.

    Reads the shape the bundle actually has rather than a field name. A
    ``ContextBundle`` groups by type and offers ``all_items``; a cached or
    reconstructed bundle may be a plain object with a flat list. Both are
    handled, and neither is guessed at from an attribute that might not exist.

    The conversation context is included because it is carried separately
    from the items and it is prose from the same conversation, so it holds
    exactly the same labels.
    """
    found = []

    flat = getattr(bundle, "all_items", None)
    if callable(flat):
        flat = flat()
    if flat:
        found.extend(flat)
    elif getattr(bundle, "items", None):
        found.extend(bundle.items)
    else:
        by_type = getattr(bundle, "items_by_type", None) or {}
        for group in by_type.values():
            found.extend(group or [])

    context = getattr(bundle, "conversation_context", None)
    if context is not None:
        found.append(context)
        for attr in ("recent_turns", "items", "summaries"):
            extra = getattr(context, attr, None)
            if isinstance(extra, (list, tuple)):
                found.extend(e for e in extra if e is not None)

    return found


# ── Search breadth, derived from the caller's top-K ──────────────────────────
#
# `max_vector_results` was a fixed 25 and nothing scaled with `max_results`, so
# a caller asking for top-100 got the same candidate pool as one asking for
# top-10 — the search never looked for more. Measured by the benchmark team on
# 2026-09-12: k=10, 25, 50 and 100 returned BYTE-IDENTICAL memory id lists on
# all 60 questions, mean 7.6 items every time. Raising k did nothing because
# there was nothing more to return.
#
# OVERFETCH: the search result is not the answer. Scope filtering, dedup and the
# relevance filter all cut the pool AFTER the store returns it, so fetching
# exactly K leaves fewer than K. 4x is the agreed multiplier.
#
# FLOOR of 25 keeps today's behaviour for every request at or under the default,
# so nothing changes for existing callers. CEILING because a derivation with no
# cap lets max_results=1000 ask the store for 4000 vectors per query, on a two
# vCPU box that already degrades past three concurrent requests.
# Minimum raw store similarity for a vector hit to be considered at all.
#
# Added 2026-02-17 (e60366ee0) as a bare `if score < 0.30: continue  # Skip
# low-similarity noise`, when embeddings were all-MiniLM. The model changed 11
# days later to Arctic Embed M v2.0, again in March to v1.5, again on 2026-05-25
# to arctic-l-v2.0 (1024-dim), and again on 2026-06-08 when queries started
# carrying the asymmetric `is_query` prefix — a scale on which query-to-document
# scores sit systematically below the document-to-document ones this number was
# first eyeballed against. The threshold never moved through any of it.
#
# Measured on the embedding model in use, 48 (hypothetical question -> the memory
# it was written for) pairs: the correct memory scores median 0.567 and ranks
# first in 31 of 48; this floor drops the correct memory in 1 of 48. So 0.30 is
# survivable today, by luck rather than derivation. Naming it, and letting an
# operator move it, makes the next model change a config change instead of a
# code change — and makes the number searchable when someone asks why recall
# looks thin.
#
# TODO(similarity-calibrator): one number for every client and every embedding
# model. To be supplied per client by SimilarityCalibrator
# (synap/cloud/application/consolidation/similarity_calibrator.py). Decisions and
# build list: synap/docs/architecture/similarity_threshold_calibrator.md
_DEFAULT_MIN_SIMILARITY = 0.30


def min_similarity() -> float:
    """The similarity floor, overridable with SYNAP_RETRIEVAL_MIN_SIMILARITY.

    ``0`` has always turned the gate off, because no cosine similarity is below
    zero; ``off`` is accepted for it too so the knob
    reads the same way as SYNAP_RETRIEVAL_DEDUP_JACCARD, and so that an operator
    reaching for the obvious word does not get the default back with a warning
    buried in the logs.

    Garbage or out-of-range values fall back to the default rather than
    silently disabling the gate or emptying every result (1.0).
    """
    raw = os.environ.get("SYNAP_RETRIEVAL_MIN_SIMILARITY", "").strip()
    if not raw:
        return _DEFAULT_MIN_SIMILARITY
    # Only the explicit word. "none" is deliberately NOT accepted: a
    # stringified Python None reaching an env var is a plausible typo, and it
    # must not silently open the gate fleet-wide (see the bad-value test).
    if raw.lower() == "off":
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "SYNAP_RETRIEVAL_MIN_SIMILARITY=%r is not a number; using %.2f",
            raw, _DEFAULT_MIN_SIMILARITY)
        return _DEFAULT_MIN_SIMILARITY
    if not 0.0 <= value <= 1.0:
        logger.warning(
            "SYNAP_RETRIEVAL_MIN_SIMILARITY=%s is outside 0..1; using %.2f",
            value, _DEFAULT_MIN_SIMILARITY)
        return _DEFAULT_MIN_SIMILARITY
    return value


VECTOR_OVERFETCH_MULT = 4
VECTOR_BREADTH_FLOOR = 25
VECTOR_BREADTH_CEILING = 400


def derive_search_breadth(max_results: int, floor: int = VECTOR_BREADTH_FLOOR) -> int:
    """How many candidates the vector search should ask for, given top-K."""
    try:
        k = int(max_results or 0)
    except (TypeError, ValueError):
        k = 0
    if k <= 0:
        return floor
    return max(floor, min(k * VECTOR_OVERFETCH_MULT, VECTOR_BREADTH_CEILING))


class RetrievalManager:
    """
    Handles context retrieval requests from SDK.

    Methods exposed as Tools for Listening & Retrieval Agent.

    Architecture:
    - Agent produces ContextDecision (WHAT to retrieve)
    - Manager executes retrieval (HOW to get it)
    - Manager returns ContextBundle (retrieved context)
    """

    def __init__(
        self,
        scope_resolver: ScopeResolver = None,
        cache_provider=None,       # Redis provider
        vector_provider=None,      # ChromaDB/Pinecone provider
        graph_provider=None,       # Neo4j provider
        postgres_provider=None,    # Postgres provider for short-term
        llm_client=None,           # Optional LLMClient for ACCURATE mode filtering
        config: RetrievalConfig = None,
        compaction_manager=None,
        compaction_storage=None,
        embedding_client=None,
    ):
        self.scope_resolver = scope_resolver or ScopeResolver()
        self.cache = cache_provider
        self.vector = vector_provider
        self.graph = graph_provider
        self.postgres = postgres_provider
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.config = config or RetrievalConfig()
        self.token_counter = TokenCounter()
        # None in Core (no usage tracker); the usage_tracking hook then does nothing.
        self._usage_tracker = edition.call("usage_tracking", "tracker")

        # Compaction integration (injectable, lazy-loaded if not provided)
        self.compaction_manager = compaction_manager
        self.compaction_storage = compaction_storage

        # L2 in-memory adjacency cache (process-local, single worker)
        self._adjacency_l2 = _AdjacencyL2Cache(
            max_entries=50,        # ~50 user scopes
            max_bytes=200_000_000, # 200 MB cap
            ttl_seconds=300,       # 5 min (matches L3 FRESH_WINDOW_SECONDS)
        )

        # Single-flight coordination for cold adjacency builds: scope key ->
        # in-flight future. Coalesces concurrent cold requests for the same scope
        # so the L3 round-trip + rebuild runs once and all callers share it.
        self._adjacency_inflight: Dict[str, "asyncio.Future"] = {}

    async def _track_vector_queries(
        self,
        scope: Optional[ScopeHierarchy],
        query_count: int,
        operation: str,
        latency_ms: Optional[float] = None,
        status: str = "success",
        correlation_id: Optional[str] = None,
    ) -> None:
        """Record vector query usage without impacting retrieval success path."""
        await edition.acall(
            "usage_tracking", "vector", self, scope, query_count, operation,
            latency_ms, status, correlation_id,
        )

    async def _track_graph_queries(
        self,
        scope: Optional[ScopeHierarchy],
        query_count: int,
        operation: str,
        latency_ms: Optional[float] = None,
        status: str = "success",
        correlation_id: Optional[str] = None,
    ) -> None:
        """Record graph query usage without impacting retrieval success path."""
        await edition.acall(
            "usage_tracking", "graph", self, scope, query_count, operation,
            latency_ms, status, correlation_id,
        )

    async def _get_maca_settings(self, instance_id: str) -> MACARetrievalSettings:
        """Get MACA settings for instance, falling back to defaults."""
        settings = await _load_maca_config(instance_id)
        return settings or MACARetrievalSettings()

    async def _get_visibility_allowed_set(
        self, client_id_raw: Optional[str], instance_id_raw: str
    ):
        """Resolve the set of source instance_ids this viewer instance may read.

        Returns the VISIBILITY_ALL sentinel (default-open — no restriction) when
        the client has no isolation policy, or a frozenset of UUID-form source
        instance_ids when isolation is active. Resolution runs in RAW id space
        (matching the policy tables), then converts to the deterministic-UUID
        form that ChromaDB metadata stores.

        Fails closed (an empty set: read nothing) when the policy cannot be
        determined at all, and says so in the response through
        ``record_retrieval_fault`` so the caller sees a partial answer, not a
        silently empty one. A client whose policy was loaded before keeps that
        policy through a store outage (see ``_load_visibility_bundle``).
        """
        try:
            client_id = client_id_raw
            if not client_id:
                client_id = await _resolve_client_id_for_instance(instance_id_raw)
            if not client_id:
                record_retrieval_fault(
                    "visibility policy", "client of this instance could not be resolved"
                )
                return frozenset()

            bundle = await _load_visibility_bundle(client_id)
            if bundle is VISIBILITY_ALL:
                return VISIBILITY_ALL
            if bundle is _VISIBILITY_UNAVAILABLE:
                record_retrieval_fault(
                    "visibility policy", "policy store unavailable, reading nothing"
                )
                return frozenset()

            return _resolve_allowed_from_bundle(bundle, instance_id_raw)
        except Exception as e:
            logger.warning(
                f"Visibility resolution failed (client={client_id_raw}, "
                f"instance={instance_id_raw}); failing closed: {e}"
            )
            record_retrieval_fault("visibility policy", e)
            return frozenset()

    async def _classify_queries(self, decision, scope, trace_id: str) -> None:
        """Rewrite the search queries so they match aliased content.

        The cost is one gate pass over a short string plus, only if it finds
        something the policy aliases, one indexed select. A client who has
        configured nothing pays the gate pass and stops.

        Never raises and never creates an alias. A query is a question: a value
        we have never been told simply stays as itself, which is the right
        answer, because no memory holds it either.
        """
        await edition.acall("retrieval_pii_classify", self, decision, scope, trace_id)

    async def _reveal_for_caller(self, bundle, scope, trace_id: str) -> None:
        """Put real values back into a bundle, for the caller's entitlement.

        PII Protection. Runs **after** the cache on both paths, which
        is what keeps plaintext out of any entry more than one caller can read.

        What gets revealed comes from the client's own table: the field types
        whose "their app" column says the real value. A client who chose
        "protect fully" for health conditions gets aliases back for those and
        real values for everything else, with nothing else to configure.

        Never raises. An alias left in place is readable text and the memory
        still says what happened, so a failure here degrades the answer rather
        than losing it. Failing the fetch would take a client's product down
        over a privacy feature that is meant to be invisible to them.

        ⚠️ This used to read ``bundle.items``, and ``ContextBundle`` has no
        such field. It carries ``items_by_type`` and exposes ``all_items``, so
        the getattr returned None on every call and the function returned
        before doing anything, for every caller, since it was written.

        The whole read half of the feature was dead. A client who chose
        "protect it" was told in as many words that their own app still
        receives the real value, and their app received a placeholder. That is
        the setting most clients pick, and the one whose entire point is that
        their product does not change.

        What hid it: the only tests of this wiring read the source of
        ``fetch_context`` with ``inspect.getsource`` and asserted the call
        appears after the cache write. It does appear there. It just does
        nothing when it gets there.
        """
        await edition.acall("retrieval_pii_reveal", self, bundle, scope, trace_id)

    async def fetch_context(
        self,
        decision: ContextDecision,
        scope: ScopeHierarchy = None,
    ) -> ContextBundle:
        """Main entry point: fetch context based on ContextDecision.

        A thin wrapper around the real fetch that opens the per-request fault
        collector and folds whatever the stores reported into the bundle. Its
        only job is that a leg which failed underneath reaches the caller: see
        ``record_retrieval_fault``. It deliberately does not swallow anything,
        so a raising fetch still raises exactly as it did.
        """
        faults: List[str] = []
        token = _RETRIEVAL_FAULTS.set(faults)
        maca_token = _REQUEST_MACA.set(None)
        visibility_token = _REQUEST_VISIBILITY.set(VISIBILITY_ALL)
        try:
            bundle = await self._fetch_context_inner(decision, scope=scope)
        finally:
            _REQUEST_VISIBILITY.reset(visibility_token)
            _REQUEST_MACA.reset(maca_token)
            _RETRIEVAL_FAULTS.reset(token)

        if faults:
            bundle.warnings.extend(faults)
            # A leg that failed means the answer was drawn from less than the
            # stores hold, which is exactly PARTIAL. Never downgrade a worse
            # verdict the fetch already reached.
            if bundle.degradation_level in (None, DegradationLevel.NONE):
                bundle.degradation_level = DegradationLevel.PARTIAL
            logger.warning(
                "retrieval_partial decision_id=%s items=%s faults=%s",
                decision.decision_id, bundle.item_count, faults,
            )
        return bundle

    async def _fetch_context_inner(
        self,
        decision: ContextDecision,
        scope: ScopeHierarchy = None,
    ) -> ContextBundle:
        """
        Fetch context based on ContextDecision.

        This is the primary Tool exposed to the ListeningRetrievalAgent.

        Args:
            decision: ContextDecision from Agent's decide_node

        Returns:
            ContextBundle with retrieved context items
        """
        start_time = time.time()
        trace_id = decision.trace_id or str(uuid.uuid4())
        # Ensure inner methods (which read ``decision.trace_id`` directly
        # for the audit-enrichment correlation_id) see the same value the
        # outer fetch generated when the SDK didn't supply one.
        decision.trace_id = trace_id

        # Step logging for request
        step_logger = StepLogger("RetrievalManager.fetch_context")
        step_logger.log_request(
            payload={
                "decision_id": decision.decision_id,
                "trace_id": trace_id,
                "retrieval_mode": decision.retrieval_mode.value,
                "trigger_type": decision.trigger_type.value if hasattr(decision.trigger_type, 'value') else str(decision.trigger_type),
                "search_queries": decision.search_queries,
                "context_types": decision.context_types,
                "instance_id": decision.instance_id,
                "user_id": decision.user_id,
                "customer_id": decision.customer_id,
                "conversation_id": decision.conversation_id,
                "max_results": decision.max_results,
                "token_budget": decision.token_budget,
                "scope": {
                    "instance_id": scope.instance_id if scope else None,
                    "client_id": scope.client_id if scope else None,
                    "customer_id": scope.customer_id if scope else None,
                    "user_id": scope.user_id if scope else None,
                } if scope else None,
            },
            correlation_id=trace_id,
        )

        logger.info(
            f"[{trace_id}] Starting retrieval: mode={decision.retrieval_mode.value}, query={[truncate_string(q, 200) for q in decision.search_queries] if decision.search_queries else []}, instance_id={decision.instance_id}, user_id={decision.user_id}, conversation_id={decision.conversation_id}, token_budget={decision.token_budget}"
        )

        # Initialize bundle
        bundle = ContextBundle(
            bundle_id=str(uuid.uuid4()),
            decision_id=decision.decision_id,
            retrieval_mode=decision.retrieval_mode.value,
            token_budget=decision.token_budget,
        )

        async with start_span(
            "synap.retrieval.fetch_context",
            {
                "synap.decision_id": decision.decision_id,
                "synap.trace_id": trace_id,
                "synap.instance_id": decision.instance_id,
                "synap.user_id": decision.user_id,
                "synap.customer_id": decision.customer_id,
                "synap.conversation_id": decision.conversation_id,
                "synap.mode": decision.retrieval_mode.value,
                "synap.query_count": len(decision.search_queries or []),
            },
        ):
            try:
                # 1. Load MACA settings for this instance (cached, 5min TTL)
                #    Must happen before scope resolution so B2C auto-resolution works.
                async with start_span(
                    "synap.retrieval.load_maca_settings",
                    {"synap.instance_id": decision.instance_id},
                ):
                    maca = await self._get_maca_settings(decision.instance_id)
                    if maca.custom_category_names and not custom_categories.is_on(decision.instance_id):
                        maca = _dc_replace(maca, custom_category_names=())
                    _REQUEST_MACA.set(maca)

                # 1b. Resolve cross-instance visibility policy (cached, opt-in).
                #     VISIBILITY_ALL = no restriction (default-open), the 99%
                #     case. Resolved here in raw-id space (before scope UUID
                #     conversion) so the policy tables can be keyed by raw ids.
                visibility = await self._get_visibility_allowed_set(
                    getattr(decision, "client_id", None), decision.instance_id
                )
                _REQUEST_VISIBILITY.set(visibility)

                # 2. Resolve scope hierarchy (use pre-built if provided)
                if scope is None:
                    async with start_span(
                        "synap.retrieval.resolve_scope",
                        {
                            "synap.instance_id": decision.instance_id,
                            "synap.user_id": decision.user_id,
                            "synap.conversation_id": decision.conversation_id,
                        },
                    ):
                        maca_scoping = {
                            "user_context_isolation": maca.user_context_isolation,
                        }
                        try:
                            from synap.cloud.shared.scope import build_scope_with_maca_mapped
                            from synap.cloud.shared.id_mapper import get_id_mapper
                            mapper = get_id_mapper()
                            scope = await build_scope_with_maca_mapped(
                                mapper=mapper,
                                client_id=decision.client_id or decision.instance_id,
                                instance_id=decision.instance_id,
                                user_id=decision.user_id,
                                customer_id=decision.customer_id if hasattr(decision, 'customer_id') else None,
                                maca_scoping=maca_scoping,
                            )
                        except Exception as scope_err:
                            logger.warning(
                                "build_scope_with_maca_mapped failed, falling back to ScopeResolver: %s "
                                "(client_id=%s, instance_id=%s, user_id=%s, customer_id=%s)",
                                scope_err,
                                decision.client_id or decision.instance_id,
                                decision.instance_id,
                                decision.user_id,
                                decision.customer_id if hasattr(decision, 'customer_id') else None,
                            )
                            scope = await self.scope_resolver.resolve(
                                instance_id=decision.instance_id,
                                user_id=decision.user_id,
                                conversation_id=decision.conversation_id,
                            )

                # 2b. Query classification (PII Protection). If the
                # store holds aliases, a search for the real value is a search
                # for characters that are not in it, and every layer of
                # retrieval finds nothing. Runs before the cache key is built,
                # so the key describes what was actually searched for.
                await self._classify_queries(decision, scope, trace_id)

                # 2c. Stamp the rung this request stands at onto the bundle.
                #
                # The prefetch path is why. `search_memories` hands what comes
                # back to `push_bundle_to_sdk`, which streams it into an SDK
                # cache that had no notion of a rung at all, so a bundle
                # retrieved for one rung could be served to a caller at
                # another. The SDK cannot resolve a chain and has never seen a
                # node id, so the stamp is written the way a caller names a
                # rung: level keys and the client's own external ids.
                #
                # Taken from THIS request's scope on both the hit and the miss
                # path, never from the cached copy. The L1 key already carries
                # the chain, so a hit is a hit at the same rung by
                # construction; reading it back off the entry would only add a
                # way for the stamp to describe whoever filled it first.
                _scope_rung = _rung_key_from_rungs(getattr(scope, "chain_rungs", None))
                bundle.scope_rung = _scope_rung

                # 3. Check L1 cache (uses resolved scope IDs for key consistency with invalidation)
                cache_key = self._build_cache_key(decision, scope)
                cached_bundle = await self._check_cache(cache_key)
                if cached_bundle:
                    cached_bundle.scope_rung = _scope_rung
                    cached_bundle.cache_hit = True
                    cached_bundle.retrieval_time_ms = int((time.time() - start_time) * 1000)
                    set_current_span_attributes(
                        {
                            "synap.cache_hit": True,
                            "synap.cache_key": cache_key,
                            "synap.result_count": cached_bundle.item_count if hasattr(cached_bundle, "item_count") else 0,
                            "synap.retrieval_time_ms": cached_bundle.retrieval_time_ms,
                        }
                    )
                    logger.info(f"[{trace_id}] Cache hit, returning cached bundle")
                    # Emit the usage event for observability; the charge is skipped
                    # for cache hits inside track_context_fetch (free, no real work).
                    # Detached: this is the fastest return path (a whole-bundle
                    # cache hit) — its telemetry INSERT must not add a blocking
                    # commit to it. Mirrors the non-cache path below.
                    _billing = edition.call(
                        "retrieval_billing_charge", self, scope, cached_bundle, decision, trace_id
                    )
                    if _billing is not None:
                        _run_detached(_billing)
                    # Reveal after the cache, never before. The cached copy
                    # stays aliased, so it belongs to no caller in particular
                    # and the next request to hit this key gets whatever its
                    # own entitlement allows rather than whatever the first
                    # caller was allowed to see.
                    await self._reveal_for_caller(cached_bundle, scope, trace_id)
                    return cached_bundle

                set_current_span_attributes({"synap.cache_hit": False, "synap.cache_key": cache_key})

                # 4. Fetch from stores based on mode
                if decision.retrieval_mode == RetrievalMode.FAST:
                    async with start_span("synap.retrieval.fast_mode"):
                        bundle = await self._fetch_fast(decision, scope, bundle)
                else:
                    async with start_span("synap.retrieval.accurate_mode"):
                        bundle = await self._fetch_accurate(decision, scope, bundle)

                # 4b. Enrich top items with source chunk evidence.
                #
                # Two switches, and they answer different questions. The env var
                # is deployment-wide. `include_source_evidence` is per request,
                # because whether the verbatim source text is wanted is the
                # caller's business, not the deployment's.
                #
                # Measured over a sample of real responses:
                # source_evidence is about 41.5% of the response body, roughly 400 bytes per
                # item, on nearly every item. A caller feeding our response to an
                # LLM pays for every one of those bytes in every prompt.
                #
                # It is not a latency knob. After the batching fix the whole
                # enrichment is p50 under 20ms over a sample of requests, under 1%
                # of a fetch. Skipping it buys bytes and tokens, not time, and
                # the docstring says so, so nobody reaches for it to go faster.
                _tp = getattr(decision, "trigger_payload", None) or {}
                _want_evidence = _tp.get("include_source_evidence", True)
                if not _want_evidence:
                    logger.info(
                        "chunk_evidence_skipped_by_caller: include_source_evidence=False"
                    )
                if _want_evidence and os.environ.get(
                    "ENABLE_CHUNK_EVIDENCE", "true"
                ).lower() in ("true", "1", "yes"):
                    async with start_span("synap.retrieval.chunk_evidence"):
                        try:
                            bundle = await self._enrich_with_chunk_evidence(
                                bundle=bundle,
                                search_queries=decision.search_queries or [],
                                scope=scope,
                            )
                        except Exception as e:
                            logger.warning(f"Chunk evidence enrichment failed (non-fatal): {e}")

                # 4c. Fill in WHO said it for items whose store never carried
                # it. The vector metadata only holds the key when extraction set
                # one, and until this change the graph never wrote it at all, so
                # anything reached through the graph had no speaker however many
                # read sites copied the field. Postgres has it on 69.4% of rows,
                # which is the real ceiling, so this closes the gap between what
                # we know and what we say.
                #
                # Placed here, not inside the evidence step, because a caller can
                # turn evidence off and attribution is not theirs to lose. Before
                # the budget, so a trimmed item is never the one that got it, and
                # before the cache, so the 300s entry carries it too.
                async with start_span("synap.retrieval.fill_assertion_source"):
                    try:
                        await self._fill_assertion_source(bundle)
                    except Exception as e:
                        logger.warning(
                            "fill_assertion_source failed (non-fatal): %s", e
                        )

                # Which thing each item is about (migration 159). Here for the
                # same three reasons as attribution above: a caller cannot turn
                # it off by turning evidence off, a trimmed item is never the
                # one that lost it, and the cached bundle carries it too.
                async with start_span("synap.retrieval.fill_subject"):
                    try:
                        await self._fill_subject(bundle)
                    except Exception as e:
                        logger.warning("fill_subject failed (non-fatal): %s", e)

                # A memory found through the hypothetical-question index is
                # served from that index's OWN copy of the text, which
                # vector_wrapper.py truncates to 500 characters when it is
                # written. The read path has the memory_id and never went back
                # for the real row, so a long memory reached the reader cut
                # off mid-word. Measured on the lme34 run: 34 memories across
                # 19 of 500 questions, every one source="discovery" and
                # context_type="fact", against 3 across 3 questions on the
                # previous run. What the reader was shown in one case:
                # "...4. Grilled Snapper with Mango Salsa - a popular dish in
                # many Caribbean countries where grilled snap" — the fifth
                # list item simply gone.
                #
                # Repairs rows already stored, so it needs no re-ingest.
                async with start_span("synap.retrieval.fill_truncated_content"):
                    try:
                        await self._fill_truncated_content(bundle)
                    except Exception as e:
                        logger.warning(
                            "fill_truncated_content failed (non-fatal): %s", e
                        )

                # 5. Apply token budget
                async with start_span(
                    "synap.retrieval.apply_token_budget",
                    {"synap.token_budget": decision.token_budget},
                ):
                    bundle = self._apply_token_budget(bundle, decision)

                # 6. Cache result
                await self._cache_result(cache_key, bundle)

                # 6b. Only now put the real values back, for this caller. The
                # order is the whole control: the entry written above holds
                # aliases, so no plaintext value ever sits in a cache that more
                # than one caller can read.
                await self._reveal_for_caller(bundle, scope, trace_id)

                # 7. Finalize
                bundle.retrieval_time_ms = int((time.time() - start_time) * 1000)
                bundle.cache_key = cache_key
                set_current_span_attributes(
                    {
                        "synap.result_count": bundle.item_count,
                        "synap.total_tokens": bundle.total_tokens,
                        "synap.retrieval_time_ms": bundle.retrieval_time_ms,
                        "synap.degradation_level": bundle.degradation_level.value if bundle.degradation_level else "none",
                    }
                )

                # Step logging for response
                step_logger.log_response(
                    response={
                        "bundle_id": bundle.bundle_id,
                        "decision_id": bundle.decision_id,
                        "item_count": bundle.item_count,
                        "total_tokens": bundle.total_tokens,
                        "retrieval_time_ms": bundle.retrieval_time_ms,
                        "cache_hit": bundle.cache_hit,
                        "degradation_level": bundle.degradation_level.value if bundle.degradation_level else "none",
                        "sources_queried": bundle.sources_queried,
                        "warnings": bundle.warnings,
                        "items_by_type": {
                            key: [
                                {
                                    "item_id": item.item_id,
                                    "content": truncate_string(item.content, 100),
                                    "source": item.source,
                                    "scope": item.scope,
                                    "confidence": item.confidence,
                                    "similarity_score": item.similarity_score,
                                }
                                for item in items[:3]  # Show first 3 items per type
                            ]
                            for key, items in bundle.items_by_type.items()
                        },
                        "total_items_by_type": {
                            key: len(items) for key, items in bundle.items_by_type.items()
                        },
                    },
                    status_code=200,
                    success=True,
                    correlation_id=trace_id,
                )

                logger.info(
                    f"[{trace_id}] Retrieval complete: item_count={bundle.item_count}, total_tokens={bundle.total_tokens}, retrieval_time_ms={bundle.retrieval_time_ms}, degradation_level={bundle.degradation_level.value if bundle.degradation_level else 'none'}, items_preview={log_data_sample([{'source': item.source, 'scope': item.scope, 'confidence': item.confidence, 'content': truncate_string(item.content, 80)} for item in bundle.all_items], sample_size=3, label='context_items')}"
                )

                # Meter the served fetch + charge retrieval credits (non-cache path).
                # Detached: metering does 2 DB round-trips (telemetry INSERT+commit
                # + credits charge) that must not sit on the response critical path.
                # _track_context_fetch is self-contained and never raises into the
                # read path (see its docstring), so fire-and-forget is safe and the
                # billing still happens — just off the caller's latency.
                # The whole billing block, its arguments included, sits behind the
                # hook: Core has no credits package, and working out the billable
                # count here would import it.
                _billing = edition.call(
                    "retrieval_billing_charge", self, scope, bundle, decision, trace_id
                )
                if _billing is not None:
                    _run_detached(_billing)

                return bundle

            except Exception as e:
                logger.error(f"[{trace_id}] Retrieval failed: {type(e).__name__}: {e}")
                set_current_span_attributes({"error": True, "synap.error_type": type(e).__name__})
                bundle.degradation_level = DegradationLevel.FAILED
                bundle.failure_code = _failure_code(e)
                # Deliberately NOT str(e): that put "embed failed after 2
                # attempts:" in front of clients, which names a service of ours
                # and gives them nothing to act on. The full text is on the
                # logger.error line above.
                bundle.warnings.append(f"Retrieval failed ({bundle.failure_code})")
                bundle.retrieval_time_ms = int((time.time() - start_time) * 1000)

                # Step logging for error response
                step_logger.log_response(
                    response={
                        "bundle_id": bundle.bundle_id,
                        "item_count": 0,
                        "degradation_level": "failed",
                        "error": str(e),
                        "retrieval_time_ms": bundle.retrieval_time_ms,
                    },
                    status_code=500,
                    success=False,
                    error=str(e),
                    correlation_id=trace_id,
                )

                return bundle

    @staticmethod
    def _llm_filter_opted_out(decision) -> bool:
        """True when the request opted out of the LLM relevance filter.

        Two request-level vehicles set this: the internal skip_llm_filter
        flag (metadata-keyed lookups, PRs #850/#851) and the public
        precision_level="medium" parameter (see ContextFetchRequest).
        Both ride in ContextDecision.trigger_payload.
        """
        tp = getattr(decision, "trigger_payload", None) or {}
        return bool(tp.get("skip_llm_filter")) or tp.get("precision_level") == "medium"

    def _should_skip_fast_filter(self, candidate_count: int, max_results: int) -> bool:
        """AI-1: whether FAST mode may skip the LLM relevance filter.

        True only when the opt-in config flag is set AND the candidate count
        already fits ``max_results`` — i.e. the filter would only relevance-drop,
        not select. OFF by default (the filter drops irrelevant items regardless
        of count, so skipping is a precision/latency tradeoff, not a no-op).
        """
        return bool(
            self.config.fast_skip_filter_when_scarce
            and max_results
            and candidate_count <= max_results
        )

    async def _fetch_fast(
        self,
        decision: ContextDecision,
        scope: ScopeHierarchy,
        bundle: ContextBundle,
    ) -> ContextBundle:
        """
        FAST mode: Direct query to vector + graph, no LLM decomposition.

        - No subquery generation
        - Parallel fetch from vector and graph
        - Basic relevance scoring
        """
        timeout = self.config.fast_timeout_ms / 1000
        fast_start = time.time()

        # Decompose compound queries (e.g. "pick up or return" → two subqueries)
        from synap.cloud.application.client_api.query_decomposition import decompose_queries
        original_queries = decision.search_queries or [""]
        expanded_queries = decompose_queries(original_queries)
        if len(expanded_queries) != len(original_queries):
            logger.info(f"COMPOUND_DECOMPOSE: original={original_queries}, expanded={expanded_queries}")
        # Which caller query each expanded string came from. Nothing was split:
        # each owns itself. Something was: ask the splitter one query at a time,
        # which is what it does internally. If the two disagree, attribute
        # nothing rather than tag an item with a string the caller never sent.
        if len(expanded_queries) == len(original_queries):
            expanded_owners = list(original_queries)
        else:
            expanded_owners = []
            for _oq in original_queries:
                expanded_owners.extend([_oq] * len(decompose_queries([_oq])))
            if len(expanded_owners) != len(expanded_queries):
                expanded_owners = [None] * len(expanded_queries)

        async with start_span(
            "synap.retrieval.fast_mode",
            {
                "synap.instance_id": str(scope.instance_id),
                "synap.user_id": decision.user_id or "",
                "synap.conversation_id": decision.conversation_id or "",
                "synap.query_count": len(decision.search_queries or []),
                "synap.context_types": ",".join(decision.context_types or []),
                "synap.timeout_s": timeout,
            },
        ):
            logger.info(
                f"FAST mode retrieval starting: timeout_s={timeout}, conversation_id={decision.conversation_id}, instance_id={scope.instance_id}, user_id={decision.user_id}, context_types={decision.context_types}, search_queries={decision.search_queries[:3] if decision.search_queries else []}"
            )

            # Initialize result containers before launching tasks so partial results
            # are available even if the timeout fires before both tasks complete.
            all_items = []
            sources = []
            conversation_context = None
            short_term_result = None
            long_term_result = None
            short_term_items: list = []

            try:
                # Encode queries first (typically <300ms); guard with full timeout
                async with asyncio.timeout(timeout):
                    query_vectors = await self._batch_encode_queries(expanded_queries)
            except asyncio.TimeoutError:
                elapsed_ms = (time.time() - fast_start) * 1000
                bundle.warnings.append(f"FAST mode timeout during embedding ({timeout}s)")
                bundle.degradation_level = DegradationLevel.MINIMAL
                set_current_span_attributes({"synap.timeout": True, "synap.elapsed_ms": round(elapsed_ms, 2)})
                logger.warning(f"FAST mode TIMEOUT during embedding: timeout_s={timeout}, elapsed_ms={round(elapsed_ms, 1)}")
                return bundle

            # Calculate remaining budget after encoding
            elapsed_so_far = time.time() - fast_start
            remaining_s = max(0.05, timeout - elapsed_so_far)

            # Launch both retrieval tasks as proper asyncio Tasks so asyncio.wait()
            # can return whichever finishes within the budget without cancelling
            # already-completed work (unlike asyncio.timeout + asyncio.gather).
            #
            # SDK-authoritative ST: when the SDK signals it has the
            # short-term block cached locally we skip the server-side ST
            # task entirely. The fetch_short_term_context call covers
            # both compacted-context retrieval and the recent_turns buffer
            # read — cutting it removes a Redis hit, an LLM-budget
            # serialization, and the L2 cache traffic.
            st_task: Optional[asyncio.Task] = None
            if getattr(decision, "include_conversation_context", True):
                st_task = asyncio.create_task(
                    self.fetch_short_term_context(
                        decision.conversation_id,
                        decision.instance_id,
                        decision.user_id,
                        filters={"types": decision.context_types},
                        customer_id=str(scope.customer_id) if scope.customer_id else None,
                        search_queries=expanded_queries,
                        precomputed_query_vectors=query_vectors,
                        client_id=scope.client_id,
                        correlation_id=decision.trace_id,
                        query_owners=expanded_owners,
                        scope=scope,
                    ),
                    name="fast_short_term",
                )
            # Honor decision-level skip flag: the anticipation agent provides
            # 5-7 broad queries via the LLM; expanding each into 4 wrapper
            # subqueries inside fetch_long_term_context just multiplies Chroma
            # work (~400ms per Chroma call) for marginal recall gain. The
            # agent already explored breadth at planning time.
            skip_sub = bool(
                (decision.trigger_payload or {}).get("skip_subquery_gen")
            )
            lt_task = asyncio.create_task(
                self.fetch_long_term_context(
                    scope,
                    queries=expanded_queries,
                    filters={"types": decision.context_types},
                    mode="fast",
                    correlation_id=decision.trace_id,
                    precomputed_query_vectors=query_vectors,
                    skip_subquery_gen=skip_sub,
                    max_results=decision.max_results,
                    query_owners=expanded_owners,
                ),
                name="fast_long_term",
            )

            wait_set = {lt_task}
            if st_task is not None:
                wait_set.add(st_task)
            done, pending = await asyncio.wait(wait_set, timeout=remaining_s)

            # Cancel tasks that didn't finish. We do NOT await them here because
            # they may be blocked inside asyncio.to_thread() (Neo4j BFS, serialization
            # threads etc.) which cannot be interrupted mid-execution. Awaiting them
            # would block the caller for the full thread duration, defeating the timeout.
            # The cancelled flag is set; once the thread returns, asyncio will raise
            # CancelledError in the task and it will clean up in the background.
            for t in pending:
                t.cancel()

            timed_out = bool(pending)

            # Extract results from completed tasks
            if st_task is not None and st_task in done:
                try:
                    short_term_result = st_task.result()
                except Exception as e:
                    short_term_result = e

            if lt_task in done:
                try:
                    long_term_result = lt_task.result()
                except Exception as e:
                    long_term_result = e

            # Process short-term result
            if isinstance(short_term_result, tuple):
                short_term_items, conversation_context = short_term_result
                if short_term_items:
                    all_items.extend(short_term_items)
                sources.append("cache")
            elif isinstance(short_term_result, Exception):
                bundle.warnings.append(f"Short-term fetch failed: {short_term_result}")
                logger.warning(f"FAST mode: short-term fetch failed: error={short_term_result}")
            elif isinstance(short_term_result, list) and short_term_result:
                # Backward compat: plain list returned
                short_term_items = short_term_result
                all_items.extend(short_term_items)
                sources.append("cache")
            # else: None (task timed out) — short_term_items stays []

            # Process long-term result
            if isinstance(long_term_result, list):
                all_items.extend(long_term_result)
                sources.extend(["vector", "graph"])
                if any(getattr(item, "source", None) == "chunks" for item in long_term_result):
                    sources.append("chunks")
            elif isinstance(long_term_result, Exception):
                bundle.warnings.append(f"Long-term fetch failed: {long_term_result}")
                bundle.degradation_level = DegradationLevel.PARTIAL
                logger.warning(f"FAST mode: long-term fetch failed: error={long_term_result}")
            # else: None (task timed out) — no long-term items

            # Set degradation level
            if timed_out:
                if all_items:
                    bundle.degradation_level = DegradationLevel.PARTIAL
                    bundle.warnings.append(f"FAST mode partial timeout ({timeout}s): {len(pending)} task(s) did not finish in time")
                else:
                    bundle.degradation_level = DegradationLevel.MINIMAL
                    bundle.warnings.append(f"FAST mode timeout ({timeout}s): no results collected")
            elif not all_items and bundle.degradation_level != DegradationLevel.NONE:
                bundle.degradation_level = DegradationLevel.FAILED
                bundle.warnings.append("All retrieval sources returned no results")

            # ── LLM filter (same as ACCURATE mode — improves precision) ──
            # Metadata-keyed lookups (e.g. LangGraph checkpoint reads keyed by
            # thread_id) opt out: the caller selects its row by exact metadata,
            # so semantic re-ranking is wasted work — a Groq round-trip whose
            # cost also grows with collection size.
            # precision_level="medium" is the public, validated form of the
            # same opt-out (see ContextFetchRequest).
            skip_filter = self._llm_filter_opted_out(decision)
            if (decision.trigger_payload or {}).get("precision_level") == "medium":
                logger.info("LLM filter skipped: precision_level=medium (mode=fast)")
            if self.llm_client is not None and all_items and not skip_filter:
                chunk_items = [item for item in all_items if item.source == "chunks"]
                non_chunk_items = [item for item in all_items if item.source != "chunks"]

                filter_start = time.time()
                pre_filter_count = len(non_chunk_items)
                query_text = " ".join(expanded_queries)
                # AI-1 (opt-in, default off): when the candidate count already
                # fits the caller's max_results there is nothing to *select* — the
                # filter would only relevance-drop. Skipping trades a little
                # precision for a Groq round-trip, so it is gated behind config
                # and NOT enabled by default.
                scarce_skip = self._should_skip_fast_filter(
                    pre_filter_count, decision.max_results
                )
                if non_chunk_items and not scarce_skip:
                    try:
                        non_chunk_items = await self.filter_items_with_llm(
                            non_chunk_items,
                            query_text,
                            scope=scope,
                            correlation_id=decision.trace_id,
                        )
                    except Exception as e:
                        logger.warning(f"FAST mode LLM filter failed, using unfiltered: {e}")
                elif scarce_skip:
                    logger.info(
                        f"FAST LLM filter skipped (scarce): items={pre_filter_count} <= max_results={decision.max_results}"
                    )
                filter_ms = (time.time() - filter_start) * 1000
                logger.info(
                    f"FAST LLM filtering applied: elapsed_ms={round(filter_ms, 1)}, items_before={pre_filter_count}, items_after={len(non_chunk_items)}, items_removed={pre_filter_count - len(non_chunk_items)}, chunks_bypassed={len(chunk_items)}"
                )
                all_items = non_chunk_items + chunk_items

            # Re-rank AFTER every drop-gate and outside the filter block, so the
            # order the top-K cut sees is the best one we can produce. Off by
            # default (SYNAP_RETRIEVAL_JEV_RERANK); never removes an item.
            all_items = await self._apply_temporal_window_filter(
                all_items, " ".join(expanded_queries), decision.query_date,
                correlation_id=decision.trace_id,
            )
            all_items = await self._apply_jev_rerank(
                all_items, " ".join(expanded_queries), correlation_id=decision.trace_id
            )

            bundle.sources_queried = sources
            bundle = await asyncio.to_thread(self.assemble_context_bundle, all_items, decision, bundle)
            bundle.conversation_context = conversation_context

            elapsed_ms = (time.time() - fast_start) * 1000
            set_current_span_attributes({
                "synap.timeout": timed_out,
                "synap.short_term_items": len(short_term_items),
                "synap.long_term_items": len(long_term_result) if isinstance(long_term_result, list) else 0,
                "synap.total_items": len(all_items),
                "synap.bundle_items": len(bundle.all_items),
                "synap.sources": ",".join(sources),
                "synap.elapsed_ms": round(elapsed_ms, 2),
                "synap.short_term_error": isinstance(short_term_result, Exception),
                "synap.long_term_error": isinstance(long_term_result, Exception),
                "synap.items_before_timeout": len(all_items) if timed_out else 0,
            })
            if timed_out:
                logger.warning(
                    f"FAST mode partial TIMEOUT: timeout_s={timeout}, elapsed_ms={round(elapsed_ms, 1)}, items_collected={len(all_items)}, sources={sources}"
                )
            else:
                logger.info(
                    f"FAST mode retrieval complete: elapsed_ms={round(elapsed_ms, 1)}, short_term_items={len(short_term_items)}, long_term_items={len(long_term_result) if isinstance(long_term_result, list) else 0}, total_items={len(all_items)}, sources={sources}, bundle_items={len(bundle.all_items)}"
                )

        return bundle

    async def _fetch_accurate(
        self,
        decision: ContextDecision,
        scope: ScopeHierarchy,
        bundle: ContextBundle,
    ) -> ContextBundle:
        """
        ACCURATE mode: LLM subquery decomposition + parallel fetch + reranking.

        Pipeline (optimized for latency):
        Initial fan-out: Fire in parallel — sub-query gen (LLM), short-term cache, main
        query long-term
        Sub-query search: When sub-queries arrive, fire additional long-term tasks for each
        Dedup: Deduplicate all items (item_id + Jaccard)
        LLM filter: LLM filter on ALL deduped items
        Assembly: Assemble context bundle
        """
        timeout = self.config.accurate_timeout_ms / 1000
        accurate_start = time.time()
        confidence_config = ConfidenceConfig(
            similarity_threshold=self.config.confidence_threshold,
            min_results=self.config.min_results_for_confidence,
        )

        all_items = []
        sources = set()
        main_query = (decision.search_queries or [""])[0]
        # What the long-term search runs on. The first query is still the one
        # the LLM writes subqueries from; the rest are searched as the caller
        # wrote them. One query: this is [main_query], as it always was.
        _all_queries = accurate_searches_all_queries()
        caller_queries = (
            list(decision.search_queries or [main_query])
            if _all_queries else [main_query]
        )

        _accurate_span_attrs = {
            "synap.instance_id": str(scope.instance_id),
            "synap.user_id": decision.user_id or "",
            "synap.conversation_id": decision.conversation_id or "",
            "synap.query_count": len(decision.search_queries or []),
            "synap.context_types": ",".join(decision.context_types or []),
            "synap.timeout_s": timeout,
        }

        logger.info(
            f"ACCURATE mode retrieval starting: timeout_s={timeout}, conversation_id={decision.conversation_id}, instance_id={scope.instance_id}, user_id={decision.user_id}, context_types={decision.context_types}, search_queries={decision.search_queries[:3] if decision.search_queries else []}, llm_client_available={self.llm_client is not None}, graph_provider_available={self.graph is not None}, vector_provider_available={self.vector is not None}"
        )

        set_current_span_attributes(_accurate_span_attrs)

        try:
            async with asyncio.timeout(timeout):
                # ── Overlapped sub-query gen, search, and sub-query search ──
                # Strategy: Fire subquery_gen + short_term + main_lt in parallel.
                # Wait for subquery_gen FIRST (typically ~2s), then immediately
                # launch the sub-query search to overlap with main_lt's tail (~3s+).
                # This saves ~1-2s vs waiting for all initial tasks before starting it.
                phase0_start = time.time()
                maca = await self._get_maca_settings(scope.instance_id)

                # Pre-compute embeddings for the main query, share with short-term and long-term
                main_query_vectors = await self._batch_encode_queries(decision.search_queries or [main_query])

                subquery_gen_task = asyncio.create_task(
                    self._generate_retrieval_subqueries(
                        main_query,
                        "accurate",
                        maca_settings=maca,
                        scope=scope,
                        correlation_id=decision.trace_id,
                    )
                )

                # SDK-authoritative ST: SDKs that have the
                # short-term block cached pass include_conversation_context=False
                # so we can skip the entire fetch.
                short_term_task: Optional[asyncio.Task] = None
                if getattr(decision, "include_conversation_context", True):
                    short_term_task = asyncio.create_task(
                        self.fetch_short_term_context(
                            decision.conversation_id,
                            decision.instance_id,
                            decision.user_id,
                            filters={"types": decision.context_types},
                            customer_id=str(scope.customer_id) if scope.customer_id else None,
                            search_queries=decision.search_queries,
                            precomputed_query_vectors=main_query_vectors,
                            client_id=scope.client_id,
                            correlation_id=decision.trace_id,
                            query_owners=decision.search_queries,
                            scope=scope,
                        )
                    )

                # Start main query immediately — don't wait for sub-query gen.
                # skip_subquery_gen=True avoids redundant LLM call inside fetch_long_term_context.
                main_lt_task = asyncio.create_task(
                    self.fetch_long_term_context(
                        scope,
                        queries=caller_queries,
                        filters={"types": decision.context_types},
                        mode="accurate",
                        correlation_id=decision.trace_id,
                        skip_subquery_gen=True,
                        precomputed_query_vectors=main_query_vectors,
                        max_results=decision.max_results,
                        anchor_on_all_queries=_all_queries,
                    )
                )

                # Wait for subquery gen FIRST — it typically completes before main_lt,
                # allowing the sub-query search to overlap with the tail of main_lt.
                try:
                    subquery_result = await subquery_gen_task
                except Exception as e:
                    logger.warning(f"Sub-query generation failed: {e}")
                    subquery_result = e
                subquery_ms = (time.time() - phase0_start) * 1000

                # Extract sub-queries immediately so we can fire the sub-query search
                additional_queries = []
                if isinstance(subquery_result, dict):
                    all_subqueries = subquery_result.get("queries", [])
                    # A subquery the caller already sent is searched by the
                    # main task above; searching it again only adds load.
                    additional_queries = [q for q in all_subqueries if q and q not in caller_queries]

                # Fire the sub-query search immediately — overlaps with main_lt + short_term
                # still running
                phase1_task = None
                if additional_queries:
                    batched_queries = additional_queries[:self.config.max_subqueries]
                    phase1_task = asyncio.create_task(
                        self.fetch_long_term_context(
                            scope,
                            queries=batched_queries,
                            filters={"types": decision.context_types},
                            mode="accurate",
                            correlation_id=decision.trace_id,
                            skip_subquery_gen=True,
                            max_results=decision.max_results,
                            # Written from the first query, so found for it.
                            query_owners=[main_query] * len(batched_queries),
                        )
                    )

                # Wait for remaining initial tasks + the sub-query search together.
                # ``short_term_task`` is None when the SDK opted out of
                # server-side ST; preserve a stable index for
                # ``main_lt_result`` regardless.
                gather_tasks = []
                if short_term_task is not None:
                    gather_tasks.append(short_term_task)
                gather_tasks.append(main_lt_task)
                if phase1_task:
                    gather_tasks.append(phase1_task)
                results = await asyncio.gather(*gather_tasks, return_exceptions=True)

                idx = 0
                short_term_result = None
                if short_term_task is not None:
                    short_term_result = results[idx]
                    idx += 1
                main_lt_result = results[idx]
                idx += 1
                phase1_result = results[idx] if phase1_task else None
                phases_ms = (time.time() - phase0_start) * 1000

                # Collect short-term context
                conversation_context = None
                if isinstance(short_term_result, tuple):
                    st_items, conversation_context = short_term_result
                    if st_items:
                        all_items.extend(st_items)
                    sources.add("cache")
                elif isinstance(short_term_result, Exception):
                    logger.warning(f"Short-term fetch failed in ACCURATE mode: {short_term_result}")
                elif isinstance(short_term_result, list) and short_term_result:
                    all_items.extend(short_term_result)
                    sources.add("cache")

                # Collect main query long-term items
                if isinstance(main_lt_result, list) and main_lt_result:
                    all_items.extend(main_lt_result)
                    sources.update(["vector", "graph"])
                    if any(getattr(item, "source", None) == "chunks" for item in main_lt_result):
                        sources.add("chunks")
                elif isinstance(main_lt_result, Exception):
                    logger.warning(f"Main query long-term fetch failed: {main_lt_result}")
                    bundle.warnings.append(f"Main long-term fetch failed: {type(main_lt_result).__name__}")

                # Collect sub-query search items
                if phase1_task:
                    if isinstance(phase1_result, list) and phase1_result:
                        all_items.extend(phase1_result)
                        sources.update(["vector", "graph"])
                        if any(getattr(item, "source", None) == "chunks" for item in phase1_result):
                            sources.add("chunks")
                    elif isinstance(phase1_result, Exception):
                        logger.warning(f"Batched subquery fetch failed: {phase1_result}")

                logger.info(
                    f"ACCURATE phases 0+1 complete (overlapped): total_ms={round(phases_ms, 1)}, "
                    f"subquery_gen_ms={round(subquery_ms, 1)}, "
                    f"main_lt_items={len(main_lt_result) if isinstance(main_lt_result, list) else 0}, "
                    f"short_term_items={len(short_term_result) if isinstance(short_term_result, (list, tuple)) else 0}, "
                    f"phase1_items={len(phase1_result) if isinstance(phase1_result, list) else 0}, "
                    f"additional_subqueries={len(additional_queries)}, "
                    f"subqueries_preview={[q[:80] for q in additional_queries[:3]]}"
                )

                # ── Deduplicate BEFORE LLM filter ──
                pre_dedup_count = len(all_items)
                async with start_span(
                    "synap.retrieval.dedup",
                    {
                        "synap.items_before": pre_dedup_count,
                    },
                ):
                    all_items = self._deduplicate_items(all_items)
                    post_exact_count = len(all_items)
                    all_items = _deduplicate_jaccard(
                        all_items, threshold=dedup_jaccard_threshold())
                post_dedup_count = len(all_items)

                logger.info(
                    f"ACCURATE dedup complete (before LLM filter): items_before={pre_dedup_count}, items_after_exact={post_exact_count}, items_after_jaccard={post_dedup_count}, items_removed={pre_dedup_count - post_dedup_count}"
                )

                # ── Degradation check (before LLM filter — based on store results, not LLM decisions) ──
                if not all_items:
                    bundle.degradation_level = DegradationLevel.FAILED
                    bundle.warnings.append("All retrieval sources returned no results")
                    logger.warning("ACCURATE mode: no items from any store — FAILED degradation")
                elif not sources.intersection({"vector", "graph", "chunks"}):
                    bundle.degradation_level = DegradationLevel.PARTIAL
                    bundle.warnings.append("Only short-term cache available; long-term stores returned no results")

                # ── LLM filtering on ALL deduped items ──
                # Metadata-keyed lookups opt out of semantic re-ranking (see
                # _fetch_fast); the caller picks its row by exact metadata.
                # precision_level="medium" is the public form of the same opt-out.
                skip_filter = self._llm_filter_opted_out(decision)
                if (decision.trigger_payload or {}).get("precision_level") == "medium":
                    logger.info("LLM filter skipped: precision_level=medium (mode=accurate)")
                # Every query that was searched, so the filter, the temporal
                # window and the re-rank below judge an item against the query
                # that found it and not only against the first one.
                all_subqueries_text = " ".join(
                    caller_queries + additional_queries
                )
                if self.llm_client is not None and all_items and not skip_filter:
                    # Chunks/sentences are already score-gated in _search_sentences — skip LLM filter
                    chunk_items = [item for item in all_items if item.source == "chunks"]
                    non_chunk_items = [item for item in all_items if item.source != "chunks"]

                    filter_start = time.time()
                    pre_filter_count = len(non_chunk_items)
                    if non_chunk_items:
                        non_chunk_items = await self.filter_items_with_llm(
                            non_chunk_items,
                            all_subqueries_text,
                            scope=scope,
                            correlation_id=decision.trace_id,
                        )
                    filter_ms = (time.time() - filter_start) * 1000
                    logger.info(
                        f"ACCURATE LLM filtering applied: elapsed_ms={round(filter_ms, 1)}, items_before={pre_filter_count}, items_after={len(non_chunk_items)}, items_removed={pre_filter_count - len(non_chunk_items)}, chunks_bypassed={len(chunk_items)}"
                    )
                    all_items = non_chunk_items + chunk_items

            # ── Semantic re-rank (order only, never drops) ──
            # Outside the filter block above on purpose: that block needs an
            # llm_client, Jev does not.
            all_items = await self._apply_temporal_window_filter(
                all_items, all_subqueries_text, decision.query_date,
                correlation_id=decision.trace_id,
            )
            all_items = await self._apply_jev_rerank(
                all_items, all_subqueries_text, correlation_id=decision.trace_id
            )

            # ── Assemble bundle ──

            bundle.sources_queried = list(sources)
            bundle = await asyncio.to_thread(self.assemble_context_bundle, all_items, decision, bundle)
            bundle.conversation_context = conversation_context

            elapsed_ms = (time.time() - accurate_start) * 1000
            set_current_span_attributes({
                "synap.total_items": len(all_items),
                "synap.bundle_items": len(bundle.all_items),
                "synap.sources": ",".join(sorted(sources)),
                "synap.elapsed_ms": round(elapsed_ms, 2),
                "synap.timeout": False,
            })
            logger.info(
                f"ACCURATE mode retrieval complete: elapsed_ms={round(elapsed_ms, 1)}, total_items={len(all_items)}, sources={list(sources)}, bundle_items={len(bundle.all_items)}, degradation_level={bundle.degradation_level.value if hasattr(bundle.degradation_level, 'value') else str(bundle.degradation_level)}"
            )

        except asyncio.TimeoutError:
            elapsed_ms = (time.time() - accurate_start) * 1000
            bundle.warnings.append(f"ACCURATE mode timeout ({timeout}s)")
            if all_items:
                bundle.degradation_level = DegradationLevel.PARTIAL
                bundle = await asyncio.to_thread(self.assemble_context_bundle, all_items, decision, bundle)
            else:
                bundle.degradation_level = DegradationLevel.MINIMAL
            set_current_span_attributes({
                "synap.timeout": True,
                "synap.elapsed_ms": round(elapsed_ms, 2),
                "synap.items_before_timeout": len(all_items),
            })
            logger.warning(
                f"ACCURATE mode TIMEOUT: timeout_s={timeout}, elapsed_ms={round(elapsed_ms, 1)}, items_collected_before_timeout={len(all_items)}, sources_collected={list(sources)}"
            )

        return bundle

    def normalize_retrieval_request(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize and validate retrieval request from SDK.

        Called before creating ContextDecision.
        """
        normalized = {
            "conversation_id": request.get("conversation_id"),
            "user_id": request.get("user_id"),
            "search_queries": request.get("searchQuery", []),
            "max_results": request.get("max_results", 10),
            "context_types": request.get("types", ["facts", "preferences", "episodes", "temporal_events"]),
        }

        # Ensure search_queries is a list
        if isinstance(normalized["search_queries"], str):
            normalized["search_queries"] = [normalized["search_queries"]]

        # Validate types
        valid_types = {"facts", "preferences", "episodes", "emotions", "temporal_events"}
        normalized["context_types"] = [
            t for t in normalized["context_types"]
            if t in valid_types
        ] or ["facts", "preferences", "episodes"]

        return normalized

    async def _l2_context_is_older_than_newest_compaction(
        self,
        cached_ctx: "ConversationContext",
        conversation_id: str,
    ) -> bool:
        """True when a compaction newer than the cached context's exists.

        The conversation-context L2 cache is keyed by client, customer, user and
        conversation, and the post-compaction invalidation rebuilds that key from
        the compaction event. It cannot: the event carries the external client and
        customer names and the key holds the internal UUIDs, so for every
        compaction after the first, the delete hits a key that does not exist and
        the old entry serves its first summary for its whole 30 minutes. In one
        observed case the cache held compaction 1 while the database held
        versions 2 to 4, and the agent answered from a summary of one message.

        So the cached entry is checked against the newest compaction directly. The
        compaction store keeps its own copy under a key that needs only the
        conversation id, and it is rewritten on every compaction, so one Redis GET
        settles it. Any failure here serves the cached entry, as before.
        """
        storage = getattr(self, "compaction_storage", None)
        if storage is None or not conversation_id:
            return False
        try:
            latest = await storage.get_from_cache(conversation_id)
            if latest is None:
                return False
            newest_id = getattr(getattr(latest, "metadata", None), "compaction_id", None)
            return bool(newest_id) and newest_id != cached_ctx.compaction_id
        except Exception as exc:  # noqa: BLE001 - freshness check must never break a fetch
            logger.debug("L2 freshness check failed, serving the cached context: %s", exc)
            return False

    async def fetch_short_term_context(
        self,
        conversation_id: str,
        instance_id: str,
        user_id: Optional[str],
        filters: Dict = None,
        customer_id: Optional[str] = None,
        search_queries: Optional[List[str]] = None,
        precomputed_query_vectors: Optional[List] = None,
        client_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        query_owners: Optional[List[str]] = None,
        scope: Optional[ScopeHierarchy] = None,
    ) -> Tuple[List[ContextItem], Optional[ConversationContext]]:
        """
        Fetch current session context as a structured ConversationContext,
        plus query-relevant short-term vector items as scored ContextItems.

        Returns a tuple of (List[ContextItem] from short-term vector search, ConversationContext).
        The ConversationContext contains the compaction narrative, key extractions,
        and recent raw turns — kept as a coherent block, NOT atomized into items.

        Args:
            conversation_id: Current conversation ID
            instance_id: Instance scope ID (from decision)
            user_id: User scope ID (from decision)
            filters: Optional filters (types, etc.)
            customer_id: Customer scope ID
            search_queries: Search queries for short-term vector similarity search
            scope: Resolved request scope. Supplies the nested-ladder chain for
                the short-term vector search; the user id used for the filter is
                still the ``user_id`` argument.

        Returns:
            Tuple of (short-term ContextItems, ConversationContext or None)
        """
        short_term_start = time.time()

        logger.info(
            f"Short-term context fetch starting: conversation_id={conversation_id}, instance_id={instance_id}, user_id={user_id}, customer_id={customer_id}"
        )
        set_current_span_attributes(
            {
                "synap.short_term.conversation_id": conversation_id,
                "synap.short_term.instance_id": instance_id,
            }
        )

        if not conversation_id:
            logger.info("Short-term fetch skipped: no conversation_id")
            return ([], None)

        # 1. Check L2 cache first (stores serialized ConversationContext)
        l2_key = CacheKeyGenerator.l2_user_context_key(
            client_id=client_id or "",
            customer_id=customer_id or "",
            user_id=user_id or "anonymous",
            conversation_id=conversation_id,
        )
        async with start_span(
            "synap.retrieval.short_term.cache_lookup",
            {"synap.cache_key": l2_key},
        ):
            cached_data = await self._get_from_cache(l2_key)
        if cached_data:
            # Handle both old format (List[ContextItem]) and new format (dict)
            conv_ctx = None
            if isinstance(cached_data, dict) and "conversation_context" in cached_data:
                cached_ctx = ConversationContext.from_dict(cached_data["conversation_context"])
                if await self._l2_context_is_older_than_newest_compaction(
                    cached_ctx, conversation_id
                ):
                    logger.info(
                        "Short-term context: L2 hit is older than the newest "
                        "compaction (%s), rebuilding", cached_ctx.compaction_id,
                    )
                else:
                    conv_ctx = cached_ctx
            elif isinstance(cached_data, list):
                # Old format — ignore stale cache, will rebuild below
                logger.info("Short-term context: L2 cache hit but old format, rebuilding")
            else:
                logger.info("Short-term context: L2 cache hit but unrecognized format, rebuilding")
            if conv_ctx is not None:
                set_current_span_attributes({"synap.short_term.cache_hit": True})
                elapsed_ms = (time.time() - short_term_start) * 1000
                logger.info(f"Short-term context: L2 cache hit (ConversationContext): elapsed_ms={round(elapsed_ms, 1)}")
                # Audit snapshot also runs on the cache-hit path so the
                # finalize-on-timeout watcher can backfill an enrichment
                # row regardless of how the ST was served.
                await self._write_audit_st_snapshot(
                    correlation_id=correlation_id,
                    conv_ctx=conv_ctx,
                    instance_id=instance_id,
                    user_id=user_id,
                    customer_id=customer_id,
                    conversation_id=conversation_id,
                )
                return ([], conv_ctx)

        set_current_span_attributes({"synap.short_term.cache_hit": False})
        logger.info("Short-term context: L2 cache miss, querying sources")

        # 2. Get compacted context (using CompactionStorage)
        compacted = None
        async with start_span(
            "synap.retrieval.short_term.compaction_lookup",
            {
                "synap.conversation_id": conversation_id,
                "synap.instance_id": instance_id,
            },
        ):
            compacted = await self._get_compacted_context(
                conversation_id=conversation_id,
                instance_id=instance_id,
                client_id=client_id or "",
                user_id=user_id or "",
                customer_id=customer_id or "",
            )

        # 3. Build ConversationContext from compacted data
        if compacted:
            formatted = self._format_compacted_for_internal_use(compacted)
            conversation_ctx = ConversationContext(
                summary=formatted.get("summary"),
                current_state=formatted.get("current_state", {}),
                key_extractions={
                    "facts": formatted.get("facts", []),
                    "decisions": formatted.get("decisions", []),
                    "preferences": formatted.get("preferences", []),
                    "constraints": formatted.get("constraints", []),
                    "emotional_states": formatted.get("emotional_states", []),
                },
                compaction_id=formatted.get("compaction_id"),
                compacted_at=formatted.get("compacted_at"),
                conversation_id=conversation_id,
            )
            logger.info(
                f"Short-term: compacted context loaded into ConversationContext: "
                f"has_summary={bool(formatted.get('summary'))}, "
                f"facts={len(formatted.get('facts', []))}, "
                f"decisions={len(formatted.get('decisions', []))}, "
                f"preferences={len(formatted.get('preferences', []))}"
            )
        else:
            conversation_ctx = ConversationContext(conversation_id=conversation_id)
            logger.info("Short-term: no compacted context available")

        # 4. Fetch un-compacted raw messages from Redis buffer → recent_turns
        raw_buffer_count = 0
        if self.compaction_storage is not None:
            async with start_span(
                "synap.retrieval.short_term.redis_raw_buffer",
                {
                    "synap.conversation_id": conversation_id,
                    "synap.instance_id": instance_id,
                },
            ):
                cutoff_ts = None
                if compacted and hasattr(compacted, 'metadata') and compacted.metadata.source_message_range:
                    cutoff_ts = compacted.metadata.source_message_range.get("end_timestamp")

                raw_messages = await self.compaction_storage.load_raw_messages_from_redis(
                    conversation_id=conversation_id,
                    after_timestamp=cutoff_ts,
                )
                conversation_ctx.recent_turns = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp),
                    }
                    for msg in _uncompacted_tail(raw_messages)
                ]
                raw_buffer_count = len(conversation_ctx.recent_turns)

            logger.info(
                f"Short-term: raw buffer → recent_turns: "
                f"total_in_buffer={len(raw_messages)}, "
                f"included={raw_buffer_count}, "
                f"cutoff_ts={cutoff_ts}"
            )

        # 5. Cache to L2 (new format: wrapped dict)
        if conversation_ctx.summary or conversation_ctx.recent_turns:
            async with start_span(
                "synap.retrieval.short_term.cache_write",
                {"synap.cache_key": l2_key},
            ):
                await self._set_to_cache(
                    l2_key,
                    {"conversation_context": conversation_ctx.to_dict()},
                    ttl=CacheKeyGenerator.TTL_L2,
                )

        # 5b. Audit-enrichment snapshot (also runs on the cache-hit branch
        # above). See _write_audit_st_snapshot for details.
        await self._write_audit_st_snapshot(
            correlation_id=correlation_id,
            conv_ctx=conversation_ctx,
            instance_id=instance_id,
            user_id=user_id,
            customer_id=customer_id,
            conversation_id=conversation_id,
        )

        # 6. Search short-term vectors for query-relevant items
        short_term_items: List[ContextItem] = []
        if search_queries and self.vector is not None:
            if scope is not None:
                # The resolved scope carries the nested-ladder chain this arm
                # needs; building a fresh one dropped it, so a turn recorded at
                # one team was read back by its siblings. The user id stays the
                # one the caller passed, which is what this arm has always
                # filtered on. The resolved scope holds the mapped internal id
                # instead, and swapping one for the other changes which turns
                # a default-open client gets back. That is a separate decision.
                st_scope = _dc_replace(scope, user_id=user_id)
            else:
                st_scope = ScopeHierarchy(
                    client_id=client_id or "",
                    instance_id=instance_id,
                    customer_id=customer_id,
                    user_id=user_id,
                )
            short_term_items = await self._search_short_term_vectors(
                scope=st_scope,
                search_queries=search_queries,
                precomputed_query_vectors=precomputed_query_vectors,
                query_owners=query_owners,
            )

        elapsed_ms = (time.time() - short_term_start) * 1000
        logger.info(
            f"Short-term context fetch complete: elapsed_ms={round(elapsed_ms, 1)}, "
            f"has_summary={bool(conversation_ctx.summary)}, "
            f"recent_turns={len(conversation_ctx.recent_turns)}, "
            f"extraction_types={list(k for k, v in conversation_ctx.key_extractions.items() if v)}, "
            f"vector_items={len(short_term_items)}"
        )

        return (short_term_items, conversation_ctx)

    async def _fetch_compacted_conversation(
        self,
        conversation_id: str,
    ) -> Optional[str]:
        """
        Fetch compacted conversation from Redis (legacy method).

        DEPRECATED: Use _get_compacted_context() instead.
        Kept for backward compatibility.
        """
        if self.cache is None:
            return None

        try:
            key = f"compacted_conv:{conversation_id}"
            compacted = await self.cache.get(key)
            return compacted
        except Exception as e:
            logger.warning(f"Failed to fetch compacted conversation: {e}")
            return None

    async def _get_compacted_context(
        self,
        conversation_id: str,
        instance_id: str,
        client_id: str = "",
        user_id: str = "",
        customer_id: str = "",
    ) -> Optional[CompactedContext]:
        """The conversation's stored compaction, or None. A read, nothing more.

        Reading never starts a compaction. It used to: from 4 Feb, when the read
        path was the only thing that compacted, a missing summary or one older
        than 15 minutes started one here, whatever the instance's thresholds
        said. The first read of a conversation therefore compacted the user's
        first message alone, and that summary is where the "verify the user's
        identity" asks came from (B1 in docs/PLAYGROUND_AGENT_FIXES_ACTION_ITEMS.md,
        27 of 47 replies in eval run B). Compaction now has its own triggers:
        the message and token thresholds (ConversationTracker), the idle sweep
        and session end. The caller serves this summary, however old, plus the
        turns after it.

        Args:
            conversation_id: Conversation ID
            instance_id / client_id / user_id / customer_id: unused; kept so
                callers and tests keep their signature.

        Returns:
            CompactedContext if one is stored, None otherwise
        """
        if self.compaction_storage is None:
            return None
        try:
            # Cache first, then Postgres. After migration 063 the
            # conversation_id columns are TEXT, so opaque-string conv ids work
            # unchanged.
            return await self.compaction_storage.get_latest(conversation_id)
        except Exception as e:
            logger.warning(f"Failed to get compacted context: {e}")
            return None

    async def on_compaction_complete(self, event) -> None:
        """Callback invoked when a compaction completes successfully.

        Invalidates the L2 user context cache so the next retrieval
        uses the fresh compacted context.
        """
        logger.info(
            f"Fresh compaction for {event.conversation_id} "
            f"(ratio={event.compression_ratio:.2f}), "
            f"invalidating retrieval cache"
        )
        if self.cache is not None:
            try:
                if getattr(event, "user_id", None):
                    l2_key = CacheKeyGenerator.l2_user_context_key(
                        client_id=getattr(event, "client_id", ""),
                        customer_id=getattr(event, "customer_id", "") or "",
                        user_id=event.user_id,
                    )
                    await self.cache.delete(l2_key)
                    logger.debug(f"Invalidated L2 cache key: {l2_key}")
            except Exception as e:
                logger.warning(f"Failed to invalidate retrieval cache: {e}")

    def _format_compacted_for_internal_use(
        self,
        compacted: CompactedContext
    ) -> Dict[str, Any]:
        """
        Format compacted context for internal retrieval use.

        Args:
            compacted: CompactedContext to format

        Returns:
            Dictionary with structured data for retrieval
        """
        return {
            "source": "compacted",
            "conversation_id": compacted.metadata.conversation_id,
            "compaction_id": compacted.metadata.compaction_id,

            # Structured extractions (for merging with long-term)
            # Convert ExtractedFact dataclass objects to dicts for downstream .get() access
            "facts": [f.to_dict() if hasattr(f, 'to_dict') else f for f in getattr(compacted.structured_extractions, "explicit_facts", [])],
            "decisions": [f.to_dict() if hasattr(f, 'to_dict') else f for f in getattr(compacted.structured_extractions, "decisions", [])],
            "preferences": [f.to_dict() if hasattr(f, 'to_dict') else f for f in getattr(compacted.structured_extractions, "user_preferences", [])],
            "constraints": [f.to_dict() if hasattr(f, 'to_dict') else f for f in getattr(compacted.structured_extractions, "constraints", [])],
            "emotional_states": [f.to_dict() if hasattr(f, 'to_dict') else f for f in getattr(compacted.structured_extractions, "emotional_states", [])],

            # Narrative (for context injection)
            "summary": compacted.narrative_summary.overview,
            "current_state": compacted.narrative_summary.current_state,

            # Recent verbatim (always included)
            "recent_messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in compacted.verbatim_recent_messages
            ],

            # Metadata
            "compression_ratio": compacted.metadata.compression_ratio,
            "original_tokens": compacted.metadata.original_token_count,
            "compacted_tokens": compacted.metadata.compacted_token_count,
            "compacted_at": compacted.metadata.created_at.isoformat()
        }

    async def get_context_for_decision(
        self,
        conversation_id: str,
        instance_id: str,
        query: str,
        client_id: str = "",
    ) -> Dict[str, Any]:
        """
        Get context for ContextDecisionAgent to make retrieval decisions.

        Uses compacted context to provide:
        - Conversation objective
        - Current state
        - Key facts
        - Open questions

        Args:
            conversation_id: Conversation ID
            instance_id: Instance ID
            query: User query
            client_id: Client ID for storage lookup

        Returns:
            Dictionary with context for decision making
        """
        compacted = await self._get_compacted_context(
            conversation_id=conversation_id,
            instance_id=instance_id,
            client_id=client_id,
        )

        if compacted:
            return {
                "objective": compacted.conversation_classification.objective,
                "current_state": compacted.narrative_summary.current_state,
                "key_facts": [
                    f.content
                    for f in getattr(compacted.structured_extractions, "explicit_facts", [])[:10]
                ],
                "open_questions": [
                    q.content
                    for q in getattr(compacted.structured_extractions, "open_questions", [])
                ],
                "user_preferences": [
                    p.content
                    for p in getattr(compacted.structured_extractions, "user_preferences", [])
                ],
                "conversation_category": compacted.conversation_classification.primary_category,
                "compression_available": True
            }

        # Fallback without compaction
        return {
            "objective": "Unknown",
            "current_state": {},
            "key_facts": [],
            "open_questions": [],
            "user_preferences": [],
            "conversation_category": "Unknown",
            "compression_available": False
        }

    async def _fetch_from_postgres_short_term(
        self,
        conversation_id: str,
        context_types: List[str],
        instance_id: Optional[str] = None,
    ) -> List[ContextItem]:
        """
        Fetch short-term context from Postgres.

        This data is persisted so SLEEP cycle can:
        - Analyze patterns
        - Elevate important items to long-term memory
        - Clean up stale items
        """
        if self.postgres is None:
            return []

        # Step logging for request
        log_substep(
            parent_step="RetrievalManager.fetch_context",
            substep_name="_fetch_from_postgres_short_term",
            data={
                "conversation_id": conversation_id,
                "context_types": context_types,
                "time_window": "24 hours",
                "limit": 50,
            },
            data_type="Query Parameters",
        )

        logger.info(
            f"retrieval_postgres_short_term_start: conversation_id={conversation_id}, context_types={context_types}"
        )

        try:
            if instance_id:
                sql = """
                    SELECT id, content, context_type, confidence, created_at
                    FROM short_term_context
                    WHERE conversation_id = :conversation_id
                      AND instance_id = CAST(:instance_id AS uuid)
                      AND context_type = ANY(:context_types)
                      AND created_at > NOW() - INTERVAL '24 hours'
                    ORDER BY created_at DESC
                    LIMIT 50
                """
                params = {
                    "conversation_id": conversation_id,
                    "instance_id": instance_id,
                    "context_types": context_types,
                }
            else:
                sql = """
                    SELECT id, content, context_type, confidence, created_at
                    FROM short_term_context
                    WHERE conversation_id = :conversation_id
                      AND context_type = ANY(:context_types)
                      AND created_at > NOW() - INTERVAL '24 hours'
                    ORDER BY created_at DESC
                    LIMIT 50
                """
                params = {
                    "conversation_id": conversation_id,
                    "context_types": context_types,
                }
            rows = await self.postgres.fetch_all(sql, params)

            logger.info(
                f"retrieval_postgres_short_term_fetched: row_count={len(rows)}, rows_preview={log_data_sample(rows, sample_size=2, label='short_term_rows')}"
            )

            items = []
            for row in rows:
                items.append(ContextItem(
                    item_id=str(row["id"]),
                    content=row["content"],
                    context_type=row.get("context_type", "episodes"),
                    source="postgres",
                    confidence=row.get("confidence", 0.5),
                    scope="user",
                    created_at=row.get("created_at"),
                    **_event_fields(row.get),
                    valid_until=_safe_parse_datetime(row.get("valid_until")),
                    temporal_category=row.get("temporal_category"),
                    temporal_confidence=row.get("temporal_confidence", 0.0) or 0.0,
                ))

            # Step logging for response
            log_substep(
                parent_step="RetrievalManager.fetch_context",
                substep_name="_fetch_from_postgres_short_term",
                data={
                    "row_count": len(rows),
                    "items_count": len(items),
                    "items": [
                        {
                            "item_id": item.item_id,
                            "content": truncate_string(item.content, 100),
                            "context_type": item.context_type,
                            "confidence": item.confidence,
                            "created_at": item.created_at.isoformat() if item.created_at and hasattr(item.created_at, 'isoformat') else item.created_at,
                        }
                        for item in items[:5]  # Show first 5 items
                    ],
                },
                data_type="Query Results",
            )

            return items

        except Exception as e:
            logger.warning(f"Failed to fetch from Postgres short-term: {e}")
            return []

    async def _fetch_from_postgres_memories(
        self,
        scope: ScopeHierarchy,
        context_types: List[str],
        time_window_hours: int = 720,  # 30 days default
        limit: int = 50,
    ) -> List[ContextItem]:
        """Fetch long-term memories from Postgres memories table.

        Args:
            scope: Scope hierarchy (client, instance, customer, user)
            context_types: List of memory types to filter (["facts", "preferences", ...])
            time_window_hours: How far back to query (default 30 days)
            limit: Maximum results

        Returns:
            List of ContextItem objects
        """
        if self.postgres is None:
            return []

        # Step logging for request
        log_substep(
            parent_step="RetrievalManager.fetch_context",
            substep_name="_fetch_from_postgres_memories",
            data={
                "scope": {
                    "instance_id": scope.instance_id,
                    "user_id": scope.user_id,
                    "customer_id": scope.customer_id,
                    "client_id": scope.client_id,
                },
                "context_types": context_types,
                "time_window_hours": time_window_hours,
                "limit": limit,
            },
            data_type="Query Parameters",
        )

        logger.info(
            f"retrieval_postgres_memories_start: instance_id={scope.instance_id}, user_id={scope.user_id}, customer_id={scope.customer_id}, context_types={context_types}, time_window_hours={time_window_hours}, limit={limit}"
        )

        # Gate user→customer rollup branches on roll_up_to_customer. When
        # disabled, user-scope callers only see their own user_id rows; the
        # `user_id IS NULL` branches that would include customer-shared rows
        # are stripped. Client-scope (customer_id IS NULL AND user_id IS NULL)
        # rollup is always on — it represents tenant-wide shared knowledge.
        maca = _request_maca(self)
        roll_up = maca.roll_up_to_customer

        try:
            # Scope chain merge: user-scope fetch returns user + customer + client data.
            # Each branch fetches memories at the requested level AND all broader scopes.
            if scope.user_id and scope.customer_id:
                # Full scope chain: user → customer → client
                customer_rollup_branch = (
                    "OR (customer_id = CAST(:customer_id AS uuid) AND user_id IS NULL)"
                    if roll_up else ""
                )
                sql = f"""
                    SELECT id, content, memory_type, confidence, category, subcategory, created_at,
                           valid_from, document_created_at, valid_until, temporal_category, temporal_confidence, source_document_id,
                           event_start, event_end, event_precision,
                           customer_id, user_id, assertion_source
                    FROM memories
                    WHERE client_id = CAST(:client_id AS uuid)
                      AND (
                        (customer_id = CAST(:customer_id AS uuid) AND user_id = CAST(:user_id AS uuid))
                        {customer_rollup_branch}
                        OR (customer_id IS NULL AND user_id IS NULL)
                      )
                      AND memory_type = ANY(:memory_types)
                      AND superseded_by IS NULL
                      AND deprecated_at IS NULL
                      AND deleted_at IS NULL
                      AND created_at > NOW() - MAKE_INTERVAL(hours => :time_window_hours)
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT :limit
                """
                params = {
                    "client_id": str(scope.client_id),
                    "customer_id": str(scope.customer_id),
                    "user_id": str(scope.user_id),
                    "memory_types": context_types,
                    "time_window_hours": time_window_hours,
                    "limit": limit,
                }
            elif scope.user_id:
                # User without customer: user + client scope
                sql = """
                    SELECT id, content, memory_type, confidence, category, subcategory, created_at,
                           valid_from, document_created_at, valid_until, temporal_category, temporal_confidence, source_document_id,
                           event_start, event_end, event_precision,
                           customer_id, user_id, assertion_source
                    FROM memories
                    WHERE client_id = CAST(:client_id AS uuid)
                      AND (
                        user_id = CAST(:user_id AS uuid)
                        OR (customer_id IS NULL AND user_id IS NULL)
                      )
                      AND memory_type = ANY(:memory_types)
                      AND superseded_by IS NULL
                      AND deprecated_at IS NULL
                      AND deleted_at IS NULL
                      AND created_at > NOW() - MAKE_INTERVAL(hours => :time_window_hours)
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT :limit
                """
                params = {
                    "client_id": str(scope.client_id),
                    "user_id": str(scope.user_id),
                    "memory_types": context_types,
                    "time_window_hours": time_window_hours,
                    "limit": limit,
                }
            elif scope.customer_id:
                # Customer scope: customer + client
                sql = """
                    SELECT id, content, memory_type, confidence, category, subcategory, created_at,
                           valid_from, document_created_at, valid_until, temporal_category, temporal_confidence, source_document_id,
                           event_start, event_end, event_precision,
                           customer_id, user_id, assertion_source
                    FROM memories
                    WHERE client_id = CAST(:client_id AS uuid)
                      AND (
                        (customer_id = CAST(:customer_id AS uuid) AND user_id IS NULL)
                        OR (customer_id IS NULL AND user_id IS NULL)
                      )
                      AND memory_type = ANY(:memory_types)
                      AND superseded_by IS NULL
                      AND deprecated_at IS NULL
                      AND deleted_at IS NULL
                      AND created_at > NOW() - MAKE_INTERVAL(hours => :time_window_hours)
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT :limit
                """
                params = {
                    "client_id": str(scope.client_id),
                    "customer_id": str(scope.customer_id),
                    "memory_types": context_types,
                    "time_window_hours": time_window_hours,
                    "limit": limit,
                }
            else:
                # Client-scope only (no customer/user data)
                sql = """
                    SELECT id, content, memory_type, confidence, category, subcategory, created_at,
                           valid_from, document_created_at, valid_until, temporal_category, temporal_confidence, source_document_id,
                           event_start, event_end, event_precision,
                           customer_id, user_id, assertion_source
                    FROM memories
                    WHERE client_id = CAST(:client_id AS uuid)
                      AND customer_id IS NULL AND user_id IS NULL
                      AND memory_type = ANY(:memory_types)
                      AND superseded_by IS NULL
                      AND deprecated_at IS NULL
                      AND deleted_at IS NULL
                      AND created_at > NOW() - MAKE_INTERVAL(hours => :time_window_hours)
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT :limit
                """
                params = {
                    "client_id": str(scope.client_id),
                    "memory_types": context_types,
                    "time_window_hours": time_window_hours,
                    "limit": limit,
                }

            rows = await self.postgres.fetch_all(sql, params)

            logger.info(
                f"retrieval_postgres_memories_fetched: row_count={len(rows)}, rows_preview={log_data_sample(rows, sample_size=2, label='memory_rows')}"
            )

            items = []
            for row in rows:
                items.append(ContextItem(
                    item_id=str(row["id"]),
                    content=row["content"],
                    context_type=row.get("memory_type", "facts"),
                    source="postgres_memories",
                    confidence=row.get("confidence", 0.5),
                    scope=derive_item_scope(
                        user_id=row.get("user_id"),
                        customer_id=row.get("customer_id"),
                    ),
                    created_at=row.get("created_at"),
                    **_event_fields(row.get),
                    valid_until=_safe_parse_datetime(row.get("valid_until")),
                    temporal_category=row.get("temporal_category"),
                    temporal_confidence=row.get("temporal_confidence", 0.0) or 0.0,
                    source_document_id=row.get("source_document_id"),
                ))

            logger.info(
                f"Converted {len(items)} postgres_memories rows to ContextItems: items_preview={log_data_sample([{'content': truncate_string(item.content, 80), 'confidence': item.confidence} for item in items], sample_size=2, label='context_items')}"
            )

            # Step logging for response
            log_substep(
                parent_step="RetrievalManager.fetch_context",
                substep_name="_fetch_from_postgres_memories",
                data={
                    "row_count": len(rows),
                    "items_count": len(items),
                    "items": [
                        {
                            "item_id": item.item_id,
                            "content": truncate_string(item.content, 100),
                            "memory_type": item.context_type,
                            "confidence": item.confidence,
                            "scope": item.scope,
                            "created_at": item.created_at.isoformat() if item.created_at and hasattr(item.created_at, 'isoformat') else item.created_at,
                        }
                        for item in items[:5]  # Show first 5 items
                    ],
                },
                data_type="Query Results",
            )

            return items

        except Exception as e:
            logger.warning(f"Failed to fetch from Postgres memories: {e}")
            return []

    async def persist_short_term_context(
        self,
        conversation_id: str,
        instance_id: str,
        user_id: Optional[str],
        items: List[ContextItem],
    ) -> None:
        """
        Persist short-term context to Postgres for SLEEP cycle.

        Called after retrieval to ensure context is available for
        background processing and potential elevation to long-term.
        """
        if self.postgres is None or not items:
            return

        sql = """
            INSERT INTO short_term_context
            (conversation_id, instance_id, user_id, content, context_type, confidence, source, created_at)
            VALUES (:conversation_id, :instance_id, :user_id, :content, :context_type, :confidence, :source, NOW())
            ON CONFLICT (conversation_id, content_hash)
            DO UPDATE SET confidence = GREATEST(short_term_context.confidence, EXCLUDED.confidence)
        """

        for item in items:
            params = {
                "conversation_id": conversation_id,
                "instance_id": instance_id,
                "user_id": user_id,
                "content": item.content,
                "context_type": item.context_type,
                "confidence": item.confidence,
                "source": item.source,
            }
            try:
                await self.postgres.fetch_all(sql, params)
            except Exception as e:
                logger.warning(f"Failed to persist short-term item: {e}")

    async def fetch_long_term_context(
        self,
        scope: ScopeHierarchy,
        queries: List[str],
        filters: Dict = None,
        mode: str = "fast",
        correlation_id: Optional[str] = None,
        skip_subquery_gen: bool = False,
        precomputed_query_vectors: Optional[List] = None,
        max_results: Optional[int] = None,
        anchor_on_all_queries: bool = False,
        query_owners: Optional[List[str]] = None,
    ) -> List[ContextItem]:
        """
        Fetch historical context from vector/graph stores.

        Unified V2 Architecture:
        1. Generate subqueries (templates for FAST, LLM for ACCURATE)
        2. Batch encode all queries
        3. Run graph traversal + direct similarity search in parallel
        4. Merge and deduplicate results

        Args:
            scope: Resolved scope hierarchy
            queries: Search queries (original or decomposed)
            filters: Type filters, time range, etc.
            mode: "fast" or "accurate"

        Returns:
            List of ContextItem from long-term stores
        """
        filters = filters or {}
        _lt_fetch_start = time.time()
        set_current_span_attributes(
            {
                "synap.long_term.mode": mode,
                "synap.long_term.instance_id": scope.instance_id if scope else None,
                "synap.long_term.query_count": len(queries or []),
            }
        )

        # Load MACA config for this instance
        async with start_span(
            "synap.retrieval.long_term.load_maca_settings",
            {"synap.instance_id": scope.instance_id if scope else None},
        ):
            maca = await self._get_maca_settings(scope.instance_id)

        # Filter context_types by MACA enabled categories
        requested_types = filters.get("types", ["facts", "preferences", "episodes"])
        context_types = [t for t in requested_types if t in maca.enabled_categories]
        if not context_types:
            context_types = maca.enabled_categories  # Fallback to all enabled

        # 1. Generate subqueries (templates for FAST, LLM for ACCURATE)
        #    When skip_subquery_gen=True (called from _fetch_accurate with pre-computed
        #    subqueries), skip the LLM call and use input queries directly.
        # Which caller query each search string belongs to, so an item can say
        # which of the caller's questions found it. ``query_owners`` runs
        # parallel to ``queries``; without it a query owns itself.
        _in_owners = (
            list(query_owners)
            if query_owners is not None and len(query_owners) == len(queries)
            else list(queries)
        )
        if skip_subquery_gen:
            all_queries = list(queries)
        else:
            # Peek at L2 adjacency cache to check which entity types exist.
            # Pure dict lookup (<1µs) — no I/O, no blocking, no thread needed.
            graph_entity_types = None
            if mode == "fast":
                l2_key, _, _, _ = self._adjacency_cache_keys(scope)
                cached_adj = self._adjacency_l2.peek(l2_key)
                if cached_adj is not None:
                    nodes = cached_adj.get("nodes", {})
                    # Iterate 500 nodes × 1 dict.get() each ≈ 50µs — not worth threading
                    graph_entity_types = {n.get("entity_type", "") for n in nodes.values()}

            async with start_span(
                "synap.retrieval.long_term.generate_subqueries",
                {
                    "synap.mode": mode,
                    "synap.initial_query_present": bool(queries[0] if queries else ""),
                },
            ):
                subquery_result = await self._generate_retrieval_subqueries(
                    queries[0] if queries else "",
                    mode,
                    maca_settings=maca,
                    scope=scope,
                    correlation_id=correlation_id,
                    graph_entity_types=graph_entity_types,
                )
            all_queries = subquery_result["queries"]
            # Merge with any additional queries passed by caller (dedup)
            for q in queries[1:]:
                if q and q not in all_queries:
                    all_queries.append(q)

        # Everything the generator wrote came from queries[0], so it carries
        # that query's owner. A string the caller also sent is the caller's.
        _owner_of = {}
        for _q, _o in zip(queries, _in_owners):
            _owner_of.setdefault(_q, _o)
        _first_owner = _in_owners[0] if _in_owners else None
        all_owners = [_owner_of.get(_q, _first_owner) for _q in all_queries]

        # 2. Batch encode all queries. Reuse any vectors the caller already computed
        #    upstream: embeddings are deterministic and per-text, so an identical
        #    query string always yields the identical vector via _batch_encode_queries.
        #    When subquery generation expands the list we encode only the newly-added
        #    strings instead of re-embedding the originals — this removes a second full
        #    encode of the caller's queries on every FAST request. Result-neutral: the
        #    reused vectors are bit-for-bit what a re-encode would produce.
        if (
            precomputed_query_vectors is not None
            and skip_subquery_gen
            and len(precomputed_query_vectors) == len(all_queries)
        ):
            query_vectors = precomputed_query_vectors
        elif (
            precomputed_query_vectors is not None
            and len(precomputed_query_vectors) == len(queries)
        ):
            # Subquery generation produced a longer/reordered all_queries, but the
            # caller's `queries` were already encoded upstream. Map string → vector
            # and encode only the strings we have not seen, preserving all_queries order.
            precomputed_by_query = dict(zip(queries, precomputed_query_vectors))
            missing = list(
                dict.fromkeys(q for q in all_queries if q not in precomputed_by_query)
            )
            if missing:
                async with start_span(
                    "synap.retrieval.long_term.batch_encode_queries",
                    {"synap.query_count": len(missing)},
                ):
                    new_vectors = await self._batch_encode_queries(missing)
                precomputed_by_query.update(zip(missing, new_vectors))
            query_vectors = [precomputed_by_query[q] for q in all_queries]
        else:
            async with start_span(
                "synap.retrieval.long_term.batch_encode_queries",
                {"synap.query_count": len(all_queries)},
            ):
                query_vectors = await self._batch_encode_queries(all_queries)

        # Consolidation J3/F4b: publish ONE tripwire event per fetch, right
        # after query_vectors materialises (whichever of the three branches
        # above produced it). This is the chokepoint every vector search path
        # below shares (quick memory search, direct similarity, questions) —
        # publishing here once instead of inside each arm (previously only
        # _search_questions did this, gated behind ENABLE_QUESTION_EMBEDDINGS)
        # covers all of them from a single call. Uses the primary query and
        # its already-computed embedding — no extra embed call on this path.
        # Fire-and-forget: must never block or fail the answer.
        if all_queries and len(query_vectors) > 0:
            try:
                edition.call("consolidation_tripwire", all_queries, query_vectors, scope)
            except Exception:  # noqa: BLE001 — tripwires never cost retrieval
                pass

        # 3. Run graph traversal + direct similarity search + neo4j in parallel
        # Gate graph traversal on env var (like Neo4j) + MACA mode config
        graph_flag = os.environ.get("ENABLE_GRAPH_TRAVERSAL", "true").lower() in ("true", "1", "yes")
        mode_config = maca.modes.get(mode, {})
        # Graph traversal is an ACCURATE-mode-only feature. Fast fetches must never
        # traverse the graph — this keeps fast low-latency and confines graph
        # cost/latency to accurate, regardless of the MACA per-mode config. The Neo4j
        # discovery paths below are already gated on mode == "accurate"; gating
        # use_graph here makes the adjacency + unified-traversal path consistent so
        # fast can never load adjacency or expand the graph.
        use_graph = (
            graph_flag
            and mode == "accurate"
            and mode_config.get("use_graph_traversal", True)
            and _graph_traversal_instance_allowed(scope.instance_id if scope else "")
        )

        # Pre-load adjacency + compute reverse anchors for ALL modes that use graph.
        # Reverse anchor selection: quick memory search → find entity names in top results.
        # This replaces the broken entity-embedding matmul approach (std=0.04, no signal).
        adjacency = None
        neo4j_task = None
        anchor_names = []

        if use_graph and self.graph is not None:
            # Phase A: Load adjacency + quick memory search IN PARALLEL
            adjacency_task = self._get_adjacency_matrix(scope)
            # The graph is entered from the entities named in each query's
            # nearest memories. Anchoring on the first vector alone meant a
            # second caller query could reach the graph only through the
            # first one's entities, so the caller-query search asks for all.
            _anchor_vectors = (
                list(query_vectors) if anchor_on_all_queries else [query_vectors[0]]
            )
            memory_search_task = self._quick_memory_search(scope, _anchor_vectors, top_k=15)
            adjacency, texts_per_query = await asyncio.gather(
                adjacency_task, memory_search_task, return_exceptions=True,
            )

            if isinstance(adjacency, Exception):
                logger.error(f"Adjacency load failed: {adjacency}")
                adjacency = None
            if isinstance(texts_per_query, Exception):
                logger.warning(f"Quick memory search failed: {texts_per_query}")
                texts_per_query = []

            if adjacency:
                entity_names = [
                    node_data.get("canonical_name", "")
                    for node_data in adjacency.get("nodes", {}).values()
                    if node_data.get("canonical_name")
                ]

                anchor_names = self._anchors_across_queries(texts_per_query, entity_names)

                if not anchor_names:
                    anchor_names = self._compute_anchor_names(adjacency, query_vectors)
                    logger.info(f"Reverse anchor: 0 matches, matmul fallback: {anchor_names[:5]}")
                else:
                    logger.info(f"Reverse anchor selected: {anchor_names[:10]}")

                # "Who do I report to?": the asker's own node. No memory text
                # carries its name, so the reverse match above cannot find it.
                from synap.cloud.agents.ingestion.services import writer_entity
                writer_anchor = writer_entity.anchor_for(queries, scope, entity_names)
                if writer_anchor and writer_anchor not in anchor_names:
                    anchor_names = [writer_anchor] + list(anchor_names)
                # "What plan are we on?": the asker's customer node, likewise.
                customer_anchor = writer_entity.customer_anchor_for(queries, scope, entity_names)
                if customer_anchor and customer_anchor not in anchor_names:
                    anchor_names = [customer_anchor] + list(anchor_names)

        # Fire Neo4j early (accurate mode only, gated by feature flag)
        neo4j_flag = os.environ.get("ENABLE_NEO4J_TRAVERSAL", "true").lower() in ("true", "1", "yes")
        if mode == "accurate" and neo4j_flag and self.graph is not None and use_graph and anchor_names:
            logger.info(
                f"Neo4j pre-computed anchors: anchor_count={len(anchor_names)}, anchor_names={anchor_names[:10]}"
            )

            neo4j_task = self._neo4j_graph_discovery(
                scope=scope,
                anchor_names=anchor_names[:10],
                query_vectors=query_vectors,
                max_depth=3,
                query_owners=all_owners,
            )

        # The unified walk over the Chroma adjacency is no longer run. Adjacency and
        # the anchors chosen above are kept because Neo4j discovery starts from
        # them; the walk's own items (labelled "graph") are not produced. The
        # anchors are handed back where the walk used to return its own, so the
        # sequential Neo4j fallback below still has them.
        _walk_anchors = list(anchor_names) if (use_graph and anchor_names) else []

        async def _no_walk():
            return [], _walk_anchors
        graph_task = _no_walk()

        # Breadth follows what the caller asked for; see derive_search_breadth.
        # Memory vectors and HQE both read it, so top-K scales them together.
        _breadth = derive_search_breadth(
            max_results or self.config.max_vector_results,
            floor=self.config.max_vector_results,
        )
        similarity_task = self._direct_similarity_search(
            scope, all_queries, query_vectors, context_types,
            top_k=_breadth,
            query_owners=all_owners,
        )

        # Sentence/chunk search disabled — returned irrelevant generic assistant content.
        # Code kept in _search_sentences() for future re-enable after content quality fix.
        async def _noop_chunks():
            return []
        chunk_task = _noop_chunks()

        # HQE: Hypothetical Question Embeddings search
        question_task = self._search_questions(
            scope=scope,
            queries=all_queries,
            query_vectors=query_vectors,
            top_k=_breadth,
            query_owners=all_owners,
        )

        # Run all tasks in parallel: graph + vector + chunks + questions + neo4j (if available)
        async with start_span(
            "synap.retrieval.long_term.parallel_search",
            {
                "synap.graph_enabled": use_graph,
                "synap.neo4j_parallel": neo4j_task is not None,
                "synap.query_count": len(all_queries),
                "synap.context_type_count": len(context_types),
            },
        ):
            if neo4j_task is not None:
                graph_result, similarity_items, chunk_items, neo4j_result, question_items = await asyncio.gather(
                    self._timed_arm("graph", graph_task, query_count=len(all_queries), correlation_id=correlation_id),
                    self._timed_arm("vector", similarity_task, query_count=len(all_queries), correlation_id=correlation_id),
                    self._timed_arm("chunk", chunk_task, query_count=len(all_queries), correlation_id=correlation_id),
                    self._timed_arm("neo4j", neo4j_task, query_count=len(all_queries), correlation_id=correlation_id),
                    self._timed_arm("question", question_task, query_count=len(all_queries), correlation_id=correlation_id),
                    return_exceptions=True,
                )
            else:
                graph_result, similarity_items, chunk_items, question_items = await asyncio.gather(
                    self._timed_arm("graph", graph_task, query_count=len(all_queries), correlation_id=correlation_id),
                    self._timed_arm("vector", similarity_task, query_count=len(all_queries), correlation_id=correlation_id),
                    self._timed_arm("chunk", chunk_task, query_count=len(all_queries), correlation_id=correlation_id),
                    self._timed_arm("question", question_task, query_count=len(all_queries), correlation_id=correlation_id),
                    return_exceptions=True,
                )
                neo4j_result = None

        # Handle exceptions gracefully
        # graph_result is Tuple[List[ContextItem], List[str]] — (items, bfs_anchor_names)
        if isinstance(graph_result, Exception):
            logger.error(f"Graph traversal failed: {graph_result}")
            graph_items = []
            bfs_anchor_names = []
        elif isinstance(graph_result, tuple):
            graph_items, bfs_anchor_names = graph_result
        else:
            graph_items = graph_result
            bfs_anchor_names = []
        if isinstance(similarity_items, Exception):
            logger.error(f"Similarity search failed: {similarity_items}")
            similarity_items = []
        if isinstance(chunk_items, Exception):
            logger.error(f"Chunk search failed: {chunk_items}")
            chunk_items = []
        if isinstance(question_items, Exception):
            logger.error(f"Question search failed: {question_items}")
            question_items = []

        all_items = list(graph_items) + list(similarity_items) + list(chunk_items) + list(question_items)

        # Handle Neo4j result (parallel or fallback)
        if isinstance(neo4j_result, Exception):
            logger.error(f"Neo4j graph discovery failed: {neo4j_result}", exc_info=True)
        elif neo4j_result is not None:
            all_items.extend(neo4j_result)
            logger.info(
                f"Neo4j graph discovery complete (parallel): neo4j_items={len(neo4j_result)}"
            )

        # Fallback: accurate mode but no pre-computed anchors — try BFS anchors sequentially
        if mode == "accurate" and neo4j_flag and self.graph is not None and neo4j_task is None:
            try:
                anchor_names = bfs_anchor_names[:10]
                anchor_source = "bfs_anchors"

                if not anchor_names:
                    anchor_names = list(set(
                        item.entity_name for item in graph_items
                        if hasattr(item, "entity_name") and item.entity_name
                    ))[:10]
                    anchor_source = "graph_items"

                if not anchor_names:
                    anchor_names = list(set(
                        item.entity_name for item in similarity_items
                        if hasattr(item, "entity_name") and item.entity_name
                    ))[:10]
                    anchor_source = "similarity_items"

                logger.info(
                    f"Neo4j anchor extraction (fallback): anchor_name_count={len(anchor_names)}, anchor_names={anchor_names[:10]}, source={anchor_source}"
                )

                if anchor_names:
                    async with start_span(
                        "synap.retrieval.long_term.neo4j_discovery",
                        {
                            "synap.anchor_count": len(anchor_names),
                            "synap.fallback": True,
                        },
                    ):
                        neo4j_items = await self._neo4j_graph_discovery(
                            scope=scope,
                            anchor_names=anchor_names,
                            query_vectors=query_vectors,
                            max_depth=3,
                            query_owners=all_owners,
                        )
                    all_items.extend(neo4j_items)
                    logger.info(
                        f"Neo4j graph discovery complete (fallback): neo4j_items={len(neo4j_items)}, anchor_names={anchor_names[:5]}"
                    )
                else:
                    logger.warning(
                        "Neo4j skipped: no entity names found in vector/graph results"
                    )
            except Exception as e:
                logger.error(f"Neo4j graph discovery failed: {e}", exc_info=True)

        logger.info(
            f"merge_pre_dedup: graph_items={len(graph_items)}, vector_items={len(similarity_items)}, neo4j_parallel={neo4j_task is not None}, combined_total={len(all_items)}"
        )

        # 4. Deduplicate (keep highest score per item)
        deduped = self._deduplicate_items(all_items)
        set_current_span_attributes(
            {
                "synap.long_term.result_count_pre_dedup": len(all_items),
                "synap.long_term.result_count": len(deduped),
            }
        )
        logger.info(
            f"long_term_fetch timing: elapsed_ms={round((time.time() - _lt_fetch_start) * 1000, 1)}, "
            f"query_count={len(queries or [])}, mode={mode}, correlation_id={correlation_id}, "
            f"first_query={(queries[0][:80] if queries else '')!r}"
        )
        return deduped

    async def _unified_graph_traversal(
        self,
        scope: ScopeHierarchy,
        queries: List[str],
        query_vectors: List[np.ndarray],
        context_types: List[str],
        max_hops: int = 2,
        adjacency: Optional[Dict] = None,
        reverse_anchor_names: Optional[List[str]] = None,
        query_owners: Optional[List[str]] = None,
    ) -> Tuple[List[ContextItem], List[str]]:
        """
        Unified V2 vector-guided graph traversal.

        Combines anchor search, BFS traversal, and hybrid edge scoring
        using pre-computed numpy matrices for vectorized operations.

        For each query vector:
        1. Vectorized anchor search via entity_matrix @ query_vector
        2. Soft anchor injection (scope root with ego-intent scoring)
        3. BFS traversal with hybrid edge scoring (0.7×edge + 0.3×target)
        4. Boomerang pruning, cycle prevention, deduplication

        Returns:
            Tuple of (items, bfs_anchor_names) where bfs_anchor_names are the
            canonical names of BFS starting entities (for Neo4j anchor selection).
        """
        if self.vector is None:
            logger.info("Unified graph traversal skipped: no vector provider")
            return [], []

        items: List[ContextItem] = []
        all_anchor_names: List[str] = []
        traversal_start = time.time()
        query_count = len(query_vectors)

        async with start_span(
            "synap.retrieval.graph_traversal",
            {
                "synap.instance_id": str(scope.instance_id),
                "synap.user_id": str(scope.user_id) if scope.user_id else "",
                "synap.query_count": query_count,
                "synap.context_types": ",".join(context_types),
                "synap.max_hops": max_hops,
            },
        ):
            logger.info(
                f"Unified graph traversal starting: instance_id={scope.instance_id}, query_count={len(queries)}, context_types={context_types}, max_hops={max_hops}, queries_preview={[q[:80] for q in queries[:3]]}"
            )

            try:
                async with start_span("synap.retrieval.graph_traversal.load_adjacency", {
                    "synap.instance_id": str(scope.instance_id),
                    "synap.pre_loaded": adjacency is not None,
                }):
                    if adjacency is None:
                        adjacency = await self._get_adjacency_matrix(scope)
                    entity_ids = adjacency.get("entity_ids", [])
                    entity_matrix = adjacency.get("entity_matrix")
                    set_current_span_attributes({
                        "synap.entity_count": len(entity_ids),
                        "synap.adjacency_loaded": entity_matrix is not None,
                    })

                if entity_matrix is None or len(entity_ids) == 0:
                    logger.info(
                        f"Unified graph traversal: no adjacency data: entity_matrix_exists={entity_matrix is not None}, entity_ids_count={len(entity_ids)}"
                    )
                    set_current_span_attributes({
                        "synap.skipped": True,
                        "synap.skip_reason": "no_adjacency_data",
                        "synap.traversal_latency_ms": round((time.time() - traversal_start) * 1000, 2),
                    })
                    _run_detached(
                        self._track_graph_queries(
                            scope=scope,
                            query_count=query_count,
                            operation="retrieval.unified_graph_traversal",
                            latency_ms=(time.time() - traversal_start) * 1000,
                        )
                    )
                    return [], []

                logger.info(
                    f"Graph adjacency loaded: entity_count={len(entity_ids)}, matrix_shape={list(entity_matrix.shape) if hasattr(entity_matrix, 'shape') else 'unknown'}"
                )

                total_anchors_found = 0
                nodes_dict = adjacency.get("nodes", {})

                for idx, query_vector in enumerate(query_vectors):
                    query_text = queries[idx] if idx < len(queries) else ""

                    if reverse_anchor_names:
                        # --- Reverse Anchor Selection ---
                        # Resolve pre-computed anchor names to entity IDs
                        name_to_id = {}
                        for eid in entity_ids:
                            node_data = nodes_dict.get(eid, {})
                            cn = node_data.get("canonical_name", "")
                            if cn:
                                name_to_id[cn.lower()] = eid
                        anchor_ids = []
                        for name in reverse_anchor_names[:10]:
                            eid = name_to_id.get(name.lower())
                            if eid:
                                anchor_ids.append(eid)
                        anchor_method = "reverse"
                    else:
                        anchor_ids = []
                        anchor_method = "reverse"

                    if not anchor_ids:
                        # Fallback: original matmul approach
                        scores = entity_matrix @ query_vector
                        top_k_a = min(10, len(entity_ids))
                        top_indices = np.argpartition(scores, -top_k_a)[-top_k_a:]
                        anchors = [(entity_ids[i], float(scores[i])) for i in top_indices]
                        anchors.sort(key=lambda x: x[1], reverse=True)
                        _ANCHOR_MIN_SCORE = 0.30
                        anchors = [a for a in anchors if a[1] >= _ANCHOR_MIN_SCORE]
                        anchor_ids = [a[0] for a in anchors[:10]]
                        anchor_method = "matmul_fallback"

                    total_anchors_found += len(anchor_ids)

                    for aid in anchor_ids:
                        node_data = nodes_dict.get(aid, {})
                        name = node_data.get("canonical_name", "")
                        if name:
                            all_anchor_names.append(name)

                    # Log anchor info
                    _anchor_details = []
                    for _aid in anchor_ids[:10]:
                        _anode = nodes_dict.get(_aid, {})
                        _aname = _anode.get("canonical_name", _anode.get("name", "?"))
                        _atype = _anode.get("entity_type", "?")
                        _anchor_details.append(f"{_aname}({_atype})")
                    logger.info(
                        f"SCORING_LOG anchor_selection: method={anchor_method}, query='{query_text[:80]}', anchors=[{', '.join(_anchor_details)}]"
                    )

                    # --- BFS Traversal with vectorized scoring ---
                    # Run sync BFS in a thread so it doesn't block the event
                    # loop (vector + BM25 ChromaDB responses need the loop).
                    async with start_span("synap.retrieval.graph_traversal.bfs", {
                        "synap.query_idx": idx,
                        "synap.anchor_count": len(anchor_ids),
                        "synap.max_hops": max_hops,
                    }):
                        traversal_items = await asyncio.to_thread(
                            self._traverse_graph,
                            adjacency,
                            anchor_ids,
                            query_vector,
                            context_types,
                            max_hops,
                        )
                        # The walk is scored against this query's vector, so
                        # what it returns is this query's.
                        _g_owner = _owner_at(query_owners, idx)
                        for _g_item in traversal_items:
                            _note_query(_g_item, _g_owner)
                        items.extend(traversal_items)
                        set_current_span_attributes({
                            "synap.items_found": len(traversal_items),
                        })

                elapsed_ms = (time.time() - traversal_start) * 1000
                set_current_span_attributes({
                    "synap.items_returned": len(items),
                    "synap.queries_processed": len(query_vectors),
                    "synap.total_anchors": total_anchors_found,
                    "synap.traversal_latency_ms": round(elapsed_ms, 2),
                })
                logger.info(
                    f"Unified graph traversal complete: elapsed_ms={round(elapsed_ms, 1)}, items_returned={len(items)}, queries_processed={len(query_vectors)}, context_type_breakdown={ {ct: sum(1 for i in items if i.context_type == ct) for ct in set(i.context_type for i in items)} if items else {} }, score_range={f'{min(i.similarity_score for i in items):.3f}-{max(i.similarity_score for i in items):.3f}' if items else 'n/a'}"
                )
                # Detached: keep the unified-graph arm's telemetry commit off
                # the response path (elapsed_ms already captured above).
                _run_detached(
                    self._track_graph_queries(
                        scope=scope,
                        query_count=query_count,
                        operation="retrieval.unified_graph_traversal",
                        latency_ms=elapsed_ms,
                    )
                )

            except Exception as e:
                elapsed_ms = (time.time() - traversal_start) * 1000
                set_current_span_attributes({
                    "synap.error": True,
                    "synap.error_type": type(e).__name__,
                    "synap.traversal_latency_ms": round(elapsed_ms, 2),
                })
                logger.error(
                    f"Unified graph traversal failed: {e}, elapsed_ms={round(elapsed_ms, 1)}",
                    exc_info=True,
                )
                _run_detached(
                    self._track_graph_queries(
                        scope=scope,
                        query_count=query_count,
                        operation="retrieval.unified_graph_traversal",
                        latency_ms=elapsed_ms,
                        status="error",
                    )
                )

        unique_anchor_names = list(dict.fromkeys(all_anchor_names))
        return items, unique_anchor_names

    def _compute_anchor_names(
        self,
        adjacency: Dict,
        query_vectors: List[np.ndarray],
    ) -> List[str]:
        """Fast anchor search: entity_matrix @ query_vectors -> top entity names.

        Pure numpy computation (~1ms). Used to fire Neo4j discovery
        in parallel with graph traversal + vector search.
        """
        entity_ids = adjacency.get("entity_ids", [])
        entity_matrix = adjacency.get("entity_matrix")
        nodes_dict = adjacency.get("nodes", {})

        if entity_matrix is None or len(entity_ids) == 0:
            return []

        all_names = []
        for query_vector in query_vectors:
            scores = entity_matrix @ query_vector
            top_k = min(10, len(entity_ids))
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            for i in sorted(top_indices, key=lambda x: scores[x], reverse=True)[:10]:
                name = nodes_dict.get(entity_ids[i], {}).get("canonical_name", "")
                if name:
                    all_names.append(name)

        return list(dict.fromkeys(all_names))  # dedupe preserving order

    async def _quick_memory_search(
        self,
        scope: ScopeHierarchy,
        query_vectors: List[np.ndarray],
        top_k: int = 15,
    ) -> List[List[str]]:
        """Lightweight ChromaDB memory search returning top-k memory texts per query.

        Used for reverse anchor extraction — find which entity names appear
        in the most semantically relevant memories. Not used for final results.
        One store call for all the vectors; the result keeps one list per
        vector, in order, so anchors can be chosen per query.
        """
        if self.vector is None or not query_vectors:
            return []

        try:
            from synap.cloud.agents.ingestion.models import Scope as IngestionScope
            from synap.cloud.shared.scope import parse_or_deterministic_uuid

            vector_scope = IngestionScope(
                client_id=parse_or_deterministic_uuid(scope.client_id),
                instance_id=parse_or_deterministic_uuid(scope.instance_id),
                customer_id=parse_or_deterministic_uuid(scope.customer_id) if scope.customer_id else None,
                user_id=parse_or_deterministic_uuid(scope.user_id) if scope.user_id else None,
            )

            embs = [
                v.tolist() if hasattr(v, "tolist") else list(v)
                for v in query_vectors
            ]
            maca = _request_maca(self)
            vis = _request_visibility(self)
            if vis is not VISIBILITY_ALL and not vis:
                return []
            # The anchors must come from memories this caller may read. An
            # unfiltered search let another instance's or another user's memory
            # choose which entities seed the graph walk.
            #
            # Same two regimes as the main memory search, for the same reason:
            # the default keeps the scope predicate in Python (~30ms against
            # ~840ms store-side), pins the instance at the store when a
            # visibility policy applies, and repairs a starved query exactly.
            nested_where = _nested_scope_where(scope, maca, vis)
            if nested_where is _NOTHING_READABLE:
                return []
            _exact_quick = _build_customer_scope_filter(scope, maca, vis)
            quick_pred = None
            pool_k = top_k
            if nested_where is not None:
                quick_filters = nested_where
            elif _use_python_post_filter_for(nested_where):
                pool_k = top_k * _POST_FILTER_OVERFETCH_MULT
                quick_pred = _scope_filter_predicate(scope, maca, vis)
                # No store-side filter, not even the instance clause. This
                # search is a cheap unfiltered query (7ms) on the critical path
                # of every accurate fetch, and pinning the instance made it a
                # filtered one (330ms), twice per request. The
                # predicate above already enforces instance visibility on what
                # comes back, and a pool the predicate starves is repaired
                # below with the exact filter, so isolation does not depend on
                # the store clause. The main memory search pins the instance
                # because it is one filtered query among several already.
                quick_filters = None
            else:
                quick_filters = _exact_quick

            batch_results = await self.vector.batch_search(
                embeddings=embs,
                scope=vector_scope,
                top_k=pool_k,
                filters=quick_filters,
                scope_level="customer",
            )
            if quick_pred is not None:
                _raw_quick = batch_results
                batch_results = _apply_post_filter(
                    batch_results, scope_predicate=quick_pred
                )

                async def _refetch_exact_quick(idx):
                    return await self.vector.batch_search(
                        embeddings=[embs[i] for i in idx],
                        scope=vector_scope,
                        top_k=top_k,
                        filters=_exact_quick or None,
                        scope_level="customer",
                    )

                batch_results = await _refetch_starved(
                    _raw_quick, batch_results,
                    pool_k=pool_k, want=top_k,
                    refetch=_refetch_exact_quick, label="quick_memory",
                )
            batch_results = [r[:top_k] for r in batch_results]

            texts_per_query: List[List[str]] = []
            for result_list in batch_results:
                texts = []
                for result in result_list:
                    content = result.get("content", "")
                    if content:
                        texts.append(content)
                texts_per_query.append(texts)
            return texts_per_query

        except Exception as e:
            logger.warning(f"Quick memory search failed: {e}")
            return []

    def _anchors_across_queries(
        self,
        texts_per_query: List[List[str]],
        entity_names: List[str],
        max_anchors: int = 10,
    ) -> List[str]:
        """Pick graph anchors from each query's memories in turn.

        Pooling every query's memories and ranking by mention count would let
        one entity-heavy query take all ``max_anchors`` places. Each query's
        own ranking is taken round-robin instead, so every query that names
        an entity gets into the graph. One query: its ranking, unchanged.
        """
        ranked = [
            self._match_entities_in_memories(texts, entity_names, max_anchors)
            for texts in texts_per_query
        ]
        if len(ranked) <= 1:
            return ranked[0] if ranked else []
        anchors: List[str] = []
        for rank in range(max_anchors):
            for names in ranked:
                if rank < len(names) and names[rank] not in anchors:
                    anchors.append(names[rank])
                    if len(anchors) >= max_anchors:
                        return anchors
        return anchors

    def _match_entities_in_memories(
        self,
        memory_texts: List[str],
        entity_names: List[str],
        max_anchors: int = 10,
    ) -> List[str]:
        """Find entity names mentioned in memory texts, ranked by mention count.

        Reverse anchor selection: instead of matching query→entity embeddings
        (broken for short labels), match query→memories (works well) then
        memories→entities (text match).
        """
        if not memory_texts or not entity_names:
            return []

        combined_text = " ".join(memory_texts).lower()

        counts = []
        for name in entity_names:
            if len(name) < 3:
                continue
            count = combined_text.count(name.lower())
            if count > 0:
                counts.append((name, count))

        counts.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in counts[:max_anchors]]

    async def _direct_similarity_search(
        self,
        scope: ScopeHierarchy,
        queries: List[str],
        query_vectors: List[np.ndarray],
        context_types: List[str],
        top_k: int = 25,
        query_owners: Optional[List[str]] = None,
    ) -> List[ContextItem]:
        """
        Direct ChromaDB similarity search for memories.

        Complements graph traversal by catching isolated memories
        not linked to entities in the graph.
        """
        if self.vector is None:
            logger.info("Vector similarity search skipped: no vector provider")
            return []

        items: List[ContextItem] = []
        search_start = time.time()
        # Diagnostic timing split (behavior-preserving): separate Chroma I/O from
        # Python-side post-filter and ContextItem build. Pre-initialized so every
        # code path (including the except-branch fallbacks) logs a valid number.
        chroma_ms = 0.0
        postfilter_ms = 0.0
        build_ms = 0.0
        total_off_chain = 0
        # id -> index in `items`, for the nested-path dedup below.
        _seen_item_idx: Dict[str, int] = {}
        query_count = len([q for q in queries if q])

        async with start_span(
            "synap.retrieval.vector_search",
            {
                "synap.instance_id": str(scope.instance_id),
                "synap.user_id": str(scope.user_id) if scope.user_id else "",
                "synap.query_count": query_count,
                "synap.context_types": ",".join(context_types),
                "synap.top_k": top_k,
            },
        ):
            try:
                from synap.cloud.agents.ingestion.models import Scope as IngestionScope
                from synap.cloud.shared.scope import parse_or_deterministic_uuid

                vector_scope = IngestionScope(
                    client_id=parse_or_deterministic_uuid(scope.client_id),
                    instance_id=parse_or_deterministic_uuid(scope.instance_id),
                    customer_id=parse_or_deterministic_uuid(scope.customer_id) if scope.customer_id else None,
                    user_id=parse_or_deterministic_uuid(scope.user_id) if scope.user_id else None,
                )

                logger.info(
                    f"Vector similarity search starting: client_id={scope.client_id}, customer_id={scope.customer_id}, user_id={scope.user_id}, query_count={len(queries)}, context_types={context_types}, top_k={top_k}, queries_preview={[q[:80] for q in queries[:3]]}"
                )

                total_raw_results = 0
                total_filtered_out = 0

                maca = _request_maca(self)
                vis = _request_visibility(self)
                if vis is not VISIBILITY_ALL and not vis:
                    return []
                base_vector_filters = {
                    **_memory_type_where(context_types),
                    **_build_customer_scope_filter(scope, maca),
                }

                # Batch all query embeddings into a single ChromaDB call
                valid_indices = [idx for idx, q in enumerate(queries) if q]
                embeddings_list = []
                for idx in valid_indices:
                    emb = query_vectors[idx] if idx < len(query_vectors) else await self._get_embedding(queries[idx])
                    embeddings_list.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))

                # When SYNAP_VECTOR_POST_FILTER is on (default true), skip the
                # Chroma `where` clause and apply the equivalent predicate in
                # Python on the returned metadata. See the helper definitions
                # above for the measurement that motivated this — Chroma's
                # in-search filter costs ~840ms per call at our scale while the
                # Python predicate is sub-millisecond on the same result set.
                #
                # Overfetch the per-call top_k so the wider raw HNSW result
                # still contains at least `top_k` items that pass the Python
                # filter. The 3× default was validated by id-overlap probe
                # (A∩B = 25/25 across three collections).
                # Two regimes for the post-filter optimization:
                #
                # (a) Full-alphabet (F=1.0 on memory_type): the caller asked
                #     for every memory_type value, so the memory_type filter
                #     is a no-op. Drop ALL filters at Chroma, overfetch by
                #     _POST_FILTER_OVERFETCH_MULT, post-filter scope in Python.
                #     ~30ms per call vs ~840ms — measured 22-31× speedup.
                #
                # (b) Selective memory_type (F<1.0): caller asked for a subset
                #     of types. Skipping the memory_type filter would lose
                #     high-similarity items beyond the overfetch window
                #     (correctness regression). Keep memory_type at Chroma,
                #     do scope filter in Python. ~225-530ms per call depending
                #     on |context_types|, still exact, no quality loss.
                #
                # Scope clauses (customer_id, user_id, sentinel) are always
                # done Python-side — their pass-rate is ≈ 1.0 by construction
                # in our deployment (collection name pins customer; B2C has
                # user_id == customer_id) so the in-search cost is pure waste.
                # Nested read (S-07). Built whole, because it can contain a
                # `$or` that would not survive being merged into the flat dicts
                # the branches below assemble. None for every request that has
                # not resolved a chain, which leaves those branches untouched.
                _nested_where = _nested_scope_where(scope, maca, vis, context_types)
                if _nested_where is _NOTHING_READABLE:
                    return []

                # R-02 and R-10. The chain drives ranking, and on a chain deeper
                # than one rung it also buys a second, narrow fetch. Both are
                # None / (None, 0) for every request that has not resolved a
                # chain, which leaves everything below untouched (SHIP-04).
                _chain_ids = _nested_chain_ids(scope)
                _cost = _per_level_cost(maca)
                # The caller deliberately widened past their own chain. See
                # `_closeness_scored`: off-chain records are then legitimate
                # rather than a pool-builder bug, and dropping them makes the
                # result depend on how far the backfill has got.
                _widened = bool(
                    maca.allow_cross_user_retrieval
                    or maca.allow_cross_customer_retrieval
                )
                _leaf_where, _leaf_top_k = (
                    _nested_leaf_reserve(scope, maca, vis, context_types, top_k)
                    if _nested_where is not None
                    else (None, 0)
                )

                use_post_filter = _use_python_post_filter_for(_nested_where)
                full_alphabet = _is_full_memory_type_alphabet(context_types)

                if _nested_where is not None:
                    _chroma_filters_customer = _nested_where
                    # The client-level collection is already scoped to the
                    # client by its name, and its records carry no customer or
                    # user id at all, so the chain predicate would match none of
                    # them. Only the type clause applies here, exactly as today.
                    _chroma_filters_client = (
                        None if full_alphabet else _memory_type_where(context_types)
                    )
                    # ⚠ The SAME candidate count as the legacy path, deliberately.
                    #
                    # This asked for `top_k` while the shipped default asks for
                    # `top_k * 3` and then post-filters at a pass rate of about
                    # 1.0, so roughly 75 candidates reach the ranker there and
                    # 25 would reach it here. Switching the feature on would
                    # have shrunk the pool threefold.
                    #
                    # The reasoning that produced the smaller number was that
                    # overfetch exists to survive post-filtering, and nothing is
                    # post-filtered on this path. True of the mechanism, false
                    # of the effect: the overfetch is what currently puts 75 in
                    # front of the ranker, and the closeness pass reorders that
                    # pool, so more candidates is strictly more to reorder.
                    # Found by audit 2026-08-23.
                    #
                    # Only bites on collections holding more than `top_k`
                    # records, so it misses a typical small collection and hits
                    # the larger tenants, which are the ones with deep ladders.
                    _chroma_top_k_customer = top_k * _POST_FILTER_OVERFETCH_MULT
                    _chroma_top_k_client = max(3, top_k // 3) * _POST_FILTER_OVERFETCH_MULT
                    # Two guards say the same thing: no Python scope predicate
                    # here, and `use_post_filter` above is already False. Either
                    # one alone makes the other redundant, and that is on
                    # purpose. `_scope_filter_predicate` only knows customer_id
                    # and user_id, so a record filed at a team, carrying node_id
                    # and neither of those, fails it. The store has already
                    # applied the chain predicate exactly; running the legacy
                    # predicate over the result would drop precisely the records
                    # nested scoping exists to return, and it would look like
                    # they were never written. Removing both is what breaks it,
                    # which is the case test_read_path_golden.py pins.
                    _scope_pred = None
                    _post_filter_context_types = None
                elif not use_post_filter:
                    # Disabled — original behavior, all filtering at Chroma.
                    _chroma_filters_customer = base_vector_filters
                    _chroma_filters_client = _memory_type_where(context_types)
                    _chroma_top_k_customer = top_k
                    _chroma_top_k_client = max(3, top_k // 3)
                    _scope_pred = None
                    _post_filter_context_types = None
                elif full_alphabet:
                    # Regime (a): post-filter EVERYTHING in Python, overfetch.
                    _chroma_filters_customer = None
                    _chroma_filters_client = None
                    _chroma_top_k_customer = top_k * _POST_FILTER_OVERFETCH_MULT
                    _chroma_top_k_client = max(3, top_k // 3) * _POST_FILTER_OVERFETCH_MULT
                    _scope_pred = _scope_filter_predicate(scope, maca, vis)
                    # context_types check is a no-op when full alphabet but
                    # we leave it in for defense-in-depth (skip-cost negligible).
                    _post_filter_context_types = context_types
                else:
                    # Regime (b): keep memory_type at Chroma (no overfetch
                    # needed — Chroma returns exact matches), scope in Python.
                    _chroma_filters_customer = _memory_type_where(context_types)
                    _chroma_filters_client = _memory_type_where(context_types)
                    _chroma_top_k_customer = top_k
                    _chroma_top_k_client = max(3, top_k // 3)
                    _scope_pred = _scope_filter_predicate(scope, maca, vis)
                    # Chroma already enforced memory_type; skip in post-filter
                    # to avoid redundant work.
                    _post_filter_context_types = None

                # Cross-instance isolation: when a visibility policy restricts
                # this viewer, pin instance_id in the CUSTOMER-scope Chroma
                # where-clause (exact — avoids the post-filter overfetch window
                # starving on a selective allowed-set). Client-scope shared
                # knowledge is never instance-filtered. The Python predicate
                # above already mirrors this for the customer task.
                # Skipped for the nested path: _nested_scope_where already
                # carries the instance clause, and merging it again would put a
                # second `instance_id` key alongside a `$or` at the same level,
                # which is not a where-clause Chroma accepts.
                if _nested_where is None and vis is not VISIBILITY_ALL:
                    _inst_clause = _visibility_where(vis)
                    _chroma_filters_customer = (
                        {**_chroma_filters_customer, **_inst_clause}
                        if _chroma_filters_customer
                        else dict(_inst_clause)
                    )

                # Scope chain merge: search customer-level + client-level collections in parallel
                client_scope_filters = _memory_type_where(context_types)
                search_tasks = []

                # The exact (store-side) form of the customer-collection filter,
                # used only to repair a query the post-filter starved.
                _exact_customer_filters = dict(base_vector_filters)
                if vis is not VISIBILITY_ALL:
                    _exact_customer_filters.update(_visibility_where(vis))

                async def _refetch_exact_memories(idx):
                    return await self.vector.batch_search(
                        embeddings=[embeddings_list[i] for i in idx],
                        scope=vector_scope,
                        top_k=top_k,
                        filters=_exact_customer_filters,
                        scope_level="customer",
                    )

                # 1. Customer-level collection (primary)
                search_tasks.append(
                    self.vector.batch_search(
                        embeddings=embeddings_list,
                        scope=vector_scope,
                        top_k=_chroma_top_k_customer,
                        filters=_chroma_filters_customer,
                        scope_level="customer",
                    )
                )
                # 2. Client-level collection (shared knowledge)
                search_tasks.append(
                    self.vector.batch_search(
                        embeddings=embeddings_list,
                        scope=vector_scope,
                        top_k=_chroma_top_k_client,
                        filters=_chroma_filters_client,
                        scope_level="client",
                    )
                )
                # 3. R-10. The requester's own node, asked for separately, so it
                # is in the pool at all. Only on a chain deeper than one rung,
                # and it reads the same collection as task 1 with a narrower
                # predicate, so its results are scored the same way.
                if _leaf_where is not None:
                    search_tasks.append(
                        self.vector.batch_search(
                            embeddings=embeddings_list,
                            scope=vector_scope,
                            top_k=_leaf_top_k,
                            filters=_leaf_where,
                            scope_level="customer",
                        )
                    )

                # Which task a record came from is the only thing that can say
                # what rung it sits on, for the client-level collection at least
                # (R-15), so it is recorded here rather than inferred later.
                _CLIENT_TASK_IDX = 1

                try:
                    _chroma_start = time.time()
                    scope_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                    chroma_ms = (time.time() - _chroma_start) * 1000
                    _postfilter_start = time.time()
                    all_search_results = []
                    for task_idx, res in enumerate(scope_results):
                        if isinstance(res, Exception):
                            # A collection that does not exist is not a failure.
                            #
                            # Each rung is searched in its own collection, and a
                            # collection is created on the first write to that
                            # rung. A client who has never written client-level
                            # knowledge has no client-level collection, so this
                            # arm raises NotFoundError, finds nothing, and the
                            # read is correct without it.
                            #
                            # It was logged at warning on every request, which buries the real
                            # failures this line exists to surface. Verified
                            # before downgrading rather than assumed: a client
                            # with 25 client-level memories in Postgres has that
                            # collection with exactly 25 in it, and a client
                            # with none gets this error. So the naming is right
                            # and the absence is data, not a fault.
                            if type(res).__name__ == "NotFoundError":
                                logger.debug(
                                    "scope chain: no collection at this rung yet "
                                    "(nothing has been written there): %s", res,
                                )
                            else:
                                logger.warning(f"Vector scope chain sub-task failed: {res}")
                                # This arm's memories were not searched. Say so
                                # in the response, not only here.
                                record_retrieval_fault(
                                    f"vector search (scope rung {task_idx})", res)
                            continue
                        if use_post_filter:
                            # task_idx 0 = customer-scope (apply full scope predicate)
                            # task_idx 1 = client-scope (no per-customer/user filter; client
                            # collection has no per-customer metadata to filter on)
                            _raw_res = res
                            res = _apply_post_filter(
                                res,
                                scope_predicate=_scope_pred if task_idx == 0 else None,
                                context_types=_post_filter_context_types,
                            )
                            if task_idx == 0:
                                res = await _refetch_starved(
                                    _raw_res, res,
                                    pool_k=_chroma_top_k_customer,
                                    want=top_k,
                                    refetch=_refetch_exact_memories,
                                    label="memories",
                                )
                        # R-15. Client-level records carry no ids at all, so the
                        # only place that knows their rung is here. The dicts are
                        # built fresh by the vector wrapper on every call, so
                        # marking them is local and cannot leak into a cache.
                        if _chain_ids and task_idx == _CLIENT_TASK_IDX:
                            for _batch in res:
                                if isinstance(_batch, Exception):
                                    continue
                                for _rec in _batch:
                                    if isinstance(_rec, dict):
                                        _rec["_synap_from_client_scope"] = True
                        all_search_results.extend(res)
                    postfilter_ms = (time.time() - _postfilter_start) * 1000
                except Exception:
                    # Ultimate fallback: single customer-level query
                    try:
                        all_search_results = await self.vector.batch_search(
                            embeddings=embeddings_list,
                            scope=vector_scope,
                            top_k=_chroma_top_k_customer,
                            filters=_chroma_filters_customer,
                        )
                        if use_post_filter:
                            all_search_results = _apply_post_filter(
                                all_search_results,
                                scope_predicate=_scope_pred,
                                context_types=_post_filter_context_types,
                            )
                    except (AttributeError, NotImplementedError):
                        async def _search_one(embedding_list):
                            return await self.vector.search(
                                embedding=embedding_list,
                                scope=vector_scope,
                                top_k=_chroma_top_k_customer,
                                filters=_chroma_filters_customer,
                            )
                        all_search_results = await asyncio.gather(
                            *[_search_one(emb) for emb in embeddings_list],
                            return_exceptions=True,
                        )
                        if use_post_filter:
                            all_search_results = _apply_post_filter(
                                all_search_results,
                                scope_predicate=_scope_pred,
                                context_types=_post_filter_context_types,
                            )

                _build_start = time.time()
                from dateutil import parser as dateutil_parser
                # Every search task returns one result list per embedding, in
                # embedding order, and the tasks are laid end to end. So a
                # list's position, modulo the embedding count, is the query
                # that produced it. If the shape is anything else, nothing is
                # attributed rather than attributed wrongly.
                _n_emb = len(embeddings_list)
                _owners_ok = bool(
                    query_owners and _n_emb and len(all_search_results) % _n_emb == 0
                )
                for _pos, batch_results in enumerate(all_search_results):
                    _owner = (
                        _owner_at(query_owners, valid_indices[_pos % _n_emb])
                        if _owners_ok else None
                    )
                    if isinstance(batch_results, Exception):
                        logger.warning(f"Vector search sub-task failed: {batch_results}")
                        continue

                    total_raw_results += len(batch_results)

                    for result in batch_results:
                        meta = result.get("metadata", {})

                        # Prefer document_created_at (original conversation time) over
                        # created_at (extraction/server time) so downstream LLMs see
                        # when the information originated, not when we processed it.
                        created_at = None
                        raw_ts = meta.get("document_created_at") or meta.get("created_at")
                        if raw_ts:
                            try:
                                created_at = dateutil_parser.isoparse(raw_ts)
                            except Exception:
                                pass
                        # KU v2 lossless-merge: latest contributor DCT, present
                        # only on merged memories (equals created_at otherwise).
                        last_observed_at = _safe_parse_datetime(
                            meta.get("document_last_observed_at")
                        )

                        score = result.get("score", 0.0)
                        # The floor is applied to the RAW score, before
                        # closeness. It is a relevance gate on what the store
                        # returned; applying it after would make it
                        # depth-dependent and drop records for being deep, which
                        # is the permission-and-ranking blend R-07 forbids.
                        if score < min_similarity():
                            total_filtered_out += 1
                            continue
                        _rel = _closeness_scored(
                            score,
                            meta,
                            _chain_ids,
                            _cost,
                            from_client_scope=bool(
                                result.get("_synap_from_client_scope")
                            ),
                            widened=_widened,
                        )
                        if _rel is None:
                            # R-08 says the pool builder made this impossible,
                            # so a non-zero count here means something upstream
                            # is wrong, not that a caller was refused.
                            total_off_chain += 1
                            continue

                        # R-09, and only on the nested path. The reserve reads
                        # the same collection as the main fetch with a narrower
                        # predicate, so anything it finds that the main fetch
                        # also found arrives twice. Two copies of one record is
                        # not a second candidate: it spends the caller's token
                        # budget twice and inflates every count downstream.
                        #
                        # Deliberately conditional on there being a chain. The
                        # legacy path already returns cross-embedding duplicates
                        # and has always relied on the dedup in bundle assembly
                        # to collapse them; changing that here would be a
                        # behaviour change for every request that has nothing to
                        # do with nested scoping (SHIP-04).
                        _rid = result.get("id")
                        if _chain_ids and _rid:
                            _prev = _seen_item_idx.get(_rid)
                            if _prev is not None:
                                # Keep the better score. The two copies normally
                                # agree, since it is one record scored against
                                # one query, but a store that reports distance
                                # for one shape and similarity for another would
                                # not, and demoting is recoverable where
                                # dropping is not.
                                if _rel > items[_prev].relevance_score:
                                    items[_prev].relevance_score = _rel
                                    items[_prev].similarity_score = score
                                _note_query(items[_prev], _owner)
                                continue
                            _seen_item_idx[_rid] = len(items)
                        items.append(ContextItem(
                            item_id=result.get("id", str(uuid.uuid4())),
                            content=result.get("content", ""),
                            context_type=meta.get("memory_type", meta.get("context_type", "facts")),
                            source="vector",
                            # similarity stays the store's own number; relevance
                            # carries the distance term, because that is the
                            # field `_compute_relevance_score` prefers and so the
                            # one the final sort actually reads.
                            similarity_score=score,
                            relevance_score=_rel,
                            confidence=meta.get("confidence", 0.5),
                            scope=meta.get("scope") or derive_item_scope(
                                user_id=meta.get("user_id"),
                                customer_id=meta.get("customer_id"),
                            ),
                            entity_id=meta.get("entity_id"),
                            created_at=created_at,
                            **_event_fields(meta.get),
                            valid_until=_safe_parse_datetime(meta.get("valid_until")),
                            temporal_category=meta.get("temporal_category"),
                            temporal_confidence=float(meta.get("temporal_confidence", 0)) if meta.get("temporal_confidence") is not None else 0.0,
                            source_document_id=meta.get("source_document_id"),
                            document_last_observed_at=last_observed_at,
                        ))
                        _note_query(items[-1], _owner)

                build_ms = (time.time() - _build_start) * 1000

                elapsed_ms = (time.time() - search_start) * 1000
                set_current_span_attributes({
                    "synap.raw_results": total_raw_results,
                    "synap.filtered_out": total_filtered_out,
                    "synap.items_returned": len(items),
                    "synap.search_latency_ms": round(elapsed_ms, 2),
                    "synap.chroma_ms": round(chroma_ms, 2),
                    "synap.postfilter_ms": round(postfilter_ms, 2),
                    "synap.build_ms": round(build_ms, 2),
                    "synap.tasks_run": len(valid_indices),
                    "synap.off_chain_dropped": total_off_chain,
                    "synap.leaf_reserve_used": bool(_leaf_where),
                })
                # Detailed vector scoring log
                if items:
                    _vec_details = []
                    for _vi in sorted(items, key=lambda x: x.similarity_score, reverse=True):
                        _vec_details.append(f"sim={round(_vi.similarity_score, 3)} type={_vi.context_type} '{_vi.content[:70]}'")
                    logger.info(
                        f"SCORING_LOG vector_results: count={len(items)}, filtered_out={total_filtered_out}, items=[{' | '.join(_vec_details)}]"
                    )
                logger.info(
                    f"Vector similarity search complete: elapsed_ms={round(elapsed_ms, 1)}, raw_results={total_raw_results}, items_returned={len(items)}"
                )
                logger.info(
                    f"Vector timing breakdown: elapsed_ms={round(elapsed_ms, 1)}, "
                    f"chroma_ms={round(chroma_ms, 1)}, postfilter_ms={round(postfilter_ms, 1)}, "
                    f"build_ms={round(build_ms, 1)}, "
                    f"in_memory_ms={round(postfilter_ms + build_ms, 1)}, "
                    f"raw_results={total_raw_results}, items_returned={len(items)}"
                )
                # Detached: telemetry INSERT+commit must not sit on the vector
                # arm's response path (latency_ms is already captured above).
                _run_detached(
                    self._track_vector_queries(
                        scope=scope,
                        query_count=query_count,
                        operation="retrieval.direct_similarity_search",
                        latency_ms=elapsed_ms,
                    )
                )

            except Exception as e:
                elapsed_ms = (time.time() - search_start) * 1000
                set_current_span_attributes({
                    "synap.error": True,
                    "synap.error_type": type(e).__name__,
                    "synap.search_latency_ms": round(elapsed_ms, 2),
                })
                logger.error(
                    f"Direct similarity search failed: {e}, elapsed_ms={round(elapsed_ms, 1)}"
                )
                _run_detached(
                    self._track_vector_queries(
                        scope=scope,
                        query_count=query_count,
                        operation="retrieval.direct_similarity_search",
                        latency_ms=elapsed_ms,
                        status="error",
                    )
                )

        return items

    async def _search_questions(
        self,
        scope: ScopeHierarchy,
        queries: List[str],
        query_vectors: List[np.ndarray],
        top_k: int = 10,
        query_owners: Optional[List[str]] = None,
    ) -> List[ContextItem]:
        """Search the hypothetical-questions collection (HQE) for memory discovery.

        The search itself is the ``retrieval_question_search`` hook: Cloud
        registers it, Core does not have HQE, so there the hook is empty and
        this returns no items.
        """
        if self.vector is None:
            return []
        found = await edition.acall(
            "retrieval_question_search", self,
            scope=scope, queries=queries, query_vectors=query_vectors,
            top_k=top_k, query_owners=query_owners,
        )
        return found or []

    # ── Sentence search constants ──
    SENTENCE_SCORE_THRESHOLD = 0.55   # min cosine for inclusion (sentence-level has wider spread than chunks)
    SENTENCE_SECOND_THRESHOLD = 0.60  # 2nd+ result needs higher bar to justify a slot
    SENTENCE_MAX_RESULTS = 3          # max sentences to return (each is ~60-90 tokens with context window)
    SENTENCE_MAX_TOTAL_TOKENS = 400   # budget cap for all sentence results combined

    async def _search_sentences(
        self,
        scope: ScopeHierarchy,
        queries: List[str],
        query_vectors: List[np.ndarray],
        top_k: int = 30,
    ) -> List[ContextItem]:
        """Search pre-indexed sentence-level embeddings from synap_{client_id}_sentences.

        Sentences are pre-embedded at ingestion/backfill time. Each document in the
        sentences collection stores a 3-sentence context window as text, but was embedded
        using only the center sentence — giving precise cosine matching with readable context.

        No ONNX, no sentence splitting, no matmul at query time. Just vector search.
        """
        if self.vector is None:
            return []

        search_start = time.time()
        items: List[ContextItem] = []

        try:
            from synap.cloud.agents.ingestion.models import Scope as IngestionScope
            from synap.cloud.shared.scope import parse_or_deterministic_uuid

            vector_scope = IngestionScope(
                client_id=parse_or_deterministic_uuid(scope.client_id),
                instance_id=parse_or_deterministic_uuid(scope.instance_id),
                customer_id=parse_or_deterministic_uuid(scope.customer_id) if scope.customer_id else None,
                user_id=parse_or_deterministic_uuid(scope.user_id) if scope.user_id else None,
            )

            # Build embeddings list from precomputed query vectors
            valid_indices = [idx for idx, q in enumerate(queries) if q]
            embeddings_list = []
            for idx in valid_indices:
                emb = query_vectors[idx] if idx < len(query_vectors) else await self._get_embedding(queries[idx])
                embeddings_list.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))

            if not embeddings_list:
                return []

            maca = _request_maca(self)
            vis = _request_visibility(self)
            if vis is not VISIBILITY_ALL and not vis:
                return []
            sent_filters = _build_customer_scope_filter(scope, maca)

            # Post-filter in Python (see helpers above for measurement +
            # rationale). Sentences collection has no `memory_type` metadata
            # so we only apply the scope predicate.
            # Nested read (S-07). Sentences carry no memory_type either.
            _s_nested = _nested_scope_where(scope, maca, vis)
            if _s_nested is _NOTHING_READABLE:
                return []

            # R-02 and R-10. Sentences read one collection, not two, so R-15's
            # client-scope case cannot arise here and no record is ever tagged.
            _chain_ids = _nested_chain_ids(scope)
            _cost = _per_level_cost(maca)
            _widened = bool(
                maca.allow_cross_user_retrieval or maca.allow_cross_customer_retrieval
            )
            _s_off_chain = 0
            _s_leaf_where, _s_leaf_top_k = (
                _nested_leaf_reserve(scope, maca, vis, None, top_k)
                if _s_nested is not None
                else (None, 0)
            )

            use_post_filter = _use_python_post_filter_for(_s_nested)

            if _s_nested is not None:
                _s_filter = _s_nested
                _s_top_k = top_k
                _s_scope_pred = None
            else:
                _s_filter = None if use_post_filter else sent_filters
                _s_top_k = (top_k * _POST_FILTER_OVERFETCH_MULT) if use_post_filter else top_k
                _s_scope_pred = _scope_filter_predicate(scope, maca, vis) if use_post_filter else None

            # Cross-instance isolation: pin instance_id in the customer-scope
            # sentences Chroma filter for exactness. Skipped for the nested
            # path, which already carries it.
            if _s_nested is None and vis is not VISIBILITY_ALL:
                _s_inst_clause = _visibility_where(vis)
                _s_filter = {**_s_filter, **_s_inst_clause} if _s_filter else dict(_s_inst_clause)

            # Exact (store-side) filter, used only to repair a starved query.
            _s_exact_filters = dict(sent_filters) if sent_filters else {}
            if vis is not VISIBILITY_ALL:
                _s_exact_filters.update(_visibility_where(vis))

            async def _refetch_exact_sentences(idx):
                return await self.vector.batch_search_sentences(
                    embeddings=[embeddings_list[i] for i in idx],
                    scope=vector_scope,
                    top_k=top_k,
                    filters=_s_exact_filters or None,
                )

            try:
                all_results = await self.vector.batch_search_sentences(
                    embeddings=embeddings_list,
                    scope=vector_scope,
                    top_k=_s_top_k,
                    filters=_s_filter,
                )
                if use_post_filter:
                    _raw_results = all_results
                    all_results = _apply_post_filter(
                        all_results,
                        scope_predicate=_s_scope_pred,
                        context_types=None,
                    )
                    all_results = await _refetch_starved(
                        _raw_results, all_results,
                        pool_k=_s_top_k,
                        want=top_k,
                        refetch=_refetch_exact_sentences,
                        label="sentences",
                    )
                # R-10. The reserve's overlap with the main fetch needs no
                # special handling here: `sent_map` below already keys on the
                # sentence id and keeps the better score.
                if _s_leaf_where is not None:
                    all_results = list(all_results) + list(
                        await self.vector.batch_search_sentences(
                            embeddings=embeddings_list,
                            scope=vector_scope,
                            top_k=_s_leaf_top_k,
                            filters=_s_leaf_where,
                        )
                    )
            except (AttributeError, NotImplementedError) as e:
                logger.info(f"Sentence search unavailable (method not supported): {e}")
                return []
            except Exception as e:
                logger.warning(f"Sentence search failed: {type(e).__name__}: {e}", exc_info=True)
                return []

            # Flatten + dedup by sentence ID (keep highest score across query variants)
            sent_map: Dict[str, Dict] = {}
            for batch_results in all_results:
                if isinstance(batch_results, Exception):
                    continue
                for result in batch_results:
                    sid = result.get("id", "")
                    score = result.get("score", 0.0)
                    if score < 0.25:
                        continue
                    if sid not in sent_map or score > sent_map[sid].get("score", 0.0):
                        sent_map[sid] = result

            if not sent_map:
                elapsed_ms = (time.time() - search_start) * 1000
                logger.info(f"Sentence search: no results above min threshold, elapsed_ms={elapsed_ms:.1f}")
                return []

            # Sort by score descending
            sorted_sents = sorted(sent_map.values(), key=lambda s: s.get("score", 0.0), reverse=True)
            best_score_overall = sorted_sents[0].get("score", 0.0) if sorted_sents else 0.0

            # R-02 here needs care, because this path gates on score and then
            # BREAKS out of a list sorted by score. Closeness has to change the
            # ORDER without changing which sentences are eligible, so the two
            # score thresholds are applied first, against the raw score, exactly
            # as the breaks below did. Because the list is sorted descending,
            # "break at the first one under X" and "keep everything at or above
            # X" select the same set, so this is the same rule written as a
            # filter. Then, and only then, the survivors are re-ordered by
            # distance.
            _floor = self.SENTENCE_SCORE_THRESHOLD
            _eligible = []
            for _rank, _sd in enumerate(sorted_sents):
                _raw = _sd.get("score", 0.0)
                if _raw < _floor:
                    break
                if _rank >= 1 and _raw < self.SENTENCE_SECOND_THRESHOLD:
                    break
                _rel = _closeness_scored(_raw, _sd.get("metadata", {}), _chain_ids,
                                         _cost, widened=_widened)
                if _rel is None:
                    _s_off_chain += 1
                    continue
                _sd["_synap_rel"] = _rel
                _eligible.append(_sd)
            if _chain_ids:
                _eligible.sort(key=lambda s: s.get("_synap_rel", 0.0), reverse=True)
            sorted_sents = _eligible

            # Apply threshold, dedup overlapping windows, build ContextItems
            from dateutil import parser as dateutil_parser
            counter = TokenCounter()
            total_tokens = 0

            for rank, sent_data in enumerate(sorted_sents):
                score = sent_data.get("score", 0.0)
                # Both score thresholds were applied above, against the raw
                # score, before the closeness re-order. Re-applying them here
                # would break out of a list that is no longer sorted by score.
                if len(items) >= self.SENTENCE_MAX_RESULTS:
                    break

                content = sent_data.get("content", "")
                if not content:
                    continue

                # Token budget check
                ptokens = counter.count(content)
                if total_tokens + ptokens > self.SENTENCE_MAX_TOTAL_TOKENS and items:
                    break

                # Dedup: skip if overlapping with already selected
                is_dup = False
                for prev_item in items:
                    if _jaccard_similarity(content, prev_item.content) > 0.5:
                        is_dup = True
                        break
                if is_dup:
                    continue

                total_tokens += ptokens
                meta = sent_data.get("metadata", {})

                created_at = None
                if meta.get("created_at"):
                    try:
                        created_at = dateutil_parser.isoparse(meta["created_at"])
                    except Exception:
                        pass

                items.append(ContextItem(
                    item_id=sent_data.get("id", str(uuid.uuid4())),
                    content=content,
                    context_type="facts",
                    source="chunks",
                    similarity_score=score,
                    relevance_score=sent_data.get("_synap_rel", score),
                    confidence=min(1.0, 0.5 + (score - self.SENTENCE_SCORE_THRESHOLD)),
                    scope=meta.get("scope") or derive_item_scope(
                        user_id=meta.get("user_id"),
                        customer_id=meta.get("customer_id"),
                    ),
                    created_at=created_at,
                    **_event_fields(meta.get),
                    valid_until=_safe_parse_datetime(meta.get("valid_until")),
                    temporal_category=meta.get("temporal_category"),
                    temporal_confidence=float(meta.get("temporal_confidence", 0)) if meta.get("temporal_confidence") is not None else 0.0,
                    source_document_id=meta.get("source_document_id"),
                    document_last_observed_at=_safe_parse_datetime(
                        meta.get("document_last_observed_at")
                    ),
                ))

            elapsed_ms = (time.time() - search_start) * 1000
            logger.info(
                f"Sentence search: elapsed_ms={elapsed_ms:.1f}, "
                f"off_chain_dropped={_s_off_chain}, "
                f"candidates={len(sent_map)}, best_score={best_score_overall:.3f}, "
                f"returned={len(items)}, "
                f"preview=[{', '.join(f'score={it.relevance_score:.3f} content={it.content[:100]!r}' for it in items)}]"
            )

        except Exception as e:
            elapsed_ms = (time.time() - search_start) * 1000
            logger.error(f"Sentence search failed: {e}, elapsed_ms={elapsed_ms:.1f}", exc_info=True)

        return items


    async def _search_short_term_vectors(
        self,
        scope: ScopeHierarchy,
        search_queries: List[str],
        top_k: int = 10,
        score_threshold: Optional[float] = None,   # None -> min_similarity()
        precomputed_query_vectors: Optional[List] = None,
        query_owners: Optional[List[str]] = None,
    ) -> List[ContextItem]:
        """
        Search the short-term vector collection for relevant session messages.

        Queries synap_{client_id}_{customer_id}_shortterm and converts results to ContextItems
        so they enter the standard scoring/dedup/filtering pipeline.
        """
        if self.vector is None:
            return []

        if not scope.customer_id:
            logger.debug("Short-term vector search skipped: no customer_id in scope")
            return []

        items: List[ContextItem] = []
        namespace = f"synap_{scope.customer_id}_st_v2"
        search_start = time.time()

        try:
            from synap.providers.vector.models import VectorQuery
            from dateutil import parser as dateutil_parser

            seen_ids = set()
            _st_by_id: Dict[str, ContextItem] = {}
            # With no owners given, a query owns what it finds.
            _st_owners = query_owners if query_owners is not None else search_queries

            for idx, query in enumerate(search_queries):
                if not query:
                    continue
                _owner = _owner_at(_st_owners, idx)

                if precomputed_query_vectors is not None and idx < len(precomputed_query_vectors):
                    embedding = precomputed_query_vectors[idx]
                else:
                    embedding = await self._get_embedding(query)
                embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

                # Customer and user, exactly as before. Short-term turns are not
                # long-term memories: the long-term builder's roll-up and
                # cross-user escape hatches describe a different collection and
                # would change which turns a caller sees.
                st_filters = {}
                if scope.customer_id:
                    st_filters["customer_id"] = scope.customer_id
                if scope.user_id:
                    st_filters["user_id"] = scope.user_id

                # The namespace is customer-wide, not instance-wide, and every
                # turn carries the instance that wrote it. When a visibility
                # policy applies, an isolated instance may read only the turns
                # of the instances it is allowed to see. Without this clause it
                # read every sibling's recent turns whenever no nested ladder
                # was active, which is every request today.
                _st_maca = _request_maca(self)
                _st_vis = _request_visibility(self)
                if _st_vis is not VISIBILITY_ALL:
                    if not _st_vis:
                        return items
                    st_filters.update(_visibility_where(_st_vis))

                # S-07, the same predicate the long-term collections compile.
                #
                # Short-term was the two-level filter and nothing else, so a
                # turn recorded at a team was read back by every sibling team
                # under the same tenant: they all produce customer_id=<tenant>,
                # user_id=<sentinel>, and they share one collection. The
                # long-term read has honoured the chain since S-07 and this did
                # not, so the same request saw a rung on one collection and the
                # whole tenant on another.
                #
                # `_nested_scope_where` returns None when there is no chain,
                # which is every request today, so the filter below is
                # unchanged for clients on three rungs. It is the same
                # function, not a second copy: S-08 exists because two
                # predicates written separately drift.
                _st_nested = _nested_scope_where(scope, _st_maca, _st_vis)
                if _st_nested is _NOTHING_READABLE:
                    # Nothing on this request's chain is readable. Returning
                    # the two-level filter here would serve the whole tenant,
                    # which is the opposite of what was asked for.
                    return items
                if _st_nested:
                    st_filters = _st_nested

                result = await asyncio.to_thread(
                    self.vector._provider.search,
                    VectorQuery(
                        embedding=embedding_list,
                        top_k=top_k,
                        namespace=namespace,
                        filters=st_filters,
                    ),
                )

                for r in result.results:
                    if r.id in seen_ids:
                        # A later query found a message an earlier one did.
                        if r.id in _st_by_id:
                            _note_query(_st_by_id[r.id], _owner)
                        continue
                    seen_ids.add(r.id)

                    score = r.score
                    if score < (score_threshold if score_threshold is not None else min_similarity()):
                        continue

                    meta = r.metadata or {}

                    created_at = None
                    if meta.get("created_at"):
                        try:
                            created_at = dateutil_parser.isoparse(meta["created_at"])
                        except Exception:
                            pass

                    items.append(ContextItem(
                        item_id=r.id,
                        content=r.content or "",
                        context_type=meta.get("memory_type", "temporal_events"),
                        source="vector",
                        similarity_score=score,
                        relevance_score=score,
                        confidence=meta.get("confidence", 0.7),
                        scope=meta.get("scope") or derive_item_scope(
                            user_id=meta.get("user_id"),
                            customer_id=meta.get("customer_id"),
                        ),
                        entity_id=None,
                        created_at=created_at,
                        **_event_fields(meta.get),
                        valid_until=_safe_parse_datetime(meta.get("valid_until")),
                        temporal_category=meta.get("temporal_category"),
                        temporal_confidence=float(meta.get("temporal_confidence", 0)) if meta.get("temporal_confidence") is not None else 0.0,
                        source_document_id=meta.get("source_document_id"),
                        document_last_observed_at=_safe_parse_datetime(
                            meta.get("document_last_observed_at")
                        ),
                    ))
                    _note_query(items[-1], _owner)
                    _st_by_id[r.id] = items[-1]

            elapsed_ms = (time.time() - search_start) * 1000
            logger.info(
                f"Short-term vector search complete: elapsed_ms={round(elapsed_ms, 1)}, "
                f"namespace={namespace}, queries={len(search_queries)}, "
                f"items_returned={len(items)}"
            )

        except Exception as e:
            elapsed_ms = (time.time() - search_start) * 1000
            logger.warning(
                f"Short-term vector search failed (non-fatal): {e}, "
                f"elapsed_ms={round(elapsed_ms, 1)}, namespace={namespace}"
            )

        return items

    async def _resolve_neo4j_by_name(
        self,
        namespace: str,
        entity_names: List[str],
    ) -> Dict[str, List[str]]:
        """Map entity canonical_names to Neo4j node IDs.

        Bridges ChromaDB → Neo4j via canonical_name (the shared semantic key).
        ChromaDB entity doc IDs and Neo4j entity_id properties are independent
        UUID spaces with zero overlap, so name-based resolution is required.

        Returns ALL matching neo4j_ids per name because entity duplication exists
        in the graph (ingestion creates separate nodes per batch). Each copy may
        have unique edges, so all must be traversed for complete discovery.

        Args:
            namespace: Neo4j namespace (customer_id or client_id)
            entity_names: Entity canonical names to resolve

        Returns:
            Dict mapping canonical_name → list of neo4j_node_ids
        """
        if not entity_names or self.graph is None:
            return {}

        cypher = """
            MATCH (n:SynapNode {_namespace: $ns})
            WHERE n.canonical_name IN $names
            RETURN n.canonical_name AS name, n.id AS neo4j_id
        """
        params = {"ns": namespace, "names": entity_names}

        try:
            records = await self.graph.run_query(cypher, params)
            result: Dict[str, List[str]] = {}
            for r in records:
                name = r.get("name")
                neo4j_id = r.get("neo4j_id")
                if name and neo4j_id:
                    result.setdefault(name, []).append(neo4j_id)

            logger.info(
                f"Neo4j name resolution complete: names_requested={len(entity_names)}, names_resolved={len(result)}, total_neo4j_ids={sum(len(ids) for ids in result.values())}, names_with_duplicates={ {name: len(ids) for name, ids in result.items() if len(ids) > 1} }"
            )
            return result
        except Exception as e:
            logger.error(f"Neo4j name resolution failed: {e}", exc_info=True)
            return {}

    async def _timed_arm(self, arm_name, coro, *, query_count=0, correlation_id=None):
        """Diagnostic: wall-clock an individual parallel-search arm. Logging only."""
        _arm_start = time.time()
        try:
            return await coro
        finally:
            logger.info(
                f"parallel_arm_timing: arm={arm_name}, query_count={query_count}, "
                f"correlation_id={correlation_id}, "
                f"elapsed_ms={round((time.time() - _arm_start) * 1000, 1)}"
            )

    async def _node_referenced_memories(self, scope, nodes, seen_memory_ids, avg_query, gate, own,
                                        edges=()):
        """The memories the reached *nodes* and *edges* point to, as (similarity, ContextItem).

        Read from the customer's memories collection by id, so a superseded
        memory (its vector is deleted) is not returned, and *gate*, the same
        scope predicate the walk applies to a memory on a relationship, is
        asked of each one's own metadata. [] when the flag is off, when nothing
        is referenced, or on any failure: this adds to the walk, never gates it.
        """
        from synap.cloud.agents.ingestion.services import node_memories
        _nodes_on = node_memories.enabled(scope.instance_id)
        _edges_on = node_memories.edges_enabled(scope.instance_id)
        if not (_nodes_on or _edges_on) or self.vector is None:
            return []

        def _closest_first(records):
            def closeness(record):
                emb = (record.properties or {}).get("embedding")
                if emb is None:
                    return 0.0
                vec = np.array(emb, dtype=np.float32)
                return float(np.dot(avg_query, vec / (np.linalg.norm(vec) + 1e-9)))
            return sorted(records, key=closeness, reverse=True)

        ids = node_memories.wanted(
            _closest_first(nodes) if _nodes_on else [], seen_memory_ids,
            edges=_closest_first(edges) if _edges_on else [])
        if not ids:
            return []
        try:
            from synap.cloud.agents.ingestion.models import Scope as IngestionScope
            from synap.cloud.shared.scope import parse_or_deterministic_uuid
            vector_scope = IngestionScope(
                client_id=parse_or_deterministic_uuid(scope.client_id),
                instance_id=parse_or_deterministic_uuid(scope.instance_id),
                customer_id=parse_or_deterministic_uuid(scope.customer_id) if scope.customer_id else None,
                user_id=parse_or_deterministic_uuid(scope.user_id) if scope.user_id else None,
            )
            namespace = self.vector._get_namespace(vector_scope, "customer")
            documents = await asyncio.to_thread(self.vector._provider.get, namespace, ids)
            # The client's shared graph points into the client's collection.
            _found = {d.id for d in documents}
            _rest = [i for i in ids if i not in _found]
            _client_ids = set()
            if _rest:
                try:
                    _shared = await asyncio.to_thread(
                        self.vector._provider.get, self.vector._get_namespace(vector_scope, "client"), _rest)
                    _client_ids = {d.id for d in _shared}
                    documents = list(documents) + list(_shared)
                except Exception:  # noqa: BLE001 - no client collection yet
                    pass
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Neo4j discovery: node memory references failed (non-fatal): {e}")
            return []

        # These come from the memories collection, so unlike a memory copied
        # onto a relationship they carry a node_id and can be demoted by
        # distance like every other vector read (R-18).
        maca = _request_maca(self)
        _chain_ids = _nested_chain_ids(scope)
        _cost = _per_level_cost(maca)
        _widened = bool(maca.allow_cross_user_retrieval or maca.allow_cross_customer_retrieval)

        out = []
        for doc in documents:
            meta = doc.metadata or {}
            # A row of the client's collection carries no customer or user by
            # construction; the walk does not gate that graph either.
            if gate is not None and doc.id not in _client_ids and not gate(meta):
                continue
            content = (doc.content or "").split("\n\n[Context:", 1)[0]
            if not content or doc.embedding is None or len(doc.embedding) == 0:
                continue
            vec = np.array(doc.embedding, dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            similarity = float(np.dot(avg_query, vec))
            relevance = _closeness_scored(similarity, meta, _chain_ids, _cost, widened=_widened)
            if relevance is None:
                continue
            own(doc.id, vec)
            out.append((relevance, ContextItem(
                item_id=doc.id,
                content=content,
                context_type=meta.get("memory_type", "facts"),
                source="neo4j",
                similarity_score=similarity,
                relevance_score=relevance,
                confidence=meta.get("confidence", 0.7),
                scope=derive_item_scope(user_id=meta.get("user_id"), customer_id=meta.get("customer_id")),
                **_event_fields(meta.get),
                valid_until=_safe_parse_datetime(meta.get("valid_until")),
                temporal_category=meta.get("temporal_category"),
                temporal_confidence=float(meta.get("temporal_confidence") or 0.0),
                graph_kind="memory",
                backing_memory_ids=[doc.id],
            )))
        return out

    async def _neo4j_graph_discovery(
        self,
        scope: ScopeHierarchy,
        anchor_names: List[str],
        query_vectors: List[np.ndarray],
        max_depth: int = 3,
        query_owners: Optional[List[str]] = None,
    ) -> List[ContextItem]:
        """Real Neo4j graph traversal for ACCURATE mode.

        Uses combined name resolution + traversal in a single Cypher query
        (saves one cross-region round-trip vs separate resolve + traverse).
        Scores discovered nodes using embeddings stored on Neo4j nodes,
        and extracts memory_contents from relationship edge properties.

        Args:
            scope: Tenant scope
            anchor_names: Entity canonical names from BFS/vector results
            query_vectors: Pre-computed query vectors for scoring
            max_depth: Maximum traversal depth (default 3)

        Returns:
            List of scored ContextItems with source="neo4j"
        """
        if self.graph is None or not anchor_names:
            return []

        # Cross-instance visibility. Graph nodes and relationships carry the list
        # of instances that wrote them (shared/graph_boundary), so a restrictive
        # policy can be applied to them, in the query and again in the gate.
        #
        # That is only safe for a client whose existing graph has been stamped:
        # a record with no instances is refused under a restrictive policy. So it
        # is switched on per instance (SYNAP_GRAPH_INSTANCE_FILTER) once the
        # backfill is verified. Until then, and whenever the policy could not be
        # loaded at all, the arm refuses to traverse rather than guess, and SAYS
        # so in the response instead of returning nothing silently.
        #
        # Default-open is untouched: a client with no
        # policy row resolves to VISIBILITY_ALL and takes the branch below with
        # no instance condition anywhere, byte for byte as before.
        _vis = _request_visibility(self)
        _allowed_instances = None
        if _vis is not VISIBILITY_ALL:
            if (
                isinstance(_vis, (set, frozenset))
                and _graph_boundary.filter_enabled(scope.instance_id if scope else None)
            ):
                _allowed_instances = sorted(str(v) for v in _vis)
            else:
                logger.warning(
                    "neo4j_discovery_skipped_visibility_policy: instance_id=%s client_id=%s "
                    "reason=%s",
                    scope.instance_id if scope else None,
                    scope.client_id if scope else None,
                    "policy not loaded" if not isinstance(_vis, (set, frozenset))
                    else "graph instance filter is not enabled for this instance",
                )
                # The wording starts "visibility policy" on purpose: the cache write
                # treats such a fault as degraded and does not remember the answer,
                # so a response that lacks graph results is never served from the
                # cache without this warning.
                record_retrieval_fault(
                    "visibility policy (graph arm)",
                    "the policy cannot yet be applied to graph records, so no graph results were returned",
                    outcome="not applied",
                )
                return []

        _discovery_start = time.time()

        customer_namespace = str(scope.customer_id)
        client_namespace = str(scope.client_id)
        items: List[ContextItem] = []
        _owner_by_id: Dict[str, Optional[str]] = {}
        graph_query_count = 1  # single combined query
        _scope_only_gate = None

        try:
            # 1. Combined name resolution + traversal — scope chain merge:
            #    customer-namespace (primary) + client-namespace (shared knowledge) in parallel
            logger.info(
                f"Neo4j discovery: starting combined name+traversal: "
                f"anchor_names={anchor_names[:10]}, max_depth={max_depth}"
            )

            traversal_maca = _request_maca(self)
            try:
                # Which graphs hold this reader's material. One per customer
                # until SYNAP_GRAPH_PER_SCOPE; with it, the reader's own graph
                # and the graph of every level above (shared/scope_key). A
                # reader the instance lets read across its users gets every
                # graph of the customer in one walk instead.
                from synap.cloud.shared import scope_key
                _customer = str(scope.customer_id) if scope.customer_id else None
                _user = str(scope.user_id) if scope.user_id else None
                if not scope_key.per_scope_enabled(scope.instance_id):
                    _scoped_walks = [{"namespace": customer_namespace, "user_id": _user}]
                elif traversal_maca.allow_cross_user_retrieval and _customer:
                    _scoped_walks = [{"namespace": customer_namespace, "user_id": None,
                                      "across_customer": True}]
                else:
                    _scoped_walks = [
                        {"namespace": ns, "user_id": _user}
                        for ns in scope_key.readable_scope_ids(
                            scope, roll_up=traversal_maca.roll_up_to_customer)
                        if ns != client_namespace
                    ]
                traversal_tasks = [
                    self.graph.combined_name_traverse(
                        anchor_names=anchor_names,
                        max_depth=max_depth,
                        per_anchor_limit=50,
                        customer_id=_customer,
                        roll_up_to_customer=traversal_maca.roll_up_to_customer,
                        **({"allowed_instances": _allowed_instances}
                           if _allowed_instances is not None else {}),
                        **walk,
                    )
                    for walk in _scoped_walks
                ] + [
                    self.graph.combined_name_traverse(
                        namespace=client_namespace,
                        anchor_names=anchor_names,
                        max_depth=max_depth,
                        per_anchor_limit=20,
                    ),
                ]
                graph_query_count = len(traversal_tasks) - 1
                traversal_results = await asyncio.gather(*traversal_tasks, return_exceptions=True)

                # One gate per sub-traversal, in the same order as the tasks.
                #
                # The customer namespace holds one tenant's material and its
                # rows carry a customer and a user, so the read filter has
                # something to say about every one of them and it is asked.
                #
                # The client namespace is the client's shared knowledge. Its
                # rows carry neither a customer nor a user by construction, and
                # the vector arm reads the matching client-level collection with
                # no scope predicate either (`_chroma_filters_client`). Gating it
                # would make the graph narrower than the read it mirrors, which
                # is a different bug in the opposite direction.
                _scoped_gate = _graph_scope_gate(scope, traversal_maca, customer_namespace, visibility=_vis)
                # The same gate with the instance policy left out, so a memory
                # refused only for its instance can be told from one refused for
                # its scope (see the counters where memories are read).
                _scope_only_gate = (
                    _graph_scope_gate(scope, traversal_maca, customer_namespace)
                    if _allowed_instances is not None else None
                )
                _traversal_gates = [_scoped_gate] * len(_scoped_walks) + [None]

                discovered_nodes = []
                # (edge, gate) rather than edge, because a memory's readability
                # depends on which namespace its edge came out of and that is
                # known here and nowhere later.
                discovered_edges = []
                seen_node_ids = set()
                nodes_refused = 0
                for res, _gate in zip(traversal_results, _traversal_gates):
                    if isinstance(res, Exception):
                        logger.warning(f"Neo4j scope chain traversal sub-task failed: {res}")
                        continue
                    if res:
                        for node in (res.nodes or []):
                            # The Cypher filters relationships and never nodes,
                            # so a path that is legitimately reachable can still
                            # end on someone else's entity. This is the check
                            # that was missing.
                            if _gate is not None and not _gate(node.properties or {}):
                                nodes_refused += 1
                                continue
                            nid = node.id or node.properties.get("canonical_name", "")
                            if nid not in seen_node_ids:
                                seen_node_ids.add(nid)
                                discovered_nodes.append(node)
                        for _edge in (res.edges or []):
                            discovered_edges.append((_edge, _gate))
                if nodes_refused:
                    logger.info(
                        "neo4j_nodes_refused_by_scope: count=%s instance_id=%s",
                        nodes_refused, scope.instance_id if scope else None,
                    )
            except Exception as e:
                logger.warning(f"Neo4j combined name+traversal failed: {e}")
                discovered_nodes = []
                discovered_edges = []

            logger.info(
                f"Neo4j discovery: traversal complete: discovered_nodes={len(discovered_nodes)}, "
                f"discovered_edges={len(discovered_edges)}, anchor_names_count={len(anchor_names)}"
            )

            if not discovered_nodes:
                return []

            # Ask Postgres, once, whether the memories behind every discovered
            # record are alive (SYNAP_GRAPH_SERVE_CHECKS_LIVENESS). Records
            # whose support is gone are not served below.
            _liveness = await _GraphLiveness.load(
                scope.instance_id, discovered_nodes, [_e for _e, _g in discovered_edges])

            # 2. Score discovered nodes using stored embeddings
            # Compute average query vector for scoring
            if query_vectors:
                avg_query = np.mean(query_vectors, axis=0)
                avg_query = avg_query / (np.linalg.norm(avg_query) + 1e-9)
            else:
                return []

            # Scoring stays against the average, as it was. Ownership is a
            # separate question with a per-query answer: which caller query is
            # this embedding nearest to. A record with no embedding of its own
            # has no answer and is left untagged.
            _unit_queries = []
            if query_owners:
                for _qv in query_vectors:
                    _qv = np.asarray(_qv, dtype=np.float32)
                    _unit_queries.append(_qv / (np.linalg.norm(_qv) + 1e-9))

            def _own(item_id, vec) -> None:
                if not _unit_queries or not item_id:
                    return
                try:
                    _best = int(np.argmax([float(np.dot(_q, vec)) for _q in _unit_queries]))
                except Exception:
                    return
                _owner_by_id[item_id] = _owner_at(query_owners, _best)

            for node in discovered_nodes:
                embedding = node.properties.get("embedding")
                if embedding is None:
                    continue

                # Compute cosine similarity between node embedding and query
                node_vec = np.array(embedding, dtype=np.float32)
                node_vec = node_vec / (np.linalg.norm(node_vec) + 1e-9)
                similarity = float(np.dot(avg_query, node_vec))

                if similarity < 0.25:
                    continue
                from synap.cloud.agents.ingestion.services.writer_entity import is_fixed as _is_fixed
                _node_ok, _node_backing = _liveness.record_ok(
                    "entity", node.properties, fixed=_is_fixed(node))
                if not _node_ok:
                    continue
                _own(node.properties.get("entity_id", node.id), node_vec)

                name = node.properties.get("canonical_name") or node.properties.get("name", "")
                # The writer's node is named by the user id; show "the user".
                from synap.cloud.agents.ingestion.services.writer_entity import shown_name
                name = shown_name(node.properties, name)
                entity_type = node.properties.get("entity_type", node.node_type)
                content = f"{name} ({entity_type})"

                items.append(ContextItem(
                    item_id=node.properties.get("entity_id", node.id),
                    content=content,
                    context_type="facts",
                    source="neo4j",
                    similarity_score=similarity,
                    relevance_score=similarity,
                    confidence=0.7,
                    scope=node.properties.get("scope") or derive_item_scope(
                        user_id=node.properties.get("user_id"),
                        customer_id=node.properties.get("customer_id"),
                    ),
                    entity_id=node.properties.get("entity_id"),
                    **_event_fields(node.properties.get),
                    valid_until=_safe_parse_datetime(node.properties.get("valid_until")),
                    temporal_category=node.properties.get("temporal_category"),
                    temporal_confidence=float(node.properties.get("temporal_confidence", 0)) if node.properties.get("temporal_confidence") is not None else 0.0,
                    graph_kind="entity",
                    backing_memory_ids=_node_backing,
                ))

            # 2b. The relationships themselves, as lines of text
            # (SYNAP_GRAPH_SERVE_RELATIONSHIPS). Until now a relationship was
            # only a route to memories: its source, type and target were never
            # shown to the agent. Scored like a node, by the embedding stored
            # on the edge. With the flag on, an edge close to the question is
            # also what exempts its memories from the cut below, in place of
            # the keyword list (BLOCK, RISK, TEST ...), which was written for
            # software work and matched inside words.
            from synap.cloud.shared.instance_flags import enabled_for as _enabled_for
            _serve_relationships = _enabled_for("SYNAP_GRAPH_SERVE_RELATIONSHIPS", scope.instance_id)
            _relevant_edge_ids = set()
            if _serve_relationships:
                from synap.cloud.agents.ingestion.services.graph_v2 import sentence as _sentence
                from synap.cloud.agents.ingestion.services.writer_entity import shown_name as _shown
                _by_id = {n.id: n for n in discovered_nodes}
                _relevant_at = float(os.environ.get("SYNAP_GRAPH_EDGE_RELEVANT_AT", "0.5"))
                _served_edges = set()
                for _edge, _gate in discovered_edges:
                    _props = _edge.properties or {}
                    _a, _b = _by_id.get(_edge.from_id), _by_id.get(_edge.to_id)
                    _emb = _props.get("embedding")
                    if _a is None or _b is None or _emb is None or _edge.id in _served_edges:
                        continue
                    if _gate is not None and not _gate(_props):
                        continue
                    # A closed relationship is neither served nor allowed to
                    # exempt its memories from the cut below.
                    _edge_ok, _edge_backing = _liveness.record_ok("relationship", _props)
                    if not _edge_ok:
                        _served_edges.add(_edge.id)
                        continue
                    _vec = np.array(_emb, dtype=np.float32)
                    _vec = _vec / (np.linalg.norm(_vec) + 1e-9)
                    _sim = float(np.dot(avg_query, _vec))
                    if _sim >= _relevant_at:
                        _relevant_edge_ids.add(_edge.id)
                    if _sim < 0.25:
                        continue
                    _served_edges.add(_edge.id)
                    _own(_edge.id, _vec)
                    _line = _sentence(
                        _shown(_a.properties, _a.properties.get("canonical_name") or _a.properties.get("name", "")),
                        _edge.edge_type,
                        _shown(_b.properties, _b.properties.get("canonical_name") or _b.properties.get("name", "")))
                    if _props.get("basis") == "implied":
                        _line += " (implied)"
                    try:
                        _conf = float(_props.get("confidence", 0.7))
                    except (TypeError, ValueError):
                        _conf = 0.7
                    items.append(ContextItem(
                        item_id=_edge.id,
                        content=_line,
                        context_type="facts",
                        source="neo4j",
                        similarity_score=_sim,
                        relevance_score=_sim,
                        confidence=_conf,
                        scope=derive_item_scope(user_id=_props.get("user_id"),
                                                customer_id=_props.get("customer_id")),
                        graph_kind="relationship",
                        backing_memory_ids=_edge_backing,
                    ))

            def _edge_is_critical(_e) -> bool:
                if _serve_relationships:
                    return _e.id in _relevant_edge_ids
                return _is_critical_edge_type(getattr(_e, "edge_type", None))

            # 3. Extract memory content from relationship edges
            # Memory content is stored as a JSON list in the `memory_contents` property
            # on real relationship edges (WORKS_AT, MANAGES, etc.)
            # Each memory may have its own embedding for per-memory scoring.
            #
            # Scope filtering: memory dicts carry customer_id/user_id.
            #
            # This used to be a hand-rolled pair of comparisons here, described
            # as mirroring `_build_customer_scope_filter`. Mirroring is the
            # problem: the read filter grew a chain arm and this copy did not,
            # so the two answers diverged silently and only in one direction.
            # The gate below IS the read filter, compiled, so they cannot.
            maca = _request_maca(self)

            seen_memory_ids = set()
            edges_with_memories = 0
            memories_with_own_embedding = 0
            scope_filtered_out = 0
            instance_filtered_out = 0
            # Collect all scored memory candidates, then take top 70% by similarity
            _all_scored_memories: list = []  # list of (similarity, ContextItem, has_emb, mem_type)
            node_by_id = {n.id: n for n in discovered_nodes}
            # Process critical edges first so a memory reachable via both a critical
            # and a non-critical edge gets flagged critical (seen_memory_ids dedups
            # on first touch).
            discovered_edges = sorted(
                discovered_edges,
                key=lambda pair: 0 if _edge_is_critical(pair[0]) else 1,
            )
            for edge, _edge_gate in discovered_edges:
                _is_critical_edge = _edge_is_critical(edge)
                props = edge.properties or {}
                memory_contents_raw = props.get("memory_contents")
                if not memory_contents_raw:
                    continue

                try:
                    memory_list = json.loads(memory_contents_raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                edges_with_memories += 1

                # Compute entity-level fallback score from connected node embedding
                connected_node = node_by_id.get(edge.from_id) or node_by_id.get(edge.to_id)
                if connected_node:
                    emb = connected_node.properties.get("embedding")
                    if emb is not None:
                        node_vec = np.array(emb, dtype=np.float32)
                        node_vec = node_vec / (np.linalg.norm(node_vec) + 1e-9)
                        entity_fallback_score = float(np.dot(avg_query, node_vec))
                    else:
                        entity_fallback_score = 0.5
                else:
                    entity_fallback_score = 0.5

                for mem in memory_list:
                    memory_id = mem.get("memory_id", "")
                    memory_content = mem.get("content", "")
                    if not memory_content or memory_id in seen_memory_ids:
                        continue

                    # Scope filter: the read filter's own answer about this
                    # memory. A memory dict carries no `node_id`, so only the
                    # transitional arm can match one today; the write side now
                    # stamps the chain onto new graph records so the nested arm
                    # starts matching as that data arrives.
                    if _edge_gate is not None and not _edge_gate(mem):
                        # A relationship is shared by every instance of the
                        # customer; the memories on it are not. Refusing another
                        # instance's memory is the policy doing its job, so it is
                        # counted apart from a scope refusal, which is a leak
                        # the Cypher should have prevented.
                        if _scope_only_gate is not None and _scope_only_gate(mem):
                            instance_filtered_out += 1
                        else:
                            scope_filtered_out += 1
                        continue

                    # A deleted memory is not a scope question, but it is the
                    # same shape of question: the store says no and the graph
                    # said yes. `soft_delete_memory_from_edges` marks the entry
                    # `is_deleted` on every edge that carries it, and nothing in
                    # this file has ever read that flag, so a memory a client
                    # deleted through the SDK route was still reachable here.
                    #
                    # No stored entries carry the flag today, so this changes
                    # nothing now and closes the gap before the first deletion
                    # on a graph-enabled instance.
                    if _is_true(mem.get("is_deleted")):
                        continue

                    # The copy on the edge says nothing about whether the
                    # memory was since superseded (`is_current` is never kept
                    # up to date); Postgres does.
                    if not _liveness.memory_alive(memory_id):
                        if _liveness.failed:
                            _liveness.dropped_unchecked["memory"] += 1
                        else:
                            _liveness.dropped_not_alive["memory"] += 1
                        continue

                    seen_memory_ids.add(memory_id)

                    # Per-memory scoring: use stored embedding if available,
                    # otherwise fall back to entity-level score
                    mem_embedding = mem.get("embedding")
                    if mem_embedding is not None:
                        mem_vec = np.array(mem_embedding, dtype=np.float32)
                        mem_vec = mem_vec / (np.linalg.norm(mem_vec) + 1e-9)
                        memory_similarity = float(np.dot(avg_query, mem_vec))
                        memories_with_own_embedding += 1
                        _own(memory_id, mem_vec)
                    else:
                        memory_similarity = entity_fallback_score

                    _has_own_emb = mem_embedding is not None
                    _mem_type = mem.get("memory_type", "facts")
                    _all_scored_memories.append((
                        memory_similarity,
                        ContextItem(
                            item_id=memory_id,
                            content=memory_content,
                            context_type=mem.get("memory_type", "facts"),
                            source="neo4j",
                            similarity_score=memory_similarity,
                            relevance_score=memory_similarity,
                            confidence=mem.get("confidence", 0.7),
                            scope=mem.get("scope") or derive_item_scope(
                                user_id=mem.get("user_id"),
                                customer_id=mem.get("customer_id"),
                            ),
                            **_event_fields(mem.get),
                            valid_until=_safe_parse_datetime(mem.get("valid_until")),
                            temporal_category=mem.get("temporal_category"),
                            temporal_confidence=float(mem.get("temporal_confidence", 0)) if mem.get("temporal_confidence") is not None else 0.0,
                            graph_kind="memory",
                            backing_memory_ids=[memory_id],
                        ),
                        _has_own_emb,
                        _mem_type,
                        _is_critical_edge,
                    ))

            # A memory that names ONE entity is on no relationship. Its node
            # holds a reference to it (SYNAP_GRAPH_NODE_MEMORIES); fetch those
            # for every node the walk reached and score them like the rest.
            from synap.cloud.agents.ingestion.services import node_memories as _refs
            _on_relevant_edge = {
                _mid for _e, _g in discovered_edges if _e.id in _relevant_edge_ids
                for _mid in (_e.properties or {}).get(_refs.PROPERTY) or []
            }
            _referenced = bool(_refs.enabled(scope.instance_id) or _refs.edges_enabled(scope.instance_id))
            # Memories Postgres says are not alive are not even fetched.
            _dead_refs = _liveness.collect_dead(
                _mid for _rec in list(discovered_nodes) + [_e for _e, _g in discovered_edges]
                for _mid in ((_rec.properties or {}).get(_refs.PROPERTY) or [])
            )
            for _sim, _item in await self._node_referenced_memories(
                scope, discovered_nodes, seen_memory_ids | _dead_refs, avg_query, _traversal_gates[0], _own,
                edges=[_e for _e, _g in discovered_edges],
            ):
                seen_memory_ids.add(_item.item_id)
                _all_scored_memories.append(
                    (_sim, _item, True, _item.context_type, _item.item_id in _on_relevant_edge))

            # Top 70% filter: sort by similarity, keep top 70% (min 5).
            # Prioritizes recall — LLM filter handles precision downstream.
            # EXEMPTION: memories reached via a decision-critical edge (blocker /
            # risk / dependency / test / supersession) always survive the cut — the
            # relationship signals decision-relevance even when the memory text isn't
            # query-similar (e.g. a failing test that blocks an in-progress task).
            _liveness.log(scope.instance_id)

            _all_scored_memories.sort(key=lambda x: x[0], reverse=True)
            _total_candidates = len(_all_scored_memories)
            _keep_count = max(5, int(_total_candidates * 0.70))
            if _referenced:
                # 70% of what references can gather is too many to hand on.
                _keep_count = min(_keep_count, _refs.walk_limit())
            _noncritical = [m for m in _all_scored_memories if not m[4]]
            _critical = [m for m in _all_scored_memories if m[4]]
            _kept_ids = set()
            _kept = []
            for m in _noncritical[:_keep_count] + _critical:
                if m[1].item_id in _kept_ids:
                    continue
                _kept_ids.add(m[1].item_id)
                _kept.append(m)
            _rejected = [m for m in _noncritical[_keep_count:] if m[1].item_id not in _kept_ids]

            for _sim, _item, _has_emb, _mtype, _crit in _kept:
                logger.info(
                    f"SCORING_LOG neo4j_memory_accepted: sim={round(_sim, 3)}, has_embedding={_has_emb}, type={_mtype}, critical_edge={_crit}, content='{_item.content[:80]}'"
                )
                items.append(_item)

            for _sim, _item, _has_emb, _mtype, _crit in _rejected:
                logger.info(
                    f"SCORING_LOG neo4j_memory_rejected_top70: sim={round(_sim, 3)}, has_embedding={_has_emb}, type={_mtype}, content='{_item.content[:80]}'"
                )

            logger.info(
                f"Neo4j discovery: scoring complete: total_candidates={_total_candidates}, kept_top70={len(_kept)}, rejected={len(_rejected)}, "
                f"scored_items={len(items)}, discovered_nodes={len(discovered_nodes)}, discovered_edges={len(discovered_edges)}, edges_with_memories={edges_with_memories}, unique_memories_found={len(seen_memory_ids)}, memories_with_own_embedding={memories_with_own_embedding}, scope_filtered_out={scope_filtered_out}"
            )
            if instance_filtered_out > 0:
                logger.info(
                    "neo4j_memories_refused_by_instance_policy: count=%s instance_id=%s",
                    instance_filtered_out, scope.instance_id if scope else None,
                )
            if scope_filtered_out > 0:
                logger.warning(
                    "neo4j_post_traversal_scope_leak: %s memories refused by the read filter after "
                    "traversal (the Cypher scoping should have excluded them). customer_id=%s user_id=%s "
                    "chain_rungs=%s",
                    scope_filtered_out,
                    scope.customer_id if scope else None,
                    scope.user_id if scope else None,
                    len(getattr(scope, "chain_rungs", None) or ()),
                )
            # Detached: neo4j-discovery telemetry commit stays off the arm's
            # response path (this arm is the accurate-mode bottleneck).
            _run_detached(
                self._track_graph_queries(
                    scope=scope,
                    query_count=graph_query_count,
                    operation="retrieval.neo4j_graph_discovery",
                )
            )

        except Exception as e:
            logger.error(f"Neo4j graph discovery failed: {e}", exc_info=True)
            _run_detached(
                self._track_graph_queries(
                    scope=scope,
                    query_count=graph_query_count,
                    operation="retrieval.neo4j_graph_discovery",
                    status="error",
                )
            )

        for _it in items:
            _note_query(_it, _owner_by_id.get(_it.item_id))

        logger.info(
            f"Neo4j discovery timing: elapsed_ms={round((time.time() - _discovery_start) * 1000, 1)}, "
            f"items_returned={len(items)}, anchor_count={len(anchor_names)}, max_depth={max_depth}"
        )
        return items

    async def _enrich_with_neo4j_memories(
        self,
        namespace: str,
        entity_node_ids: List[str],
        avg_query: np.ndarray,
        items: List[ContextItem],
        seen_node_ids: set,
    ) -> None:
        """Find memory nodes connected to discovered entities via 'mentions' edges.

        Memory nodes are created during ingestion as memory:{uuid} nodes
        linked to entity nodes via 'mentions' edges.
        """
        if not entity_node_ids or self.graph is None:
            return

        try:
            for node_id in entity_node_ids[:10]:  # Limit to avoid excessive queries
                neighbors = await asyncio.to_thread(
                    self.graph.get_neighbors,
                    namespace, node_id, "mentions", "in"
                )
                for mem_node in neighbors:
                    if mem_node.id in seen_node_ids:
                        continue
                    seen_node_ids.add(mem_node.id)

                    # Memory nodes have content in properties
                    content = mem_node.properties.get("content", "")
                    if not content:
                        continue

                    # Score using embedding if available
                    embedding = mem_node.properties.get("embedding")
                    if embedding:
                        mem_vec = np.array(embedding, dtype=np.float32)
                        mem_vec = mem_vec / (np.linalg.norm(mem_vec) + 1e-9)
                        similarity = float(np.dot(avg_query, mem_vec))
                    else:
                        similarity = 0.5  # Default score for unembedded memories

                    if similarity < 0.30:
                        continue

                    items.append(ContextItem(
                        item_id=mem_node.id,
                        content=content,
                        context_type=mem_node.properties.get("memory_type", "facts"),
                        source="neo4j",
                        similarity_score=similarity,
                        relevance_score=similarity,
                        confidence=mem_node.properties.get("confidence", 0.5),
                        scope=mem_node.properties.get("scope") or derive_item_scope(
                            user_id=mem_node.properties.get("user_id"),
                            customer_id=mem_node.properties.get("customer_id"),
                        ),
                        **_event_fields(mem_node.properties.get),
                        valid_until=_safe_parse_datetime(mem_node.properties.get("valid_until")),
                        temporal_category=mem_node.properties.get("temporal_category"),
                        temporal_confidence=float(mem_node.properties.get("temporal_confidence", 0)) if mem_node.properties.get("temporal_confidence") is not None else 0.0,
                    ))
        except Exception as e:
            logger.warning(f"Neo4j memory enrichment failed: {e}")

    def assemble_context_bundle(
        self,
        items: List[ContextItem],
        decision: ContextDecision,
        bundle: ContextBundle,
    ) -> ContextBundle:
        """
        Assemble ContextBundle from retrieved items.

        Steps:
        1. Deduplicate with Jaccard similarity
        2. Score and sort by relevance
        3. Trim to budget
        4. Categorize by type and sort within each category
        """
        # Get query for filtering
        query = " ".join(decision.search_queries) if decision.search_queries else ""

        # A fetch with one query found everything it returns for that query,
        # including what no search arm attributes (a graph discovery, a stored
        # memory read by recency). With several queries those stay untagged:
        # naming one would be a guess.
        _distinct_queries = list(dict.fromkeys(q for q in (decision.search_queries or []) if q))
        if len(_distinct_queries) == 1:
            for _it in items:
                if not _it.matched_queries:
                    _note_query(_it, _distinct_queries[0])

        # Log items before filtering
        items_before = {
            ctx_type: sum(1 for item in items if item.context_type == ctx_type)
            for ctx_type in set(item.context_type for item in items)
        }

        logger.info(
            f"Assembling context bundle: total_items_before_filter={len(items)}, items_by_type_before_filter={items_before}, query={truncate_string(query, 100)}, max_results={decision.max_results}"
        )

        # Get MACA settings for enhanced scoring
        maca = _request_maca_or_none(self)

        # Separate chunk items (already score-gated, max 2) — they get reserved slots
        chunk_items = [item for item in items if item.source == "chunks"]
        non_chunk_items = [item for item in items if item.source != "chunks"]

        # Deduplicate non-chunk items only (chunks/sentences are already deduped in _search_sentences)
        deduped = _deduplicate_jaccard(
            non_chunk_items, threshold=dedup_jaccard_threshold())

        _pre_jaccard_source = {}
        _post_jaccard_source = {}
        for _item in non_chunk_items:
            _pre_jaccard_source[_item.source] = _pre_jaccard_source.get(_item.source, 0) + 1
        for _item in deduped:
            _post_jaccard_source[_item.source] = _post_jaccard_source.get(_item.source, 0) + 1
        logger.info(
            f"assembly_jaccard_dedup: pre={len(non_chunk_items)}, post={len(deduped)}, "
            f"removed={len(non_chunk_items) - len(deduped)}, "
            f"pre_source={_pre_jaccard_source}, post_source={_post_jaccard_source}"
        )

        # Score and sort by relevance
        _prov_discount = _provenance_discount()
        for _item in deduped:
            _item._final_score = _compute_relevance_score(_item, maca=maca)
            # Provenance discount (SYNAP_RETRIEVAL_PROVENANCE_DISCOUNT, unset
            # = off): assistant-inferred facts rank below user-stated ones at
            # equal relevance, so an agent's own earlier speculation can't
            # crowd out what the user actually said. Applied at this single
            # choke point so every source arm (vector/discovery/postgres/
            # graph) is covered; None assertion_source (legacy rows) is never
            # discounted.
            if (
                _prov_discount is not None
                and getattr(_item, "assertion_source", None) == "assistant"
            ):
                _item._final_score *= _prov_discount
        # Chunks are scored with the same formula, so they can be placed in the
        # ranked list on the same scale as memories. Only the score: they still
        # skip the dedup, the trim and the per-type order below, so which chunks
        # come back, and where they sit in the per-type lists, is unchanged. No
        # provenance discount, since a chunk is a quote, not an inference.
        for _chunk in chunk_items:
            _chunk._final_score = _compute_relevance_score(
                _chunk, maca=maca, confidence=_CHUNK_SCORING_CONFIDENCE)

        # When Jev is ranking, its verdict is the key and the computed score is
        # the tiebreak (see _ranking_key). When it is off, no item carries a
        # verdict and this is identical to sorting on _final_score alone.
        _jev_ranks = jev_is_primary_ranker()
        if _jev_ranks:
            _order_key = _ranking_key
        else:
            def _order_key(x):
                return x._final_score
        deduped.sort(key=_order_key, reverse=True)

        # Log final scoring breakdown for every item
        for _rank, _item in enumerate(deduped):
            logger.info(
                f"SCORING_LOG final_ranking: rank={_rank+1}, source={_item.source}, type={_item.context_type}, "
                f"sim={round(_item.similarity_score, 3)}, rel={round(_item.relevance_score, 3)}, "
                f"conf={round(_item.confidence, 2)}, final={round(_item._final_score, 3)}, "
                f"event_date={_item.event_date}, hop={getattr(_item, 'hop_distance', None)}, "
                f"content='{_item.content[:80]}'"
            )

        # Log all graph items and top vector items for manual relevance comparison
        _graph_items_detail = [
            (i, _item) for i, _item in enumerate(deduped) if _item.source == "graph"
        ]
        if _graph_items_detail:
            for rank, _item in _graph_items_detail:
                logger.info(
                    f"assembly_graph_item: rank={rank+1}/{len(deduped)}, "
                    f"final_score={_item._final_score:.4f}, sim={_item.similarity_score:.4f}, "
                    f"rel={_item.relevance_score:.4f}, conf={_item.confidence}, "
                    f"type={_item.context_type}, temporal={getattr(_item, 'temporal_category', '')}, "
                    f"content={_item.content[:150]!r}"
                )
        _vector_top = [(i, _item) for i, _item in enumerate(deduped) if _item.source == "vector"][:20]
        for rank, _item in _vector_top:
            logger.info(
                f"assembly_vector_item: rank={rank+1}/{len(deduped)}, "
                f"final_score={_item._final_score:.4f}, sim={_item.similarity_score:.4f}, "
                f"rel={_item.relevance_score:.4f}, conf={_item.confidence}, "
                f"type={_item.context_type}, temporal={getattr(_item, 'temporal_category', '')}, "
                f"content={_item.content[:150]!r}"
            )

        # Trim non-chunk items, leaving room for reserved chunk slots.
        # Use max_results as budget (not *2) so chunks appear in the final top-N.
        budget = decision.max_results
        # max(0, ...) because a negative slice is not an empty slice: with more
        # chunk items than max_results this went negative and `deduped[:-4]`
        # returns everything BUT the last four, so the trim inverted into a
        # near-passthrough and the response EXCEEDED the cap the caller set.
        # Observed live: 16 of 60 questions asked for 10 items and got more,
        # one got 19.
        non_chunk_budget = max(0, budget - len(chunk_items))
        _floor = 0
        try:
            _floor = int(getattr(maca, "per_type_floor", 0) or 0)
        except (TypeError, ValueError):
            _floor = 0
        _per_query = decision.max_results_per_query
        if (
            _per_query and _per_query > 0
            and len(_distinct_queries) > 1
            and non_chunk_budget > 0
        ):
            # The caller asked for a share per search query. Takes the place of
            # the single cut by rank, and of the per-type floor, which answers a
            # different question about the same places. One query, or no
            # per-query value: this branch is not reached and the cut below is
            # the one it always was.
            trimmed = _reserve_per_query(
                deduped, non_chunk_budget, int(_per_query), _distinct_queries)
        elif _floor > 0 and non_chunk_budget > 0:
            # Per-type reserve pass (Workstream B lever 2) — guarantee a floor
            # per context_type before filling the rest by global relevance, so a
            # lone low-frequency fact isn't crowded out of the top-N. Inert
            # (floor 0) for any instance without the MACA key.
            trimmed = _reserve_per_type(deduped, non_chunk_budget, _floor)
        else:
            trimmed = deduped[:non_chunk_budget]

        _trimmed_source = {}
        _assistant_sourced = 0
        for _item in trimmed:
            _trimmed_source[_item.source] = _trimmed_source.get(_item.source, 0) + 1
            if getattr(_item, "assertion_source", None) == "assistant":
                _assistant_sourced += 1
        logger.info(
            f"Assembly complete: {len(trimmed)+len(chunk_items)}/{len(items)} items after dedup+trim, "
            f"items_removed={len(items) - len(trimmed) - len(chunk_items)}, chunk_slots_reserved={len(chunk_items)}, "
            f"trimmed_source={_trimmed_source}, assistant_sourced={_assistant_sourced}, "
            f"provenance_discount={_prov_discount if _prov_discount is not None else 'off'}"
        )
        if chunk_items:
            logger.info(
                f"Assembly chunk detail: count={len(chunk_items)}, "
                f"chunks=[{', '.join(f'score={c.relevance_score:.3f} sim={c.similarity_score:.3f} conf={c.confidence} content={c.content[:120]!r}' for c in chunk_items)}]"
            )

        # Categorize by type. The list arrives ALREADY RANKED: `deduped` was
        # sorted on `_ranking_key` above and `trimmed` preserves that order, so
        # appending in sequence leaves every category in global relevance order.
        items_by_type: Dict[str, List[ContextItem]] = {}
        for item in trimmed:
            items_by_type.setdefault(item.context_type, []).append(item)

        # ⚠ Do not add a per-category sort here. A second sort cannot produce an
        # order, only replace the correct one above with a different one. The one
        # removed on 2026-10-08 recomputed relevance from scratch, rounded it to
        # 0.1 and then ordered by date: that banded the 14,934 items served on the
        # 29 Sep run into 11 values, 56% of them identical, so the date decided
        # most of the sequence. It also never read `_final_score`, so every
        # adjustment made at the scoring choke point was dropped from the order
        # the reader reads. Near ties used to go to the newer item and now go to
        # the higher score, which is the one behaviour that changed.

        # Interleave chunk items into their type list at position 2 (after top 2 long-term items)
        # so they appear in the visible portion of results, not buried at the end.
        for chunk in chunk_items:
            type_list = items_by_type.setdefault(chunk.context_type, [])
            insert_pos = min(2, len(type_list))
            type_list.insert(insert_pos, chunk)

        bundle.items_by_type = items_by_type

        # THE SAME ITEMS AS ONE RANKED LIST, for `response_shape="ranked"`.
        #
        # The per-type lists above lose the order across types, and they put
        # each chunk third by rule rather than by score. This is the order the
        # engine actually ranked: every served item once, chunks placed by their
        # own score. Built from the same `trimmed` and `chunk_items`, so it holds
        # exactly the items the per-type lists hold, and it changes nothing about
        # them: the default response is built from `items_by_type` as before.
        #
        # Sorted with the key the memories were sorted with. Python's sort is
        # stable and `trimmed` is already in that order, so memories keep their
        # sequence and a chunk that ties a memory goes after it.
        #
        # Ids rather than items, so the bundle holds each item once. Anything
        # that later drops an item (the hard token budget) leaves a dangling id,
        # and the reader skips ids it cannot find.
        _ranked = sorted(list(trimmed) + list(chunk_items), key=_order_key, reverse=True)
        for _item in _ranked:
            _item.rank_score = _published_rank_score(_item, _jev_ranks)
        bundle.ranked_item_ids = [_item.item_id for _item in _ranked]

        logger.info(
            f"Context bundle assembled: items_by_type={ {ctx_type: len(items) for ctx_type, items in items_by_type.items()} }, total_items={sum(len(items) for items in items_by_type.values())}"
        )

        return bundle

    async def _lookup_source_documents_for_memories(
        self,
        memory_ids: List[str],
    ) -> Dict[str, List[str]]:
        """Batch-resolve memory_id → list of source_document_ids via the
        memory_source_documents join table. Merged memories (KU v2 ENRICH)
        have N rows here while their `memories.source_document_id` is NULL;
        legacy memories have one row backfilled by migration 052.

        Returns {} for any id not found in the table — caller should fall
        back to the in-bundle `source_document_id` for those.
        """
        if not memory_ids or self.postgres is None:
            return {}

        # Filter to ids that look like UUIDs — graph items can have non-UUID
        # ids and the cast would explode the whole query.
        uuid_ids: List[str] = []
        for mid in memory_ids:
            try:
                uuid.UUID(str(mid))
                uuid_ids.append(str(mid))
            except (ValueError, TypeError):
                continue
        if not uuid_ids:
            return {}

        try:
            rows = await self.postgres.fetch_all(
                """
                SELECT memory_id::text AS memory_id, source_document_id
                FROM memory_source_documents
                WHERE memory_id = ANY(CAST(:ids AS uuid[]))
                """,
                {"ids": uuid_ids},
            )
        except Exception as e:
            logger.warning(f"chunk_evidence_msd_lookup_failed: {e}")
            return {}

        result: Dict[str, List[str]] = {}
        for row in rows:
            mid = row["memory_id"]
            doc_id = row["source_document_id"]
            if mid and doc_id:
                result.setdefault(mid, []).append(doc_id)
        return result

    async def _document_said_dates(
        self,
        document_ids: List[str],
    ) -> Dict[str, datetime]:
        """When was each of these documents said: document_id -> date.

        Every memory row carries the date of the document it came from in
        `document_created_at`, so this is a group-by, not a new field and not a
        new table. Asked only for the documents the bundle could not supply for
        free -- a single-document item already knows its own document's date.

        MIN, not MAX: 36 of the 2,730 documents on the benchmark instance have
        rows claiming two different document dates (merge bookkeeping), and the
        earlier one is the document's own.
        """
        if not document_ids or self.postgres is None:
            return {}
        try:
            rows = await self.postgres.fetch_all(
                """
                SELECT source_document_id, MIN(document_created_at) AS said
                FROM memories
                WHERE source_document_id = ANY(CAST(:ids AS text[]))
                  AND document_created_at IS NOT NULL
                GROUP BY source_document_id
                """,
                {"ids": [str(d) for d in document_ids]},
            )
        except Exception as e:  # noqa: BLE001
            # An excerpt with no date is what shipped until now, so a failed
            # lookup costs the label and nothing else.
            logger.warning(f"evidence_said_dates_lookup_failed: {e}")
            return {}
        out: Dict[str, datetime] = {}
        for row in rows:
            doc_id, said = row["source_document_id"], row["said"]
            if doc_id and said is not None:
                out[str(doc_id)] = said
        return out

    # The hypothetical-question index stores its own copy of each memory's
    # text, truncated to 500 characters at write time
    # (agents/ingestion/storage/vector_wrapper.py). An item built from that
    # index therefore carries a truncated fact. 500 is the exact cap, so an
    # item at exactly 500 characters is the one to re-read; a memory that is
    # genuinely 500 characters long re-reads to the same value and costs
    # nothing.
    _HQ_CONTENT_CAP = 500

    async def _fill_truncated_content(self, bundle: ContextBundle) -> None:
        """Replace a truncated discovery item's text with the stored memory.

        One batched query, and only for items sitting exactly on the cap, so a
        bundle with no truncated item costs nothing. Non-UUID ids are dropped
        before the cast, as in _fill_assertion_source: a graph item can carry
        an id that is not a memory id and one of those fails the statement.
        """
        if self.postgres is None:
            return

        truncated: Dict[str, List[ContextItem]] = {}
        for items_list in bundle.items_by_type.values():
            for item in items_list:
                content = getattr(item, "content", None) or ""
                if len(content) != self._HQ_CONTENT_CAP:
                    continue
                mid = getattr(item, "item_id", None)
                if not mid:
                    continue
                try:
                    uuid.UUID(str(mid))
                except (ValueError, TypeError):
                    continue
                truncated.setdefault(str(mid), []).append(item)

        if not truncated:
            return

        _started = time.time()
        rows = await self.postgres.fetch_all(
            """
            SELECT id::text AS id, content
            FROM memories
            WHERE id = ANY(CAST(:ids AS uuid[]))
            """,
            {"ids": list(truncated.keys())},
        )
        filled = 0
        for row in rows:
            full = row["content"] or ""
            # Only ever lengthen. A shorter or equal row means this item was
            # not truncated after all, and overwriting it would be a silent
            # change to what we serve.
            if len(full) <= self._HQ_CONTENT_CAP:
                continue
            for item in truncated.get(row["id"], []):
                item.content = full
                filled += 1

        logger.info(
            "fill_truncated_content: at_cap=%d restored=%d elapsed_ms=%.1f",
            len(truncated), filled, (time.time() - _started) * 1000,
        )

    async def _fill_assertion_source(self, bundle: ContextBundle) -> None:
        """Set `assertion_source` on bundle items that arrived without one.

        Who asserted a memory is stored on `memories.assertion_source` for
        most rows, and it reached only a third of served items:
        of the ten places a ContextItem is built, seven never copied it, and
        the graph never stored it to be copied. Measured 67.1% unset across
        4,352 memories served for the lme17 counting questions.

        It matters because a reader cannot otherwise tell what the user said
        from what we said back. On "how many different types of citrus fruits
        have I used in my cocktail recipes" (gold 3) every mention of a fourth
        fruit sat in the assistant's suggestions, 32 of 40 items carried no
        speaker, and the answer was 4.

        One batched query, and only for the items still missing a value, so a
        bundle whose stores already carried it costs nothing. Non-UUID ids are
        dropped before the cast: a graph item can have an id that is not a
        memory id and one of those fails the whole statement.
        """
        if self.postgres is None:
            return

        missing: Dict[str, List[ContextItem]] = {}
        for items_list in bundle.items_by_type.values():
            for item in items_list:
                if getattr(item, "assertion_source", None):
                    continue
                mid = getattr(item, "item_id", None)
                if not mid:
                    continue
                try:
                    uuid.UUID(str(mid))
                except (ValueError, TypeError):
                    continue
                missing.setdefault(str(mid), []).append(item)

        if not missing:
            return

        _started = time.time()
        rows = await self.postgres.fetch_all(
            """
            SELECT id::text AS id, assertion_source
            FROM memories
            WHERE id = ANY(CAST(:ids AS uuid[]))
              AND assertion_source IS NOT NULL
            """,
            {"ids": list(missing.keys())},
        )
        filled = 0
        for row in rows:
            for item in missing.get(row["id"], []):
                item.assertion_source = row["assertion_source"]
                filled += 1

        logger.info(
            "fill_assertion_source: missing=%d filled=%d elapsed_ms=%.1f",
            len(missing), filled, (time.time() - _started) * 1000,
        )

    async def _fill_subject(self, bundle: ContextBundle) -> None:
        """Stamp every served item with the thing it is about (migration 159).

        Ingestion groups memories that state different facts about one
        real-world thing, and writes the group on `memories.subject_id` with the
        person's own words in `memories.subject_label`. Without this the group
        exists in the database and never reaches the reader, which is the whole
        point of having written it.

        One batched query, same shape as `_fill_assertion_source` above and for
        the same reason: the subject is written after a memory's own vector, so
        Chroma metadata cannot carry it without a backfill. Non-UUID ids are
        dropped before the cast — a graph item can have an id that is not a
        memory id, and one of those fails the whole statement.

        Silent and harmless before the migration is applied: the query raises,
        it is caught, and every item keeps the None it already had.
        """
        if self.postgres is None:
            return

        by_id: Dict[str, List[ContextItem]] = {}
        for items_list in bundle.items_by_type.values():
            for item in items_list:
                if getattr(item, "subject_id", None):
                    continue
                mid = getattr(item, "item_id", None)
                if not mid:
                    continue
                try:
                    # Keyed on the CANONICAL spelling, because that is what
                    # Postgres returns. An upper-case or un-hyphenated uuid
                    # passes the cast, matches in SQL, and then fails to find
                    # its own bucket on the way back.
                    key = str(uuid.UUID(str(mid)))
                except (ValueError, TypeError):
                    continue
                by_id.setdefault(key, []).append(item)

        if not by_id:
            return

        _started = time.time()
        try:
            rows = await self.postgres.fetch_all(
                """
                SELECT id::text AS id, subject_id::text AS subject_id, subject_label
                FROM memories
                WHERE id = ANY(CAST(:ids AS uuid[]))
                  AND subject_id IS NOT NULL
                """,
                {"ids": list(by_id.keys())},
            )
        except Exception as e:  # noqa: BLE001 — a missing column is not a failed fetch
            logger.info("fill_subject skipped: %s", str(e)[:160])
            return

        filled = 0
        for row in rows:
            for item in by_id.get(row["id"], []):
                item.subject_id = row["subject_id"]
                item.subject_label = row["subject_label"]
                filled += 1

        logger.info(
            "fill_subject: candidates=%d filled=%d distinct_subjects=%d elapsed_ms=%.1f",
            len(by_id), filled, len({r["subject_id"] for r in rows}),
            (time.time() - _started) * 1000,
        )

    async def _enrich_with_chunk_evidence(
        self,
        bundle: ContextBundle,
        search_queries: List[str],
        scope: ScopeHierarchy,
        max_items_to_enrich: Optional[int] = None,
        max_chunks_per_item: int = 1,
    ) -> ContextBundle:
        """Enrich top-N bundle items with source chunk evidence.

        For each item, look up its source documents in `memory_source_documents`
        (KU v2 chunk fan-out), then fetch the original conversation chunks from
        ChromaDB and select the most relevant FULL CHUNK as evidence. No
        sentence splitting — tables, lists, and structured content are
        preserved intact.

        Merged memories (ENRICH outcome) have NULL `source_document_id` on
        the row but multiple entries in `memory_source_documents`; the fan-out
        unions chunks from every contributor doc before scoring. Non-merged
        memories have exactly one source doc and behave as before.

        Chunks are deduplicated across items: if two sibling memories share
        a source document, the same chunk won't be attached twice.

        Args:
            bundle: Assembled context bundle with final items
            search_queries: Original search queries for scoring
            scope: Scope hierarchy for ChromaDB namespace resolution
            max_items_to_enrich: Maximum number of items to enrich.
                When None or <= 0, enrich all items currently in the bundle
                (already capped by decision.max_results during assembly).
            max_chunks_per_item: Maximum evidence chunks per item
        """
        if self.vector is None:
            return bundle

        query_text = " ".join(search_queries) if search_queries else ""
        if not query_text:
            return bundle

        # Collect candidate items in bundle order. Include every item — the
        # source-document lookup decides eligibility, not the in-row column,
        # because merged memories have NULL `source_document_id`.
        # By default (max_items_to_enrich=None), enrich all bundle items.
        item_limit = (
            max_items_to_enrich
            if (max_items_to_enrich is not None and max_items_to_enrich > 0)
            else None
        )
        enrichable_items: List[ContextItem] = []
        for items_list in bundle.items_by_type.values():
            for item in items_list:
                if item_limit is not None and len(enrichable_items) >= item_limit:
                    break
                enrichable_items.append(item)
            if item_limit is not None and len(enrichable_items) >= item_limit:
                break

        if not enrichable_items:
            return bundle

        # Batch-resolve memory_id → [source_document_id, ...] from the join table.
        # Only items that do NOT already carry a source document need the join
        # table, and in practice that is none of them.
        #
        # The join table is authoritative because a merged memory (KU v2 ENRICH)
        # has N rows there while its in-row `source_document_id` is NULL. A
        # memory that DOES carry an in-row value has exactly one join row, equal
        # to it, backfilled by migration 052. Checked against stored data rather than
        # taken from that docstring: no memory has a NULL in-row value, and
        # every join row matches the in-row value with at most one per memory.
        # So for every item now in a bundle the query returns what the item
        # already carries.
        #
        # It cost 290-408 ms per request to learn that, because the database is
        # in another cloud (Part 22). Querying only for the items that need it
        # keeps the merged-memory case exactly as it was and skips the round
        # trip entirely when there is nothing to resolve.
        needs_lookup = [
            it.item_id for it in enrichable_items
            if it.item_id and not getattr(it, "source_document_id", None)
        ]
        _msd_started = time.time()
        msd_map = (
            await self._lookup_source_documents_for_memories(needs_lookup)
            if needs_lookup else {}
        )
        _msd_ms = (time.time() - _msd_started) * 1000

        # Per-item resolved source docs: prefer the join table, fall back to
        # the in-row column for items the lookup didn't cover (e.g. items
        # whose item_id is not a memory UUID, or pre-backfill stragglers).
        item_to_docs: Dict[int, List[str]] = {}  # id(item) -> [doc_id, ...]
        for item in enrichable_items:
            docs = msd_map.get(item.item_id) if item.item_id else None
            if not docs:
                fallback = getattr(item, "source_document_id", None)
                docs = [fallback] if fallback else []
            if docs:
                item_to_docs[id(item)] = docs

        if not item_to_docs:
            return bundle

        # Build IngestionScope for vector wrapper calls
        from synap.cloud.agents.ingestion.models import Scope as IngestionScope
        from synap.cloud.shared.scope import parse_or_deterministic_uuid

        try:
            vector_scope = IngestionScope(
                client_id=parse_or_deterministic_uuid(scope.client_id),
                instance_id=parse_or_deterministic_uuid(scope.instance_id),
                customer_id=parse_or_deterministic_uuid(scope.customer_id) if scope.customer_id else None,
                user_id=parse_or_deterministic_uuid(scope.user_id) if scope.user_id else None,
            )
        except Exception as e:
            logger.warning(f"chunk_evidence_skip: scope conversion failed: {e}")
            return bundle

        # Unique source docs across all candidate items — fetched once each.
        unique_doc_ids: List[str] = sorted({d for docs in item_to_docs.values() for d in docs})

        from synap.cloud.agents.retrieval.sentence_scorer import (
            extract_evidence, evidence_min_support)

        # How much of a fact a raw chunk must repeat before we attach it to
        # that fact. 0 (the default) attaches whatever scores best, which is
        # what shipped until now. See sentence_scorer for why that is a
        # problem and what raising it was measured to be worth.
        _min_support = evidence_min_support()

        # Read once, not per item: all three change only on a deploy.
        _dated = dated_evidence_enabled()
        _said_from_doc = evidence_said_from_document_enabled()
        _speaker_label = evidence_speaker_label_enabled()

        enrich_start = time.time()

        # One query for every document, not one per document.
        #
        # These all live in the SAME chunks collection, and `asyncio.gather` made
        # the fan-out look free: it is concurrent, so the cost is the slowest
        # call rather than the sum. It is not free, because the store serialises
        # them. With ten documents, one `$in` query is roughly 8x faster than ten
        # concurrent single fetches, with byte-identical
        # rows. The wrapper falls back to the per-document shape if the batched
        # read comes back full, since one global limit can starve a document.
        # Fetch and scoring are timed separately. Merged, there was no way
        # to tell how much of the time is the store and how much is
        # `extract_evidence` scoring on the CPU. One number that cannot be
        # attributed is not a measurement.
        _fetch_started = time.time()
        doc_chunk_rows: Dict[str, List[Dict[str, Any]]] = {}
        try:
            doc_chunk_rows = await self.vector.get_chunks_by_document_ids(
                scope=vector_scope,
                source_document_ids=unique_doc_ids,
                limit_per_document=50,
            )
            fetch_results = [
                (doc_id, [c["content"] for c in doc_chunk_rows.get(doc_id, [])
                          if c.get("content")])
                for doc_id in unique_doc_ids
            ]
        except Exception as e:  # noqa: BLE001
            # Never let an enrichment optimisation cost the caller their bundle.
            logger.warning("chunk_evidence_batched_fetch_failed, falling back: %s", e)

            async def _fetch_chunks_for_doc(doc_id: str):
                chunks = await self.vector.get_chunks_by_document_id(
                    scope=vector_scope,
                    source_document_id=doc_id,
                    limit=50,
                )
                return doc_id, [c["content"] for c in chunks if c.get("content")]

            fetch_results = await asyncio.gather(
                *[_fetch_chunks_for_doc(d) for d in unique_doc_ids],
                return_exceptions=True,
            )

        _chunk_fetch_ms = (time.time() - _fetch_started) * 1000

        # Build doc_id -> chunk_texts mapping
        doc_chunks: Dict[str, List[str]] = {}
        for result in fetch_results:
            if isinstance(result, Exception):
                logger.warning(f"chunk_evidence_fetch_error: {result}")
                continue
            doc_id, texts = result
            if texts:
                doc_chunks[doc_id] = texts

        # When was each document said: document_id -> date.
        #
        # Free for a document that any single-document item in this bundle came
        # from, because that item's `said_at` IS that document's date. The query
        # is only for the rest -- documents that reach the bundle solely as a
        # contributor to a merged memory.
        doc_said: Dict[str, datetime] = {}
        if _said_from_doc:
            for _item in enrichable_items:
                _docs = item_to_docs.get(id(_item)) or []
                _said = getattr(_item, "said_at", None)
                if len(_docs) == 1 and _said is not None:
                    doc_said.setdefault(_docs[0], _said)
            _missing = [d for d in unique_doc_ids if d not in doc_said]
            if _missing:
                doc_said.update(await self._document_said_dates(_missing))

        # Extract evidence per item — union chunks from ALL its source docs
        # before scoring so a merged memory can cite any contributor.
        enriched_count = 0
        merged_enriched_count = 0
        unsupported_count = 0
        seen_chunks: set = set()  # dedup across all items
        for item in enrichable_items:
            docs = item_to_docs.get(id(item))
            if not docs:
                continue

            unioned: List[str] = []
            seen_in_item: set = set()
            # Which document each chunk came from, and where in that document.
            # The union below is what loses it, and without it an excerpt
            # cannot be dated or attributed to a speaker.
            chunk_home: Dict[str, Tuple[str, int]] = {}
            for d in docs:
                for pos, txt in enumerate(doc_chunks.get(d, [])):
                    if txt not in seen_in_item:
                        seen_in_item.add(txt)
                        unioned.append(txt)
                        chunk_home[txt] = (d, pos)
            if not unioned:
                continue

            evidence = extract_evidence(
                chunk_texts=unioned,
                query=query_text,
                memory_content=item.content,
                max_chunks=max_chunks_per_item,
                seen_chunks=seen_chunks,
                min_support=_min_support,
            )
            if not evidence and _min_support > 0:
                # The item keeps its fact and every other field; only the raw
                # excerpt is withheld, because nothing in its source documents
                # repeats enough of it to be evidence FOR it.
                unsupported_count += 1
            if evidence:
                # Say when the text was said, and who said it.
                #
                # The old rule, still what runs with both new flags off: the
                # date this row's own source document carries, and only for a
                # single-document item because a merged row's evidence could
                # come from any contributor. That dates 55.5% of them.
                #
                # It used to reach that date through `event_date`, valid only
                # when `event_date_is_mention` confirmed the read path had just
                # substituted it. Same value, now read from the field that
                # holds it by name.
                _old_said = (
                    getattr(item, "said_at", None) if len(docs) == 1 else None
                )
                if _said_from_doc or _speaker_label:
                    evidence = [
                        _excerpt_label(
                            e, chunk_home.get(e), doc_chunks, doc_said,
                            _old_said if _dated else None,
                            _said_from_doc, _speaker_label,
                        )
                        for e in evidence
                    ]
                elif _dated and _old_said is not None:
                    evidence = [f"{_said_prefix(_old_said)}{e}" for e in evidence]
                item.source_evidence = evidence
                enriched_count += 1
                if len(docs) > 1:
                    merged_enriched_count += 1

        enrich_ms = (time.time() - enrich_start) * 1000
        logger.info(
            f"chunk_evidence_complete: enrichable={len(enrichable_items)}, "
            f"items_with_docs={len(item_to_docs)}, unique_docs={len(unique_doc_ids)}, "
            f"docs_with_chunks={len(doc_chunks)}, items_enriched={enriched_count}, "
            f"merged_items_enriched={merged_enriched_count}, "
            f"elapsed_ms={round(enrich_ms, 1)}, "
            # Split, so the store and the CPU can be told apart. One merged
            # number sent this optimisation looking in the wrong place twice.
            f"fetch_ms={round(_chunk_fetch_ms, 1)}, "
            f"score_ms={round(enrich_ms - _chunk_fetch_ms, 1)}, "
            f"source_doc_lookup_ms={round(_msd_ms, 1)}, "
            f"source_doc_lookups={len(needs_lookup)}, "
            # 0/0 means the floor is off. Any other pair is the one number that
            # says whether raising it is costing evidence on real traffic.
            f"min_support={_min_support}, unsupported={unsupported_count}"
        )

        return bundle


    async def _filter_batch_with_llm(
        self,
        batch_items: List[ContextItem],
        batch_index: int,
        global_offset: int,
        query: str,
        system_prompt: str,
        scope: Optional[ScopeHierarchy] = None,
        correlation_id: Optional[str] = None,
    ) -> List[int]:
        """Filter a single batch of items with LLM. Returns global indices of kept items.

        FAIL-OPEN: a batch that errors, times out, or comes back unparseable
        keeps ALL of its items. Dropping them (the old ``return []``) silently
        deleted up to BATCH_SIZE candidates on a single Groq timeout — an
        infrastructure hiccup became permanent memory loss for that turn, and
        the whole-call top-K fallback never fired because the OTHER batches
        still returned results. Only an explicit ``KEEP: NONE`` (the model
        actually judged the batch irrelevant) drops items.
        """
        import time as _time

        items_text = "\n".join([
            f"{i+1}. [{item.context_type}]{_format_temporal_tag(item)} {item.content}"
            for i, item in enumerate(batch_items)
        ])

        from synap.cloud.application.client_api.prompts.relevance_filter import build_user_prompt

        user_prompt = build_user_prompt(query, items_text)

        filter_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Fail-open result: every index in this batch survives untouched.
        keep_all = list(range(global_offset, global_offset + len(batch_items)))

        t0 = _time.monotonic()
        try:
            try:
                response = await self.llm_client.complete(
                    messages=filter_messages,
                    # 0.0, not 0.1. Two
                    # cold-cache runs of the same build returned a DIFFERENT SET
                    # of memories on 9 of 20 non-empty requests, mean overlap
                    # 0.53. Deliberate sampling on the step that decides which
                    # memories a user gets back buys nothing and costs
                    # reproducibility. See Part 27b.
                    #
                    # `_LLM_FILTER_MAX_TOKENS` rather than a literal 200 comes
                    # from main: the filter was measured returning an empty
                    # string on 7 of 7 calls because a reasoning model spent the
                    # entire 200-token budget before emitting a visible
                    # character. Both changes belong here, and they are about
                    # different things: one makes the step repeatable, the other
                    # makes it capable of answering at all.
                    temperature=0.0, max_tokens=_LLM_FILTER_MAX_TOKENS, json_mode=False,
                    reasoning_effort="low",
                    telemetry_scope=scope,
                    telemetry_operation="retrieval.llm_filter",
                    telemetry_correlation_id=correlation_id,
                )
            except TypeError:
                response = await self.llm_client.complete(
                    messages=filter_messages,
                    temperature=0.0, max_tokens=_LLM_FILTER_MAX_TOKENS, json_mode=False,
                )
        except Exception as e:
            elapsed = (_time.monotonic() - t0) * 1000
            logger.warning(
                f"LLM filter batch {batch_index} failed ({type(e).__name__}: {e}) — "
                f"failing open, kept {len(keep_all)} items, "
                f"elapsed_ms={elapsed:.1f}, batch_size={len(batch_items)}"
            )
            return keep_all

        elapsed = (_time.monotonic() - t0) * 1000
        content = response.get("content", "").strip()

        # Parse KEEP line if present, otherwise parse entire content
        keep_line = content
        for line in content.split("\n"):
            if line.strip().upper().startswith("KEEP:"):
                keep_line = line.split(":", 1)[1].strip()
                break

        logger.info(
            f"LLM filter batch {batch_index}: elapsed_ms={elapsed:.1f}, batch_size={len(batch_items)}, "
            f"response_preview={content[:150]}"
        )

        # An explicit "NONE" is a real judgement (model saw the batch and kept
        # nothing) — honour it. Everything else that yields no usable indices is
        # a PARSE failure, not a judgement, so it fails open below.
        if keep_line.upper().strip() == "NONE":
            return []

        # Extract numbers and map to global indices
        kept_global = []
        for part in keep_line.split(","):
            try:
                idx = int(part.strip()) - 1  # 1-indexed to 0-indexed within batch
                if 0 <= idx < len(batch_items):
                    kept_global.append(global_offset + idx)
            except ValueError:
                continue

        if not kept_global:
            reason = "empty response" if not content else "unparseable KEEP line"
            logger.warning(
                f"LLM filter batch {batch_index} failed ({reason}) — "
                f"failing open, kept {len(keep_all)} items, "
                f"response_preview={content[:150]!r}"
            )
            return keep_all

        return kept_global

    async def _apply_temporal_window_filter(
        self,
        items: List[ContextItem],
        query: str,
        query_date: Optional[datetime],
        correlation_id: Optional[str] = None,
    ) -> List[ContextItem]:
        """Drop items whose KNOWN event date falls outside the period the question names.

        Off unless ``SYNAP_RETRIEVAL_TEMPORAL_WINDOW_FILTER`` is on AND the
        caller sent ``query_date``. Without the date there is nothing to count
        back from, so the filter cannot run; that is why it is not gated on the
        flag alone.

        THE WORK IS SPLIT BY WHAT EACH PART IS GOOD AT. A model reads the
        question and returns a structured reading; this code does the calendar
        arithmetic. Measured against the real Jev API on 2026-10-01, asked
        directly whether a memory's date fell inside "two months ago" from
        2023-03-11, it placed the answer-bearing January memory OUTSIDE and a
        February distractor INSIDE. It classified the question correctly at
        confidence 1.00. So it classifies and never counts.

        THREE REASONS AN ITEM IS NEVER DROPPED, all in ``item_is_outside``: no
        date, a date that is only the conversation date standing in for a
        missing one, or a window that overlaps. That caps the filter's reach at
        the third of served memories carrying a real date, and it is the
        property that makes it safe on multi-hop questions, where the evidence
        can sit outside the window the question names.

        Every failure path returns ``items`` unchanged.
        """
        from synap.cloud.application.client_api.temporal_question import (
            READING_SYSTEM_PROMPT, item_is_outside, parse_reading, resolve_window,
        )
        from synap.cloud.application.client_api.prompts.temporal_question import (
            build_question_message,
        )

        if not temporal_window_filter_enabled() or query_date is None or not items:
            return items
        if self.llm_client is None:
            record_retrieval_fault("Temporal window filter", "no LLM client, cannot read the question")
            return items

        messages = [
            {"role": "system", "content": READING_SYSTEM_PROMPT},
            {"role": "user", "content": build_question_message(query, query_date)},
        ]
        try:
            # temperature 0 and json_mode for the same reason the subquery
            # generator uses them: a different reading is a different window,
            # and a window that moves between identical requests churns the
            # served set.
            try:
                response = await asyncio.wait_for(
                    self.llm_client.complete(
                        messages=messages, temperature=0.0, max_tokens=300, json_mode=True,
                        telemetry_operation="retrieval.temporal_question_reading",
                        telemetry_correlation_id=correlation_id,
                    ),
                    timeout=_TEMPORAL_READING_TIMEOUT_SECS,
                )
            except TypeError:
                # Adapters that do not take the telemetry kwargs, matching the
                # fallback the subquery generator already carries.
                response = await asyncio.wait_for(
                    self.llm_client.complete(
                        messages=messages, temperature=0.0, max_tokens=300, json_mode=True,
                    ),
                    timeout=_TEMPORAL_READING_TIMEOUT_SECS,
                )
        except Exception as exc:
            logger.warning(
                f"Temporal window filter: reading the question failed, keeping every item: "
                f"{type(exc).__name__}: {exc} (correlation_id={correlation_id})"
            )
            record_retrieval_fault("Temporal window filter", f"question reading unavailable: {type(exc).__name__}")
            return items

        text = response.get("content", "") if isinstance(response, dict) else str(response)
        reading = parse_reading(text)
        if reading is None:
            logger.warning(
                f"Temporal window filter: unparseable reading, keeping every item "
                f"(correlation_id={correlation_id}, reply={str(text)[:160]!r})"
            )
            record_retrieval_fault("Temporal window filter", "unparseable question reading")
            return items

        window = resolve_window(reading, query_date)
        if window is None:
            logger.info(
                f"Temporal window filter: the question restricts no period, keeping all "
                f"{len(items)} items (correlation_id={correlation_id})"
            )
            return items

        kept = [it for it in items if not item_is_outside(it, window)]
        dropped = len(items) - len(kept)
        if not kept:
            # A window that empties the bundle is a wrong window, not a correct
            # verdict that the user has no relevant memories. Keep everything.
            logger.error(
                f"Temporal window filter: window {window} would have dropped ALL "
                f"{len(items)} items; discarding the filter (correlation_id={correlation_id})"
            )
            record_retrieval_fault("Temporal window filter", "window matched nothing, filter discarded")
            return items
        logger.info(
            f"Temporal window filter: phrase={window.phrase!r} window={window} "
            f"dropped={dropped} kept={len(kept)} (correlation_id={correlation_id})"
        )
        return kept

    async def _apply_jev_rerank(
        self,
        items: List[ContextItem],
        query: str,
        correlation_id: Optional[str] = None,
    ) -> List[ContextItem]:
        """Re-order candidates by Jev's relevance verdict. Never drops one.

        Deliberately independent of ``self.llm_client``: Jev is its own service,
        so a deployment with no LLM client still re-ranks. It is also called
        OUTSIDE the LLM filter block for that reason — inside it, a missing
        llm_client would silently skip the re-rank too.

        Chunks bypass, matching ``filter_items_with_llm``: they are already
        score-gated in ``_search_sentences``, and they carry sentence text rather
        than a memory, which the rubric is not written for.
        """
        if not items:
            return items
        chunk_items = [item for item in items if item.source == "chunks"]
        non_chunk_items = [item for item in items if item.source != "chunks"]
        if len(non_chunk_items) < 2:
            return items
        reranked = await jev_rerank_items(
            non_chunk_items,
            query,
            correlation_id=correlation_id,
            on_fault=lambda reason: record_retrieval_fault("Jev re-rank", reason),
        )
        # An item count that changed means the re-ranker dropped or duplicated
        # something, which it is contracted never to do. Trust the input over a
        # contract violation rather than serve a short bundle.
        if len(reranked) != len(non_chunk_items):
            logger.error(
                f"Jev re-rank changed the item count "
                f"({len(non_chunk_items)} -> {len(reranked)}); discarding its order"
            )
            record_retrieval_fault("Jev re-rank", "returned a different item count")
            return items
        return reranked + chunk_items

    async def filter_items_with_llm(
        self,
        items: List[ContextItem],
        query: str,
        scope: Optional[ScopeHierarchy] = None,
        correlation_id: Optional[str] = None,
    ) -> List[ContextItem]:
        """
        LLM-based filtering for ACCURATE mode.

        Splits items into batches of ~30, filters in parallel for better accuracy.
        Returns fallback on error or empty result (safety net).

        Failure handling is fail-open at BOTH levels: a batch that errors,
        times out, or returns an unparseable KEEP line keeps all of its items
        (see ``_filter_batch_with_llm``), and a whole-call failure still falls
        back to the top-K by similarity.
        """
        import time as _time

        if not items:
            return []

        if not _LLM_FILTER_ENABLED:
            # Same result the fail-open path produces today, minus the round
            # trip. See _LLM_FILTER_ENABLED for the measurements.
            logger.info(
                f"LLM filter disabled (SYNAP_RETRIEVAL_LLM_FILTER=off): "
                f"kept all {len(items)} items, skipped the model call"
            )
            return items

        sorted_for_llm = sorted(items, key=lambda x: x.similarity_score, reverse=True)

        FALLBACK_TOP_K = 20
        fallback = sorted_for_llm[:FALLBACK_TOP_K]

        from synap.cloud.application.client_api.prompts.relevance_filter import (
            RELEVANCE_FILTER_SYSTEM_PROMPT,
        )

        system_prompt = RELEVANCE_FILTER_SYSTEM_PROMPT

        BATCH_SIZE = 30
        batches = []
        for i in range(0, len(sorted_for_llm), BATCH_SIZE):
            batches.append(sorted_for_llm[i:i + BATCH_SIZE])

        logger.info(
            f"LLM filter starting: total_items={len(sorted_for_llm)}, "
            f"num_batches={len(batches)}, batch_size={BATCH_SIZE}"
        )

        t0_total = _time.monotonic()

        try:
            # Bound the per-request fan-out so a large candidate set doesn't open
            # ceil(N/30) simultaneous Groq calls. The semaphore only limits how many
            # batches run at once; each batch is still filtered independently and the
            # results are merged by set-union below, so the kept set is identical to
            # an unbounded gather.
            sem = asyncio.Semaphore(max(1, self.config.llm_filter_max_concurrency))

            async def _run_batch(batch_index, batch_items):
                async with sem:
                    return await self._filter_batch_with_llm(
                        batch_items=batch_items,
                        batch_index=batch_index,
                        global_offset=batch_index * BATCH_SIZE,
                        query=query,
                        system_prompt=system_prompt,
                        scope=scope,
                        correlation_id=correlation_id,
                    )

            tasks = [_run_batch(i, batch) for i, batch in enumerate(batches)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_kept_indices = set()
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Same fail-open contract as inside _filter_batch_with_llm:
                    # an exception that escaped the batch helper (semaphore /
                    # cancellation / programming error) must not silently delete
                    # that batch's candidates.
                    keep_all = list(
                        range(i * BATCH_SIZE, i * BATCH_SIZE + len(batches[i]))
                    )
                    logger.warning(
                        f"LLM filter batch {i} failed "
                        f"({type(result).__name__}: {result}) — failing open, "
                        f"kept {len(keep_all)} items"
                    )
                    all_kept_indices.update(keep_all)
                    continue
                all_kept_indices.update(result)

            total_elapsed = (_time.monotonic() - t0_total) * 1000
            kept_indices = sorted(all_kept_indices)

            kept_after_40 = [i for i in kept_indices if i >= 40]
            logger.info(
                f"LLM filter complete: total_elapsed_ms={total_elapsed:.1f}, "
                f"num_batches={len(batches)}, items_kept={len(kept_indices)}, "
                f"items_total={len(sorted_for_llm)}, "
                f"kept_positions={kept_indices}, "
                f"kept_after_40={len(kept_after_40)}/{len(kept_indices)}"
            )

            if not kept_indices:
                logger.info(
                    f"LLM filter kept 0 items across all batches; falling back to top-K: fallback_k={FALLBACK_TOP_K}"
                )
                return fallback

            filtered = [sorted_for_llm[i] for i in kept_indices]
            return filtered

        except Exception as e:
            total_elapsed = (_time.monotonic() - t0_total) * 1000
            logger.warning(
                f"LLM filtering failed, falling back to top-K: {e}, "
                f"fallback_k={FALLBACK_TOP_K}, elapsed_ms={total_elapsed:.1f}"
            )
            return fallback

    def _apply_token_budget(
        self,
        bundle: ContextBundle,
        decision: ContextDecision,
    ) -> ContextBundle:
        """Apply token budget to bundle, truncating if needed.

        If conversation_context is present, reserves up to 30% of the budget
        (capped at 600 tokens) for it. Trims recent_turns if over allocation.
        """
        budget = decision.token_budget
        budget_mode = decision.budget_mode

        # Reserve tokens for conversation_context (30% of budget, capped at 600).
        # The reservation is a HARD cap on how much budget conv_ctx can eat — if
        # the actual conv_ctx payload exceeds its allocation we truncate it to
        # fit, never the other way around. Previously the deduction used the
        # raw count, which could go larger than the entire budget when a
        # compaction summary was long, leaving negative budget for items and
        # truncating every memory to zero (silently).
        conv_ctx_tokens = 0
        if bundle.conversation_context:
            # The compacted summary IS the point of conv_ctx — the previous
            # 600-token ceiling truncated v4 summaries (~880 tokens) and
            # surfaced as ``…[truncated]`` in the playground demo even though
            # the DB row was complete. 2000 fits the longest summaries we've
            # observed (3500 chars ≈ 880 tokens) while still leaving items
            # budget proportional to the caller's request. The 30%-of-budget
            # gate alone was insufficient: the SDK default ``token_budget=2000``
            # produced ``conv_ctx_budget = min(600, 2000) = 600`` — back to
            # the very ceiling we raised. A 1500-token floor guarantees the
            # demo summary fits even on the smallest caller budgets, at the
            # cost of items budget (default 500 instead of 1400). For the
            # playground demo the summary is what the user actually sees in
            # the panel, so this is the correct tradeoff.
            conv_ctx_budget = min(max(int(budget * 0.30), 1500), 2000)
            conv_ctx_text = json.dumps(bundle.conversation_context.to_dict(), default=str)
            conv_ctx_tokens = self.token_counter.count(conv_ctx_text)

            # Trim recent_turns first (cheapest content to drop).
            while conv_ctx_tokens > conv_ctx_budget and bundle.conversation_context.recent_turns:
                bundle.conversation_context.recent_turns.pop(0)  # Remove oldest
                conv_ctx_text = json.dumps(bundle.conversation_context.to_dict(), default=str)
                conv_ctx_tokens = self.token_counter.count(conv_ctx_text)

            # If still over allocation, truncate the summary itself.
            if conv_ctx_tokens > conv_ctx_budget and bundle.conversation_context.summary:
                # Estimate target chars; refine in a small loop because tokens-per-char
                # varies. Cap at 4 iterations to bound work.
                summary = bundle.conversation_context.summary
                for _ in range(4):
                    if conv_ctx_tokens <= conv_ctx_budget:
                        break
                    overshoot_ratio = conv_ctx_budget / max(conv_ctx_tokens, 1)
                    new_len = max(int(len(summary) * overshoot_ratio) - 16, 0)
                    if new_len >= len(summary):
                        break
                    summary = summary[:new_len].rstrip() + "…[truncated]"
                    bundle.conversation_context.summary = summary
                    conv_ctx_text = json.dumps(bundle.conversation_context.to_dict(), default=str)
                    conv_ctx_tokens = self.token_counter.count(conv_ctx_text)

            # Deduct at most the allocation. If truncation above failed to bring
            # actual tokens under the cap (pathological case), we still only
            # charge the allocation against the items budget — the alternative
            # is wiping every memory, which is strictly worse for the caller.
            budget = budget - min(conv_ctx_tokens, conv_ctx_budget)

        # Count current tokens and items before budget
        total_tokens = 0
        items_before_budget = {
            ctx_type: len(items)
            for ctx_type, items in bundle.items_by_type.items()
        }

        for items in bundle.items_by_type.values():
            total_tokens += self.token_counter.count_items(
                [i.to_dict() for i in items]
            )

        bundle.total_tokens = total_tokens + conv_ctx_tokens

        logger.info(
            f"Token budget check: total_tokens={total_tokens}, conv_ctx_tokens={conv_ctx_tokens}, budget={budget}, budget_mode={budget_mode}, over_budget={total_tokens > budget}, items_before_budget={items_before_budget}, total_items_before={sum(items_before_budget.values())}"
        )

        if total_tokens <= budget:
            return bundle

        # Over budget
        if budget_mode == "soft":
            bundle.budget_exceeded = True
            bundle.warnings.append(
                f"Token budget exceeded: {total_tokens}/{budget}"
            )
            logger.warning(
                f"Token budget exceeded (soft mode): {total_tokens}/{budget}"
            )
            return bundle

        # Hard mode: truncate
        logger.info(
            f"Applying hard token budget truncation: {total_tokens}/{budget}"
        )

        remaining_budget = budget
        truncated_by_type: Dict[str, List[ContextItem]] = {}

        # What the client configured, if anything. `category_priorities` is a
        # per-client MACA setting, so when it is present it still decides who is
        # served first. When it is absent (almost every client) there is one
        # tier and the fill below follows the ranking alone.
        maca = _request_maca_or_none(self)
        priority_map = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        configured = bool(maca and maca.category_priorities)
        if configured:
            type_priority = sorted(
                maca.enabled_categories,
                key=lambda t: priority_map.get(maca.category_priorities.get(t, "medium"), 2),
            )

            def _tier_of(t: str) -> int:
                if t not in maca.enabled_categories:
                    return 99
                return priority_map.get(maca.category_priorities.get(t, "medium"), 2)
        else:
            type_priority = ["facts", "preferences", "episodes", "emotions", "temporal_events"]

            def _tier_of(t: str) -> int:
                return 0

        # One ranked list, best first. The old fill visited whole types in a
        # fixed order (facts, preferences, episodes, ...) and let each take what
        # it could, so a weak fact could use the budget a top-ranked episode
        # needed. Now the budget goes to the best items wherever they sit.
        #
        # An item that does not fit CLOSES its tier: nothing ranked below it in
        # that tier is served in its place. That keeps what is served a true
        # top-of-the-ranking prefix, and it is deliberately not "skip it and
        # try the next": that would serve lower-ranked items while a
        # higher-ranked one is missing. With no configured priorities there is
        # one tier, so this is simply "stop at the first item that does not fit".
        closed_tiers: set = set()
        for tier, item in _ranked_across_types(bundle.items_by_type, _tier_of):
            if tier in closed_tiers:
                continue
            item_tokens = self.token_counter.count(item.content)
            # Count source_evidence tokens if present
            if item.source_evidence:
                for ev_sent in item.source_evidence:
                    item_tokens += self.token_counter.count(ev_sent)
            if remaining_budget >= item_tokens:
                truncated_by_type.setdefault(item.context_type, []).append(item)
                remaining_budget -= item_tokens
            else:
                closed_tiers.add(tier)

        # Keep the response's type order as it has always been.
        _type_order = sorted(
            truncated_by_type,
            key=lambda t: type_priority.index(t) if t in type_priority else 99,
        )
        truncated_by_type = {t: truncated_by_type[t] for t in _type_order}

        # Log truncation results
        items_after_budget = {
            ctx_type: len(items)
            for ctx_type, items in truncated_by_type.items()
        }

        logger.info(
            f"Token budget truncation applied: items_before={items_before_budget}, items_after={items_after_budget}, items_dropped={ {ctx_type: items_before_budget.get(ctx_type, 0) - items_after_budget.get(ctx_type, 0) for ctx_type in items_before_budget.keys()} }, total_items_before={sum(items_before_budget.values())}, total_items_after={sum(items_after_budget.values())}, tokens_used={budget - remaining_budget}, budget={budget}"
        )

        bundle.items_by_type = truncated_by_type
        bundle.total_tokens = (budget - remaining_budget) + conv_ctx_tokens
        bundle.budget_exceeded = False

        return bundle

    async def _write_audit_st_snapshot(
        self,
        *,
        correlation_id: Optional[str],
        conv_ctx,
        instance_id: str,
        user_id: Optional[str],
        customer_id: Optional[str],
        conversation_id: str,
    ) -> None:
        """Write the short-term snapshot the EnrichmentWatcher reads.

        Key: ``st_snapshot:{correlation_id}``. Value: JSON-serialized
        conversation_context + scope ids. TTL: 30s (watcher runs every
        5s with a 10s window, so the snapshot must survive at least one
        sweep after entering the eligible-age window).

        Silently no-ops when correlation_id is empty, cache is unavailable,
        or the ConversationContext has no summary/recent_turns to audit.
        """
        if not correlation_id or self.cache is None:
            return
        if not (
            getattr(conv_ctx, "summary", None)
            or getattr(conv_ctx, "recent_turns", None)
        ):
            return
        try:
            snapshot = {
                "conversation_context": conv_ctx.to_dict(),
                "instance_id": instance_id,
                "user_id": user_id or "",
                "customer_id": customer_id or "",
                "conversation_id": conversation_id,
            }
            await self.cache.set(
                f"st_snapshot:{correlation_id}",
                json.dumps(snapshot, default=str),
                ex=30,
            )
        except Exception as e:
            logger.warning(
                "st_snapshot write failed (non-fatal): "
                f"correlation_id={correlation_id} err={e}"
            )

    async def _get_from_cache(self, key: str) -> Optional[List[ContextItem] | Dict[str, Any]]:
        """Get items from cache, deserialize."""
        if self.cache is None:
            return None

        try:
            data = await self.cache.get(key)
            if data:
                parsed = json.loads(data) if isinstance(data, str) else data
                if isinstance(parsed, dict):
                    # A bundle dict (warm_cache / _cache_result) carries "items".
                    # Other dict payloads — e.g. the short-term L2 entry
                    # {"conversation_context": ...} written by
                    # fetch_short_term_context — are returned raw so the caller
                    # can interpret them. Without this the ST L2 cache never hit:
                    # .get("items") was empty -> None, so every ST fetch rebuilt
                    # from compaction storage (P2).
                    if "items" not in parsed:
                        return parsed
                    items_raw = parsed.get("items") or []
                    if not items_raw:
                        return None
                    return [
                        ContextItem.from_dict(item_dict)
                        for item_dict in items_raw
                    ]
                return [
                    ContextItem.from_dict(item_dict)
                    for item_dict in parsed
                ]
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")

        return None

    async def _set_to_cache(
        self,
        key: str,
        items,
        ttl: int,
    ) -> None:
        """Set items to cache, serialize.

        Accepts either List[ContextItem] or a plain dict/list (e.g. conversation context).
        """
        if self.cache is None:
            return

        try:
            if isinstance(items, dict):
                data = items
            elif isinstance(items, list) and items and hasattr(items[0], "to_dict"):
                data = [item.to_dict() for item in items]
            else:
                data = items
            await self.cache.set(key, json.dumps(data), ex=ttl)
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def _build_cache_key(self, decision: ContextDecision, scope: ScopeHierarchy = None) -> str:
        """Build cache key for decision, using resolved scope IDs when available.

        Uses resolved scope UUIDs (matching invalidation keys from ingestion)
        instead of raw external IDs from the decision.
        """
        # Derive scope level for cache key isolation
        if decision.conversation_id:
            scope_level = "conversation"
        elif decision.user_id and decision.user_id != "anonymous":
            scope_level = "user"
        elif decision.customer_id:
            scope_level = "customer"
        else:
            scope_level = "client"

        # Use resolved scope IDs when available (matches invalidation keys from ingestion)
        resolved_client_id = str(scope.client_id) if scope and scope.client_id else decision.client_id
        resolved_customer_id = str(scope.customer_id) if scope and scope.customer_id else (decision.customer_id or "global")

        # Skip-flag lookups return a different (unfiltered / unexpanded) result
        # set than a normal fetch, so they must not share a cache entry with one.
        # Suffix only when a flag is set → existing keys are byte-identical.
        _payload = decision.trigger_payload or {}
        _skip_suffix = ""
        if _payload.get("skip_subquery_gen"):
            _skip_suffix += "|sqg0"
        if _payload.get("skip_llm_filter"):
            _skip_suffix += "|lf0"
        if _payload.get("precision_level") == "medium":
            _skip_suffix += "|pmed"
        # Two request flags change what the bundle CONTAINS, and neither was in
        # the key. A fetch that opted out of source evidence (41.5% of a
        # response body) shared an entry with one that asked for it, so the
        # opt-out came back with the evidence and the opt-in came back without.
        # Same for the conversation block. Suffixed only when the caller opted
        # OUT, so every existing key (the default is on) stays byte-identical.
        if not _payload.get("include_source_evidence", True):
            _skip_suffix += "|noev"
        if not getattr(decision, "include_conversation_context", True):
            _skip_suffix += "|noconv"
        # The question's own date changes which items the temporal window filter
        # keeps, so two fetches differing only in query_date are different
        # results and must not share an entry. Omitted at first, which meant an
        # A/B of with-date against without-date cross-contaminated inside the
        # 300s TTL whenever both arms sent the same question text. Found by the
        # benchmark team, 2026-10-01. Suffixed only when a date is present, so
        # every existing key stays byte-identical.
        if decision.query_date is not None:
            try:
                _skip_suffix += f"|qd{decision.query_date.date().isoformat()}"
            except Exception:
                pass

        # `max_results` is the assembly budget (`budget = decision.max_results`
        # in _assemble_bundle), so it decides how many items the bundle holds.
        # It was missing from this key, and two fetches differing only in
        # max_results therefore shared an entry for the TTL: asking for 20 after
        # asking for 5 returned 5, and asking for 5 after 20 returned 20.
        # Confirmed in both directions.
        #
        # Nothing raises. The caller receives a bundle that looks complete and is
        # the wrong size, for up to five minutes, which is the same shape as the
        # SDK cache bug where the key omitted instance_id and two instances of
        # one account shared a local database.
        #
        # Unlike the skip flags above, this is included **unconditionally**
        # rather than suffixed only when it differs from the default. That
        # changes every existing key, and that is the point: entries already in
        # the cache were written without it and any one of them may be the wrong
        # size for the next caller that finds it. A single cold TTL after deploy
        # is the cost of not serving those.
        _budget_suffix = f"|n{int(decision.max_results or 0)}"
        _skip_suffix += _budget_suffix
        # The per-query share decides WHICH items fill those places, so a fetch
        # that names one and a fetch that does not are different results for
        # the same queries. Suffixed only when present, so every existing key
        # is byte-identical.
        if decision.max_results_per_query:
            _skip_suffix += f"|pq{int(decision.max_results_per_query)}"

        # ⚠ The resolved chain IS the scope, and it was not in this key.
        #
        # `ScopeHierarchy.to_dict` argues that the rungs "are derived from ids
        # already in the key, so carrying them would add length without
        # distinguishing anything". True while the only source of a chain was
        # the legacy adapter, which derives it from exactly `customer_id` and
        # `user_id`. False the moment `scope` (G-03) arrived: an explicit path
        # resolves a chain those ids do not determine.
        #
        # Measured on a four-rung ladder with the same external user id under
        # two teams: chains [client, acme, payments, dana] and
        # [client, acme, security, dana] produced ONE key. Two different people,
        # one cache entry, and whichever request arrived first served the other
        # for the whole TTL. No store is involved in that, so a correct filter
        # underneath cannot prevent it, and nothing in any log looks wrong.
        #
        # Appended to the query component because that is already the part of
        # the key that carries "what exactly was asked", and it is hashed, so a
        # deep ladder costs no key length.
        #
        # Conditional on purpose, exactly like `nested_depth` in `to_dict`.
        # Adding it unconditionally would cold-start the L1 cache for every
        # instance in the fleet, and almost none of them have the feature on.
        # This way the cold start lands only where a chain actually resolves.
        _chain = _nested_chain_ids(scope) if scope is not None else None
        if _chain:
            _skip_suffix += "|chain:" + ",".join(_chain)

        _query_component = "|".join(decision.search_queries) if decision.search_queries else ""

        cache_key = CacheKeyGenerator.l1_query_key(
            client_id=resolved_client_id,
            customer_id=resolved_customer_id,
            user_id=decision.user_id,
            query=_query_component + _skip_suffix,
            context_types=decision.context_types,
            conversation_id=decision.conversation_id,
            scope_level=scope_level,
            mode=decision.retrieval_mode.value if hasattr(decision.retrieval_mode, 'value') else str(decision.retrieval_mode),
            # Section 14 — include instance_id so this writer's keys collide
            # with the anticipation writer's for the same logical scope.
            instance_id=getattr(decision, "instance_id", "") or "",
        )

        logger.debug(
            f"cache_key_generated: cache_key={cache_key}, client_id={resolved_client_id}, customer_id={resolved_customer_id}, user_id={decision.user_id}, search_queries={decision.search_queries}, context_types={decision.context_types}, scope_level={scope_level}"
        )

        return cache_key

    async def _check_cache(self, cache_key: str) -> Optional[ContextBundle]:
        """Check L1 cache for existing bundle."""
        if self.cache is None:
            logger.debug("cache_check_skipped: no cache provider")
            return None

        logger.debug(
            f"cache_check_start: cache_key={cache_key}"
        )

        async with start_span(
            "synap.retrieval.cache_lookup",
            {"synap.cache_key": cache_key},
        ):
            try:
                cached = await self.cache.get(cache_key)
                if cached:
                    data = cached if isinstance(cached, dict) else json.loads(cached)
                    bundle = ContextBundle.from_dict(data)
                    set_current_span_attributes(
                        {
                            "synap.cache_hit": True,
                            "synap.result_count": bundle.item_count if hasattr(bundle, "item_count") else 0,
                        }
                    )

                    logger.info(
                        f"cache_hit: cache_key={cache_key}, item_count={bundle.item_count if hasattr(bundle, 'item_count') else 0}, bundle_id={bundle.bundle_id if hasattr(bundle, 'bundle_id') else None}"
                    )

                    return bundle
                else:
                    set_current_span_attributes({"synap.cache_hit": False})
                    logger.info(
                        f"cache_miss: cache_key={cache_key}"
                    )
            except Exception as e:
                set_current_span_attributes({"error": True, "synap.error_type": type(e).__name__})
                logger.warning(
                    f"cache_check_failed: cache_key={cache_key}, error={e}",
                    exc_info=True
                )

        return None

    async def _cache_result(self, cache_key: str, bundle: ContextBundle) -> None:
        """Cache bundle to L1, unless it is empty or degraded.

        A fetch that returned zero items used to write the
        key anyway, and the next identical request read it back as a hit with
        item_count=0. The TTL is 300s, so one bad moment answered the same
        question with nothing for five minutes.

        "Bad moment" is not hypothetical: FAST times out at 8s and ACCURATE at
        30s, and a timeout produces a MINIMAL or PARTIAL bundle, which the old
        code cached as readily as a complete one. An empty answer is the one
        answer never worth remembering — recomputing it costs a query, while
        serving it costs a question.
        """
        if self.cache is None:
            return

        item_count = int(getattr(bundle, "item_count", 0) or 0)
        level = getattr(bundle, "degradation_level", None)
        degraded = level is not None and level != DegradationLevel.NONE
        # Faults are folded into the degradation level only after this write, so
        # a fetch that read less because the visibility policy was unknown would
        # otherwise be remembered, under this instance's key, for the full TTL.
        if any(
            f.startswith(("visibility policy", _GRAPH_LIVENESS_FAULT))
            for f in (_RETRIEVAL_FAULTS.get() or [])
        ):
            degraded = True
        if item_count == 0 or degraded:
            logger.info(
                "cache_write_skipped: cache_key=%s reason=%s item_count=%d degradation=%s",
                cache_key,
                "empty" if item_count == 0 else "degraded",
                item_count,
                getattr(level, "value", level),
            )
            return

        async with start_span(
            "synap.retrieval.cache_write",
            {
                "synap.cache_key": cache_key,
                "synap.bundle_id": bundle.bundle_id if hasattr(bundle, "bundle_id") else None,
                "synap.result_count": bundle.item_count if hasattr(bundle, "item_count") else 0,
            },
        ):
            try:
                ttl = CacheKeyGenerator.get_ttl("L1")
                await self.cache.set(cache_key, json.dumps(bundle.to_dict(), default=str), ex=ttl)
                set_current_span_attributes({"synap.cache_ttl_seconds": ttl})

                logger.info(
                    f"cache_write_success: cache_key={cache_key}, ttl_seconds={ttl}, bundle_id={bundle.bundle_id if hasattr(bundle, 'bundle_id') else None}, item_count={bundle.item_count if hasattr(bundle, 'item_count') else 0}"
                )
            except Exception as e:
                set_current_span_attributes({"error": True, "synap.error_type": type(e).__name__})
                logger.error(
                    f"cache_write_failed: cache_key={cache_key}, error={e}",
                    exc_info=True
                )

    async def invalidate_user_cache(
        self,
        instance_id: str,
        user_id: Optional[str] = None,
        client_id: str = "",
        customer_id: str = "",
    ) -> int:
        """
        Invalidate L1 query cache and L2 adjacency cache for a user after memory updates.

        Args:
            instance_id: Instance identifier (legacy, used as fallback)
            user_id: User identifier (if None, invalidates all queries for scope)
            client_id: Client identifier for cache key scoping
            customer_id: Customer identifier for cache key scoping

        Returns:
            Number of L1 keys deleted
        """
        # An adjacency entry is partitioned by instance and read boundary, and
        # an ingestion event names only the scope. Drop every variant of this
        # scope's entry (and no other scope's), and mark the scope dirty for the
        # singleton RM in the API process, because this method usually runs on a
        # throwaway RM in the worker.
        effective_client = client_id or instance_id
        effective_customer = customer_id or "global"
        l2_base = f"{effective_client}:{effective_customer}:{user_id or 'global'}"
        dropped = self._adjacency_l2.delete_scope(l2_base)
        if dropped:
            logger.info(
                "adjacency_l2_invalidated: key=%s, variants=%d, source=direct",
                l2_base, dropped,
            )
        _l2_dirty_keys.add(l2_base)  # belt-and-suspenders for throwaway RM instances

        if self.cache is None:
            logger.debug("cache_invalidation_skipped: no cache provider")
            return 0

        try:
            # Pattern: query:{client_id}:{customer_id}:*
            pattern = f"query:{effective_client}:{effective_customer}:*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self.cache.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                if keys:
                    await self.cache.delete(*keys)
                    deleted += len(keys)

                if cursor == 0:
                    break

            logger.info(
                f"Invalidated L1 cache: client_id={effective_client}, customer_id={effective_customer}, "
                f"user_id={user_id}, keys_deleted={deleted}"
            )
            return deleted

        except Exception as e:
            logger.warning(
                f"Cache invalidation failed (non-fatal): {e}, client_id={effective_client}, user_id={user_id}"
            )
            return 0

    def _traverse_graph(
        self,
        adjacency: Dict,
        start_nodes: List[str],
        query_unit: np.ndarray,
        context_types: List[str],
        max_hops: int,
    ) -> List[ContextItem]:
        """
        V2 BFS traversal with vectorized edge scoring.

        Args:
            adjacency: Pre-built adjacency with numpy matrices
            start_nodes: Anchor entity IDs to start traversal from
            query_unit: Normalized query vector
            context_types: Allowed context types
            max_hops: Maximum traversal depth

        Returns:
            List of ContextItem discovered via graph traversal
        """
        items: List[ContextItem] = []
        visited = set(start_nodes)
        current_nodes = list(set(start_nodes))
        matrices = adjacency.get("matrices", {})
        confidence_config = ConfidenceConfig()

        for hop in range(max_hops):
            next_nodes = []

            # Deduplicate current frontier
            unique_nodes = list(set(current_nodes))

            for node_id in unique_nodes:
                edges = adjacency.get("edges", {}).get(node_id, [])
                if not edges:
                    continue

                # Vectorized edge scoring using pre-computed matrices
                scored_edges = self._score_edges_hybrid(
                    query_unit,
                    edges,
                    node_id,
                    matrices,
                    confidence_config,
                )

                for edge, score in scored_edges:
                    target_id = edge["target_id"]

                    if target_id in visited:
                        continue
                    visited.add(target_id)

                    target_node = adjacency.get("nodes", {}).get(target_id, {})

                    # Entity nodes are transparent waypoints — traverse through
                    # them but don't emit as results. Only memory nodes produce
                    # ContextItems. This eliminates triple-text noise like
                    # "John INTERESTED_IN dance" from filling context slots.
                    if target_node.get("entity_type") != "memory":
                        next_nodes.append(target_id)
                        continue

                    node_type = target_node.get("context_type", target_node.get("memory_type", "facts"))

                    if node_type not in context_types:
                        next_nodes.append(target_id)
                        continue

                    # Parse created_at from node metadata
                    created_at = None
                    if target_node.get("created_at"):
                        try:
                            from dateutil import parser
                            created_at = parser.isoparse(target_node["created_at"])
                        except Exception:
                            pass

                    # Hop-distance decay: penalize items further from anchor
                    # hop 0: 1.0, hop 1: 0.77, hop 2: 0.63
                    hop_penalty = 1.0 / (1.0 + 0.3 * hop)
                    decayed_score = score * hop_penalty

                    logger.debug(
                        f"graph_item_constructed: item_id={target_id}, memory_type={node_type}, hop={hop}, raw_score={round(score, 3)}, hop_penalty={round(hop_penalty, 3)}, decayed_score={round(decayed_score, 3)}, raw_context_type={target_node.get('context_type')}, raw_memory_type={target_node.get('memory_type')}"
                    )

                    items.append(ContextItem(
                        item_id=target_id,
                        content=target_node.get("content", ""),
                        context_type=node_type,
                        source="graph",
                        similarity_score=decayed_score,
                        relevance_score=decayed_score,
                        confidence=target_node.get("confidence", 0.5),
                        scope=target_node.get("scope") or derive_item_scope(
                            user_id=target_node.get("user_id"),
                            customer_id=target_node.get("customer_id"),
                        ),
                        entity_id=target_id,
                        # Carry canonical_name for Neo4j anchor resolution in accurate mode
                        entity_name=target_node.get("canonical_name", ""),
                        created_at=created_at,
                        **_event_fields(target_node.get),
                        valid_until=_safe_parse_datetime(target_node.get("valid_until")),
                        temporal_category=target_node.get("temporal_category"),
                        temporal_confidence=float(target_node.get("temporal_confidence", 0)) if target_node.get("temporal_confidence") is not None else 0.0,
                        hop_distance=hop,
                        source_document_id=target_node.get("source_document_id"),
                    ))
                    next_nodes.append(target_id)

            current_nodes = next_nodes
            if not current_nodes:
                break

        if items:
            _type_breakdown = {}
            for _item in items:
                _type_breakdown[_item.context_type] = _type_breakdown.get(_item.context_type, 0) + 1
            logger.info(
                f"traverse_graph_type_breakdown: total_items={len(items)}, type_breakdown={_type_breakdown}, start_nodes_count={len(start_nodes)}"
            )

        return items

    def _score_edges_hybrid(
        self,
        query_unit: np.ndarray,
        edges: List[Dict],
        node_id: str,
        matrices: Dict,
        config: ConfidenceConfig,
    ) -> List[Tuple[Dict, float]]:
        """
        Vectorized hybrid edge scoring using pre-computed numpy matrices.

        Score = 0.7 × edge_similarity + 0.3 × target_similarity
        Uses single matrix multiply instead of per-edge loops.
        Memory edges use a stricter threshold than entity edges.

        Args:
            query_unit: Pre-normalized query vector (dim,)
            edges: List of edge dicts for this node
            node_id: Source node ID (for matrix lookup)
            matrices: Pre-computed {node_id: {edge_matrix, target_matrix, memory_mask}}
            config: Confidence thresholds
        """
        if not edges:
            return []

        node_matrices = matrices.get(node_id)

        if node_matrices is not None:
            # --- VECTORIZED PATH: use pre-computed matrices ---
            edge_matrix = node_matrices["edge_matrix"]      # (E, dim) pre-normalized
            target_matrix = node_matrices["target_matrix"]   # (E, dim) pre-normalized
            memory_mask = node_matrices["memory_mask"]        # (E,) boolean

            # Single matrix multiply — scores all edges at once
            edge_scores = edge_matrix @ query_unit            # (E,)
            target_scores = target_matrix @ query_unit        # (E,)

            final_scores = 0.7 * edge_scores + 0.3 * target_scores

            # Vectorized memory boost
            if memory_mask.any():
                final_scores[memory_mask] *= config.memory_boost

            # Vectorized threshold filtering
            thresholds = np.full(len(edges), config.edge_safety_floor, dtype=np.float32)
            thresholds[memory_mask] = config.memory_threshold
            valid_mask = final_scores >= thresholds

            valid_indices = np.where(valid_mask)[0]
            results = [(edges[i], float(final_scores[i])) for i in valid_indices]
        else:
            # --- FALLBACK: per-edge computation (for edges without pre-computed matrices) ---
            results = []
            for edge in edges:
                edge_vector = np.array(edge.get("edge_vector", []), dtype=np.float32)
                target_vector = np.array(edge.get("target_vector", []), dtype=np.float32)

                if edge_vector.size == 0 or target_vector.size == 0:
                    continue

                edge_norm = edge_vector / (np.linalg.norm(edge_vector) + 1e-9)
                target_norm = target_vector / (np.linalg.norm(target_vector) + 1e-9)

                edge_sim = float(np.dot(query_unit, edge_norm))
                target_sim = float(np.dot(query_unit, target_norm))
                score = 0.7 * edge_sim + 0.3 * target_sim

                if edge.get("is_memory", False):
                    score *= config.memory_boost

                threshold = config.memory_threshold if edge.get("is_memory") else config.edge_safety_floor
                if score >= threshold:
                    results.append((edge, score))

        results.sort(key=lambda x: x[1], reverse=True)

        # Log edge scoring details for debugging relevance
        if results:
            _edge_log = []
            for _e, _s in results[:15]:
                _content = _e.get("content", "")[:60]
                _target = _e.get("target_name", _e.get("target_id", "?"))[:30]
                _is_mem = _e.get("is_memory", False)
                _edge_log.append(f"{'[MEM]' if _is_mem else '[ENT]'} {_target}: score={round(_s, 3)}, content='{_content}'")
            _rejected = len(edges) - len(results)
            logger.info(
                f"SCORING_LOG edge_scoring: node={node_id[:20]}, total_edges={len(edges)}, passed={len(results)}, rejected={_rejected}, details=[{' | '.join(_edge_log[:8])}]"
            )

        return results

    def _adjacency_cache_keys(
        self, scope: ScopeHierarchy
    ) -> Tuple[str, Optional[str], Optional[str], str]:
        """Return (l2_key, l3_data_key, l3_meta_key, boundary) for this request.

        The adjacency matrix is a *filtered view*: it holds the records the
        read filter admits, so two callers share one only if the filter is the
        same. The filter is a function of the scope ids, the resolved chain,
        three MACA flags and the visibility set, and that is the whole read
        boundary (``_build_customer_scope_filter`` and ``_nested_scope_where``
        read nothing else, and the collection namespace pins the customer).

        Standard boundary (no chain, nothing widened, no visibility policy, the
        default for almost every client): the graph depends on the scope ids
        alone, so it keeps the shared key it has always had, in L2 and in Redis,
        across every instance of the client. Every existing invalidation and
        purge path (ingestion, consolidation, instance deletion) already knows
        that key.

        Any other boundary is private to this process: the L2 key carries the
        instance and a fingerprint of the boundary, and there is NO Redis key,
        so a graph built under one set of grants can never be loaded under
        another. These are the rare isolated or widened callers, and a rebuild
        after a restart is the price of never sharing across a boundary.
        """
        user_key = scope.user_id or scope.internal_user_id or "global"
        maca = _request_maca(self)
        visibility = _request_visibility(self)
        chain = _nested_chain_ids(scope) or []
        visibility_key = (
            "all" if visibility is VISIBILITY_ALL
            else ",".join(sorted(str(instance_id) for instance_id in visibility))
        )
        boundary = {
            "chain": chain,
            "visibility": visibility_key,
            "roll_up_to_customer": maca.roll_up_to_customer,
            "allow_cross_user_retrieval": maca.allow_cross_user_retrieval,
            "allow_cross_customer_retrieval": maca.allow_cross_customer_retrieval,
        }
        fingerprint = hashlib.sha256(
            json.dumps(boundary, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        scope_key = f"{scope.client_id}:{scope.customer_id}:{user_key}"
        standard = (
            not chain
            and visibility is VISIBILITY_ALL
            and maca.roll_up_to_customer
            and not maca.allow_cross_user_retrieval
            and not maca.allow_cross_customer_retrieval
        )
        if standard:
            return (
                scope_key,
                CacheKeyGenerator.l3_graph_data_key(scope.client_id, scope.customer_id or "none", user_key),
                CacheKeyGenerator.l3_graph_meta_key(scope.client_id, scope.customer_id or "none", user_key),
                fingerprint,
            )
        return f"{scope_key}:{scope.instance_id}:{fingerprint}", None, None, fingerprint

    async def _get_adjacency_matrix(self, scope: ScopeHierarchy) -> Dict:
        """
        Get or build adjacency matrix (lazy + cached).

        Cache hierarchy: L2 (in-memory) → L3 (Redis) → Build (ChromaDB).
        """
        user_key = scope.user_id or scope.internal_user_id or "global"
        l2_key, data_key, meta_key, boundary_fingerprint = self._adjacency_cache_keys(scope)
        legacy_l2_key = f"{scope.client_id}:{scope.customer_id}:{user_key}"

        logger.info(
            f"adjacency_matrix_get_start: client_id={scope.client_id}, customer_id={scope.customer_id}, user_key={user_key}, instance_id={scope.instance_id}, boundary={boundary_fingerprint}, has_cache={self.cache is not None}, data_key={data_key}"
        )

        # Check dirty flags from stage4 (module-level set)
        global _l2_flush_all
        if _l2_flush_all:
            # D-007: subscription (re)established — missed messages possible; flush all.
            _l2_flush_all = False
            _l2_dirty_keys.clear()
            self._adjacency_l2.clear()
            logger.info("adjacency_l2_flushed: source=invalidation_resubscribe")
        if l2_key in _l2_dirty_keys or legacy_l2_key in _l2_dirty_keys:
            _l2_dirty_keys.discard(l2_key)
            _l2_dirty_keys.discard(legacy_l2_key)
            # The flag names the scope, not this request's read boundary. Drop
            # every variant of the scope so a request from another instance does
            # not find the entry this one was spared.
            dropped = self._adjacency_l2.delete_scope(legacy_l2_key)
            if dropped:
                logger.info(
                    "adjacency_l2_invalidated: key=%s, variants=%d, source=dirty_flag",
                    legacy_l2_key, dropped,
                )
            # A request that observes the invalidation must not be served a build
            # that STARTED before it (pre-ingestion data). Drop the in-flight
            # entries, for this boundary and every other boundary of the scope,
            # so this request leads a fresh post-invalidation build and no
            # variant's older build lands in L2 afterwards; waiters already
            # attached to an old flight keep its result — their requests predate
            # the invalidation, matching pre-single-flight behaviour.
            for _flight_key in [
                k for k in self._adjacency_inflight
                if k == legacy_l2_key or k.startswith(legacy_l2_key + ":")
            ]:
                self._adjacency_inflight.pop(_flight_key, None)
                logger.info(
                    "adjacency_inflight_dropped: key=%s, source=dirty_flag", _flight_key
                )

        # L2 in-memory check
        l2_result = self._adjacency_l2.get(l2_key)
        if l2_result is not None:
            return l2_result

        # Single-flight: coalesce concurrent cold requests for the same scope.
        # The prologue above (dirty-flag + L2 read) is fully synchronous, so
        # between the L2 miss here and registering the in-flight future below
        # there is no await point — exactly one task becomes the "leader" that
        # runs the L3 round-trip + rebuild; every other concurrent caller awaits
        # its result. Sharing the leader's output is identical to each having
        # built independently (same scope, same stores), just without the
        # duplicate work; a build error propagates to waiters, who already
        # degrade to adjacency=None exactly as an independent failure would.
        inflight = self._adjacency_inflight.get(l2_key)
        if inflight is None:
            inflight = asyncio.ensure_future(
                self._load_or_build_adjacency(
                    scope, l2_key=l2_key, data_key=data_key, meta_key=meta_key
                )
            )
            self._adjacency_inflight[l2_key] = inflight

            def _clear(fut, key=l2_key):
                # Only remove our own future, never a replacement registered later.
                if self._adjacency_inflight.get(key) is fut:
                    self._adjacency_inflight.pop(key, None)

            inflight.add_done_callback(_clear)

        return await inflight

    async def _load_or_build_adjacency(
        self,
        scope: ScopeHierarchy,
        *,
        l2_key: str,
        data_key: Optional[str],
        meta_key: Optional[str],
    ) -> Dict:
        """L3-cache read + fresh build for one scope (the single-flight leader body).

        ``data_key`` / ``meta_key`` are None for a non-standard read boundary:
        that graph is never shared through Redis (see ``_adjacency_cache_keys``).

        Extracted from ``_get_adjacency_matrix`` so exactly one task runs it per
        scope at a time. Behaviour is unchanged from the original inline path.
        """
        # Check L3 cache
        if self.cache and data_key and meta_key:
            try:
                # Check staleness via meta key (cheap)
                meta_raw = await self.cache.get(meta_key)
                if meta_raw:
                    meta = json.loads(meta_raw)
                    cached_count = meta.get("doc_count")

                    # Time-gated staleness: if built within the last 5 minutes, skip
                    # the doc_count graph query entirely — it's guaranteed fresh.
                    FRESH_WINDOW_SECONDS = 300
                    built_at_str = meta.get("built_at")
                    is_fresh = False
                    if built_at_str:
                        try:
                            built_at = datetime.fromisoformat(built_at_str)
                            age_s = (datetime.now(timezone.utc) - _ensure_aware(built_at)).total_seconds()
                            is_fresh = age_s < FRESH_WINDOW_SECONDS
                        except Exception:
                            pass

                    if is_fresh:
                        current_count = cached_count  # trust cache, skip graph query
                        is_stale = False
                    else:
                        current_count = await self._get_graph_doc_count(scope)
                        is_stale = current_count != -1 and cached_count != current_count

                    logger.info(
                        f"adjacency_l3_cache_check: cached_doc_count={cached_count}, current_doc_count={current_count}, is_stale={is_stale}, is_fresh={is_fresh}, built_at={meta.get('built_at')}"
                    )

                    # -1 means graph unavailable; skip staleness check and use cache
                    if not is_stale:
                        # Not stale, load from cache
                        data_b64 = await self.cache.get(data_key)
                        if data_b64:
                            data = base64.b64decode(data_b64)
                            # Offload CPU-heavy msgpack+numpy deserialization to thread
                            # so concurrent tasks (ChromaDB, LLM) aren't blocked
                            adjacency = await asyncio.to_thread(self._deserialize_adjacency, data)
                            entity_ids = adjacency.get("entity_ids", [])
                            entity_matrix = adjacency.get("entity_matrix")
                            logger.info(
                                f"adjacency_l3_cache_hit: entity_ids_count={len(entity_ids)}, entity_matrix_shape={list(entity_matrix.shape) if hasattr(entity_matrix, 'shape') else None}, nodes_count={len(adjacency.get('nodes', {}))}"
                            )
                            # Populate L2 from L3 hit
                            self._adjacency_l2.put(l2_key, adjacency)
                            return adjacency
                else:
                    logger.info("adjacency_l3_cache_miss")
            except Exception as e:
                logger.warning(f"L3 cache check failed: {e}", exc_info=True)

        # Build fresh adjacency matrix
        logger.info("adjacency_building_fresh")
        adjacency = await self._build_adjacency_matrix(scope)

        # Cache to L3
        if self.cache and adjacency and data_key and meta_key:
            try:
                entity_ids = adjacency.get("entity_ids", [])
                if entity_ids:  # Only cache non-empty adjacency
                    serialized = await asyncio.to_thread(self._serialize_adjacency, adjacency)
                    await self.cache.set(data_key, base64.b64encode(serialized).decode(), ex=CacheKeyGenerator.TTL_L3)
                    await self.cache.set(meta_key, json.dumps({
                        "doc_count": await self._get_graph_doc_count(scope),
                        "built_at": datetime.now(timezone.utc).isoformat(),
                    }), ex=CacheKeyGenerator.TTL_L3)
                    logger.info(f"adjacency_l3_cache_written: entity_ids_count={len(entity_ids)}")
                else:
                    logger.warning("adjacency_l3_cache_skip_empty: empty adjacency, not caching")
            except Exception as e:
                logger.warning(f"L3 cache write failed: {e}", exc_info=True)

        # Populate L2 from fresh build
        if adjacency and adjacency.get("entity_ids"):
            self._adjacency_l2.put(l2_key, adjacency)

        return adjacency

    async def _build_adjacency_matrix(self, scope: ScopeHierarchy) -> Dict:
        """
        Build adjacency matrix from ChromaDB entity + triples collections.

        V2 Architecture:
        1. Fetch entities from entity collection (with embeddings)
        2. Fetch triples from triples collection (with embeddings)
        3. Build adjacency dict with edge vectors
        4. Inject memory edges from mentioned entities to high-confidence memories
        5. Pre-compute numpy matrices per node for vectorized scoring

        Returns dict with:
            nodes: {entity_id: {name, entity_type, content, embedding, ...}}
            edges: {source_id: [edge_dicts with edge_vector, target_vector]}
            matrices: {node_id: {edge_matrix, target_matrix, memory_mask}}
            entity_ids: [str] — ordered list matching entity_matrix rows (excludes memory nodes)
            entity_matrix: np.ndarray (N, dim) — pre-normalized entity embeddings
        """
        if self.vector is None:
            logger.warning("adjacency_build_skip: no vector provider")
            return {"nodes": {}, "edges": {}, "matrices": {}, "entity_ids": [], "entity_matrix": None}

        build_start = time.time()

        try:
            from synap.cloud.agents.ingestion.models import Scope as IngestionScope
            from synap.cloud.shared.scope import parse_or_deterministic_uuid

            vector_scope = IngestionScope(
                client_id=parse_or_deterministic_uuid(scope.client_id),
                instance_id=parse_or_deterministic_uuid(scope.instance_id),
                customer_id=parse_or_deterministic_uuid(scope.customer_id) if scope.customer_id else None,
                user_id=parse_or_deterministic_uuid(scope.user_id) if scope.user_id else None,
            )

            # --- 1. Build scoping filters from MACA config ---
            maca = _request_maca(self)
            vis = _request_visibility(self)
            if vis is not VISIBILITY_ALL and not vis:
                # Nothing is readable (the policy could not be determined).
                # `instance_id $in []` is not a clause the store accepts, and an
                # empty graph is the answer anyway.
                return {"nodes": {}, "edges": {}, "matrices": {}, "entity_ids": [], "entity_matrix": None}
            scope_filters = _build_customer_scope_filter(scope, maca, vis)

            # The graph pool must be the set the read filter admits, not a
            # narrower one. `_build_customer_scope_filter` names a customer and
            # a user and has no field for a rung between them, so on a ladder
            # like client -> customer -> practice -> care_team -> user every
            # record filed at a practice or a care team is invisible to the
            # graph while being perfectly readable through the vector arm. The
            # graph then answers a question about a smaller world and nothing
            # about it looks broken.
            #
            # `_nested_scope_where` is the same clause the vector read sends, so
            # the two arms cannot disagree about what this request may see. It
            # returns None for every request without a resolved chain, which is
            # every client on the default ladder today, and that branch is byte for byte
            # what ran before.
            _adj_nested = _nested_scope_where(scope, maca, vis)
            if _adj_nested is _NOTHING_READABLE:
                logger.info("adjacency_build_skip: requester can read nothing")
                return {"nodes": {}, "edges": {}, "matrices": {}, "entity_ids": [], "entity_matrix": None}
            _customer_filters = scope_filters if _adj_nested is None else _adj_nested

            logger.info(
                f"adjacency_build_start: client_id={scope.client_id}, instance_id={scope.instance_id}, customer_id={scope.customer_id}, user_id={scope.user_id}, scope_filters={_customer_filters}, nested={_adj_nested is not None}, vector_scope_client_id={str(vector_scope.client_id)}"
            )

            # --- 2 & 3. Fetch entities, triples, and memories in parallel ---
            # Scope chain: fetch from both customer-level and client-level collections.
            # Strip instance_id too: client-scope shared knowledge is never
            # instance-isolated (mirrors the customer-only rule in the vector sites).
            client_scope_filters = {k: v for k, v in scope_filters.items() if k not in ("user_id", "customer_id", "instance_id")}

            fetch_results = await asyncio.gather(
                # Customer-level (primary)
                self._fetch_all_from_collection(
                    vector_scope, collection_type="entities", filters=_instance_lists_where(_customer_filters), scope_level="customer",
                ),
                self._fetch_all_from_collection(
                    vector_scope, collection_type="triples", filters=_instance_lists_where(_customer_filters), scope_level="customer",
                ),
                self._fetch_all_from_collection(
                    vector_scope, collection_type="memories", scope_level="customer",
                    # ⚠ Built as an explicit `$and` rather than `{**a, **b}`.
                    # The nested clause can be an `$or`, and merging one into a
                    # flat dict silently drops it: the request would then read
                    # the whole collection filtered only on confidence. Same
                    # trap `_direct_similarity_search` documents at its own
                    # nested branch.
                    filters=_and_where(
                        _customer_filters,
                        {"confidence": {"$gte": 0.7}},
                        {"entities_mentioned": {"$ne": ""}},
                    ),
                    top_k=200,
                ),
                # Client-level (shared/global knowledge)
                self._fetch_all_from_collection(
                    vector_scope, collection_type="entities", filters=client_scope_filters, scope_level="client",
                ),
                self._fetch_all_from_collection(
                    vector_scope, collection_type="triples", filters=client_scope_filters, scope_level="client",
                ),
                self._fetch_all_from_collection(
                    vector_scope, collection_type="memories", scope_level="client",
                    filters={**client_scope_filters, "confidence": {"$gte": 0.7}, "entities_mentioned": {"$ne": ""}},
                    top_k=50,
                ),
                return_exceptions=True,
            )

            # Unpack and merge, deduplicating by ID
            def _safe_list(result):
                if isinstance(result, Exception):
                    logger.warning(f"Adjacency fetch failed: {result}")
                    return []
                return result

            def _merge_dedup(customer_results, client_results):
                seen_ids = set()
                merged = []
                for item in customer_results:
                    if item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        merged.append(item)
                for item in client_results:
                    if item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        merged.append(item)
                return merged

            entity_results = _merge_dedup(_safe_list(fetch_results[0]), _safe_list(fetch_results[3]))
            triple_results = _merge_dedup(_safe_list(fetch_results[1]), _safe_list(fetch_results[4]))
            memory_results = _merge_dedup(_safe_list(fetch_results[2]), _safe_list(fetch_results[5]))

            logger.info(
                f"adjacency_entities_and_triples_fetched: entity_count={len(entity_results)}, triple_count={len(triple_results)}, sample_ids={[e['id'][:20] for e in entity_results[:3]] if entity_results else []}, sample_has_embedding={[e.get('embedding') is not None and len(e.get('embedding', [])) > 0 for e in entity_results[:3]] if entity_results else []}, sample_embedding_dims={[len(e.get('embedding', [])) for e in entity_results[:3]] if entity_results else []}"
            )

            # --- 4. Build nodes dict from entities ---
            nodes = {}
            for entity in entity_results:
                eid = entity["id"]
                meta = entity.get("metadata", {})
                nodes[eid] = {
                    "name": meta.get("name", ""),
                    "canonical_name": meta.get("canonical_name", ""),
                    "entity_type": meta.get("entity_type", ""),
                    "content": entity.get("content", ""),
                    "embedding": entity.get("embedding", []),
                    "context_type": meta.get("memory_type", "facts"),
                    "confidence": meta.get("confidence", 0.5),
                    "scope": "user",
                }

            logger.info(
                f"adjacency_nodes_built: node_count={len(nodes)}, nodes_with_embeddings={sum(1 for n in nodes.values() if n.get('embedding') is not None and len(n.get('embedding', [])) > 0)}, entity_types={list(set(n.get('entity_type', '') for n in nodes.values()))[:10]}"
            )

            # --- 5. Build edges dict from triples ---
            edges = {}
            skipped_triples = 0
            for triple in triple_results:
                meta = triple.get("metadata", {})
                source_id = meta.get("source_entity_id", "")
                target_id = meta.get("target_entity_id", "")

                if not source_id or not target_id:
                    skipped_triples += 1
                    continue

                # Edge vector = triple embedding, target vector = target entity embedding
                edge_vector = triple.get("embedding", [])
                target_node = nodes.get(target_id, {})
                target_vector = target_node.get("embedding", [])

                edge_dict = {
                    "target_id": target_id,
                    "edge_vector": edge_vector,
                    "target_vector": target_vector,
                    "relationship_type": meta.get("relationship_type", ""),
                    "source_entity_name": meta.get("source_entity_name", ""),
                    "target_entity_name": meta.get("target_entity_name", ""),
                    "is_memory": False,
                    "content": triple.get("content", ""),
                }

                if source_id not in edges:
                    edges[source_id] = []
                edges[source_id].append(edge_dict)

                # --- Reverse edge for bidirectional traversal ---
                source_node = nodes.get(source_id, {})
                source_vector = source_node.get("embedding", [])
                reverse_edge = {
                    "target_id": source_id,
                    "edge_vector": edge_vector,       # Same triple embedding
                    "target_vector": source_vector,    # Swap: target now points to source
                    "relationship_type": f"{meta.get('relationship_type', '')}_REVERSE",
                    "source_entity_name": meta.get("target_entity_name", ""),
                    "target_entity_name": meta.get("source_entity_name", ""),
                    "is_memory": False,
                    "content": triple.get("content", ""),
                }
                if target_id not in edges:
                    edges[target_id] = []
                edges[target_id].append(reverse_edge)

            total_edge_entries = sum(len(el) for el in edges.values())
            logger.info(
                f"adjacency_edges_built: edge_sources={len(edges)}, total_edge_entries={total_edge_entries}, skipped_triples={skipped_triples}"
            )

            # --- 6. Link memory edges (high-confidence memories → mentioned entities) ---
            # Memory results were already fetched in parallel with entities/triples above
            nodes_before_mem = len(nodes)
            self._link_memory_edges(memory_results, nodes, edges)
            memory_nodes_added = len(nodes) - nodes_before_mem

            logger.info(
                f"adjacency_memory_edges_injected: memory_nodes_added={memory_nodes_added}, total_nodes_now={len(nodes)}"
            )

            # --- 7. Build entity matrix for vectorized anchor search ---
            # Exclude memory nodes: they have no outgoing edges and would be dead-end anchors
            entity_ids = [eid for eid in nodes if nodes[eid].get("entity_type") != "memory" and nodes[eid].get("embedding") is not None and len(nodes[eid].get("embedding", [])) > 0]
            entity_matrix = None

            if entity_ids:
                # Check for mismatched embedding dimensions
                embedding_dims = set(len(nodes[eid]["embedding"]) for eid in entity_ids)
                if len(embedding_dims) > 1:
                    logger.warning(
                        f"adjacency_mismatched_dims: unique_dims={sorted(embedding_dims)}, counts_per_dim={ {d: sum(1 for eid in entity_ids if len(nodes[eid]['embedding']) == d) for d in embedding_dims} }"
                    )
                    # Use only the most common dimension
                    from collections import Counter
                    dim_counts = Counter(len(nodes[eid]["embedding"]) for eid in entity_ids)
                    target_dim = dim_counts.most_common(1)[0][0]
                    entity_ids = [eid for eid in entity_ids if len(nodes[eid]["embedding"]) == target_dim]
                    logger.info(f"Filtered to {len(entity_ids)} entities with dim={target_dim}")

                # Filter out zero-length embeddings
                entity_ids = [eid for eid in entity_ids if len(nodes[eid]["embedding"]) > 0]

                embeddings = [nodes[eid]["embedding"] for eid in entity_ids]
                entity_matrix = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(entity_matrix, axis=1, keepdims=True) + 1e-9
                entity_matrix = entity_matrix / norms

            # --- 8. Pre-compute per-node edge matrices ---
            matrices = {}
            for source_id, edge_list in edges.items():
                valid_edges = [
                    e for e in edge_list
                    if e.get("edge_vector") is not None and len(e.get("edge_vector", [])) > 0
                    and e.get("target_vector") is not None and len(e.get("target_vector", [])) > 0
                ]
                if not valid_edges:
                    continue

                edge_vecs = np.array([e["edge_vector"] for e in valid_edges], dtype=np.float32)
                target_vecs = np.array([e["target_vector"] for e in valid_edges], dtype=np.float32)
                memory_mask = np.array([e.get("is_memory", False) for e in valid_edges], dtype=bool)

                # Normalize
                edge_norms = np.linalg.norm(edge_vecs, axis=1, keepdims=True) + 1e-9
                target_norms = np.linalg.norm(target_vecs, axis=1, keepdims=True) + 1e-9
                edge_vecs = edge_vecs / edge_norms
                target_vecs = target_vecs / target_norms

                matrices[source_id] = {
                    "edge_matrix": edge_vecs,
                    "target_matrix": target_vecs,
                    "memory_mask": memory_mask,
                }

                # Replace edge list with only valid edges (matching matrix rows)
                edges[source_id] = valid_edges

            build_elapsed_ms = (time.time() - build_start) * 1000
            logger.info(
                f"adjacency_build_complete: elapsed_ms={round(build_elapsed_ms, 1)}, node_count={len(nodes)}, edge_sources={len(edges)}, matrices_count={len(matrices)}, entity_ids_count={len(entity_ids)}, entity_matrix_shape={list(entity_matrix.shape) if entity_matrix is not None else None}"
            )

            return {
                "nodes": nodes,
                "edges": edges,
                "matrices": matrices,
                "entity_ids": entity_ids,
                "entity_matrix": entity_matrix,
            }

        except Exception as e:
            build_elapsed_ms = (time.time() - build_start) * 1000
            logger.error(
                f"Failed to build adjacency matrix: {repr(e)}, error_type={type(e).__name__}, elapsed_ms={round(build_elapsed_ms, 1)}",
                exc_info=True,
            )
            return {"nodes": {}, "edges": {}, "matrices": {}, "entity_ids": [], "entity_matrix": None}

    def _link_memory_edges(
        self,
        memory_results: List[Dict],
        nodes: Dict,
        edges: Dict,
    ) -> None:
        """
        Link pre-fetched high-confidence memories as HAS_MEMORY edges from mentioned entities.

        Creates virtual memory nodes and HAS_MEMORY edges from each mentioned
        entity to the memory node, making them discoverable via entity-anchored BFS.
        Orphan memories (no entity mentions) are caught by direct similarity search.

        Note: Memory results are fetched in parallel with entities/triples in
        _build_adjacency_matrix, then passed here for CPU-only linking.
        """
        try:
            for mem in memory_results:
                mem_id = f"memory:{mem['id']}"
                meta = mem.get("metadata", {})
                mem_embedding = mem.get("embedding", [])

                if mem_embedding is None or (hasattr(mem_embedding, '__len__') and len(mem_embedding) == 0):
                    continue

                # Add HAS_MEMORY edges from each mentioned entity to this memory
                entities_str = meta.get("entities_mentioned", "")
                if not entities_str:
                    continue

                # Only add the memory node if at least one mentioned entity exists in the graph
                has_linked_entity = False
                for entity_id in entities_str.split(","):
                    entity_id = entity_id.strip()
                    if entity_id and entity_id in nodes:
                        has_linked_entity = True
                        break

                if not has_linked_entity:
                    continue

                # Add memory as a node
                nodes[mem_id] = {
                    "name": mem.get("content", "")[:50],
                    "entity_type": "memory",
                    "content": mem.get("content", ""),
                    "embedding": mem_embedding,
                    "context_type": meta.get("memory_type", "facts"),
                    "memory_type": meta.get("memory_type", "facts"),
                    "confidence": meta.get("confidence", 0.7),
                    "scope": "user",
                    # Temporal metadata for BFS→ContextItem propagation
                    "valid_from": meta.get("valid_from"),
                    "valid_until": meta.get("valid_until"),
                    "temporal_category": meta.get("temporal_category"),
                    "temporal_confidence": float(meta.get("temporal_confidence", 0)) if meta.get("temporal_confidence") is not None else 0.0,
                }

                # Create edges from mentioned entities to this memory
                for entity_id in entities_str.split(","):
                    entity_id = entity_id.strip()
                    if entity_id and entity_id in nodes:
                        if entity_id not in edges:
                            edges[entity_id] = []
                        edges[entity_id].append({
                            "target_id": mem_id,
                            "edge_vector": mem_embedding,
                            "target_vector": mem_embedding,
                            "relationship_type": "HAS_MEMORY",
                            "is_memory": True,
                        })

        except Exception as e:
            logger.warning(f"Memory edge linking failed: {e}")

    async def _fetch_all_from_collection(
        self,
        scope,
        collection_type: str = "memories",
        filters: Optional[Dict] = None,
        top_k: int = 500,
        scope_level: str = "customer",
    ) -> List[Dict]:
        """
        Fetch all documents from a ChromaDB collection WITH embeddings.

        Uses get_all (ChromaDB collection.get) instead of search to retrieve
        all matching documents with their stored embedding vectors.

        Returns dicts with: id, content, metadata, embedding
        """
        fetch_start = time.time()
        logger.info(
            f"fetch_all_collection_start: collection_type={collection_type}, filters={filters}, limit={top_k}, scope_level={scope_level}, scope_client_id={str(scope.client_id) if hasattr(scope, 'client_id') else 'unknown'}"
        )
        try:
            results = await self.vector.get_all_from_collection(
                scope=scope,
                collection_type=collection_type,
                filters=filters,
                limit=top_k,
                scope_level=scope_level,
            )
            fetch_elapsed_ms = (time.time() - fetch_start) * 1000
            logger.info(
                f"fetch_all_collection_complete: collection_type={collection_type}, result_count={len(results)}, elapsed_ms={round(fetch_elapsed_ms, 1)}, has_embeddings={sum(1 for r in results if r.get('embedding') is not None and len(r.get('embedding', [])) > 0) if results else 0}"
            )
            return results
        except Exception as e:
            fetch_elapsed_ms = (time.time() - fetch_start) * 1000
            if type(e).__name__ == "NotFoundError" or "does not exist" in str(e):
                # A collection is created with a user's first memory. Before that, an empty answer is the
                # right one, not a failure with a traceback.
                logger.info(
                    f"No {collection_type} collection yet (nothing stored for this scope): elapsed_ms={round(fetch_elapsed_ms, 1)}"
                )
                return []
            logger.warning(
                f"Failed to fetch from {collection_type} collection: {repr(e)}, error_type={type(e).__name__}, elapsed_ms={round(fetch_elapsed_ms, 1)}",
                exc_info=True,
            )
            return []

    def _serialize_adjacency(self, adjacency: Dict) -> bytes:
        """
        Serialize adjacency to MessagePack with numpy array support.

        Numpy arrays are converted to {__numpy__: True, data: bytes, shape: tuple, dtype: str}.
        """
        import msgpack

        def _encode(obj):
            if isinstance(obj, np.ndarray):
                return {
                    "__numpy__": True,
                    "data": obj.tobytes(),
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
            if isinstance(obj, dict):
                return {k: _encode(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_encode(v) for v in obj]
            return obj

        serializable = _encode(adjacency)
        return msgpack.packb(serializable, use_bin_type=True)

    def _deserialize_adjacency(self, data: bytes) -> Dict:
        """
        Deserialize adjacency from MessagePack with numpy array support.
        """
        import msgpack

        def _decode(obj):
            if isinstance(obj, dict):
                if obj.get("__numpy__"):
                    return np.frombuffer(
                        obj["data"],
                        dtype=np.dtype(obj["dtype"]),
                    ).reshape(obj["shape"]).copy()
                return {k: _decode(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_decode(v) for v in obj]
            return obj

        raw = msgpack.unpackb(data, raw=False)
        return _decode(raw)

    async def _get_embedding(self, text: str) -> list:
        """Get embedding for text (via embedding client or vector provider)."""
        t0 = time.time()
        async with start_span(
            "synap.retrieval.embedding.single",
            {
                "synap.text_len": len(text),
            },
        ):
            # 1. Use dedicated embedding client (default: Snowflake/snowflake-arctic-embed-l-v2.0, 1024 dims)
            if self.embedding_client and hasattr(self.embedding_client, "embed"):
                result = await self.embedding_client.embed(text, is_query=True)
                latency_ms = round((time.time() - t0) * 1000, 2)
                set_current_span_attributes({
                    "synap.embedding_backend": "embedding_client",
                    "synap.embedding_dims": len(result) if result else 0,
                    "synap.embedding_latency_ms": latency_ms,
                })
                return list(result) if not isinstance(result, list) else result

            # 2. Try vector provider's embed method
            if self.vector and hasattr(self.vector, "embed"):
                result = await self.vector.embed(text)
                latency_ms = round((time.time() - t0) * 1000, 2)
                set_current_span_attributes({
                    "synap.embedding_backend": "vector_provider",
                    "synap.embedding_dims": len(result) if result else 0,
                    "synap.embedding_latency_ms": latency_ms,
                })
                return list(result) if not isinstance(result, list) else result

            # A fabricated vector can retrieve unrelated private memories.
            # Fail explicitly so the request degrades instead of returning noise.
            set_current_span_attributes({"synap.embedding_backend": "unavailable"})
            raise RuntimeError("No embedding backend is configured for retrieval")

    async def _get_graph_doc_count(self, scope: ScopeHierarchy) -> int:
        """Get document count for staleness check."""
        if self.graph is None:
            return -1
        try:
            return await self.graph.count(filter={"customer_id": scope.customer_id or scope.client_id})
        except Exception:
            return 0

    async def _generate_retrieval_subqueries(
        self,
        original_query: str,
        mode: str = "fast",
        maca_settings: MACARetrievalSettings = None,
        scope: Optional[ScopeHierarchy] = None,
        correlation_id: Optional[str] = None,
        graph_entity_types: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Generate subqueries for retrieval decomposition.

        FAST mode: template-based (instant, no LLM cost)
        ACCURATE mode: LLM decomposition. With no LLM client, or when the call
        fails, the spaCy splitter (rule-based, no model call) splits the question
        and the fixed templates are added.

        Args:
            graph_entity_types: Entity types present in the user's graph (from
                L2 cache peek). When provided in FAST mode, B2B templates are
                skipped if no B2B entity types are found — saves encode + search
                time without affecting quality.

        Returns:
            Dict with "queries" (List[str]). The graph relationship types the
            LLM used to be asked for were only logged, never used, and are gone.
        """
        queries = [original_query]

        # ── Determinism, and why this cache exists ────────────────────────────
        #
        # Three identical `accurate` fetches
        # could produce candidate pools of different sizes, and the divergence was
        # here. The subquery COUNT was stable; the TEXT was not
        # ("does the user have" became "does the system have", "hobbies" became
        # "habits"). Different subqueries are different vector searches, so a
        # different pool, so a different top ten.
        #
        # `temperature=0.0` reduced that and did not remove it. Providers still
        # vary at temperature 0, so pinning it was right and was never the fix.
        # It is also why `accurate` was unstable while FAST was not: FAST
        # decomposes with `_CORE_RETRIEVAL_TEMPLATES`, which are fixed.
        #
        # The subqueries are a pure function of the question and the instance's
        # own prompt inputs. They do not depend on the caller, the scope, or the
        # stored data, so the same question can safely reuse them. That makes a
        # repeated question return a repeated answer, and skips an LLM round trip
        # on the slowest path.
        _subq_cache_key = None
        if mode == "accurate" and self.llm_client:
            _subq_cache_key = "subq:v1:" + hashlib.sha256(
                json.dumps(
                    {
                        "q": original_query,
                        # Both of these change the prompt, so both belong in the key.
                        "edges": sorted(maca_settings.graph_edge_types)
                        if maca_settings and maca_settings.graph_edge_types
                        else [],
                        "ents": sorted(graph_entity_types) if graph_entity_types else [],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:32]
            _cached_subq = await self._get_from_cache(_subq_cache_key)
            if isinstance(_cached_subq, dict) and _cached_subq.get("queries"):
                logger.info(
                    "subquery cache HIT: reusing %d subqueries for this question",
                    len(_cached_subq["queries"]),
                )
                return _cached_subq

        if mode == "accurate" and self.llm_client:
            # Build domain-specific relationship types from MACA config
            from synap.cloud.application.client_api.prompts.subqueries import (
                DOMAIN_REL_TYPES_HEADER,
                build_system_prompt,
                build_user_prompt,
            )

            domain_rel_types = ""
            if maca_settings and maca_settings.graph_edge_types:
                domain_rel_types = (
                    DOMAIN_REL_TYPES_HEADER
                    + ", ".join(maca_settings.graph_edge_types)
                )

            # Try LLM-based subquery generation (also infer relationship types for Neo4j)
            try:
                system_prompt = build_system_prompt(domain_rel_types)
                user_prompt = build_user_prompt(original_query)

                subquery_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                try:
                    response = await self.llm_client.complete(
                        messages=subquery_messages,
                        # 0.0 for the same reason as the relevance filter: a
                        # different subquery is a different vector search, which
                        # is the likeliest explanation for a top-scoring item
                        # being absent from one run entirely rather than merely
                        # reordered.
                        temperature=0.0,
                        max_tokens=1000,
                        json_mode=True,
                        reasoning_effort="medium",
                        telemetry_scope=scope,
                        telemetry_operation="retrieval.subquery_generation",
                        telemetry_correlation_id=correlation_id,
                    )
                except TypeError:
                    response = await self.llm_client.complete(
                        messages=subquery_messages,
                        temperature=0.0,
                        max_tokens=1000,
                        json_mode=True,
                    )

                content = response.get("content", "")
                if not content or not content.strip():
                    logger.warning("LLM subquery generation returned empty content, falling back to templates")
                    raise ValueError("Empty LLM response")

                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                parsed = json.loads(content.strip())

                # Support rank_N format ("rank_1": ..., "rank_2": ...) and array format ("subqueries": [...])
                subqueries = [parsed.get(f"rank_{i}") for i in range(1, 9)]
                subqueries = [q for q in subqueries if q and isinstance(q, str)]
                if not subqueries:
                    # Fallback to array format
                    subqueries = parsed.get("subqueries", [])
                    if not isinstance(subqueries, list):
                        subqueries = []
                subqueries = [q for q in subqueries if q and isinstance(q, str)]
                if subqueries:
                    queries.extend(subqueries)

                # Store for the next identical question. Short TTL on purpose:
                # this exists to make one question answer the same way twice,
                # not to pin a decomposition while a prompt or MACA setting
                # changes underneath it. A miss costs exactly what today costs.
                if _subq_cache_key:
                    await self._set_to_cache(
                        _subq_cache_key,
                        {"queries": queries},
                        ttl=300,
                    )
                return {"queries": queries}
            except Exception as e:
                logger.warning(f"LLM subquery generation failed, falling back to templates: {e}")

        # Accurate mode with no LLM client (Core): split the question by rule, as
        # FAST mode does. Parts come before the templates. A failed call with a
        # client keeps the templates only, as Cloud always did.
        if mode == "accurate" and not self.llm_client:
            from synap.cloud.application.client_api.query_decomposition import decompose_queries
            for _part in decompose_queries([original_query]):
                if _part and _part not in queries:
                    queries.append(_part)
            if len(queries) > 1:
                logger.info(f"COMPOUND_DECOMPOSE (accurate, no LLM subqueries): original={original_query!r}, parts={queries[1:]}")

        # Template-based fallback (used for FAST mode or when LLM fails)
        # Prefer MACA-configured fallback queries when available
        topic = original_query
        if maca_settings and maca_settings.fallback_queries:
            fallback_templates = maca_settings.fallback_queries
        else:
            # Start with core templates (always relevant)
            fallback_templates = list(_CORE_RETRIEVAL_TEMPLATES)
            # Add B2B templates only when the graph has meaningful B2B presence.
            # Require at least 3 distinct B2B entity types to avoid false triggers
            # (e.g. a restaurant tagged as "organization" shouldn't fire B2B queries).
            # graph_entity_types=None means L2 cache was cold — skip B2B as safe default.
            b2b_overlap = graph_entity_types & _B2B_ENTITY_TYPES if graph_entity_types else set()
            if len(b2b_overlap) >= 3:
                fallback_templates.extend(_B2B_RETRIEVAL_TEMPLATES)

        for template in fallback_templates:
            subquery = template.format(topic=topic) if "{topic}" in template else template
            if subquery != original_query:
                queries.append(subquery)

        b2b_skipped = graph_entity_types is not None and not (graph_entity_types & _B2B_ENTITY_TYPES)
        logger.info(
            f"Subquery generation complete: mode={mode}, query_count={len(queries)}, b2b_skipped={b2b_skipped}, graph_entity_types={sorted(graph_entity_types)[:8] if graph_entity_types else None}, queries_preview={[q[:80] for q in queries[:3]]}"
        )

        # Store for the next identical question. Short TTL on purpose:
        # this exists to make one question answer the same way twice,
        # not to pin a decomposition while a prompt or MACA setting
        # changes underneath it. A miss costs exactly what today costs.
        if _subq_cache_key:
            await self._set_to_cache(
                _subq_cache_key,
                {"queries": queries},
                ttl=300,
            )
        return {"queries": queries}

    async def _batch_encode_queries(self, queries: List[str]) -> List[np.ndarray]:
        """Batch encode queries into normalized vectors.

        Tries batch encoding first, falls back to individual encoding.
        """
        t0 = time.time()
        async with start_span(
            "synap.retrieval.embedding.batch_encode",
            {
                "synap.query_count": len(queries),
                "synap.total_text_len": sum(len(q) for q in queries),
            },
        ):
            vectors = []

            # Try batch encode via embedding client
            if self.embedding_client and hasattr(self.embedding_client, "embed_batch"):
                try:
                    results = await self.embedding_client.embed_batch(queries, is_query=True)
                    for r in results:
                        v = np.array(r, dtype=np.float32)
                        v = v / (np.linalg.norm(v) + 1e-9)
                        vectors.append(v)
                    set_current_span_attributes({
                        "synap.encoding_method": "batch",
                        "synap.vector_count": len(vectors),
                        "synap.embedding_latency_ms": round((time.time() - t0) * 1000, 2),
                    })
                    return vectors
                except Exception as e:
                    logger.warning(f"Batch encoding failed, falling back to individual: {e}")
                    set_current_span_attributes({"synap.batch_encode_error": type(e).__name__})

            # Individual encoding fallback
            for query in queries:
                emb = await self._get_embedding(query)
                v = np.array(emb, dtype=np.float32)
                v = v / (np.linalg.norm(v) + 1e-9)
                vectors.append(v)

            set_current_span_attributes({
                "synap.encoding_method": "individual_fallback",
                "synap.vector_count": len(vectors),
                "synap.embedding_latency_ms": round((time.time() - t0) * 1000, 2),
            })
            return vectors

    def _deduplicate_items(self, items: List[ContextItem]) -> List[ContextItem]:
        """Deduplicate items, keeping highest scoring.

        There is NO cross-source boost. An item found by more than one pipeline
        (e.g. graph and vector) is only counted and logged; its score is the
        best of its copies. This docstring used to promise a 1.2x multiplier
        the code does not apply (see "no boost applied" below).
        """
        seen: Dict[str, ContextItem] = {}
        seen_sources: Dict[str, set] = {}  # track all sources per item_id
        _collisions: Dict[str, int] = {}  # track source collisions: "graph→vector" etc.

        for item in items:
            # Normalise item_id so graph ("memory:UUID") and vector ("UUID")
            # items for the same memory share a single dedup key.
            key = item.item_id
            if key.startswith("memory:"):
                key = key[len("memory:"):]
            if key not in seen:
                seen[key] = item
                seen_sources[key] = {item.source}
            else:
                seen_sources[key].add(item.source)
                # Keep higher scoring
                existing = seen[key]
                item_score = max(item.relevance_score, item.similarity_score)
                existing_score = max(existing.relevance_score, existing.similarity_score)
                loser_source = item.source if existing_score >= item_score else existing.source
                winner_source = existing.source if existing_score >= item_score else item.source
                collision_key = f"{loser_source}→{winner_source}"
                _collisions[collision_key] = _collisions.get(collision_key, 0) + 1
                if item_score > existing_score:
                    _merge_matched_queries(item, existing)
                    seen[key] = item
                else:
                    _merge_matched_queries(existing, item)

        # Log cross-source items (no boost applied — LLM filter handles precision)
        _cross_source_count = 0
        for key in seen:
            if len(seen_sources[key]) > 1:
                _cross_source_count += 1
        if _cross_source_count:
            logger.info(f"cross_source_items: count={_cross_source_count} (no boost applied)")

        if _collisions:
            logger.info(f"merge_dedup_collisions: {_collisions}")

        deduped = list(seen.values())
        if items:
            _in_breakdown: Dict[str, int] = {}
            _out_breakdown: Dict[str, int] = {}
            _in_source: Dict[str, int] = {}
            _out_source: Dict[str, int] = {}
            for _item in items:
                _in_breakdown[_item.context_type] = _in_breakdown.get(_item.context_type, 0) + 1
                _in_source[_item.source] = _in_source.get(_item.source, 0) + 1
            for _item in deduped:
                _out_breakdown[_item.context_type] = _out_breakdown.get(_item.context_type, 0) + 1
                _out_source[_item.source] = _out_source.get(_item.source, 0) + 1
            logger.info(
                f"merge_dedup_complete: input_total={len(items)}, merged_unique={len(deduped)}, duplicates_removed={len(items) - len(deduped)}, "
                f"input_type_breakdown={_in_breakdown}, output_type_breakdown={_out_breakdown}, "
                f"input_source_breakdown={_in_source}, output_source_breakdown={_out_source}"
            )
        return deduped
