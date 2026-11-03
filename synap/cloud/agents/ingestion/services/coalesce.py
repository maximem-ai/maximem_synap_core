"""One row per fact, instead of one row per type that mentions the fact.

Stage 3 runs one extraction call per memory type on the same conversation. Each
call writes its own version of whatever it finds, and until now nothing compared
those versions to each other:

  - a per-type dedupe used to sit in ``stage3_extraction``: it grouped BY type,
    was gated on ``is_chunked`` which a normal conversation never trips, and keyed
    on ``source_span``, which only the ENTITY prompts ask for, so for memories the
    key was always None and it kept every row unconditionally. Dead three ways
    over, and deleted on 2026-10-03;
  - knowledge update is the stage built to coalesce, and it finds rivals with a
    vector search. The new memories reach Chroma at ``stage4_organization:895``
    and KU runs at 616, so a memory is only ever compared against OLDER
    documents, never against the siblings born in the same breath;
  - KU also skips EPISODES and EMOTIONS outright, and a quarter to a half of the
    copies live there.

So this runs where the problem is: the one point in the pipeline that holds every
type's output at once, after the chunk loop and before stage 4.

What it is NOT allowed to do
----------------------------
Lose a detail. Coalescing that trades a duplication bug for a data-loss bug is
the worse trade, so every group passes ``find_dropped_entities`` before it is
accepted, and that guard now protects bare numbers, clock times, spelled-out
numerals and whole dates — which it did not when this was designed. A refused
group leaves every member standing, exactly as today.

Merge two events into one. If two siblings carry event windows that cannot both
describe the same event, the group is refused rather than merged with the window
NULLed out. ⚠ This paragraph used to say ``knowledge_update.resolve_merge_metadata``
took the opposite choice and silently destroyed a date. It no longer does: it
refuses with ``merge_refused_reason`` exactly as this does, and both now share one
adjudicator in ``event_anchor`` (``precision_order``, ``windows_conflict``). The
sentence outlived the behaviour it described by long enough to be quoted back as
evidence of a live bug, which is its own lesson about docstrings that name another
module's choices.

Decide anything by word overlap. Grouping is the model's, with no similarity
threshold anywhere: a 0.45 Jaccard dedup once collapsed non-duplicates that
differed only in a number or a date and cost 4.04 points.

What it reuses
--------------
Everything except the grouping question. The merge itself is the existing
``MERGE_SYSTEM_PROMPT`` from prompts/merge.py, which already preserves numbers,
dates and named entities verbatim and already requires its output to be at least
as long as the longest contributor. The detail guard is ``find_dropped_entities``.
The window rule is ``event_anchor.windows_overlap``. Both LLM calls are injected,
so this module is testable without a model and the caller keeps ownership of
telemetry and model choice, which is how date_prepass and prior_context are wired
too.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from synap.cloud.agents.ingestion.models import ExtractedMemory, MemoryType
from synap.cloud.agents.ingestion.services.event_anchor import (
    precision_order, windows_conflict)

logger = logging.getLogger(__name__)

# Which type a merged row carries as its single primary, so every reader that
# filters on `memory_type` keeps working. Facts first because the measured
# failures say so: in a08a253f the `facts` row was exactly right and the
# `preferences` restatement reframed an attendance as an intention. Emotions and
# preferences last because they are readings OF a fact rather than the fact.
TYPE_PRECEDENCE = (
    MemoryType.FACTS,
    MemoryType.TEMPORAL_EVENTS,
    MemoryType.EPISODES,
    MemoryType.PREFERENCES,
    MemoryType.EMOTIONS,
)

# ⚠ Deliberately NOT inheriting knowledge_update.APPEND_ONLY_TYPES.
#
# KU skips EPISODES and EMOTIONS entirely, and refuses any merge cluster
# containing one, because collapsing two DIFFERENT episodes into one perpetual
# statement loses the per-occurrence detail that is the point of the type. That
# reasoning is right for KU and does not apply here: this step only ever groups
# rows that state the SAME fact from the SAME conversation, and the window check
# below is what enforces "same event". Inheriting the exclusion would miss a
# quarter to a half of the copies, measured — 15 of 60 rows in the model-kit
# case, 4 of 8 in the yoga case.
#
# If that turns out wrong, this is the line to change.
COALESCE_ALL_TYPES = True


def type_labels_metadata(labels: Optional[Sequence[str]]) -> Dict[str, str]:
    """Vector metadata for a merged row's extra type labels, or ``{}``.

    Comma-joined because ChromaDB metadata values must be scalars, which is
    already how ``entities_mentioned`` is stored. Same shape as
    ``event_window.event_window_metadata``: a dict to update the metadata with,
    empty when there is nothing to say.

    Empty for every row coalescing did not merge, which is most of them, so
    their metadata stays byte-identical to what it was.

    ⚠ ``memory_type`` beside this stays the single primary, and retrieval still
    buckets on it. A merged row is therefore served under one heading, not
    several — it is NOT dropped, because items_by_type is a partition of the
    already-trimmed flat list (retrieval_manager:7474). Teaching retrieval to
    read these labels is a separate change with its own decision.
    """
    if not labels:
        return {}
    return {"memory_type_labels": ",".join(sorted(set(labels)))}


@dataclass
class CoalesceResult:
    memories: List[ExtractedMemory]
    counts: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Refusal:
    """Why a proposed group was not merged. Passed to ``on_refusal`` when the
    caller supplies one, so refusals can be counted or escalated rather than
    only logged.

    Left unwired on purpose. ``adjudication_cases`` already exists for exactly
    this shape of question (``case_type`` includes 'merge' and 'contradiction'),
    and ``tidy_window`` already takes an injected ``open_case_fn``. Pointing this
    at it is one line at the call site when someone wants the review queue; it is
    not something to add to the ingest hot path without asking.
    """
    reason: str
    detail: str
    memory_ids: Sequence[str]


GroupFn = Callable[[List[str], List[str]], Awaitable[Optional[Dict[str, Any]]]]
MergeFn = Callable[[List[str], List[str]], Awaitable[Optional[Dict[str, Any]]]]


def _windows_conflict(members: Sequence[ExtractedMemory]) -> bool:
    """Any two members whose event windows cannot both be the same event."""
    return windows_conflict([(m.event_start, m.event_end) for m in members])


def _agreed_window(members: Sequence[ExtractedMemory]):
    """The window a merged row should carry: the most precise among members that
    carry one, having already established they do not conflict.

    Precision order matches event_anchor's: a day beats a month, because a month
    window containing a precise range is the same event at two precisions and the
    precise one is the better answer.
    """
    windowed = [m for m in members if m.event_start is not None]
    if not windowed:
        return None, None, None
    best = min(windowed,
               key=lambda m: (precision_order(m.event_precision), m.event_start))
    return best.event_start, best.event_end, best.event_precision


def identical_key(content: Optional[str]) -> str:
    """The key under which two extractions count as the SAME TEXT.

    Whitespace only, and case is kept. Deliberately stricter than
    ``event_anchor.normalise``, which also strips punctuation and the rendered
    date prefix: that is the right tool for noticing two rows are the same
    sentence, and the wrong one for deciding a row can be discarded, because
    "(Feb 10) X" and "X" would collapse and one of them would lose its date.

    Equal keys mean the stored text is the same string, so keeping one copy
    cannot lose a word. Anything looser needs the model.
    """
    return " ".join((content or "").split())


def fold_identical(memories: Sequence[ExtractedMemory]
                   ) -> Tuple[List[ExtractedMemory], int, int]:
    """Collapse byte-identical siblings with no model call at all.

    M2 measured this on 40 lme3 documents: 32 groups of identical text across
    21 of the 40, worth 39 rows, 4.1% of every memory extracted. The set-level
    grouping call caught most of them and MISSED several, including two pairs at
    cosine 1.000 ("The user is interested in Ori and the Blind Forest due to its
    expansive open-world design", written once by `emotions` and once by
    `preferences`). A model deciding whether one string equals itself is a model
    being asked the wrong question, and it answers wrong often enough to matter.

    One case must still be refused: identical text with contradictory windows.
    The real corpus holds it. ``event_anchor`` records the scope that stores one
    American Airlines sentence eight times, four rows dated 2023-02-10 and four
    with no date, and M2 found the same sentence written by four different types
    at cosine 1.000. Folding those would pick one date and silence the other, so
    the window guard runs here exactly as it does for a model-proposed group.

    Returns ``(memories, folded_groups, rows_removed)``.
    """
    by_key: Dict[str, List[int]] = {}
    for i, memory in enumerate(memories):
        by_key.setdefault(identical_key(memory.content), []).append(i)

    merged_by_index: Dict[int, ExtractedMemory] = {}
    dropped: set = set()
    folded = 0
    for idxs in by_key.values():
        if len(idxs) < 2:
            continue
        members = [memories[i] for i in idxs]
        if _windows_conflict(members):
            continue                      # same words, incompatible dates
        # The content is already every member's content, so no merge call is
        # needed and no detail check is possible to fail: there is nothing to
        # synthesise and nothing a synthesis could drop.
        merged_by_index[idxs[0]] = _build_merged(
            members, members[0].content or "",
            "identical text from several extractors")
        dropped.update(idxs[1:])
        folded += 1

    if not merged_by_index:
        return list(memories), 0, 0

    out: List[ExtractedMemory] = []
    for i, memory in enumerate(memories):
        if i in merged_by_index:
            out.append(merged_by_index[i])
        elif i not in dropped:
            out.append(memory)
    return out, folded, len(dropped)


def parse_groups(raw: Any, count: int) -> List[Tuple[List[int], str]]:
    """Validate the model's groups into ``(zero-based indices, reason)`` pairs.

    The reason travels with the group because it is the only explanation this
    step ever produces. It is asked for, and kept, for two separate reasons:
    naming the shared fact makes the model's own grouping sharper (it is what
    separates "same fact" from "same topic"), and it is the audit record for a
    row that used to be several.

    Everything questionable is dropped rather than repaired: an out-of-range
    index, a member claimed by two groups, a group of one. A model that returns
    nonsense therefore produces no merges, which is today's behaviour.
    """
    if not isinstance(raw, dict):
        return []
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, list):
        return []

    claimed: set = set()
    out: List[Tuple[List[int], str]] = []
    for entry in groups_raw:
        if not isinstance(entry, dict):
            continue
        members = entry.get("members")
        if not isinstance(members, list):
            continue
        idxs: List[int] = []
        for m in members:
            if isinstance(m, bool) or not isinstance(m, int):
                continue
            zero = m - 1
            if 0 <= zero < count and zero not in claimed and zero not in idxs:
                idxs.append(zero)
        if len(idxs) < 2:
            continue                      # a group of one is not a group
        claimed.update(idxs)
        out.append((idxs, str(entry.get("reason") or "").strip()))
    return out


def _primary_type(members: Sequence[ExtractedMemory]) -> MemoryType:
    present = {m.memory_type for m in members}
    for candidate in TYPE_PRECEDENCE:
        if candidate in present:
            return candidate
    return members[0].memory_type


def _build_merged(members: Sequence[ExtractedMemory],
                  content: str, reason: str = "") -> ExtractedMemory:
    """One memory carrying the union of its members.

    Field choices worth stating, because a merge that quietly picks the wrong
    one loses information just as surely as dropped text:
      - confidence: the MAX. The group states one fact, and the most confident
        extractor's reading of it is the one to keep;
      - document_created_at: the MIN, and extracted_at the min, matching
        resolve_merge_metadata's "earliest known mention";
      - is_episode / pii flags: True if ANY member says so, because a flag exists
        to make something handled more carefully, not less;
      - event_anchor: the first member that has one. A merged row could not carry
        an anchor at all before this (knowledge_update never mentions the column),
        so taking one is strictly better than dropping it.
    """
    primary = _primary_type(members)
    base = next(m for m in members if m.memory_type == primary)
    start, end, precision = _agreed_window(members)

    # The merged row takes the HIGHEST confidence of its members, so it has to
    # take the destination that confidence earned. Taking `base.destination`
    # instead sent merged rows holding 0.90 and 0.95 confidence to the
    # `candidates` table, which retrieval never reads, so they could never be
    # served. `base` is the first member of the highest-precedence type, and
    # facts and temporal_events lead that list, so a sub-threshold facts row
    # decided the destination while a 0.90 sibling of another type decided the
    # confidence. Across the benchmark scopes, candidate rows with
    # differing confidences are common, for example "The user attended a
    # digital marketing workshop ... Rachel Lee" at 0.95 and "The user picked up
    # Hendrick's gin around August 4, 2023" at 0.90. Reproduced with no model
    # and no database from two byte-identical siblings at 0.35 and 0.90.
    #
    # The provenance is unrecoverable after the fact: `candidates` has no
    # `coalesced_from` column, and the candidates loop in
    # storage/coordinator.py writes no action row, so a folded candidate leaves
    # no audit trail at all. That is why this is argued from the mechanism.
    #
    # It only ever failed in this direction, because a merge never promoted a
    # row out of CANDIDATES either. `confidence` is unchanged by this: the max
    # and the best member's value are the same number.
    best = max(members, key=lambda m: m.confidence)

    entities: List[str] = []
    questions: List[str] = []
    for m in members:
        for e in m.entities_mentioned:
            if e not in entities:
                entities.append(e)
        for q in m.hypothetical_questions:
            if q not in questions:
                questions.append(q)

    dcts = [m.document_created_at for m in members if m.document_created_at]
    anchors = [m.event_anchor for m in members if m.event_anchor]
    pii_fields: List[str] = []
    for m in members:
        for f in m.pii_fields_obfuscated:
            if f not in pii_fields:
                pii_fields.append(f)

    return ExtractedMemory(
        memory_id=uuid4(),
        memory_type=primary,
        type_labels=sorted({m.memory_type.value for m in members}),
        # The only record that this row used to be several. See migration 156:
        # the folded rows never persist, so nothing else can carry the evidence.
        coalesced_from={
            "reason": reason or "",
            "contributors": [m.content for m in members],
            "types": [m.memory_type.value for m in members],
        },
        content=content,
        confidence=best.confidence,
        destination=best.destination,
        # The first one any member has, so a custom category's name is not
        # lost when its member is merged with a plain fact. Without a custom category
        # every member's is None, as before.
        category=next((m.category for m in members if m.category), None),
        subcategory=base.subcategory,
        source_document_id=base.source_document_id,
        assertion_source=base.assertion_source,
        entities_mentioned=entities,
        pii_obfuscated=any(m.pii_obfuscated for m in members),
        pii_fields_obfuscated=pii_fields,
        pii_detected=(True if any(m.pii_detected for m in members)
                      else base.pii_detected),
        temporal_category=base.temporal_category,
        valid_from=base.valid_from,
        valid_until=base.valid_until,
        temporal_confidence=max(m.temporal_confidence for m in members),
        event_start=start,
        event_end=end,
        event_precision=precision,
        event_anchor=anchors[0] if anchors else None,
        is_episode=any(m.is_episode for m in members),
        episode_metadata=base.episode_metadata,
        hypothetical_questions=questions,
        extracted_at=min(m.extracted_at for m in members),
        document_created_at=min(dcts) if dcts else None,
        extraction_trace_id=base.extraction_trace_id,
    )


async def coalesce_memories(
    memories: List[ExtractedMemory],
    *,
    group_fn: GroupFn,
    merge_fn: MergeFn,
    correlation_id: Optional[str] = None,
    on_refusal: Optional[Callable[[Refusal], None]] = None,
) -> CoalesceResult:
    """Collapse same-fact siblings into one memory each.

    Returns the input unchanged on anything unexpected: too few memories, a
    missing or failing model call, an unparseable answer, a group that fails a
    guard. Behaving exactly like today is always an available answer, and is the
    reason this needs no flag to be safe.
    """
    counts = {"in": len(memories), "groups_proposed": 0, "groups_merged": 0,
              "merged_away": 0, "refused_detail": 0, "refused_window": 0,
              "refused_no_merge": 0, "folded_identical": 0,
              "folded_away": 0, "out": len(memories)}
    if len(memories) < 2:
        return CoalesceResult(memories=list(memories), counts=counts)

    # Identical text first, in code. This runs even when no model is wired,
    # because asking a model whether a string equals itself is both wasteful
    # and, measured, unreliable. See fold_identical.
    memories, folded, folded_away = fold_identical(memories)
    counts["folded_identical"] = folded
    counts["folded_away"] = folded_away
    counts["out"] = len(memories)
    if folded:
        logger.info("coalesce_folded_identical groups=%s rows_removed=%s "
                    "correlation_id=%s", folded, folded_away, correlation_id or "")

    if len(memories) < 2 or group_fn is None or merge_fn is None:
        return CoalesceResult(memories=list(memories), counts=counts)

    contents = [m.content or "" for m in memories]
    types = [m.memory_type.value for m in memories]

    try:
        raw = await group_fn(contents, types)
    except Exception:  # noqa: BLE001 — never fails an ingest
        logger.warning("coalesce_group_failed correlation_id=%s",
                       correlation_id or "", exc_info=True)
        return CoalesceResult(memories=list(memories), counts=counts)

    groups = parse_groups(raw, len(memories))
    counts["groups_proposed"] = len(groups)
    if not groups:
        return CoalesceResult(memories=list(memories), counts=counts)

    def refuse(reason: str, detail: str, members: Sequence[ExtractedMemory]) -> None:
        counts[reason] = counts.get(reason, 0) + 1
        logger.info("coalesce_refused reason=%s detail=%s ids=%s "
                    "correlation_id=%s", reason, detail,
                    ",".join(str(m.memory_id) for m in members),
                    correlation_id or "")
        if on_refusal is not None:
            try:
                on_refusal(Refusal(reason=reason, detail=detail,
                                   memory_ids=[str(m.memory_id) for m in members]))
            except Exception:  # noqa: BLE001 — reporting never fails an ingest
                logger.warning("coalesce_on_refusal_failed", exc_info=True)

    from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
        find_dropped_entities,
    )

    merged_by_index: Dict[int, ExtractedMemory] = {}
    dropped_indexes: set = set()

    for idxs, reason in groups:
        members = [memories[i] for i in idxs]

        # Step 4 of the plan, and the one place this disagrees with
        # resolve_merge_metadata on purpose: two windows that cannot both be the
        # same event mean these are not one fact. Refuse, rather than merge and
        # NULL the window, which destroys a date silently.
        if _windows_conflict(members):
            refuse("refused_window", "members give windows that cannot both be "
                                     "the same event", members)
            continue

        member_contents = [m.content or "" for m in members]
        member_types = [m.memory_type.value for m in members]
        try:
            merged = await merge_fn(member_contents, member_types)
        except Exception:  # noqa: BLE001
            logger.warning("coalesce_merge_failed correlation_id=%s",
                           correlation_id or "", exc_info=True)
            merged = None
        content = ((merged or {}).get("merged_content") or "").strip()
        if not content:
            refuse("refused_no_merge", "merge produced no content", members)
            continue

        dropped = find_dropped_entities(member_contents, content)
        if dropped:
            refuse("refused_detail",
                   "merge dropped " + ",".join(sorted(dropped)[:6]), members)
            continue

        merged_by_index[idxs[0]] = _build_merged(members, content, reason)
        dropped_indexes.update(idxs[1:])
        counts["groups_merged"] += 1
        counts["merged_away"] += len(idxs) - 1

    # Rebuilt in the original order, with each accepted group's merged row
    # standing where its first member was. Stage 4 computes embeddings from this
    # list afterwards (stage4_organization:556) and zips them back by position,
    # so nothing here can desync them; order is kept because a stable order makes
    # a before/after diff of a run readable.
    out: List[ExtractedMemory] = []
    for i, memory in enumerate(memories):
        if i in merged_by_index:
            out.append(merged_by_index[i])
        elif i not in dropped_indexes:
            out.append(memory)
    counts["out"] = len(out)
    return CoalesceResult(memories=out, counts=counts)
