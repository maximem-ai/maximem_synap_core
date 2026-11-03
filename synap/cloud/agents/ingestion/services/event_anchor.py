"""Settle a date that was only knowable relative to another event.

"I had to book three months in advance" of a wedding dated in a different
conversation. Migration 155 stores that as an ``event_anchor`` with ``event_start``
left NULL, and this module is what later turns it into a date.

Every guardrail here exists because a measured run produced the failure it stops.
A date-aware derivation was tried against the benchmark corpus and of
eight dated outputs exactly one was right:

- **five** carried a zero offset, which only restates the anchor's own date;
- **one** double-counted, because its anchor was *"has been playing a Fender CD-60S
  guitar for three weeks as of March 25"* already stored as 2023-03-04 — taking
  three weeks off again gave 11 February instead of 4 March;
- **one** picked a side of a contested date. That scope stores one American
  Airlines sentence eight times, four rows dated 2023-02-10 and four with no date,
  and asserting one of them cost the question on the frontier reader.

So the guardrails are not defensive padding; without them this makes things worse,
which is also why it must not ship before the pre-pass stops the per-type calls
disagreeing in the first place.

The arithmetic is in code. The model's only job upstream is to say which event the
date hangs off and by how much.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Sequence, Tuple

from synap.cloud.agents.ingestion.services.date_prepass import add_months
from synap.cloud.agents.ingestion.services.event_window import validate_event_window

logger = logging.getLogger(__name__)

_FLAG_ENV = "SYNAP_EVENT_ANCHOR"
_BY_INSTANCE_ENV = "SYNAP_EVENT_ANCHOR_BY_INSTANCE"
_TRUTHY = {"1", "true", "on", "yes"}

VALID_UNITS = ("days", "weeks", "months", "years")
VALID_DIRECTIONS = ("before", "after")

# Guardrail 2. A duration in the ANCHOR's own words means its stored date was very
# likely produced by applying that duration already, so applying ours as well
# subtracts the same interval twice. Cheap to detect, and the failure it prevents
# is silent: a plausible date that is simply wrong.
_ANCHOR_STATES_A_DURATION = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+"
    r"(?:day|week|month|year)s?\b"
    r"(?:\s+\w+){0,3}?\s*\b(?:ago|as of|in advance|earlier|later|before|after|since)\b"
    r"|\b(?:for|been|since)\b(?:\s+\w+){0,2}\s+"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+"
    r"(?:day|week|month|year)s?\b",
    re.IGNORECASE)

# A stated distance from ANOTHER event, read out of the memory's own words.
#
# ⚠ Why this exists at all. Two mechanisms already asked the MODEL to volunteer
# this, and both return nothing on real conversations:
#   * the extraction prompt's "event_anchor" field: asked 5,220 times on the
#     lme14 run, returned an anchor 0 times;
#   * date_prepass's "offset_from_event" phrase kind: 0 classifications in a
#     live ingest.
# A third mechanism of that shape would fail the same way, so this one reads the
# text we have already stored instead of asking for anything new.
#
# It extracts ONLY the number, the unit and the direction. Deciding WHICH stored
# memory is the event stays with the Jev screen in anchor_resolver, which is what
# it was built for and which already searches the whole scope across documents.
# That division matters: word comparison deciding a date is the mistake this file
# exists to stop, and parsing "three" into 3 is not that.
#
# Deliberately NOT matched:
#   "two weeks ago"            counted from the conversation, date_prepass owns it
#   "for six weeks"            a duration, no origin, must stay undated
#   "a while before the move"  no number, so no arithmetic is possible
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                 "twelve": 12, "a": 1, "an": 1}
_BEFORE_WORDS = ("in advance", "before", "prior to", "ahead of", "earlier than")
_AFTER_WORDS = ("after", "later than", "following")
# The words that turn a number and a unit into a LIMIT rather than a distance:
# "within 12 months", "up to two weeks", "every three days". Matched against the
# words immediately before the number.
_BOUND = re.compile(
    r"\b(?:within|up\s+to|at\s+least|at\s+most|no\s+more\s+than|no\s+less\s+than|"
    r"less\s+than|more\s+than|every|each|per)\s*$",
    re.IGNORECASE)
# A RATE: "20-30 pages a day", "twice a week". Only ever "a"/"an" plus the unit,
# so it is checked only for those, which keeps "paid $500 three months in
# advance" out of it.
_RATE = re.compile(r"(?:\d|\bonce\b|\btwice\b|[a-z]s)\s+\w*\s*$", re.IGNORECASE)
_STATED_OFFSET = re.compile(
    r"\b(?P<n>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|an?)\s+"
    r"(?P<unit>day|week|month|year)s?\s+"
    r"(?P<rel>in advance|before|prior to|ahead of|earlier than|after|later than|following)\b",
    re.IGNORECASE)


def offset_from_text(content: Optional[str]) -> Optional[Dict[str, Any]]:
    """An ``event_anchor`` built from a memory's own words, or None.

    ``anchor_text`` is the WHOLE sentence rather than a guessed noun phrase. The
    screen that uses it is asked "which stored memory is the event this is
    counted from", and the full sentence answers that better than two words
    lifted out of it could. Guessing the noun phrase would be the word matching
    this module already deleted once.
    """
    text_in = (content or "").strip()
    if not text_in:
        return None
    m = _STATED_OFFSET.search(text_in)
    if not m:
        return None
    # ⚠ A BOUND or a RATE is not a distance from an event, and both match the
    # pattern above. Measured on 4,863 memories: 2 of the 4 anchors this
    # detector produced were "should be used within 12 months after opening"
    # (a shelf life) and "reads 20-30 pages a day before bed" (a nightly
    # routine). Neither names another event, so neither has a date to count
    # from, and the screen downstream matched something anyway.
    before = text_in[:m.start("n")][-24:]
    if _BOUND.search(before):
        return None
    if m.group("n").lower() in ("a", "an") and _RATE.search(before):
        return None
    raw_n = m.group("n").lower()
    n = _WORD_NUMBERS.get(raw_n)
    if n is None:
        try:
            n = int(raw_n)
        except ValueError:
            return None
    if n <= 0:
        return None
    rel = m.group("rel").lower()
    direction = "after" if any(rel.startswith(w) for w in _AFTER_WORDS) else "before"
    return {"anchor_text": text_in[:300],
            "offset": {"n": n, "unit": m.group("unit").lower() + "s",
                       "direction": direction},
            "source_phrase": m.group(0),
            "derived_from": "memory_text"}


def enabled_for(instance_id: Optional[object]) -> bool:
    """Whether the event anchor is on for this instance.

    One switch for the whole set, ``SYNAP_TEMPORAL_V2``; this feature's own pair
    of variables still overrides it. See services/temporal_flags for precedence.
    """
    from synap.cloud.agents.ingestion.services import temporal_flags
    return temporal_flags.enabled(_FLAG_ENV, _BY_INSTANCE_ENV, instance_id)


_PUNCT = re.compile(r"[^a-z0-9\s]")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class AnchorCandidate:
    """A stored memory the anchor might refer to."""
    memory_id: str
    content: str
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None


@dataclass(frozen=True)
class AnchorResolution:
    """The outcome. ``event_start is None`` means nothing was written, and
    ``reason`` says which guardrail refused — worth logging, because a resolver
    that silently declines is indistinguishable from one that never ran."""
    event_start: Optional[datetime] = None
    event_end: Optional[datetime] = None
    event_precision: Optional[str] = None
    anchor_memory_id: Optional[str] = None
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.event_start is not None


def normalise(content: Optional[str]) -> str:
    """Content with its rendered date prefix, punctuation and spacing removed.

    Used ONLY to notice that two rows are the same sentence. It never decides
    that a memory should be dropped — the one thing word comparison has already
    cost us points for.
    """
    text = (content or "").strip()
    text = re.sub(r"^\((?:said |around )?[^)]{3,30}\)\s*", "", text)
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def parse_offset(raw: Any) -> Optional[Tuple[int, str, str]]:
    """(n, unit, direction), or None when the offset cannot be trusted."""
    if not isinstance(raw, dict):
        return None
    n = raw.get("n")
    if isinstance(n, bool):
        return None
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    unit = str(raw.get("unit") or "").strip().lower()
    direction = str(raw.get("direction") or "").strip().lower()
    if unit not in VALID_UNITS or direction not in VALID_DIRECTIONS:
        return None
    return n, unit, direction


def shift(dt: Optional[datetime], n: int, unit: str, direction: str) -> Optional[datetime]:
    if dt is None:
        return None
    sign = -1 if direction == "before" else 1
    if unit == "days":
        return dt + timedelta(days=sign * n)
    if unit == "weeks":
        return dt + timedelta(weeks=sign * n)
    if unit == "months":
        return add_months(dt, sign * n)
    if unit == "years":
        return add_months(dt, sign * n * 12)
    return None


# Precisions whose end date is the width of our uncertainty, not a duration.
_END_IS_UNCERTAINTY = {"week", "weekend", "month", "season", "year", "approximate"}


def _end_is_a_duration(candidate: AnchorCandidate) -> bool:
    """Does this candidate's ``event_end`` say the event LASTED that long?

    "range" says so outright. A missing or unknown precision with an end more
    than a day wide is treated as a duration too, because inheriting a length we
    cannot account for asserts a stay that never happened, while declining to
    inherit a width only costs precision.
    """
    precision = (candidate.event_precision or "").lower()
    if precision in _END_IS_UNCERTAINTY:
        return False
    if precision == "range":
        return True
    return bool(candidate.event_end
                and (candidate.event_end - candidate.event_start).days >= 1)


def _contested(candidate: AnchorCandidate,
               all_candidates: Sequence[AnchorCandidate]) -> bool:
    """Another row holds the SAME sentence with a DIFFERENT date.

    Guardrail 3. When the corpus disagrees with itself about when something
    happened, deriving from either side launders a contradiction into a new,
    confident memory. Refusing is the only safe answer, and it is why the
    per-type date pre-pass has to land before this does.
    """
    key = normalise(candidate.content)
    if not key:
        return False
    dates = set()
    for other in all_candidates:
        if normalise(other.content) == key and other.event_start is not None:
            dates.add(other.event_start.date())
    return len(dates) > 1



def windows_overlap(a_start, a_end, b_start, b_end) -> bool:
    """Can these two windows both describe one event?

    Overlap is the test, not equality: a month window and a precise range inside
    it are the same event at two precisions. Public because the coalesce step
    asks the same question of extraction's siblings, and two definitions of
    "these cannot be the same event" would eventually disagree.
    """
    a_end = a_end or a_start
    b_end = b_end or b_start
    return a_start <= b_end and b_start <= a_end


def _overlaps(a: AnchorCandidate, b: AnchorCandidate) -> bool:
    return windows_overlap(a.event_start, a.event_end,
                           b.event_start, b.event_end)


def _windows_conflict(candidates: Sequence[AnchorCandidate]) -> bool:
    """Any two candidates whose windows cannot both describe one event.

    Overlap is the test, not equality: a month window and a precise range inside
    it are the same event at two precisions, which is normal and resolvable. Two
    windows that never touch are not, whatever their wording, and deriving from
    either would make one of them louder.
    """
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if not _overlaps(a, b):
                return True
    return False


# Which of two event windows is the better answer for one fact, stated once.
#
# ⚠ This table existed verbatim in THREE places: here, coalesce._agreed_window and
# knowledge_update.resolve_merge_metadata. All three face the identical question,
# "several rows describe one event, which window does the surviving row carry", and
# three copies of an answer is three chances for them to drift apart. They had not
# drifted yet. The window-conflict rule beside it had: coalesce refused a merge
# whose windows could not be one event while knowledge_update was documented as
# merging and NULLing the date, and by the time anyone checked, knowledge_update had
# been fixed and the docstring saying otherwise had not.
#
# Lower is better. A month window containing a day range is the same event said at
# two precisions, and the precise one is the better answer.
PRECISION_RANK = {"day": 0, "range": 1, "weekend": 2, "week": 3,
                  "approximate": 4, "month": 5, "season": 6, "year": 7}
_UNRANKED = 9


def precision_order(precision: Optional[str]) -> int:
    """Sort key for an event precision. An unknown precision sorts last."""
    return PRECISION_RANK.get((precision or "").lower(), _UNRANKED)


def windows_conflict(windows: Sequence[Tuple[Any, Any]]) -> bool:
    """True when no single event could have produced all of these windows.

    *windows* are (start, end) pairs; a None start is ignored, because "we do not
    know when" contradicts nothing.
    """
    dated = [(s, e) for s, e in windows if s is not None]
    for i, (s_a, e_a) in enumerate(dated):
        for s_b, e_b in dated[i + 1:]:
            if not windows_overlap(s_a, e_a, s_b, e_b):
                return True
    return False


def resolve_anchor(anchor: Optional[Dict[str, Any]],
                   candidates: Sequence[AnchorCandidate],
                   *, dct: Optional[datetime] = None) -> AnchorResolution:
    """Turn an ``event_anchor`` plus the scope's candidate rows into a window.

    *candidates* are the memories that might BE the anchor; picking them is the
    caller's job. Only one may win, and every guardrail below prefers refusing.
    """
    if not isinstance(anchor, dict):
        return AnchorResolution(reason="no anchor")

    offset = parse_offset(anchor.get("offset"))
    if offset is None:
        return AnchorResolution(reason="offset missing or malformed")
    n, unit, direction = offset

    # Guardrail 1: a zero offset is not a relationship, it is a restatement of
    # the anchor's own date. Five of eight measured outputs were this.
    if n == 0:
        return AnchorResolution(reason="offset is zero, which only restates the anchor")

    dated = [c for c in candidates if c.event_start is not None]
    if not dated:
        return AnchorResolution(reason="no candidate carries a date yet")

    # Guardrail 2: an anchor whose own words state a duration has probably had
    # that duration applied already.
    usable = [c for c in dated if not _ANCHOR_STATES_A_DURATION.search(c.content or "")]
    if not usable:
        return AnchorResolution(
            reason="anchor states a duration itself, so its date may already "
                   "include the offset (double-count risk)")

    # Guardrail 3: refuse while the corpus contradicts itself about this fact.
    uncontested = [c for c in usable if not _contested(c, candidates)]
    if not uncontested:
        return AnchorResolution(
            reason="the anchor's date is contested by another copy of the same fact")

    # Guardrail 3b. _contested above compares identical SENTENCES, which misses the
    # case tracing the real Airbnb scope turned up: three differently worded
    # memories all matched the same anchor and carried two different windows. There
    # they were compatible (Mar 1-31 contains Mar 27-31) and the most precise one
    # was right. Differently worded candidates with INCOMPATIBLE windows are the
    # same contradiction wearing different words, and nothing above would notice.
    if _windows_conflict(uncontested):
        return AnchorResolution(
            reason="candidates for this anchor give windows that cannot both be "
                   "the same event")

    # More than one survivor: take the most precise, then the earliest, so the
    # choice is deterministic rather than dependent on row order.
    chosen = min(uncontested,
                 key=lambda c: (precision_order(c.event_precision),
                                c.event_start, c.memory_id))

    # ⚠ An ``event_end`` means one of two opposite things and they need opposite
    # treatment here.
    #
    # On a month, a season or a year it is HOW VAGUE the date is: the wedding
    # "in March 2023" is stored 03-01..03-31 and nobody claims it lasted a
    # month. Shifting both edges keeps that width, which is what makes the real
    # win in this file work: three months before March 2023 is December 2022.
    #
    # On a range it is a real LENGTH, and shifting both edges handed the new
    # event the old event's duration. The stay dated 2023-03-27..03-31, counted
    # back three months, was written as a booking running 2022-12-27..12-31, so
    # the context asserted the user was in Haight-Ashbury for five days in
    # December 2022, which never happened. A booking is one act.
    #
    # Which edge to count FROM follows the words: "three months before the trip"
    # is three months before it started, "a week after getting back" is a week
    # after it ended.
    if chosen.event_end and not _end_is_a_duration(chosen):
        start = shift(chosen.event_start, n, unit, direction)
        end = shift(chosen.event_end, n, unit, direction)
        precision = chosen.event_precision
    else:
        base = chosen.event_end if (direction == "after" and chosen.event_end) \
            else chosen.event_start
        start, end = shift(base, n, unit, direction), None
        # A point counted off a multi-day event, or counted in months or years,
        # is not a day-precise claim. Calling it "day" would let a later window
        # filter drop evidence on a boundary we never actually knew.
        precision = ("approximate"
                     if (chosen.event_end or unit in ("months", "years"))
                     else chosen.event_precision)
    if start is None:
        return AnchorResolution(reason=f"unsupported unit {unit!r}")

    start, end, precision = validate_event_window(
        start, end, precision, dct or chosen.event_start)
    if start is None:
        return AnchorResolution(reason="computed window failed validation")

    return AnchorResolution(event_start=start, event_end=end,
                            event_precision=precision,
                            anchor_memory_id=chosen.memory_id,
                            reason="ok")
