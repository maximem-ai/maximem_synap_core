"""Resolve a document's temporal phrases ONCE, before the per-type fan-out.

Why this exists
---------------
Stage 3 extracts a document with up to five parallel per-type calls, and each one
does its own calendar arithmetic on the same words. So one sentence is stored with
two different dates: on the lme3 cohort **87 of 470 answerable questions** serve a
fact under conflicting dates, and one scope holds the same American Airlines
sentence eight times, four rows dated 2023-02-10 and four with no date at all.

Measured separately, extraction returns a different memory set on **26% of
identical prompts**, so the five calls are not only doing the arithmetic five
times, they are five independent samples. One resolution replaces all of that
with one answer, which is why this attacks both defects at once.

The split that matters
----------------------
The model reads, code computes. It returns a STRUCTURED READING of each phrase —
which kind of reference it is, and its numbers — and never a date. Measured on a
calendar stress set, code-computes scored 98.5% against 93.4% for the model
computing its own dates, and 171 of 184 real life-event phrases against 139.

The rules below are the rules in ``prompts/extraction/temporal_addendum.py``,
which is the table the extraction prompts already follow. They are implemented
here, not restated: two copies of a calendar WILL drift, and a drift between the
pre-pass and the addendum would put the two halves of Stage 3 back into
disagreement, which is the whole defect.

Fail-open
---------
Every failure path returns an empty table, and an empty table means the per-type
prompts go out exactly as they do today. A missing resolution must never cost a
memory.
"""

from __future__ import annotations

import asyncio
import calendar
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from synap.cloud.agents.ingestion.services.event_window import validate_event_window
from synap.cloud.agents.ingestion.prompts.date_prepass import (
    READING_SYSTEM_PROMPT,
    READING_USER_PROMPT,
    CONVERSATION_DATE_LINE,
    NO_CONVERSATION_DATE_NOTICE,
    RESOLVED_DATES_HEADER,
    NOT_DATED_HEADER,
    ASSISTANT_ONLY_HEADER,
    NO_DATE_HEADER,
)

logger = logging.getLogger(__name__)

_TIMEOUT_ENV = "SYNAP_STAGE3_DATE_PREPASS_TIMEOUT_SECS"
_FLAG_ENV = "SYNAP_STAGE3_DATE_PREPASS"
_BY_INSTANCE_ENV = "SYNAP_STAGE3_DATE_PREPASS_BY_INSTANCE"
_TRUTHY = {"1", "true", "on", "yes"}


# ──────────────────────────────────────────────────────────────────────────────
# flags
# ──────────────────────────────────────────────────────────────────────────────

def timeout_secs() -> float:
    """Seconds to wait for the one reading call.

    ⚠ This was 3.0, copied from services/temporal_gate, and that was wrong. The
    gate is a short classification; this reads a WHOLE document and may return
    1500 output tokens. Measured on 172 lme3 documents, a 3.0s budget timed out
    on 6 of them and silently returned an empty table, which makes the whole
    pre-pass a no-op on exactly the long multi-topic documents whose dates are
    hardest. Generous on purpose: the call is once per document, it is additive,
    and a timeout costs the feature rather than the ingest.
    """
    try:
        return float(os.environ.get(_TIMEOUT_ENV) or "30.0")
    except (TypeError, ValueError):
        return 30.0


def enabled_for(instance_id: Optional[object]) -> bool:
    """Whether the date pre-pass is on for this instance.

    One switch for the whole set, ``SYNAP_TEMPORAL_V2``; this feature's own pair
    of variables still overrides it. See services/temporal_flags for precedence.
    """
    from synap.cloud.agents.ingestion.services import temporal_flags
    return temporal_flags.enabled(_FLAG_ENV, _BY_INSTANCE_ENV, instance_id)


# ──────────────────────────────────────────────────────────────────────────────
# the reading the model returns — structure only, never a date
# ──────────────────────────────────────────────────────────────────────────────

# ⚠ The "next_*" half of this set is as load-bearing as the "last_*" half.
#
# It did not exist until 2026-10-11, and the kind list handed "next month" to
# ``n_months_ahead``, a POINT kind that resolves as add_months(DCT, +1) keeping
# the day number. "I am flying to Miami next month", said on 2023-05-21, became
# a point at 2023-06-21 with precision "approximate", and the served text read
# "planned for around Jun 21, 2023". The speaker named a month; we answered with
# a day in the middle of it that nobody said, which is the same fabrication
# class as substituting the conversation date for a missing event date.
#
# A calendar phrase is a WINDOW in both directions: "next month" is the whole of
# the following month exactly as "last month" is the whole of the previous one.
# "in two months" stays a point in ``n_months_ahead``, because that phrase really
# does mean roughly one day about two months out.
WINDOW_KINDS = {
    "last_week", "this_week", "next_week",
    "last_weekend", "this_weekend", "next_weekend",
    "last_month", "this_month", "next_month", "named_month",
    "last_season", "named_season",
    "last_year", "this_year", "next_year", "named_year",
    "explicit_range", "since_year", "n_years_ago",
}
POINT_KINDS = {
    "today", "yesterday", "tomorrow", "n_days_ago", "n_days_ahead",
    "weekday_past", "weekday_future", "explicit_date",
    "n_weeks_ago", "n_weeks_ahead", "n_months_ago", "n_months_ahead",
    "duration_ongoing",
}

# ⚠ Kinds this pre-pass must RECOGNISE and must never DATE.
#
# Every phrase above is counted from the conversation date. These are not, and
# until they existed the reader had nowhere to put them, so a phrase with a
# number and a unit was read as one of the kinds above and silently became
# "DCT minus n". Measured on the real corpus:
#
#   "three months in advance"  (of a wedding dated in ANOTHER session)
#        read as n_months_ago -> 2023-02-27. The wedding was 2023-03-27, so the
#        booking was 2022-12-27 and the benchmark's own answer is "five months
#        ago". We stored three months, and the resolved-dates block told the
#        extractor to "use it exactly and do NOT recompute", which overrode the
#        event-anchor rule telling it not to guess from DCT. Our own prompt
#        contradicted itself and the wrong half won.
#   "can be stored in the fridge for up to 2 weeks"
#        read as duration_ongoing -> DCT minus 2 weeks, giving a shelf life an
#        event date. 4 of 85 dated rows in one 10-scope run.
#
# Both resolve to NOTHING here. ``offset_from_event`` is the event anchor's
# input and gets its date later, from the other event, by anchor_resolver;
# ``duration_property`` has no origin at all and must stay undated.
NO_DCT_ORIGIN_KINDS = {"offset_from_event", "duration_property"}

ALL_KINDS = WINDOW_KINDS | POINT_KINDS | NO_DCT_ORIGIN_KINDS | {"none"}

# Kinds whose date is ARITHMETIC from the conversation date, as opposed to a day,
# a month, a season or a year the speaker named outright. Only these are refused
# when the assistant is the one who said them: a date the assistant states
# outright is usually read off a record ("your order shipped on 3 July"), while a
# relative phrase is the assistant doing the subtraction, which is the one thing
# this module exists to stop a model doing.
COUNTED_FROM_DCT = (WINDOW_KINDS | POINT_KINDS) - {
    "explicit_date", "explicit_range", "named_month", "named_season",
    "named_year", "since_year",
}

# What an ``offset_from_event`` must carry before code will apply it. Same values
# as event_anchor.VALID_UNITS / VALID_DIRECTIONS; stated here so a malformed
# reading is refused before any arithmetic runs.
VALID_OFFSET_UNITS = ("days", "weeks", "months", "years")
VALID_OFFSET_DIRECTIONS = ("before", "after")

_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
# the addendum's seasons: spring Mar-May, summer Jun-Aug, fall Sep-Nov, winter Dec-Feb
_SEASONS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11),
            "autumn": (9, 11), "winter": (12, 2)}


@dataclass(frozen=True)
class PhraseReading:
    """One temporal phrase as the model read it. Carries no computed date."""
    phrase: str
    kind: str
    n: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None
    weekday: Optional[str] = None
    date: Optional[str] = None          # ISO, only for explicit_date
    end_date: Optional[str] = None      # ISO, only for explicit_range
    season: Optional[str] = None
    # "past" | "future", for a bare month, season or month-and-day with no year.
    # ⚠ Without it the year was always the most recent PAST occurrence, so
    # "planning a road trip in June", said on 2023-05-05, resolved to
    # 2022-06-01. The extraction prompt's own rule four lines away says "that
    # month, in the year that fits the TENSE", and this is the field that lets
    # code honour it.
    #
    # ⚠ It took until 2026-10-08 to cover explicit_date and explicit_range too,
    # and they are the kinds where the missing year cost the most, because they
    # assert a DAY. "I just attended a digital marketing workshop at the city
    # convention center on March 15-16", said 2023-02-26, was stored as
    # 2023-03-15, seventeen days AFTER the day it was said. Six memories about
    # that one workshop all carried it, the read path labelled every one of them
    # "planned for", and the reader excluded the $500 the user paid because our
    # own label said the workshop had not happened: gpt4_731e37d7, gold $720,
    # answered $220. Same shape on a9f6b44c: "I remember cleaning and
    # lubricating my bike chain on March 22nd", said 2023-03-20, stored as
    # 2023-03-22, two days ahead, which made one chain cleaning look like two.
    tense: Optional[str] = None
    # "user" | "assistant": whose turn the phrase is in.
    #
    # ⚠ A conversation has two speakers and only one of them was there. On
    # gpt4_85da3956 the user said "I just got back from an amazing day at
    # Universal Studios Hollywood" in a session dated 2023-07-15, and the
    # assistant replied "I don't have personal experiences ... so I didn't know
    # you went to Universal Studios Hollywood three weeks ago". The user's own
    # words carried no datable phrase, because "just" used to be deliberately
    # not dated (changed 2026-10-11: "just <past tense>" is now kind "today",
    # which is what this very paragraph argues for), so the
    # assistant's invented three weeks was the only date on offer and the
    # festival was stored as 2023-06-24. Asked three weeks later how long ago
    # it was, the reader read our date and answered six weeks.
    said_by: Optional[str] = None
    # ``unit`` is also what ``duration_ongoing`` counts in, as of 2026-10-08. The
    # rest of this group is for ``offset_from_event`` only: how far, in which
    # direction, and which OTHER phrase in this same conversation dates the event
    # being counted from.
    #
    # ⚠ Why these exist. "I had to book it three months in advance" of a wedding
    # the SAME conversation dates to Mar 27-31 was stored with no date at all,
    # and the benchmark's answer is "five months ago". Three parts of the system
    # each declined it, correctly and in isolation: this pre-pass refuses it
    # because it is not counted from the conversation date; the extraction
    # prompt tells the model to "resolve it yourself" and the model does not do
    # arithmetic; the event anchor only covers the case where the other event is
    # dated in a DIFFERENT conversation, so the model returns null here (measured:
    # asked 5,220 times, returned an anchor 0 times). Nobody owned it. The words
    # needed to resolve it were in the conversation the whole time.
    unit: Optional[str] = None          # days | weeks | months | years
    direction: Optional[str] = None     # before | after
    anchor_phrase: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> Optional["PhraseReading"]:
        """Lenient: a malformed reading becomes None rather than an exception.
        One bad phrase must not cost the whole table."""
        if not isinstance(raw, dict):
            return None
        phrase = str(raw.get("phrase") or "").strip()
        kind = str(raw.get("kind") or "").strip().lower()
        if not phrase or kind not in ALL_KINDS or kind == "none":
            return None

        def as_int(key: str) -> Optional[int]:
            v = raw.get(key)
            if isinstance(v, bool):
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        return cls(phrase=phrase, kind=kind, n=as_int("n"), month=as_int("month"),
                   year=as_int("year"),
                   weekday=(str(raw.get("weekday")).strip().lower()
                            if raw.get("weekday") else None),
                   date=(str(raw.get("date")).strip() if raw.get("date") else None),
                   end_date=(str(raw.get("end_date")).strip()
                             if raw.get("end_date") else None),
                   season=(str(raw.get("season")).strip().lower()
                           if raw.get("season") else None),
                   tense=(str(raw.get("tense")).strip().lower()
                          if raw.get("tense") else None),
                   said_by=(str(raw.get("said_by")).strip().lower()
                            if raw.get("said_by") else None),
                   unit=(str(raw.get("unit")).strip().lower()
                         if raw.get("unit") else None),
                   direction=(str(raw.get("direction")).strip().lower()
                              if raw.get("direction") else None),
                   anchor_phrase=(str(raw.get("anchor_phrase")).strip()
                                  if raw.get("anchor_phrase") else None))


# ──────────────────────────────────────────────────────────────────────────────
# calendar helpers — the addendum's rules, in code
# ──────────────────────────────────────────────────────────────────────────────

def _day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def add_months(dt: datetime, months: int) -> datetime:
    """Calendar month shift keeping the day number, clamped to the target month.

    The addendum is explicit that "a month ago" said on 2023-05-30 is 2023-04-30
    and NOT 2023-04-01. The old prompt's first-of-month rule turned "had the rug a
    week when I rearranged" into five weeks, so the day number is load-bearing.
    """
    total = (dt.year * 12 + dt.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    return dt.replace(year=year, month=month,
                      day=min(dt.day, calendar.monthrange(year, month)[1]))


_DURATION_UNITS = ("years", "months", "weeks", "days")
# The prompt's own convention for the vague quantities, so that a phrase and a
# reading are counted the same way: "a few days ago (n=3)", "a couple (n=2)".
#
# ⚠ Ordered, and the bare articles come LAST on purpose. "for a few months now"
# contains the word "a", so a lookup that tried the articles first read it as ONE
# month and overruled the model's correct 3. Caught against the live model on the
# real gpt4_2f56ae70 conversation before this shipped anywhere.
_QUANTITY_WORDS = (("couple", 2), ("few", 3), ("several", 3),
                   ("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5),
                   ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9), ("ten", 10),
                   ("eleven", 11), ("twelve", 12), ("an", 1), ("a", 1))


def _duration_unit_and_n(reading: "PhraseReading", n: int) -> Tuple[Optional[str], int]:
    """Which unit ``n`` is in, and the number, preferring the person's own words.

    ⚠ The phrase wins over the reading, on both counts. "for three months now"
    with n 12 is a model that converted months to weeks the way the old prompt
    told it to, and taking 12 as months would date the thing a year back instead
    of three months. The speaker said three and said months; they were there.
    The reading's own ``unit`` is used only when the phrase names no unit.
    """
    phrase = (reading.phrase or "").lower()
    unit = None
    for u in _DURATION_UNITS:
        if u[:-1] in phrase:                      # year, month, week, day
            unit = u
            break
    if unit is None:
        bare = (reading.unit or "").lower().rstrip("s")
        unit = f"{bare}s" if f"{bare}s" in _DURATION_UNITS else None
    if unit in ("months", "years"):
        said = _quantity_in(phrase)
        if said is not None and said != n:
            logger.info("date_prepass_duration_number_disagrees phrase=%r "
                        "model_n=%s phrase_n=%s unit=%s", reading.phrase, n, said, unit)
            n = said
    return unit, n


def _quantity_in(phrase: str) -> Optional[int]:
    """The number the phrase itself states, as digits or as a word. None if it
    states none ("for months now"), in which case the reading's n stands."""
    m = re.search(r"\b(\d{1,3})\b", phrase)
    if m:
        return int(m.group(1))
    for word, value in _QUANTITY_WORDS:
        if re.search(rf"\b{word}\b", phrase):
            return value
    return None


def week_start_sunday(dt: datetime) -> datetime:
    """The Sunday of dt's week. Weeks run Sunday to Saturday per the addendum."""
    return _day(dt) - timedelta(days=(dt.weekday() + 1) % 7)


def month_bounds(year: int, month: int) -> Tuple[datetime, datetime]:
    return (datetime(year, month, 1),
            datetime(year, month, calendar.monthrange(year, month)[1]))


def _most_recent_weekday(dt: datetime, weekday: int, *, strictly_before: bool) -> datetime:
    """Latest `weekday` at or before dt, or strictly before it."""
    base = _day(dt)
    delta = (base.weekday() - weekday) % 7
    if strictly_before and delta == 0:
        delta = 7
    return base - timedelta(days=delta)


def _next_weekday(dt: datetime, weekday: int) -> datetime:
    base = _day(dt)
    return base + timedelta(days=(weekday - base.weekday()) % 7)


def _season_bounds(season: str, year: int) -> Tuple[datetime, datetime]:
    first, last = _SEASONS[season]
    if first == 12:                             # winter straddles the new year
        return datetime(year, 12, 1), datetime(year + 1, 2,
                                               calendar.monthrange(year + 1, 2)[1])
    return (datetime(year, first, 1),
            datetime(year, last, calendar.monthrange(year, last)[1]))


_YEAR_IN_PHRASE = re.compile(r"\b(?:19|20)\d{2}\b")


def _phrase_states_a_year(phrase: str) -> bool:
    """Did the speaker's own words carry a year?

    ``phrase`` is the words exactly as they appear in the conversation, which is
    what the reading prompt asks for. So this is the one thing we can check
    without trusting the year the model wrote into ``date``.
    """
    return bool(_YEAR_IN_PHRASE.search(phrase or ""))


def _occurrence_of(month: int, day: int, base: datetime, *,
                   before: bool) -> Optional[datetime]:
    """The nearest month-and-day at or before (or at or after) ``base``.

    A year at a time, so February 29 lands on a leap year instead of being
    refused. Four steps is always enough; nine is slack.
    """
    year = base.year
    for _ in range(9):
        try:
            candidate = datetime(year, month, day)
        except ValueError:                       # Feb 29 in a common year
            year += -1 if before else 1
            continue
        if (candidate <= base) if before else (candidate >= base):
            return candidate
        year += -1 if before else 1
    return None


def _year_from_tense(d: datetime, tense: Optional[str],
                     base: datetime) -> Optional[datetime]:
    """``d``'s month and day, in the year the sentence's tense puts them in.

    ⚠ The no-tense branch is narrow on purpose, and the first version of it was
    wrong. It refused every untensed phrase, on the reasoning that a day is
    asserted at day precision and both defaults assert something nobody said.
    Measured on the four re-ingested scopes the same hour: four phrases came
    back with no tense — "February 22" and "December 12" said on 2023-02-26,
    "on February 20th" and "March 10th" said on 2023-03-20 — and all four were
    already RIGHT at the year the model wrote, because all four land before the
    day they were said. Refusing them lost four correct dates to fix two wrong
    ones.

    So the line is not "was I told the tense", it is "could this date be the
    defect at all". A month-and-day at or before the speaking day asserts
    nothing about the future, and the past reading and the model's reading agree
    there, so it is kept. One landing AFTER the speaking day is refused, because
    that is exactly where a mis-yeared past event and a real plan look
    identical, and a wrong day-precise date is worse than none.
    """
    if tense == "past":
        return _occurrence_of(d.month, d.day, base, before=True)
    if tense == "future":
        return _occurrence_of(d.month, d.day, base, before=False)
    return d if d <= base else None


def _end_after(start: datetime, end: datetime) -> datetime:
    """``end``'s month and day in ``start``'s year, or the next year when that
    would fall before the start. "from Dec 28 to Jan 3" is eleven days, not
    minus three hundred and fifty."""
    for year in (start.year, start.year + 1):
        try:
            candidate = datetime(year, end.month, end.day)
        except ValueError:                       # Feb 29 in a common year
            continue
        if candidate >= start:
            return candidate
    return start


def _parse_iso_day(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", value.strip())
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# resolution — pure code, no model
# ──────────────────────────────────────────────────────────────────────────────

def _with_year(d: datetime, year: int) -> Optional[datetime]:
    """``d`` moved into ``year``, or None when that day does not exist there.

    29 February is the whole reason. ``datetime.replace(year=...)`` raises
    ValueError moving a leap day into a year that has none, and the raise landed
    in ``resolve_all``'s catch-all, which dropped the phrase from the table
    instead of naming it as undated. A phrase missing from the table is one the
    extraction prompt tells the model to "date by the rules you were given",
    which is the guess this module exists to prevent, so a crash here was worse
    than returning nothing.
    """
    try:
        return d.replace(year=year)
    except ValueError:
        return None


def _dates_itself(reading: PhraseReading) -> bool:
    """Can this phrase be dated with NO conversation date to count from?

    Only when a year is available without one: stated in the phrase itself
    ("March 15 2023", "summer 2021"), or read off the conversation by the model
    and reported in ``year``. Everything else is counted from the day the
    conversation happened, and without that day there is nothing to count from.
    """
    if reading.kind in ("explicit_date", "explicit_range"):
        return _phrase_states_a_year(reading.phrase) or reading.year is not None
    if reading.kind in ("named_year", "since_year", "named_month", "named_season"):
        return reading.year is not None
    return False


def resolve_reading(reading: PhraseReading,
                    dct: Optional[datetime]) -> Tuple[Optional[datetime], Optional[datetime], Optional[str]]:
    """(event_start, event_end, event_precision) for one reading, or all None.

    Returning all None is always safe: the caller leaves that phrase out of the
    table and the per-type calls behave exactly as they do today.

    ``dct`` is None when nothing said when the conversation happened. Then only
    a phrase carrying its own year resolves; every branch below that reaches
    ``base`` is guarded by :func:`_dates_itself`, so ``base`` cannot be None
    where it is used.
    """
    k, n = reading.kind, reading.n

    # Not counted from this conversation, so there is no arithmetic to do here.
    # First, so no later branch can reach a phrase whose origin is elsewhere.
    if k in NO_DCT_ORIGIN_KINDS:
        return None, None, None

    if dct is None and not _dates_itself(reading):
        return None, None, None

    base = _day(dct) if dct is not None else None

    if k == "today":
        return base, None, "day"
    if k == "yesterday":
        return base - timedelta(days=1), None, "day"
    if k == "tomorrow":
        return base + timedelta(days=1), None, "day"

    if k in ("n_days_ago", "n_days_ahead"):
        if n is None:
            return None, None, None
        sign = -1 if k.endswith("ago") else 1
        return base + timedelta(days=sign * n), None, "day"

    if k in ("n_weeks_ago", "n_weeks_ahead"):
        if n is None:
            return None, None, None
        sign = -1 if k.endswith("ago") else 1
        return base + timedelta(weeks=sign * n), None, "approximate"

    if k in ("n_months_ago", "n_months_ahead"):
        if n is None:
            return None, None, None
        sign = -1 if k.endswith("ago") else 1
        return add_months(base, sign * n), None, "approximate"

    if k == "duration_ongoing":
        # "has been taking lessons for six weeks": the event STARTED n units back
        # and has no end. A reading without n is unusable rather than guessed at.
        #
        # ⚠ `n` used to be counted in WEEKS whatever the phrase said, because the
        # reading prompt asked the model to turn months and years into weeks
        # itself ("a month = 4, a year = 52"). Two things were wrong with that.
        # Measured on an ingest, across the 10 month- or
        # year-long phrases whose conversation date was recoverable from a
        # sibling phrase in the same document: ONE came back unconverted, so
        # "six months now" said on 2023-05-20 started the thing on 2023-04-08,
        # 42 days back instead of about 183; and all 8 that were converted land
        # EARLY, because four weeks is 28 days and a month is not ("the past six
        # months" -> 168 days, a fortnight late; "three months" -> 84 vs 91).
        # Code does calendar arithmetic everywhere else in this file and the
        # model is asked for a reading, not a sum. Same rule here now.
        if n is None:
            return None, None, None
        unit, n = _duration_unit_and_n(reading, n)
        if unit == "months":
            return add_months(base, -n), None, "approximate"
        if unit == "years":
            return add_months(base, -12 * n), None, "approximate"
        if unit == "days":
            return base - timedelta(days=n), None, "approximate"
        # No unit in the phrase or the reading: weeks, which is what this branch
        # has always done. A model that ignores the new field is also a model
        # still converting to weeks, so the old reading keeps the old arithmetic.
        return base - timedelta(weeks=n), None, "approximate"

    if k == "weekday_past":
        wd = _WEEKDAYS.get(reading.weekday or "")
        if wd is None:
            return None, None, None
        return _most_recent_weekday(base, wd, strictly_before=True), None, "day"

    if k == "weekday_future":
        wd = _WEEKDAYS.get(reading.weekday or "")
        if wd is None:
            return None, None, None
        return _next_weekday(base, wd), None, "day"

    # ⚠ These two kinds are the only ones where the MODEL hands back a finished
    # date instead of a reading for code to resolve, and the reading prompt asks
    # for a full YYYY-MM-DD even when the phrase states no year. So the year came
    # from the model, with nothing telling it which year to pick, and it used the
    # conversation's. The resolved block then tells all five per-type calls to
    # "use them exactly as given" and not to recompute, so one unstated year
    # became the same wrong date on every memory about the event.
    #
    # A year the speaker did not say is not a reading, it is a guess, and this
    # file does not take guesses from the model. So when the phrase carries no
    # year we keep the month and the day and put them in the year the TENSE
    # implies, which is the rule the extraction prompt already states for
    # "on February 24th" and which code now honours here as well.
    if k == "explicit_date":
        d = _parse_iso_day(reading.date)
        if d is None:
            return None, None, None
        if not _phrase_states_a_year(reading.phrase):
            if base is None:
                # No conversation date to pick the year from, so the only year
                # available is the one the model read off the conversation.
                if reading.year is None:
                    return None, None, None
                d = _with_year(d, reading.year)
                if d is None:
                    return None, None, None
            else:
                d = _year_from_tense(d, reading.tense, base)
                if d is None:
                    return None, None, None
        return d, None, "day"

    if k == "explicit_range":
        start, end = _parse_iso_day(reading.date), _parse_iso_day(reading.end_date)
        if not start:
            return None, None, None
        if not _phrase_states_a_year(reading.phrase):
            if base is None:
                if reading.year is None:
                    return None, None, None
                shifted = _with_year(start, reading.year)
            else:
                shifted = _year_from_tense(start, reading.tense, base)
            if shifted is None:
                return None, None, None
            # Only when the start actually moved. A start we are keeping as the
            # model wrote it keeps the model's end too, rather than having its
            # end recomputed against a year nothing changed.
            if shifted != start:
                start, end = shifted, (_end_after(shifted, end) if end else None)
        return start, end, "range"

    if k in ("this_week", "last_week", "next_week"):
        start = week_start_sunday(base)
        if k == "last_week":
            start -= timedelta(days=7)
        elif k == "next_week":
            start += timedelta(days=7)
        return start, start + timedelta(days=6), "week"

    if k in ("this_weekend", "last_weekend", "next_weekend"):
        if k == "next_weekend":
            # The coming Saturday. Same as the no-tense branch below, named
            # explicitly so a speaker who said "next weekend" is not resolved by
            # a tense the pre-pass may not have reported.
            sat = _next_weekday(base, _WEEKDAYS["saturday"])
        elif k == "last_weekend":
            # The addendum: "the most recent Saturday and Sunday before DCT".
            # ⚠ On a Sunday this yields yesterday and today. The phrase is
            # genuinely ambiguous then; we follow the table rather than invent a
            # second rule the extraction prompts do not know about.
            sat = _most_recent_weekday(base, _WEEKDAYS["saturday"], strictly_before=True)
        elif reading.tense == "past":
            # "I finally decluttered the kitchen this weekend", said on Monday
            # 2023-05-29, used to resolve to the COMING Saturday, 2023-06-03: an
            # event dated five days in the future of the conversation that
            # reported it as done. Caught on a real ingest through the API.
            #
            # "this weekend" is genuinely ambiguous in English and the sentence's
            # tense is what settles it, exactly as it already settles a bare
            # month or season. At-or-before rather than strictly-before, because
            # said ON a Saturday or Sunday the weekend in question is the one
            # happening now.
            sat = _most_recent_weekday(base, _WEEKDAYS["saturday"], strictly_before=False)
        else:
            # No tense reported, or future: unchanged. Guessing when we were not
            # told is how the bug above got written in the first place.
            sat = _next_weekday(base, _WEEKDAYS["saturday"])
        return sat, sat + timedelta(days=1), "weekend"

    if k in ("this_month", "last_month", "next_month"):
        shift = {"this_month": 0, "last_month": -1, "next_month": 1}[k]
        anchor = (base if shift == 0
                  else add_months(base.replace(day=1), shift))
        start, end = month_bounds(anchor.year, anchor.month)
        return start, end, "month"

    if k == "named_month":
        if not reading.month or not 1 <= reading.month <= 12:
            return None, None, None
        year = reading.year
        if year is None:
            if reading.tense == "future":
                # Still to come: the NEXT occurrence at or after DCT's month.
                year = (base.year if reading.month >= base.month
                        else base.year + 1)
            else:
                # Already happened: the most recent occurrence at or before it.
                year = base.year if reading.month <= base.month else base.year - 1
        start, end = month_bounds(year, reading.month)
        return start, end, "month"

    if k in ("this_year", "last_year", "next_year", "named_year"):
        year = (reading.year if k == "named_year"
                else base.year + {"last_year": -1, "this_year": 0,
                                  "next_year": 1}[k])
        if year is None:
            return None, None, None
        return datetime(year, 1, 1), datetime(year, 12, 31), "year"

    if k == "n_years_ago":
        if n is None:
            return None, None, None
        year = base.year - n
        return datetime(year, 1, 1), datetime(year, 12, 31), "year"

    if k == "since_year":
        if reading.year is None:
            return None, None, None
        return datetime(reading.year, 1, 1), None, "year"

    if k in ("last_season", "named_season"):
        season = reading.season
        if season not in _SEASONS:
            return None, None, None
        year = reading.year
        if year is None and reading.tense == "future" and k != "last_season":
            # Still to come: this year's, unless it has already ended.
            year = base.year
            _, end = _season_bounds(season, year)
            if end < base:
                year += 1
        if year is None:
            year = base.year
            start, _ = _season_bounds(season, year)
            if start > base:                      # not reached yet this year
                year -= 1
            elif k == "last_season":
                _, end = _season_bounds(season, year)
                if end >= base:                   # we are inside it; go back one
                    year -= 1
        start, end = _season_bounds(season, year)
        return start, end, "season"

    return None, None, None


def resolve_all(readings: List[PhraseReading],
                dct: Optional[datetime]) -> Dict[str, Dict[str, Any]]:
    """{phrase: {event_start, event_end, event_precision}} for every phrase we
    could resolve. Each window goes through ``validate_event_window`` so it is
    checked against DCT by the same code that checks extraction's own windows."""
    table: Dict[str, Dict[str, Any]] = {}
    deferred: List[PhraseReading] = []
    for reading in readings:
        # Kept, not dropped. A phrase that is simply absent from the block is one
        # the block tells the extractor to "date by the rules you were given",
        # which is the guess this whole change exists to stop. So it is carried
        # through and named as undated on purpose.
        if reading.kind in NO_DCT_ORIGIN_KINDS:
            # ...unless it is an offset from an event THIS conversation dates, in
            # which case the second pass below does the subtraction. Held back
            # rather than resolved here because the phrase it counts from may not
            # have been resolved yet.
            if reading.kind == "offset_from_event":
                deferred.append(reading)
            else:
                table[reading.phrase] = {"event_start": None, "event_end": None,
                                         "event_precision": None,
                                         "no_dct_origin": reading.kind}
            continue
        # ⚠ The assistant was not there. See PhraseReading.said_by: its invented
        # "three weeks ago" dated a festival the user had just got back from.
        # Refused rather than dropped, for the same reason as the kinds above:
        # a phrase missing from the block is one the extractor is told to date
        # by its own rules, which is the guess being prevented.
        if reading.said_by == "assistant" and reading.kind in COUNTED_FROM_DCT:
            table[reading.phrase] = {"event_start": None, "event_end": None,
                                     "event_precision": None,
                                     "assistant_only": True}
            continue
        try:
            start, end, precision = resolve_reading(reading, dct)
        except Exception:  # noqa: BLE001 — one bad phrase never costs the table
            logger.warning("date_prepass_resolve_failed phrase=%r kind=%r",
                           reading.phrase, reading.kind, exc_info=True)
            # ⚠ Not a bare continue when there is no conversation date. Dropping
            # the phrase hands it back to the model's own rules, and with no
            # date those rules are the guess being prevented. A phrase we failed
            # to resolve is still a phrase we cannot date.
            if dct is None:
                table[reading.phrase] = {"event_start": None, "event_end": None,
                                         "event_precision": None,
                                         "no_conversation_date": True}
            continue
        if start is None:
            # With no conversation date, a phrase we could not date has to be
            # NAMED as undated. A phrase missing from the block entirely is one
            # the extraction prompt tells the model to "date by the rules you
            # were given", which is the guess this module exists to stop.
            if dct is None:
                table[reading.phrase] = {"event_start": None, "event_end": None,
                                         "event_precision": None,
                                         "no_conversation_date": True}
            continue
        start, end, precision = validate_event_window(start, end, precision, dct)
        if start is None:
            continue
        window = {"event_start": start, "event_end": end,
                  "event_precision": precision}
        # ⚠ The table is keyed by the phrase TEXT, so one conversation can only
        # give one phrase one date. When the same words are used twice with
        # different meanings the second reading used to overwrite the first
        # silently, and BOTH memories were then stamped with the surviving date.
        #
        # Measured on a real ingest: "I finally decluttered the
        # kitchen this weekend" (past) and "I'm going hiking this weekend"
        # (future), said on the same Monday, both came out dated to the COMING
        # weekend. One of those two is always wrong.
        #
        # So a phrase that resolves two different ways is dated neither way. The
        # extraction prompt then sees it in the NOT DATED list and leaves the
        # date null, which is this module's standing rule: refusing is always
        # safe, a wrong date is not. Keying by occurrence instead would be the
        # larger fix, and it changes the block's contract with five prompts, so
        # it is not something to do without a measurement.
        prior = table.get(reading.phrase)
        if prior is not None and not prior.get("ambiguous_in_document"):
            same = (prior.get("event_start") == start
                    and prior.get("event_end") == end
                    and prior.get("event_precision") == precision)
            if not same:
                logger.info(
                    "date_prepass_phrase_is_ambiguous phrase=%r first=%s second=%s",
                    reading.phrase, prior.get("event_start"), start)
                table[reading.phrase] = {"event_start": None, "event_end": None,
                                         "event_precision": None,
                                         "no_dct_origin": "two readings in one conversation",
                                         "ambiguous_in_document": True}
                continue
        if prior is not None and prior.get("ambiguous_in_document"):
            continue
        table[reading.phrase] = window

    # Second pass: "three months in advance" of something this conversation
    # already dated. The arithmetic is the event anchor's, reused rather than
    # written twice, so a change to how an offset is applied cannot mean two
    # different things in two places.
    for reading in deferred:
        resolved = _resolve_offset_from_event(reading, table, dct)
        if resolved is None:
            # The conversation does not date the other event, or the offset is
            # unusable. Named as undated exactly as before, so the extraction
            # prompt still sees "we could not date this" rather than nothing.
            table[reading.phrase] = {"event_start": None, "event_end": None,
                                     "event_precision": None,
                                     "no_dct_origin": reading.kind}
            continue
        table[reading.phrase] = resolved
    return table


def _resolve_offset_from_event(reading: PhraseReading,
                               table: Dict[str, Dict[str, Any]],
                               dct: Optional[datetime]) -> Optional[Dict[str, Any]]:
    """The window for an offset counted from another phrase in this conversation.

    None whenever anything is missing or unusable, which leaves the phrase
    undated exactly as it was before this existed. Refusing is always safe; a
    wrong date is not.
    """
    from synap.cloud.agents.ingestion.services.event_anchor import shift
    anchor = (reading.anchor_phrase or "").strip()
    if not anchor:
        return None
    base = table.get(anchor)
    if base is None:
        # The model may quote the anchor with different surrounding whitespace or
        # case than the phrase key. One forgiving lookup, no fuzzy matching: word
        # comparison deciding a DATE is the mistake this file exists to stop.
        want = " ".join(anchor.lower().split())
        for key, value in table.items():
            if " ".join(key.lower().split()) == want:
                base = value
                break
    if not base or base.get("event_start") is None:
        return None
    if reading.n in (None, 0) or reading.unit not in VALID_OFFSET_UNITS \
            or reading.direction not in VALID_OFFSET_DIRECTIONS:
        return None
    start = shift(base["event_start"], reading.n, reading.unit, reading.direction)
    if start is None:
        return None
    end = shift(base.get("event_end"), reading.n, reading.unit, reading.direction) \
        if base.get("event_end") else None
    start, end, precision = validate_event_window(
        start, end, base.get("event_precision"), dct)
    if start is None:
        return None
    return {"event_start": start, "event_end": end, "event_precision": precision,
            "offset_from": anchor}


# ──────────────────────────────────────────────────────────────────────────────
# the reading prompt — structure out, never a date
# ──────────────────────────────────────────────────────────────────────────────


def _render_conversation_date(dct: Optional[datetime]) -> str:
    """The one line of the reading prompt that says when the conversation was.

    Two forms, because there are two truths. Stating a date we invented is how
    "three months ago" in a 2023 conversation became July 2026; saying nothing
    at all would leave the model to assume today without being told to.
    """
    if dct is not None:
        return CONVERSATION_DATE_LINE % (dct.strftime("%Y-%m-%d"), dct.strftime("%A"))
    return NO_CONVERSATION_DATE_NOTICE


def _parse_reading_response(text: str) -> List[PhraseReading]:
    try:
        payload = json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:
            return []
        try:
            payload = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return []
    raw = payload.get("phrases") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        reading = PhraseReading.from_raw(item)
        if reading is not None:
            out.append(reading)
    return out


async def run_prepass(*, content: str, dct: Optional[datetime], llm_client,
                      correlation_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """One LLM call per DOCUMENT, then code resolves. Never raises.

    An empty table is the safe answer and means every per-type prompt goes out
    unchanged, so a timeout, a refusal or a malformed response costs nothing.
    """
    # ⚠ `dct is None` was once in this condition, so a conversation
    # whose date the client never sent got no reading at all. That threw away
    # far more than the phrases it had to: classifying a phrase needs no date,
    # a phrase that states its own date resolves without one, and a phrase
    # counted from ANOTHER event this conversation dates resolves without one
    # too. The only phrases that genuinely cannot be dated are the ones counted
    # from today, and those are reported as undated rather than dropped.
    if not content or llm_client is None:
        return {}

    messages = [
        {"role": "system", "content": READING_SYSTEM_PROMPT.format(
            conversation_date=_render_conversation_date(dct))},
        {"role": "user", "content": READING_USER_PROMPT.format(content=content)},
    ]
    try:
        response = await asyncio.wait_for(
            llm_client.complete(messages=messages, temperature=0.0,
                                max_tokens=1500, json_mode=True),
            timeout=timeout_secs())
    except asyncio.TimeoutError:
        logger.warning("date_prepass_timeout secs=%.1f correlation_id=%s",
                       timeout_secs(), correlation_id or "")
        return {}
    except TypeError:
        # Some clients do not take every keyword. Try the narrow signature once
        # rather than losing the pre-pass to a plumbing mismatch — the same
        # mistake that made an earlier temporal filter a guaranteed no-op.
        try:
            response = await asyncio.wait_for(
                llm_client.complete(messages=messages, json_mode=True),
                timeout=timeout_secs())
        except Exception:  # noqa: BLE001
            logger.warning("date_prepass_llm_signature_mismatch correlation_id=%s",
                           correlation_id or "", exc_info=True)
            return {}
    except Exception:  # noqa: BLE001
        logger.warning("date_prepass_llm_failed correlation_id=%s",
                       correlation_id or "", exc_info=True)
        return {}

    try:
        return _table_from_response(response, dct, correlation_id)
    except Exception:  # noqa: BLE001 — the docstring promises it, and stage 3 relies on it
        logger.warning("date_prepass_resolve_failed correlation_id=%s",
                       correlation_id or "", exc_info=True)
        return {}


def _table_from_response(response, dct: Optional[datetime],
                         correlation_id: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Parse the model's reply and resolve it. May raise; ``run_prepass`` catches."""
    text = (response.get("content") if isinstance(response, dict)
            else getattr(response, "content", response))
    readings = _parse_reading_response(str(text or ""))
    table = resolve_all(readings, dct)
    # Split, because the table now also carries phrases that are deliberately NOT
    # dated. Counting them as "resolved" made the one number an operator reads to
    # check the pre-pass mean two different things at once.
    dated = sum(1 for w in table.values() if w.get("event_start") is not None)
    logger.info("date_prepass phrases_read=%d dated=%d not_dated=%d correlation_id=%s",
                len(readings), dated, len(table) - dated, correlation_id or "")
    # ⚠ Two counts cannot answer "why is this memory dated that?". The table is
    # what five per-type prompts are then told not to argue with, so a wrong
    # date on a stored row has no trail back to the phrase that produced it, or
    # to whether the year came from the speaker, from the model or from the
    # tense. Finding that out for ONE row cost a re-ingest of four scopes. The phrases are
    # short time expressions and the median
    # document yields one, so the line is cheap.
    if readings:
        logger.info("date_prepass_table %s correlation_id=%s",
                    " | ".join(_describe_reading(r, table) for r in readings),
                    correlation_id or "")
    return table


def _describe_reading(reading: PhraseReading,
                      table: Dict[str, Dict[str, Any]]) -> str:
    """One phrase, as read and as resolved, for the log line above.

    ``model_date`` is printed next to the resolved window on purpose: when the
    phrase states no year those two disagree, and the disagreement is the thing
    worth seeing.
    """
    window = table.get(reading.phrase) or {}
    start, end = window.get("event_start"), window.get("event_end")
    if start is None:
        when = ("said by the assistant" if window.get("assistant_only")
                else window.get("no_dct_origin") or "not dated")
    else:
        when = f"{start:%Y-%m-%d}" + (f"..{end:%Y-%m-%d}" if end else "")
    tense = f" tense={reading.tense}" if reading.tense else ""
    model_date = f" model_date={reading.date}" if reading.date else ""
    return f'"{reading.phrase}" [{reading.kind}{tense}{model_date}] -> {when}'


# ──────────────────────────────────────────────────────────────────────────────
# the block each per-type call receives
# ──────────────────────────────────────────────────────────────────────────────

def render_resolved_block(table: Dict[str, Dict[str, Any]]) -> str:
    """The resolved dates, as a block prepended to each per-type user prompt.

    Empty table -> empty string, so the prompt is byte-identical to today.
    """
    if not table:
        return ""
    lines, undated, assistant_only, undatable = [], [], [], []
    for phrase, window in table.items():
        if window.get("assistant_only"):
            assistant_only.append(f'  "{phrase}"')
            continue
        # Its own section, because the reason is different from the one below.
        # These are not counted from another event; there is simply no day to
        # count from, because nothing said when this conversation happened.
        if window.get("no_conversation_date"):
            undatable.append(f'  "{phrase}"')
            continue
        if window.get("no_dct_origin"):
            undated.append(f'  "{phrase}"')
            continue
        start = window["event_start"]
        if start is None:
            continue
        end = window.get("event_end")
        precision = window.get("event_precision") or "day"
        when = (f"{start:%Y-%m-%d} to {end:%Y-%m-%d}" if end else f"{start:%Y-%m-%d}")
        # An offset says what it was counted from. Without it the block asserts a
        # date the conversation never spells out, and the extraction prompt's own
        # rule about distances from other events reads as a contradiction. With
        # it the two halves agree and the arithmetic is auditable in the prompt.
        via = window.get("offset_from")
        suffix = f' (counted from "{via}")' if via else ""
        lines.append(f'  "{phrase}" -> event_start {when}, precision {precision}{suffix}')

    out = ""
    if lines:
        out += (RESOLVED_DATES_HEADER
                + "\n".join(lines) + "\n\n")
    if undated:
        # ⚠ This section is why the whole pre-pass stopped contradicting the
        # event-anchor rule. The resolved list above is framed as already decided,
        # so whatever it says wins over any rule further down the prompt. When a
        # distance from another event landed in it, the extractor was ordered to
        # use a date computed from the wrong origin, and the anchor rule saying
        # "do not guess from DCT" lost. Naming these phrases here makes the two
        # halves agree instead.
        out += (NOT_DATED_HEADER
                + "\n".join(undated) + "\n\n")
    if assistant_only:
        out += (ASSISTANT_ONLY_HEADER
                + "\n".join(assistant_only) + "\n\n")
    if undatable:
        out += (NO_DATE_HEADER
                + "\n".join(undatable) + "\n\n")
    return out
