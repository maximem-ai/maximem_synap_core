"""Does a question restrict its answer to a period of time, and which dates is that?

WHY THE WORK IS SPLIT THE WAY IT IS

Two jobs live here and they are deliberately given to different things, because
measurement said so. Asked against the real Jev API on 2026-10-01, with the
question "I mentioned visiting a museum two months ago. Did I visit with a
friend or not?" and TODAY set to 2023-03-11:

  * "Does the question refer to a period of time?"  -> yes, confidence 1.00
  * "Is the memory dated 2023-01-15 inside that period?" -> OUTSIDE (0.65)
  * "Is the memory dated 2023-02-17 inside that period?" -> INSIDE  (0.75)

Two months before 11 March is 11 January, so January 15 is inside and
February 17 is not. The model got the classification right and both pieces of
arithmetic backwards. The same split showed up in the September temporal work:
Jev's NOUL beat our regex as the gate for "needs temporal rules" (93% against
27% recall) while regex date resolution answered only 16-36% of relative dates.

So: a model reads the question and returns a STRUCTURED READING. This module
does the calendar arithmetic. No model is asked what date anything falls on.

THE DISTINCTION THAT KEEPS THIS SAFE

"Two months ago", "in March" and "last week" RESTRICT the answer to a period.
Those can drive a filter.

"How many weeks ago did I attend X" and "how long had I been doing Y" ask FOR a
date or a duration. They name a period only in the sense that the answer is
one. Filtering on those is wrong: the evidence can sit anywhere on the
calendar, and on 982b5123 the answer needs a memory from outside any window the
question implies. ``restricts_to_period`` is false for them, and nothing is
dropped.
"""

from __future__ import annotations

import calendar
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from synap.cloud.application.client_api.prompts.temporal_question import READING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# the structured reading a model is asked for
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalWindow:
    """An inclusive [start, end] the question restricts its answer to."""
    start: datetime
    end: datetime
    precision: str
    phrase: str = ""

    def __str__(self) -> str:
        return f"{self.start.date()}..{self.end.date()} ({self.precision})"


def parse_reading(raw: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON object out of a model reply. Returns None on anything odd.

    Tolerant of a code fence and of leading prose, because the cost of being
    strict here is losing the whole filter for a formatting wobble.
    """
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.S)
        if not brace:
            return None
        text = brace.group(0)
    try:
        out = json.loads(text)
    except Exception:
        return None
    return out if isinstance(out, dict) else None


# ---------------------------------------------------------------------------
# the arithmetic — no model involved past this line
# ---------------------------------------------------------------------------

# How far past a week boundary a memory may sit and still count. See the week
# branch of resolve_window for the measurement behind 3.
_WEEK_SLACK_DAYS = 3

_DAY_END = {"hour": 23, "minute": 59, "second": 59, "microsecond": 0}


def _start_of_day(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(d: datetime) -> datetime:
    return d.replace(**_DAY_END)


def _add_months(d: datetime, months: int) -> datetime:
    """Shift by whole months, clamping the day to the target month's length."""
    total = (d.year * 12 + (d.month - 1)) + months
    year, month = divmod(total, 12)
    month += 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return d.replace(year=year, month=month, day=day)


def _month_window(year: int, month: int, ref: datetime) -> TemporalWindow:
    last = calendar.monthrange(year, month)[1]
    return TemporalWindow(
        _start_of_day(ref.replace(year=year, month=month, day=1)),
        _end_of_day(ref.replace(year=year, month=month, day=last)),
        "month",
    )


def resolve_window(reading: Dict[str, Any], query_date: datetime) -> Optional[TemporalWindow]:
    """Turn a structured reading plus the date the question was asked into a window.

    Returns None whenever the question does not restrict to a period, or the
    reading is not usable. None means "do not filter", which is the safe answer:
    every failure mode here degrades to today's behaviour.

    TOLERANCE IS PART OF THE WINDOW, NOT A SEPARATE KNOB. "Two months ago" is
    not a two-month-wide claim about a single day, it is a vague reference to a
    region of the calendar, so a relative_past window spans from n+1 units back
    to n-1 units back rather than landing on one date. Measured need: on
    0bc8ad93 the question is asked 2023-03-11 and says "two months ago"; the
    answer-bearing memory is dated January 2023 at month precision, which a
    single-day window would exclude.
    """
    if not reading or not reading.get("restricts_to_period"):
        return None

    kind = (reading.get("kind") or "none").strip()
    phrase = str(reading.get("phrase") or "")[:120]
    qd = query_date if query_date.tzinfo else query_date.replace(tzinfo=timezone.utc)

    def _int(key):
        v = reading.get(key)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    unit, n = (reading.get("unit") or "").strip(), _int("n")

    if kind == "relative_past":
        if not unit or n is None or n < 0:
            return None
        if unit == "day":
            centre = qd - timedelta(days=n)
            pad = timedelta(days=1)
            return TemporalWindow(_start_of_day(centre - pad), _end_of_day(centre + pad), "day", phrase)
        if unit == "week":
            # "the past two weeks" reads as the span up to today, not a point.
            #
            # Three days of slack at the far end, not one. Measured on
            # gpt4_59149c78: asked 2023-02-01, "two weeks ago", and the Met visit
            # that answers it is dated 2023-01-15. A one-day pad ends on 17 Jan
            # and deletes the answer by two days. Nobody saying "two weeks ago"
            # means a 14-day boundary to the day, and the cost of the slack is
            # keeping a few more items, which this filter is allowed to do.
            start = _start_of_day(qd - timedelta(weeks=n, days=_WEEK_SLACK_DAYS))
            return TemporalWindow(start, _end_of_day(qd), "week", phrase)
        if unit == "month":
            centre = _add_months(qd, -n)
            return TemporalWindow(_start_of_day(_add_months(centre, -1)),
                                  _end_of_day(_add_months(centre, 1)), "month", phrase)
        if unit == "year":
            centre = qd.replace(year=qd.year - n)
            return TemporalWindow(_start_of_day(centre.replace(month=1, day=1)),
                                  _end_of_day(centre.replace(month=12, day=31)), "year", phrase)
        return None

    if kind == "relative_future":
        if not unit or n is None:
            return None
        if unit == "day":
            centre = qd + timedelta(days=n)
            return TemporalWindow(_start_of_day(centre - timedelta(days=1)),
                                  _end_of_day(centre + timedelta(days=1)), "day", phrase)
        if unit == "week":
            return TemporalWindow(_start_of_day(qd), _end_of_day(qd + timedelta(weeks=n, days=1)), "week", phrase)
        if unit == "month":
            centre = _add_months(qd, n)
            return TemporalWindow(_start_of_day(_add_months(centre, -1)),
                                  _end_of_day(_add_months(centre, 1)), "month", phrase)
        return None

    if kind == "named_month":
        month = _int("month")
        if not month or not 1 <= month <= 12:
            return None
        year = _int("year")
        if year is None:
            # No year given: the most recent occurrence at or before the question.
            year = qd.year if month <= qd.month else qd.year - 1
        return TemporalWindow(_month_window(year, month, qd).start,
                              _month_window(year, month, qd).end, "month", phrase)

    if kind == "weekday_past":
        wd = _int("weekday")
        if wd is None or not 0 <= wd <= 6:
            return None
        back = (qd.weekday() - wd) % 7 or 7          # "last Saturday" is never today
        target = qd - timedelta(days=back)
        return TemporalWindow(_start_of_day(target), _end_of_day(target), "day", phrase)

    if kind == "explicit_date":
        raw = reading.get("date")
        if not raw:
            return None
        try:
            d = datetime.fromisoformat(str(raw)).replace(tzinfo=qd.tzinfo)
        except Exception:
            return None
        return TemporalWindow(_start_of_day(d), _end_of_day(d), "day", phrase)

    return None


# ---------------------------------------------------------------------------
# applying it to an item
# ---------------------------------------------------------------------------

def item_is_outside(item: Any, window: TemporalWindow) -> bool:
    """Is this item's date KNOWN to fall outside the window?

    False whenever we are not sure, and that covers three cases on purpose:

      * no date at all. Two thirds of served memories are in this state
        (2,164 of 3,259 measured on the lme3 failure scopes). They used to
        arrive carrying the conversation date instead, flagged
        ``event_date_is_mention``, and this function had to test that flag to
        avoid dropping evidence for a date the read path had invented;
        `event_date` is now simply empty for them and the first test below
        catches it. 982b5123 is the case: "had to book three months in advance"
        carries no real date and is exactly the memory the answer needs;
      * an item carrying its own window that overlaps the question's.

    So only a genuine, non-overlapping event date is ever reported as outside.
    """
    if window is None:
        return False
    start = getattr(item, "event_date", None)
    if start is None:
        return False
    end = getattr(item, "event_end", None) or start
    if start.tzinfo is None:
        start = start.replace(tzinfo=window.start.tzinfo)
    if end.tzinfo is None:
        end = end.replace(tzinfo=window.start.tzinfo)
    # overlap test, inclusive on both sides
    return end < window.start or start > window.end
