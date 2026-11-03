"""The event window: WHEN a memory's event happened.

Extraction (the temporal addendum) asks the model for ``event_start``,
``event_end`` and ``event_precision``. "Bought a cat last month" said on
23 March 2023 is 2023-02-01 .. 2023-02-28, precision "month". The conversation
date stays in ``document_created_at``; valid_from/valid_until keep meaning how
long a STATE held. See migration 154 for why those are separate.

The model does the resolving. This module only does what code is better at:
parse leniently, reject windows that cannot be right, and render one for the
reader. It never invents a date.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, time
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

PRECISIONS = frozenset(
    {"day", "week", "weekend", "month", "season", "year", "range", "approximate"}
)

# A window this far from the conversation is a parse or model error, not a
# memory: TimeNorm turned "December 8-11" into 1911 on the TempEval test.
_MIN_YEAR = 1900
_MAX_YEARS_AHEAD = 20


def parse_event_day(value: Any) -> Optional[datetime]:
    """'2023-04-01' or an ISO timestamp -> midnight of that day. Anything else
    -> None. Never raises: a bad date must not cost the memory."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return datetime.combine(value.date(), time.min)
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if not isinstance(value, str):
        return None
    text = value.strip().strip("\"'`")
    if len(text) < 10:
        return None
    try:
        return datetime.combine(date.fromisoformat(text[:10]), time.min)
    except ValueError:
        return None


def _infer_precision(start: date, end: Optional[date]) -> str:
    if end is None or end == start:
        return "day"
    if (start.month, start.day, end.month, end.day) == (1, 1, 12, 31) and start.year == end.year:
        return "year"
    if start.day == 1 and (start.year, start.month) == (end.year, end.month) \
            and end.day == calendar.monthrange(end.year, end.month)[1]:
        return "month"
    return "range"


def validate_event_window(start, end, precision, dct: Optional[datetime]):
    """Return a cleaned (start, end, precision), or (None, None, None).

    ``end`` comes back as the last second of its day (the window is inclusive).
    Datetimes carry ``dct``'s tzinfo so they compare and store like the other
    timestamps on the row."""
    start, end = parse_event_day(start), parse_event_day(end)
    if start is None:
        return None, None, None
    if end is not None and end < start:
        end = None
    horizon = (dct.year if dct else datetime.utcnow().year) + _MAX_YEARS_AHEAD
    if not (_MIN_YEAR <= start.year <= horizon) or (end and end.year > horizon):
        return None, None, None
    if not isinstance(precision, str) or precision.strip().lower() not in PRECISIONS:
        precision = _infer_precision(start.date(), end.date() if end else None)
    else:
        precision = precision.strip().lower()
    tz = dct.tzinfo if dct is not None else None
    start = start.replace(tzinfo=tz)
    if end is not None:
        end = datetime.combine(end.date(), time(23, 59, 59), tzinfo=tz)
    return start, end, precision


def validate_event_windows(memories: Iterable[Any], dct: Optional[datetime]) -> int:
    """Clean every memory's window in place. Returns how many carry one.

    ⚠ DO NOT clear the window on a `perpetual` memory here. It is tempting: our
    own definition of perpetual is "No temporal bounds", and 36 rows in one
    ingest violated it. That check was written and reverted
    after reading all 36.

    Half of them carry a date somebody actually stated, and the label is what is
    wrong, not the date: "The user completed their first full marathon on April
    10th", "The user's anniversary is on July 22nd, 2023", "ran 5 kilometers in
    the 'Run for the Cure' event on October 15th", "volunteered at the 'Food for
    Thought' charity gala on September 25th, 2023", ten research papers dated by
    their publication year. Four of those are the gold fact for a question in
    the 25-question set, so the check would have deleted the answers it was
    meant to protect.

    The reason the label loses this argument: `temporal_category` DEFAULTS to
    "perpetual" when the model omits it (stage3_extraction `_or_default`), and
    the model also mislabels plain occurrences as perpetual. The window came
    from words in the conversation; the label came from a classifier with a
    default. When they disagree, the window is the better evidence, so code must
    not delete data on the label's word.

    The same rule belongs in the PROMPT, where it shapes a judgement instead of
    destroying a value, and it is there: "perpetual never has an event window".
    """
    kept = 0
    for mem in memories:
        try:
            s, e, p = validate_event_window(
                getattr(mem, "event_start", None), getattr(mem, "event_end", None),
                getattr(mem, "event_precision", None), dct)
        except Exception:
            logger.warning("event_window_invalid memory_id=%s",
                           getattr(mem, "memory_id", None), exc_info=True)
            s = e = p = None
        mem.event_start, mem.event_end, mem.event_precision = s, e, p
        kept += s is not None
    return kept


def _day(d: date, weekday: bool = False) -> str:
    stem = f"{d.strftime('%b')} {d.day}, {d.year}"
    return f"{d.strftime('%a')} {stem}" if weekday else stem


_SEASONS = {3: "Spring", 6: "Summer", 9: "Fall", 12: "Winter"}


def format_event_window(start, end, precision, weekday: bool = False) -> Optional[str]:
    """How the window reads in front of a memory, e.g. 'Feb 2023',
    'Apr 2 – Apr 8, 2023', 'around May 13, 2023'. None when there is none.

    *weekday* names the day of the week on the forms that state a real day:
    'Sat Mar 18 – Sun Mar 19, 2023'. The reader is routinely asked which thing
    happened "the past weekend" or "last Friday", and the calendar arithmetic
    that turns a date into a weekday is arithmetic it gets wrong. On
    gpt4_e414231f ("which bike did I service the past weekend?", gold road bike)
    we served a mountain-bike repair dated Mar 15 — a Wednesday — and both
    readers put it in the answer beside the Mar 19 road bike.

    It is deliberately NOT added to the coarser forms. A year, a month, a season
    and an "around" date do not know which day they fell on, so naming a weekday
    there would assert a precision nobody gave us.
    """
    if start is None or not hasattr(start, "date"):
        return None
    a = start.date()
    b = end.date() if end is not None and hasattr(end, "date") else None
    p = (precision or "").lower()
    if p == "year":
        return f"{a.year}" if b else f"since {a.year}"
    if p == "month":
        return a.strftime("%b %Y")
    if p == "season" and a.month in _SEASONS:
        return f"{_SEASONS[a.month]} {a.year}"
    if p == "approximate":
        return f"around {_day(a)}"
    if b is None or b == a:
        return _day(a, weekday)
    if a.year != b.year:
        return f"{_day(a, weekday)} – {_day(b, weekday)}"
    head = f"{a.strftime('%a')} " if weekday else ""
    tail = f"{b.strftime('%a')} " if weekday else ""
    return (f"{head}{a.strftime('%b')} {a.day} – "
            f"{tail}{b.strftime('%b')} {b.day}, {a.year}")


def event_window_metadata(start, end, precision) -> dict:
    """The window as vector-store metadata. Chroma rejects None values, so a
    missing part is left out rather than stored as null."""
    meta = {}
    if start is not None:
        meta["event_start"] = start.isoformat() if hasattr(start, "isoformat") else str(start)
        if end is not None:
            meta["event_end"] = end.isoformat() if hasattr(end, "isoformat") else str(end)
        if precision:
            meta["event_precision"] = str(precision)
    return meta
