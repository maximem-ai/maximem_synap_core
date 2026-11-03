"""Keep working on a database that has not had migration 154 or 155 applied.

The event-window columns (event_start, event_end, event_precision) are read and
written on the hot ingestion and retrieval paths, and migration 155 adds
event_anchor beside them. Migrations may be applied by hand per deployment,
and a deployment can run for weeks without later migrations.
Without this, deploying this code to such a database would fail every memory
insert and every Postgres read arm.

So each SQL site passes its statement through :func:`adapt`, and on the first
"column event_* does not exist" error calls :func:`mark_missing` and runs the
adapted statement again. From then on the process leaves the missing columns out:
the memory is stored and served exactly as before they existed, and one error line
says which migration is missing.

⚠ The window and the anchor are tracked SEPARATELY. A database can have 154 and
not 155, which is the normal state right after 155 ships, and one shared flag
would make a missing anchor column silently disable the event window too.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Named `_available` (not `_window_available`) because tests and callers
# already reach for it by that name; the anchor gets its own flag.
_available = True
_anchor_available = True

_WINDOW_COLUMNS = ("event_start", "event_end", "event_precision")
_ANCHOR_COLUMNS = ("event_anchor",)

_SELECT_COLS = re.compile(r"\s*event_start,\s*event_end,\s*event_precision,")
# ⚠ The lookaheads tolerate a following event_anchor (migration 155). Without
# that, adding any column AFTER event_precision silently defeats the window
# strip: the pattern required event_precision to be followed straight by ")",
# so on an unmigrated database the window stopped being removed and every
# insert failed again.
_ANCHOR_TAIL = r"(?:,\s*event_anchor)?"
_INSERT_COLS = re.compile(r",\s*event_start,\s*event_end,\s*event_precision"
                          r"(?=" + _ANCHOR_TAIL + r"\s*(?:\)|FROM\b))")
_INSERT_VALS = re.compile(
    r",\s*:event_start,\s*:event_end,\s*:event_precision"
    r"(?=(?:,\s*(?:CAST\(\s*:event_anchor\s+AS\s+JSONB\s*\)|:event_anchor))?\s*\))")
_UPSERT_SET = re.compile(
    r"\s*event_start = EXCLUDED\.event_start,\s*event_end = EXCLUDED\.event_end,"
    r"\s*event_precision = EXCLUDED\.event_precision,")

# The anchor rides the same statements one column further along. Its value may be
# spelled bare or wrapped in a cast, so both forms are stripped.
# ⚠ The negative lookbehind is load-bearing. Without it this also matches the
# TAIL of "event_anchor = EXCLUDED.event_anchor," and leaves "event_anchor =
# EXCLUDED." behind — broken SQL that still has balanced parentheses, so a
# paren-counting test passes while every insert fails.
_ANCHOR_SELECT_COL = re.compile(r"\s*(?<![.:])\bevent_anchor,")
_ANCHOR_INSERT_COL = re.compile(r",\s*event_anchor(?=\s*(?:\)|FROM\b))")
_ANCHOR_INSERT_VAL = re.compile(
    r",\s*(?:CAST\(\s*:event_anchor\s+AS\s+JSONB\s*\)|:event_anchor)(?=\s*\))",
    re.IGNORECASE)
_ANCHOR_UPSERT_SET = re.compile(r"\s*event_anchor = EXCLUDED\.event_anchor,")


def available() -> bool:
    """Whether the event WINDOW columns are usable. Kept for existing callers."""
    return _available


def anchor_available() -> bool:
    return _anchor_available


def _mentions(exc_text: str, columns) -> bool:
    return any(c in exc_text for c in columns)


def is_missing_column_error(exc: BaseException) -> bool:
    """True for a missing window OR anchor column.

    Deliberately covers both so every existing call site — `except` block, mark,
    retry — picks up the anchor without being touched. :func:`mark_missing` works
    out which one it was.
    """
    text = str(exc)
    if "does not exist" not in text:
        return False
    return _mentions(text, _WINDOW_COLUMNS) or _mentions(text, _ANCHOR_COLUMNS)


def mark_missing(exc: BaseException) -> None:
    """Record which family this database lacks, from the error's own text."""
    global _available, _anchor_available
    text = str(exc)
    if _mentions(text, _ANCHOR_COLUMNS) and _anchor_available:
        logger.error(
            "event_anchor_column_missing: migration 155_memory_event_anchor.sql is not "
            "applied to this database; a date that depends on another event cannot be "
            "stored for later resolution until it is. error=%s", exc)
        _anchor_available = False
    if _mentions(text, _WINDOW_COLUMNS) and _available:
        logger.error(
            "event_window_columns_missing: migration 154_memory_event_window.sql is not "
            "applied to this database; memories are stored and served without their event "
            "window until it is. error=%s", exc)
        _available = False


def adapt(sql: str) -> str:
    """The statement as this database can run it."""
    if not _anchor_available:
        # Upsert clause FIRST: it is the only place the column name appears
        # twice on one line, so removing it whole stops any later pattern
        # chewing half of it.
        sql = _ANCHOR_UPSERT_SET.sub("", sql)
        sql = _ANCHOR_INSERT_VAL.sub("", sql)
        sql = _ANCHOR_INSERT_COL.sub("", sql)
        sql = _ANCHOR_SELECT_COL.sub("", sql)
    if not _available:
        sql = _SELECT_COLS.sub("", sql)
        sql = _INSERT_COLS.sub("", sql)
        sql = _INSERT_VALS.sub("", sql)
        sql = _UPSERT_SET.sub("", sql)
    return sql


def _reset_for_tests() -> None:
    """Both families available again. Tests only — the flags are process-global."""
    global _available, _anchor_available
    _available = True
    _anchor_available = True
