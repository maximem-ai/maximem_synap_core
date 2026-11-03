"""The guard for the anchor column added by migration 155.

A database can have 154 and not 155 — the normal state right after 155 ships — so
the window and the anchor are tracked separately and have to strip independently.
"""

import inspect
import re

import pytest

from synap.cloud.shared import event_window_schema as ews
from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import (
    SQLAlchemyRDBMSWrapper,
)

WRITERS = ("insert_memory", "insert_candidate")


def sql_of(method):
    return re.search(r'"""(.*?)"""',
                     inspect.getsource(getattr(SQLAlchemyRDBMSWrapper, method)),
                     re.S).group(1)


@pytest.fixture(autouse=True)
def _reset():
    ews._reset_for_tests()
    yield
    ews._reset_for_tests()


def test_both_writers_actually_persist_the_anchor():
    """If a writer drops the column the field is silently lost, which is how a
    new field normally dies: the model has it, the row never does."""
    for m in WRITERS:
        sql = sql_of(m)
        assert "event_anchor" in sql, m
        assert "CAST(:event_anchor AS JSONB)" in sql, m
        assert "event_anchor = EXCLUDED.event_anchor" in sql, m


@pytest.mark.parametrize("method", WRITERS)
@pytest.mark.parametrize("window_ok", [True, False])
def test_a_database_without_155_still_inserts(method, window_ok):
    ews._available = window_ok
    ews._anchor_available = False
    out = ews.adapt(sql_of(method))
    assert "event_anchor" not in out
    assert out.count("(") == out.count(")")
    assert not re.search(r",\s*\)", out)
    # the window is independent: having 154 and not 155 must keep the window
    assert ("event_start" in out) is window_ok


def test_stripping_the_anchor_does_not_maim_the_upsert_clause():
    """Regression. ``\\s*event_anchor,`` also matches the TAIL of
    "event_anchor = EXCLUDED.event_anchor," and left "event_anchor = EXCLUDED."
    behind: broken SQL with balanced parentheses, so a paren-counting check
    passed while every insert would have failed."""
    ews._anchor_available = False
    for m in WRITERS:
        out = ews.adapt(sql_of(m))
        assert "EXCLUDED." not in out or not re.search(r"EXCLUDED\.\s*(?:\n|$)", out), out
        assert "= EXCLUDED.\n" not in out


def test_adapt_is_a_no_op_while_both_families_exist():
    for m in WRITERS:
        sql = sql_of(m)
        assert ews.adapt(sql) == sql


def test_a_missing_anchor_column_does_not_disable_the_window():
    """One shared flag would make a missing 155 silently switch off 154's window
    for the whole process."""
    ews.mark_missing(Exception('column "event_anchor" of relation "memories" '
                               'does not exist'))
    assert ews.anchor_available() is False
    assert ews.available() is True


def test_a_missing_window_column_does_not_disable_the_anchor():
    ews.mark_missing(Exception('column "event_start" of relation "memories" '
                               'does not exist'))
    assert ews.available() is False
    assert ews.anchor_available() is True


def test_the_existing_retry_path_recognises_an_anchor_error():
    """Call sites catch, mark and retry through is_missing_column_error, so it has
    to answer for the anchor too or the retry never happens."""
    assert ews.is_missing_column_error(
        Exception('column "event_anchor" does not exist')) is True
    assert ews.is_missing_column_error(
        Exception('column "event_start" does not exist')) is True
    assert ews.is_missing_column_error(Exception("connection reset")) is False


def test_a_column_after_event_precision_does_not_defeat_the_window_strip():
    """Regression, and the reason to keep this test when the next column lands.

    The window patterns required event_precision to be followed straight by ")".
    Adding event_anchor after it meant that on an unmigrated database the window
    was no longer removed, so every insert failed again — the exact outage the
    guard exists to prevent, reintroduced by a one-column change elsewhere.
    """
    ews._available = False
    ews._anchor_available = True          # has 155, lacks 154
    for m in WRITERS:
        out = ews.adapt(sql_of(m))
        assert not re.search(r"\bevent_(start|end|precision)\b", out), out
        assert "event_anchor" in out      # 155 is present, so it stays
        assert out.count("(") == out.count(")")
        assert not re.search(r",\s*\)", out)
        assert ",," not in out
