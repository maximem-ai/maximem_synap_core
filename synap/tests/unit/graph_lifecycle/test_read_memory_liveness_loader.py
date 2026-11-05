"""memory_liveness.load_supporter_rows: one query, by id, with the superseder's dates."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from synap.cloud.shared import memory_liveness


A = "3c1f4817-d3c8-4ad1-8fee-b26d9cfbe082"
B = "58bee2aa-8733-4d93-b167-38e0d6210781"


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows=(), fail=None):
        self.calls = []
        self._rows = list(rows)
        self._fail = fail

    async def execute(self, stmt, params):
        self.calls.append((str(stmt), params))
        if self._fail:
            raise self._fail
        return _Result(self._rows)


async def test_one_query_by_id_any_and_rows_mapped():
    dca = datetime(2026, 9, 6, tzinfo=timezone.utc)
    s = _Session(rows=[{
        "id": A, "superseded_by": B, "superseded_at": dca, "deprecated_at": None,
        "deleted_at": None, "is_episode": False, "temporal_category": "temporal_fact",
        "document_created_at": dca, "created_at": dca, "event_start": None,
        "n_document_created_at": dca, "n_created_at": dca, "n_event_start": dca,
    }])
    rows = await memory_liveness.load_supporter_rows([A, B, A, "not-a-uuid", None], session=s)
    assert len(s.calls) == 1
    sql, params = s.calls[0]
    assert "= ANY(" in sql and "LEFT JOIN memories n" in sql
    assert params == {"ids": [A, B]}
    assert set(rows) == {A}
    assert rows[A].superseded_by == B and rows[A].superseder_event_start == dca
    assert not rows[A].alive
    # B has no row: hard-deleted, so not alive.
    assert not memory_liveness.is_alive(rows.get(B))


async def test_empty_input_touches_nothing():
    s = _Session()
    assert await memory_liveness.load_supporter_rows([], session=s) == {}
    assert await memory_liveness.load_supporter_rows(["x", ""], session=s) == {}
    assert s.calls == []


async def test_database_error_propagates():
    s = _Session(fail=RuntimeError("db down"))
    with pytest.raises(RuntimeError):
        await memory_liveness.load_supporter_rows([A], session=s)


def test_alive_sql_matches_the_python_predicate():
    for col in ("superseded_by", "deprecated_at", "deleted_at"):
        assert f"m.{col} IS NULL" in memory_liveness.ALIVE_SQL
