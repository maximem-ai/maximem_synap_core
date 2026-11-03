"""Tests for the subject queries in storage/sqlalchemy_rdbms_wrapper.py.

No database: ``fetch_all`` / ``_execute`` are replaced by recorders, so these pin
the SQL shape (the filters serving relies on) and the Python around it.
"""
from types import SimpleNamespace

import pytest

from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import SQLAlchemyRDBMSWrapper


def wrapper(rows=None, raise_on=None):
    w = SQLAlchemyRDBMSWrapper.__new__(SQLAlchemyRDBMSWrapper)
    w.calls = []

    async def fetch_all(sql, params=None):
        w.calls.append((sql, params))
        if raise_on:
            raise raise_on
        return rows if rows is not None else []

    async def _execute(sql, params=None):
        w.calls.append((sql, params))
    w.fetch_all = fetch_all
    w._execute = _execute
    return w


def norm(sql):
    return " ".join(sql.split())


class TestSubjectsInScope:
    async def test_groups_contents_by_subject_in_row_order(self):
        w = wrapper([{"subject_id": "S1", "content": "a"}, {"subject_id": "S1", "content": "b"},
                     {"subject_id": "S2", "content": "c"}])
        assert await w.subjects_in_scope(SimpleNamespace(user_id="u1")) == {"S1": ["a", "b"], "S2": ["c"]}

    async def test_empty(self):
        assert await wrapper([]).subjects_in_scope(SimpleNamespace(user_id="u1")) == {}

    async def test_null_content_becomes_empty_string_not_none(self):
        w = wrapper([{"subject_id": "S1", "content": None}])
        assert await w.subjects_in_scope(SimpleNamespace(user_id="u1")) == {"S1": [""]}

    async def test_only_reaching_rows_are_asked_for(self):
        w = wrapper([])
        await w.subjects_in_scope(SimpleNamespace(user_id="u1"))
        sql = norm(w.calls[0][0])
        assert "deleted_at IS NULL" in sql and "superseded_by IS NULL" in sql
        assert "subject_id IS NOT NULL" in sql
        assert "user_id = CAST(:uid AS uuid)" in sql

    async def test_scoped_to_the_users_id(self):
        w = wrapper([])
        await w.subjects_in_scope(SimpleNamespace(user_id="u-42"))
        assert w.calls[0][1] == {"uid": "u-42"}

    @pytest.mark.parametrize("scope", [SimpleNamespace(), SimpleNamespace(user_id=None), None])
    async def test_missing_user_id_is_passed_as_empty_string(self, scope):
        w = wrapper([])
        await w.subjects_in_scope(scope)
        assert w.calls[0][1] == {"uid": ""}

    async def test_database_error_propagates_for_the_caller_to_handle(self):
        w = wrapper(raise_on=RuntimeError("down"))
        with pytest.raises(RuntimeError):
            await w.subjects_in_scope(SimpleNamespace(user_id="u1"))


class TestMergeSubjects:
    @pytest.mark.parametrize("frm,into", [("", "S1"), ("S1", ""), (None, "S1"), ("S1", None),
                                          (None, None), ("S1", "S1")])
    async def test_falsy_or_equal_ids_are_a_no_op_without_touching_the_db(self, frm, into):
        w = wrapper([{"id": "m"}])
        assert await w.merge_subjects(frm, into) == 0
        assert w.calls == []

    async def test_returns_rows_moved(self):
        w = wrapper([{"id": "a"}, {"id": "b"}, {"id": "c"}])
        assert await w.merge_subjects("S2", "S1") == 3

    async def test_returns_zero_when_nothing_moved(self):
        assert await wrapper([]).merge_subjects("S2", "S1") == 0

    async def test_repoints_from_to_into_and_nothing_else(self):
        w = wrapper([])
        await w.merge_subjects("FROM", "INTO")
        sql, params = w.calls[0]
        assert params == {"frm": "FROM", "into": "INTO"}
        s = norm(sql)
        assert "UPDATE memories SET subject_id = CAST(:into AS uuid)" in s
        assert "WHERE subject_id = CAST(:frm AS uuid)" in s
        assert "RETURNING" in s
        assert "subject_label" not in s              # the label is rewritten by the join, not here

    async def test_error_propagates(self):
        with pytest.raises(RuntimeError):
            await wrapper(raise_on=RuntimeError("x")).merge_subjects("a", "b")


class TestUnchangedHelpers:
    async def test_subjects_of_empty_input_does_not_query(self):
        w = wrapper([])
        assert await w.subjects_of([]) == {}
        assert w.calls == []

    async def test_subjects_of_maps_rows(self):
        w = wrapper([{"id": "m1", "subject_id": "S1", "subject_label": "L"}])
        assert await w.subjects_of(["m1", "m2"]) == {"m1": ("S1", "L")}

    async def test_assign_subject_empty_ids_does_not_write(self):
        w = wrapper([])
        await w.assign_subject([], "S1", "L")
        assert w.calls == []

    async def test_assign_subject_writes_label_for_every_id(self):
        w = wrapper([])
        await w.assign_subject(["a", "b"], "S1", "L")
        assert w.calls[0][1] == {"ids": ["a", "b"], "sid": "S1", "label": "L"}

    def test_removed_methods_are_gone(self):
        for name in ("clear_event_window", "subject_members"):
            assert not hasattr(SQLAlchemyRDBMSWrapper, name), name
