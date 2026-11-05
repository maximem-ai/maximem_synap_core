"""The soft delete: bound to one user, marks first, then clears every copy."""

import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.application.memory_forget.soft_delete import soft_delete_memories


def _scope(user=True):
    # Not the real Scope: importing it pulls in the MACA agent stack (langchain),
    # which these tests have no need of.
    return SimpleNamespace(client_id=uuid.uuid4(), instance_id=uuid.uuid4(),
                           customer_id=uuid.uuid4(), user_id=uuid.uuid4() if user else None,
                           scope_level="user" if user else "customer")


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeSession:
    """Marks every id it is given, unless told which ones the UPDATE refuses."""

    def __init__(self, marked=None, events=None):
        self.marked, self.events, self.statements = marked, events if events is not None else [], []

    async def execute(self, stmt, params=None):
        self.statements.append((str(stmt), params))
        self.events.append("update")
        ids = self.marked if self.marked is not None else params["ids"]
        return _Result([(i, f"text of {i}") for i in ids])

    async def commit(self):
        self.events.append("commit")


class Recorder:
    def __init__(self, events, fail=False):
        self.events, self.fail, self.calls = events, fail, []

    async def delete_memory_vectors(self, memory_ids, scope):
        self.events.append("vectors")
        if self.fail:
            raise RuntimeError("chroma down")
        self.calls.append(list(memory_ids))

    async def soft_delete_memory_from_edges(self, memory_id, scope, redact=False):
        self.events.append("graph")
        self.redact = redact
        if self.fail:
            raise RuntimeError("neo4j down")
        self.calls.append(memory_id)


async def test_the_update_is_bound_to_one_user_and_skips_deleted_rows():
    s = FakeSession()
    scope = _scope()
    await soft_delete_memories(s, scope=scope, memory_ids=[str(uuid.uuid4())], reason="r")
    sql, params = s.statements[0]
    assert "SET deleted_at = NOW()" in sql
    for needle in ("client_id = CAST(:client_id", "customer_id = CAST(:customer_id",
                   "user_id = CAST(:user_id", "deleted_at IS NULL"):
        assert needle in sql
    assert params["user_id"] == str(scope.user_id)
    assert "DELETE FROM" not in sql                       # the row stays: it is a soft delete


async def test_a_scope_without_a_user_is_refused_before_any_write():
    s = FakeSession()
    with pytest.raises(ValueError):
        await soft_delete_memories(s, scope=_scope(user=False), memory_ids=["x"], reason="r")
    assert s.statements == []


async def test_no_ids_does_nothing():
    s = FakeSession()
    res = await soft_delete_memories(s, scope=_scope(), memory_ids=[], reason="r")
    assert res.deleted_ids == [] and s.statements == []


async def test_the_mark_is_committed_before_vectors_graph_and_caches_are_touched(monkeypatch):
    """Core runs the graph lifecycle: the graph's outbox row is written in the
    mark's transaction (before the commit), then the graph step goes through
    the lifecycle service (erase with redact) and its row is marked done."""
    from synap.cloud.application.memory_forget import soft_delete

    monkeypatch.setenv("SYNAP_GRAPH_LIFECYCLE", "true")
    events = []
    s = FakeSession(events=events)
    rec = Recorder(events)
    outbox = []

    async def enqueue_graph(session, **kw):
        assert session is s                               # the mark's own session
        events.append("outbox")
        outbox.append(kw)
        return ["row-1"]

    async def complete_graph_rows(row_ids):
        events.append("outbox done")

    class Lifecycle:
        async def release(self, scope, memory_ids, *, mode, reason, superseded_by=None,
                          redact=False):
            events.append("graph")
            rec.calls.append(list(memory_ids))
            rec.mode, rec.redact = mode, redact

    monkeypatch.setattr(soft_delete, "enqueue_graph", enqueue_graph)
    monkeypatch.setattr(soft_delete, "complete_graph_rows", complete_graph_rows)
    monkeypatch.setattr(soft_delete, "lifecycle_for", lambda graph, vector=None: Lifecycle())

    async def invalidate():
        events.append("cache")

    ids = [str(uuid.uuid4())]
    res = await soft_delete_memories(s, scope=_scope(), memory_ids=ids, reason="r",
                                     vector=rec, graph=rec, invalidate=invalidate)
    assert events == ["update", "outbox", "commit", "vectors", "graph", "outbox done", "cache"]
    assert res.deleted_ids == ids and res.still_present_in == []
    assert rec.calls[0] == ids                             # vectors
    assert rec.calls[1] == ids and rec.mode == "erase"     # graph
    assert rec.redact is True                              # the edge text is blanked too
    assert (outbox[0]["memory_ids"], outbox[0]["op"], outbox[0]["redact"]) == (ids, "erase", True)
    assert res.contents == {ids[0]: f"text of {ids[0]}"}


async def test_ids_the_update_refused_are_reported_skipped_and_not_cleaned_elsewhere():
    mine, theirs = str(uuid.uuid4()), str(uuid.uuid4())
    events = []
    s = FakeSession(marked=[mine], events=events)
    rec = Recorder(events)
    res = await soft_delete_memories(s, scope=_scope(), memory_ids=[mine, theirs],
                                     reason="r", vector=rec)
    assert res.deleted_ids == [mine] and res.skipped_ids == [theirs]
    assert rec.calls == [[mine]]                           # vectors only for what was marked


async def test_when_nothing_was_marked_no_other_store_is_touched():
    events = []
    s = FakeSession(marked=[], events=events)
    rec = Recorder(events)
    res = await soft_delete_memories(s, scope=_scope(), memory_ids=[str(uuid.uuid4())],
                                     reason="r", vector=rec, graph=rec)
    assert res.deleted_ids == [] and "commit" not in events and "vectors" not in events


async def test_each_store_failure_is_named_and_does_not_stop_the_others():
    events = []
    s = FakeSession(events=events)
    bad = Recorder(events, fail=True)

    async def bad_invalidate():
        raise RuntimeError("redis down")

    res = await soft_delete_memories(s, scope=_scope(), memory_ids=[str(uuid.uuid4())],
                                     reason="r", vector=bad, graph=bad,
                                     invalidate=bad_invalidate)
    assert res.still_present_in == ["search index", "graph", "cache"]
    assert len(res.deleted_ids) == 1                      # the row is still marked


async def test_the_audit_row_is_written_without_any_memory_text():
    s = FakeSession()
    seen = []

    async def audit(**kw):
        seen.append(kw)

    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    await soft_delete_memories(s, scope=_scope(), memory_ids=ids, reason="because",
                               correlation_id="c1", audit=audit)
    assert [k["memory_id"] for k in seen] == ids
    assert all(set(k) == {"correlation_id", "scope", "memory_id", "reason"} for k in seen)


async def test_a_failing_audit_does_not_fail_the_delete():
    s = FakeSession()

    async def audit(**kw):
        raise RuntimeError("audit table down")

    res = await soft_delete_memories(s, scope=_scope(), memory_ids=[str(uuid.uuid4())],
                                     reason="r", correlation_id="c1", audit=audit)
    assert len(res.deleted_ids) == 1


async def test_duplicate_ids_are_marked_once():
    s = FakeSession()
    i = str(uuid.uuid4())
    await soft_delete_memories(s, scope=_scope(), memory_ids=[i, i], reason="r")
    assert s.statements[0][1]["ids"] == [i]
