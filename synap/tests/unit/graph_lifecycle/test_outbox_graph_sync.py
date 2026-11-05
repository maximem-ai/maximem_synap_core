"""Outbox for the graph side of retire/delete (W4).

Before: no outbox, no retry; a failed or skipped graph step was lost for good.
"""
from __future__ import annotations

import asyncio
import pathlib
from uuid import UUID

import pytest

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.storage import graph_sync_outbox as ob
from synap.cloud.agents.ingestion.storage.graph_lifecycle import GraphLifecycle
from synap.cloud.database import schema_compat

from _glc_fakes import FakeGraph, supporters_loader

CLIENT = UUID("00000000-0000-0000-0000-00000000000c")
INST = UUID("00000000-0000-0000-0000-00000000000a")
CUST = UUID("00000000-0000-0000-0000-0000000000c1")
USER = UUID("00000000-0000-0000-0000-0000000000a1")
NS = str(USER)


@pytest.fixture(autouse=True)
def _per_scope(monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_PER_SCOPE", "true")


def _scope():
    return Scope(client_id=CLIENT, instance_id=INST, customer_id=CUST, user_id=USER, scope_level="user")


class _Session:
    def __init__(self):
        self.sql = []

    async def execute(self, stmt, params=None):
        self.sql.append((str(stmt), params))


class FakeStore:
    """In-memory twin of PostgresOutboxStore."""

    def __init__(self, rows):
        self.rows = {r["id"]: {**r, "attempts": 0, "done": False, "last_error": None} for r in rows}

    async def available(self):
        return True

    async def claim(self, *, limit=50, ingestion_id=None, row_ids=None, lease_seconds=120):
        out = []
        for r in self.rows.values():
            if r["done"] or (row_ids is not None and r["id"] not in row_ids) \
                    or (ingestion_id is not None and r.get("ingestion_id") != ingestion_id):
                continue
            r["attempts"] += 1
            out.append(dict(r))
        return out[:limit]

    async def complete(self, ids):
        for i in ids:
            self.rows[i]["done"] = True

    async def fail(self, ids, error, attempts):
        for i in ids:
            self.rows[i]["last_error"] = error
            self.rows[i]["wait"] = ob.backoff_seconds(attempts)


def _row(rid, mid, op="erase", reason="deleted", **kw):
    return {"id": rid, "memory_id": mid, "client_id": str(CLIENT), "instance_id": str(INST),
            "customer_id": str(CUST), "user_id": str(USER), "scope_level": "user",
            "namespaces": [NS], "op": op, "reason": reason, "superseded_by": None, "redact": False, **kw}


def _graph():
    g = FakeGraph()
    g.node(NS, "u", "the user", is_writer=True)
    g.node(NS, "acme", "Acme Corp", "organization", memory_ids=["m1"])
    g.edge(NS, "e1", "u", "acme", "WORKS_AT", memory_ids=["m1"])
    return g


def test_migration_file_is_where_the_runner_reads():
    from synap.db import runner
    files = [p.name for p in runner.migration_files(runner.MIGRATIONS)]
    assert "001_graph_sync_pending.sql" in files
    sql = (runner.MIGRATIONS / "001_graph_sync_pending.sql").read_text()
    for col in ("memory_id", "op", "reason", "superseded_by", "attempts", "last_error",
                "next_attempt_at", "done_at", "namespaces", "ingestion_id"):
        assert col in sql


async def test_enqueue_writes_in_the_callers_session_without_committing(monkeypatch):
    async def has_table(t, session=None):
        return True
    monkeypatch.setattr(schema_compat, "has_table", has_table)
    s = _Session()
    ids = await ob.enqueue(s, memory_ids=["11111111-1111-1111-1111-111111111111"], scope=_scope(),
                           op="retire", reason="superseded", superseded_by="22222222-2222-2222-2222-222222222222",
                           ingestion_id="33333333-3333-3333-3333-333333333333")
    assert len(ids) == 1 and len(s.sql) == 1
    stmt, params = s.sql[0]
    assert "INSERT INTO graph_sync_pending" in stmt
    assert params["op"] == "retire" and params["namespaces"][0] == NS
    assert not hasattr(s, "committed")


async def test_missing_table_skips_enqueue_with_one_warning(monkeypatch, caplog):
    async def has_table(t, session=None):
        return False
    monkeypatch.setattr(schema_compat, "has_table", has_table)
    s = _Session()
    with caplog.at_level("WARNING"):
        assert await ob.enqueue(s, memory_ids=["m"], scope=_scope(), op="erase", reason="deleted") == []
        assert await ob.enqueue(s, memory_ids=["m"], scope=_scope(), op="erase", reason="deleted") == []
    assert s.sql == []
    assert sum("graph_sync_pending table is missing" in r.message for r in caplog.records) == 1


async def test_bad_op_is_refused():
    with pytest.raises(ValueError):
        await ob.enqueue(_Session(), memory_ids=["m"], scope=_scope(), op="drop", reason="x")


async def test_drain_applies_rows_and_marks_them_done():
    g = _graph()
    store = FakeStore([_row("r1", "m1")])
    lc = GraphLifecycle(g, load_supporters=supporters_loader({}))
    res = await ob.apply_rows(await store.claim(), store, lc)
    assert res.done == 1 and store.rows["r1"]["done"]
    assert g.e(NS, "e1") is None


async def test_failed_release_stays_pending_with_backoff_then_succeeds():
    g = _graph()
    g.fail_release = True
    store = FakeStore([_row("r1", "m1")])
    lc = GraphLifecycle(g, load_supporters=supporters_loader({}))
    res = await ob.apply_rows(await store.claim(), store, lc)
    assert res.failed == 1 and not store.rows["r1"]["done"]
    assert "graph down" in store.rows["r1"]["last_error"] and store.rows["r1"]["wait"] == 5
    res = await ob.apply_rows(await store.claim(), store, lc)  # attempt 2, still down
    assert store.rows["r1"]["attempts"] == 2 and store.rows["r1"]["wait"] == 10
    g.fail_release = False
    res = await ob.apply_rows(await store.claim(), store, lc)
    assert res.done == 1 and store.rows["r1"]["done"] and g.e(NS, "e1") is None


def test_backoff_is_capped():
    assert [ob.backoff_seconds(a) for a in (1, 2, 3)] == [5, 10, 20]
    assert ob.backoff_seconds(50) == ob.MAX_BACKOFF_SECONDS


async def test_drain_for_ingestion_only_takes_that_ingestion(monkeypatch):
    import synap.cloud.agents.ingestion.storage.graph_lifecycle as glc
    monkeypatch.setattr(glc, "_default_loader", lambda: supporters_loader({}))
    g = _graph()
    store = FakeStore([_row("r1", "m1", ingestion_id="ing-1"), _row("r2", "m9", ingestion_id="ing-2")])
    assert await ob.drain_for_ingestion("ing-1", graph=g, store=store) == 1
    assert store.rows["r1"]["done"] and not store.rows["r2"]["done"]


async def test_drain_for_ingestion_never_raises(monkeypatch):
    class Broken(FakeStore):
        async def claim(self, **kw):
            raise RuntimeError("db down")
    assert await ob.drain_for_ingestion("ing", graph=_graph(), store=Broken([])) == 0


async def test_worker_loop_drains_and_stops(monkeypatch):
    import synap.cloud.agents.ingestion.storage.graph_lifecycle as glc
    monkeypatch.setattr(glc, "_default_loader", lambda: supporters_loader({}))
    g = _graph()
    store = FakeStore([_row("r1", "m1")])
    stop = asyncio.Event()
    task = asyncio.create_task(ob.run_drain_worker(graph=g, store=store, interval=0.01, stop=stop))
    for _ in range(100):
        if store.rows["r1"]["done"]:
            break
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, 1)
    assert store.rows["r1"]["done"]


async def test_retire_with_audit_enqueues_in_the_same_transaction(monkeypatch):
    """enqueue_graph=True puts the outbox row in the retire's own session,
    before its single commit."""
    from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import SQLAlchemyRDBMSWrapper

    events = []

    class _Res:
        rowcount = 1

    class Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, stmt, params=None):
            events.append(("execute", str(stmt).split()[0:3]))
            return _Res()

        async def commit(self):
            events.append(("commit",))

        async def rollback(self):
            events.append(("rollback",))

    import synap.cloud.database.connection as conn
    monkeypatch.setattr(conn, "get_session_factory", lambda: Sess)

    async def has_table(t, session=None):
        return True
    monkeypatch.setattr(schema_compat, "has_table", has_table)

    ok = await SQLAlchemyRDBMSWrapper().retire_memory_with_audit(
        memory_id="11111111-1111-1111-1111-111111111111", action="SUPERSEDE", scope=_scope(),
        previous_content="x", superseded_by="22222222-2222-2222-2222-222222222222",
        ingestion_id="33333333-3333-3333-3333-333333333333", enqueue_graph=True)
    assert ok is True
    kinds = [e for e in events if e[0] == "execute"]
    assert any("graph_sync_pending" in " ".join(e[1]) or e[1][:2] == ["INSERT", "INTO"] for e in kinds)
    inserts = [e for e in events if e[0] == "execute" and e[1][:2] == ["INSERT", "INTO"]]
    assert len(inserts) == 2, "outbox row + memory_actions row"
    assert events[-1] == ("commit",) and events.count(("commit",)) == 1


async def test_retire_without_enqueue_writes_no_outbox_row(monkeypatch):
    from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import SQLAlchemyRDBMSWrapper
    seen = []

    class _Res:
        rowcount = 1

    class Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, stmt, params=None):
            seen.append(str(stmt))
            return _Res()

        async def commit(self):
            pass

    import synap.cloud.database.connection as conn
    monkeypatch.setattr(conn, "get_session_factory", lambda: Sess)
    await SQLAlchemyRDBMSWrapper().retire_memory_with_audit(
        memory_id="11111111-1111-1111-1111-111111111111", action="SUPERSEDE", scope=_scope(),
        previous_content="x", superseded_by="22222222-2222-2222-2222-222222222222")
    assert not any("graph_sync_pending" in s for s in seen)


def test_startup_skips_worker_without_graph(monkeypatch):
    from synap.cloud.application.ingestion import startup
    monkeypatch.setenv("SYNAP_GRAPH_LIFECYCLE", "true")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    assert startup._start_graph_sync_worker() is False
