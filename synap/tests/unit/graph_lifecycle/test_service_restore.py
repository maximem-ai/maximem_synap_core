"""Deleting a newer fact restores the older fact it replaced (owner decision).

Memory N superseded memory S; the graph line S supported was closed (T2).
Deleting N clears S.superseded_by in Postgres (foreign key ON DELETE SET NULL),
so S is alive again, and the graph must follow: S goes back into memory_ids of
the records that released it and the rule reopens the line. Before this change
the line stayed closed for good.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import SupporterRow
from synap.cloud.agents.ingestion.storage import graph_sync_outbox as ob
from synap.cloud.agents.ingestion.storage.graph_lifecycle import GraphLifecycle
from synap.cloud.database import schema_compat

from _glc_fakes import FakeGraph, supporters_loader

CLIENT = UUID("00000000-0000-0000-0000-00000000000c")
INST = UUID("00000000-0000-0000-0000-00000000000a")
CUST = UUID("00000000-0000-0000-0000-0000000000c1")
USER = UUID("00000000-0000-0000-0000-0000000000a1")
NS = str(USER)
D = lambda day: datetime(2026, 10, day, tzinfo=timezone.utc)  # noqa: E731


@pytest.fixture(autouse=True)
def _per_scope(monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_PER_SCOPE", "true")


def _scope():
    return Scope(client_id=CLIENT, instance_id=INST, customer_id=CUST, user_id=USER, scope_level="user")


def _graph():
    """WORKS_AT Acme backed by S; WORKS_AT Globex backed by N."""
    g = FakeGraph()
    g.node(NS, "u", "the user", is_writer=True)
    g.node(NS, "acme", "Acme Corp", "organization", memory_ids=["S"])
    g.node(NS, "globex", "Globex", "organization", memory_ids=["N"])
    g.edge(NS, "e_acme", "u", "acme", "WORKS_AT", memory_ids=["S"])
    g.edge(NS, "e_globex", "u", "globex", "WORKS_AT", memory_ids=["N"])
    return g


SUPERSEDED = {
    "S": SupporterRow(id="S", superseded_by="N", document_created_at=D(1), superseder_document_created_at=D(26)),
    "N": SupporterRow(id="N", document_created_at=D(26)),
}


async def _supersede(g):
    await GraphLifecycle(g, load_supporters=supporters_loader(SUPERSEDED)).release(
        _scope(), ["S"], mode="retire", reason="superseded", superseded_by="N")
    e = g.e(NS, "e_acme")
    assert e["retired_at"] and e["retired_by_memory_id"] == "N" and e["memory_ids"] == []


async def test_deleting_the_newer_memory_reopens_the_older_line():
    g = _graph()
    await _supersede(g)
    # N deleted: its row is gone and the foreign key made S alive again.
    after = {"S": SupporterRow(id="S", document_created_at=D(1))}
    out = await GraphLifecycle(g, load_supporters=supporters_loader(after)).release(
        _scope(), ["N"], mode="erase", reason="deleted", restore_ids=["S"])
    acme = g.e(NS, "e_acme")
    assert acme["memory_ids"] == ["S"] and "S" not in acme["released_memory_ids"]
    assert "retired_at" not in acme and "retired_by_memory_id" not in acme
    assert "retired_at" not in g.n(NS, "acme") and g.n(NS, "acme")["memory_ids"] == ["S"]
    # N's own line is erased as before.
    assert g.e(NS, "e_globex") is None and g.n(NS, "globex") is None
    assert out.memories_restored == 1 and out.records_reopened >= 1 and out.edges_deleted == 1


async def test_nothing_reopens_when_the_older_memory_was_itself_deleted():
    g = _graph()
    await _supersede(g)
    # S was deleted too (row absent): nothing to restore.
    out = await GraphLifecycle(g, load_supporters=supporters_loader({})).release(
        _scope(), ["N"], mode="erase", reason="deleted", restore_ids=["S"])
    acme = g.e(NS, "e_acme")
    assert acme["retired_at"] and acme["memory_ids"] == [] and acme["released_memory_ids"] == ["S"]
    assert out.memories_restored == 0


async def test_nothing_reopens_when_the_older_memory_is_still_superseded():
    g = _graph()
    await _supersede(g)
    still = {"S": SupporterRow(id="S", superseded_by="X", superseder_document_created_at=D(27))}
    await GraphLifecycle(g, load_supporters=supporters_loader(still)).release(
        _scope(), ["N"], mode="erase", reason="deleted", restore_ids=["S"])
    assert g.e(NS, "e_acme")["retired_at"]


async def test_the_reconciler_restores_a_memory_whose_superseder_is_gone():
    from synap.cloud.agents.ingestion.graph_reconcile import reconcile_namespace
    g = _graph()
    await _supersede(g)
    # N was deleted by a path that never ran the graph step: N is still on its
    # edge and node, its row is gone, S is alive again.
    after = {"S": SupporterRow(id="S", document_created_at=D(1))}
    load = supporters_loader(after)
    lc = GraphLifecycle(g, load_supporters=load)
    dry = await reconcile_namespace(lc, NS, commit=False, load_rows=load)
    assert dry.ids_to_restore == 1 and "retired_at" in g.e(NS, "e_acme")
    rep = await reconcile_namespace(lc, NS, commit=True, load_rows=load)
    assert rep.memories_restored == 1
    acme = g.e(NS, "e_acme")
    assert acme["memory_ids"] == ["S"] and "retired_at" not in acme
    assert g.e(NS, "e_globex") is None, "N's own line is erased"
    again = await reconcile_namespace(lc, NS, commit=False, load_rows=load)
    assert again.ids_to_restore == 0


# --- capturing S before the delete, and carrying it through the outbox --------------

class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Session:
    def __init__(self, superseded=(), audited=()):
        self.superseded, self.audited = list(superseded), list(audited)
        self.sql = []

    async def execute(self, stmt, params=None):
        text = str(stmt)
        self.sql.append((text, params))
        if "FROM memories" in text and "superseded_by = ANY" in text:
            return _Rows(self.superseded)
        if "FROM memory_actions" in text:
            return _Rows(self.audited)
        return _Rows([])


N_ID = "11111111-1111-1111-1111-111111111111"
S_ID = "22222222-2222-2222-2222-222222222222"
S2_ID = "33333333-3333-3333-3333-333333333333"


async def test_superseded_by_any_reads_the_column_and_the_audit_trail():
    s = _Session(superseded=[(N_ID, S_ID)], audited=[(N_ID, S_ID), (N_ID, S2_ID)])
    assert await ob.superseded_by_any(s, [N_ID]) == {N_ID: [S_ID, S2_ID]}


async def test_erase_enqueue_records_the_memories_to_restore(monkeypatch):
    async def yes(*a, **k):
        return True
    monkeypatch.setattr(schema_compat, "has_table", yes)
    monkeypatch.setattr(schema_compat, "has_column", yes)
    s = _Session(superseded=[(N_ID, S_ID)])
    await ob.enqueue(s, memory_ids=[N_ID], scope=_scope(), op="erase", reason="deleted")
    insert = [p for t, p in s.sql if "INSERT INTO graph_sync_pending" in t]
    assert insert and insert[0]["restore"] == [S_ID]
    assert "restore_memory_ids" in [t for t, _ in s.sql if "INSERT" in t][0]


async def test_erase_enqueue_without_migration_002_still_works(monkeypatch, caplog):
    async def yes(*a, **k):
        return True

    async def no(*a, **k):
        return False
    monkeypatch.setattr(schema_compat, "has_table", yes)
    monkeypatch.setattr(schema_compat, "has_column", no)
    s = _Session(superseded=[(N_ID, S_ID)])
    with caplog.at_level("WARNING"):
        ids = await ob.enqueue(s, memory_ids=[N_ID], scope=_scope(), op="erase", reason="deleted")
    assert len(ids) == 1
    assert "restore_memory_ids" not in [t for t, _ in s.sql if "INSERT" in t][0]
    assert any("migration 002" in r.message for r in caplog.records)


async def test_drained_erase_row_restores():
    g = _graph()
    await _supersede(g)

    class Store:
        done = []

        async def complete(self, ids):
            self.done += ids

        async def fail(self, *a):
            raise AssertionError("must not fail")

    row = {"id": "r1", "memory_id": "N", "client_id": str(CLIENT), "instance_id": str(INST),
           "customer_id": str(CUST), "user_id": str(USER), "scope_level": "user", "namespaces": [NS],
           "op": "erase", "reason": "deleted", "superseded_by": None, "redact": False,
           "restore_memory_ids": ["S"], "attempts": 1}
    lc = GraphLifecycle(g, load_supporters=supporters_loader({"S": SupporterRow(id="S")}))
    res = await ob.apply_rows([row], Store(), lc)
    assert res.done == 1 and "retired_at" not in g.e(NS, "e_acme")


def test_migration_002_adds_the_column():
    from synap.db import runner
    names = [p.name for p in runner.migration_files(runner.MIGRATIONS)]
    assert names[:2] == ["001_graph_sync_pending.sql", "002_graph_sync_restore.sql"]
    assert "restore_memory_ids" in (runner.MIGRATIONS / "002_graph_sync_restore.sql").read_text()
