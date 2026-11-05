"""An outbox row written by an ingestion is Stage 4's to drain after the graph
write; the background worker must not take it first (it would release before
the new edges exist and hold the lease while Stage 4 looks for the row)."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

from synap.cloud.agents.ingestion.storage import graph_sync_outbox as outbox


class Session:
    def __init__(self):
        self.calls = []

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))


def _scope():
    return SimpleNamespace(client_id=uuid.uuid4(), instance_id=uuid.uuid4(),
                           customer_id=uuid.uuid4(), user_id=uuid.uuid4(),
                           scope_level="user", chain=None)


async def _enqueue(monkeypatch, **kw):
    async def yes(session=None):
        return True

    async def no_column(*a, **k):
        return False

    monkeypatch.setattr(outbox, "table_available", yes)
    from synap.cloud.database import schema_compat
    monkeypatch.setattr(schema_compat, "has_column", no_column)
    s = Session()
    await outbox.enqueue(s, memory_ids=[str(uuid.uuid4())], scope=_scope(), op="retire",
                         reason="superseded", **kw)
    sql, params = s.calls[-1]
    return sql, params


async def test_ingestion_rows_wait_for_stage4(monkeypatch):
    monkeypatch.delenv("SYNAP_GRAPH_SYNC_INGESTION_GRACE_SECONDS", raising=False)
    sql, params = await _enqueue(monkeypatch, ingestion_id=str(uuid.uuid4()))
    assert "next_attempt_at" in sql and "now() + make_interval(secs => CAST(:grace AS double precision))" in sql
    assert params["grace"] == 300.0 and isinstance(params["grace"], float)


async def test_rows_without_an_ingestion_are_due_at_once(monkeypatch):
    sql, params = await _enqueue(monkeypatch)
    assert params["grace"] == 0.0 and isinstance(params["grace"], float)


async def test_stage4_drain_ignores_the_grace(monkeypatch):
    """Stage 4 claims its rows by ingestion id, whatever next_attempt_at says."""
    import inspect
    src = inspect.getsource(outbox.PostgresOutboxStore.claim)
    by_ingestion = src[src.index("elif ingestion_id is not None"):src.index("else:")]
    assert "next_attempt_at" not in by_ingestion


# --- a row the caller applies itself is leased to it (live finding 2026-11-05) ---
# The delete route wrote its row, committed, then applied it inline; the
# worker claimed the same unleased row at the same moment, and the two graph
# transactions raced (the loser read a just-deleted relationship's endpoint as
# null and crashed).

async def test_inline_rows_are_written_leased(monkeypatch):
    sql, params = await _enqueue(monkeypatch, inline=True)
    assert "leased_until" in sql
    assert "CASE WHEN CAST(:inline AS boolean) THEN now() + make_interval(secs => CAST(:lease AS double precision))" in sql
    assert params["inline"] is True and params["lease"] == float(outbox.LEASE_SECONDS)


async def test_worker_rows_are_not_leased(monkeypatch):
    sql, params = await _enqueue(monkeypatch)
    assert params["inline"] is False


async def test_the_route_and_forget_enqueue_inline(monkeypatch):
    """Both callers that apply their rows after the commit ask for the lease."""
    import inspect
    from synap.cloud.application.sdk_api import sdk_memories
    from synap.cloud.application.memory_forget import soft_delete
    assert "inline=True" in inspect.getsource(sdk_memories._enqueue_graph)
    src = inspect.getsource(soft_delete)
    assert 'reason="deleted", redact=True,\n                                         inline=True)' in src


def test_release_ignores_a_null_endpoint():
    """An endpoint deleted by a concurrent transaction comes back as null: the
    orphan sweep skips it instead of failing the whole release."""
    from synap.providers.graph.adapters import neo4j as m

    class Rec:
        def __init__(self, d):
            self._d = d

        def data(self):
            return self._d

    seen = {}

    class Tx:
        def run(self, q, p):
            if q is m._RELEASE_EDGES:
                return [Rec({"id": "e1", "type": "T", "memory_ids": [], "released_memory_ids": ["x"],
                             "unsupported": True, "newly": True, "deleted": True,
                             "from_id": None, "from_name": None, "from_type": None, "from_vector_name": None,
                             "to_id": "n1", "to_name": "Lisbon", "to_type": "location",
                             "to_vector_name": None})]
            if q is m._RELEASE_NODES:
                return []
            seen["candidates"] = p["candidates"]
            return []

    adapter = m.Neo4jGraphStore.__new__(m.Neo4jGraphStore) if hasattr(m, "Neo4jGraphStore") else None
    cls = next(c for c in vars(m).values() if isinstance(c, type) and hasattr(c, "release_memory_ids")
               and hasattr(c, "_run_in_write_tx"))
    adapter = cls.__new__(cls)
    adapter._run_in_write_tx = lambda work: work(Tx())
    adapter.release_memory_ids("ns", ["x"], mode="erase")
    assert seen["candidates"] == ["n1"]
