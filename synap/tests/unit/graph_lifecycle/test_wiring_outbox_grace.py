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
    assert "next_attempt_at" in sql and "now() + CAST(:grace AS interval)" in sql
    assert params["grace"] == "300 seconds"


async def test_rows_without_an_ingestion_are_due_at_once(monkeypatch):
    sql, params = await _enqueue(monkeypatch)
    assert params["grace"] == "0 seconds"


async def test_stage4_drain_ignores_the_grace(monkeypatch):
    """Stage 4 claims its rows by ingestion id, whatever next_attempt_at says."""
    import inspect
    src = inspect.getsource(outbox.PostgresOutboxStore.claim)
    by_ingestion = src[src.index("elif ingestion_id is not None"):src.index("else:")]
    assert "next_attempt_at" not in by_ingestion
