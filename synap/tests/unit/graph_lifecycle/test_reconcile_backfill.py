"""graph_reconcile: dry-run reports, --commit applies, unbacked records are
left alone, and a second run changes nothing (W6)."""
from __future__ import annotations

from datetime import datetime, timezone

from synap.cloud.agents.ingestion import graph_reconcile
from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import SupporterRow

D1 = datetime(2026, 10, 1, tzinfo=timezone.utc)
D2 = datetime(2026, 10, 26, tzinfo=timezone.utc)


class FakeProvider:
    """Minimal in-memory graph with the adapter's lifecycle primitives."""

    def __init__(self, records):
        self.records = {r["id"]: dict(r) for r in records}
        self.release_calls, self.state_calls = [], []

    def list_namespaces(self):
        return ["ns1"]

    def lifecycle_records(self, namespace):
        return [dict(r) for r in self.records.values()]

    def release_memory_ids(self, namespace, ids, mode="retire", now=None, reason="superseded",
                           superseded_by=None, cap=500):
        self.release_calls.append((list(ids), mode, reason, superseded_by))
        out = {"edges_retired": 0, "edges_deleted": 0, "nodes_retired": 0, "nodes_deleted": 0}
        for rid, r in list(self.records.items()):
            gone = [i for i in r.get("memory_ids", []) if i in ids]
            if not gone:
                continue
            r["memory_ids"] = [i for i in r["memory_ids"] if i not in ids]
            r["released_memory_ids"] = list(r.get("released_memory_ids", [])) + gone
            if not r["memory_ids"]:
                k = "edges" if r["kind"] == "edge" else "nodes"
                if mode == "erase":
                    del self.records[rid]
                    out[f"{k}_deleted"] += 1
                elif r.get("retired_at") is None:
                    r["retired_at"] = now
                    out[f"{k}_retired"] += 1
        return out

    def set_record_states(self, namespace, states):
        self.state_calls.append(states)
        n = 0
        for s in states:
            r = self.records.get(s["id"])
            if r is None:
                continue
            r["retired_at"] = None if s["open"] else (r.get("retired_at") or s["now"])
            n += 1
        return n


def _rows():
    return {
        # superseded by m_new (which supports nothing here)
        "m_old": SupporterRow(id="m_old", superseded_by="m_new", superseded_at=D2,
                              document_created_at=D1, superseder_document_created_at=D2),
        # live, said before the supersession
        "m_live": SupporterRow(id="m_live", document_created_at=D1),
        # m_gone has no row: hard-deleted
    }


async def _load(ids, session=None):
    rows = _rows()
    return {i: rows[i] for i in ids if i in rows}


def _graph():
    return FakeProvider([
        {"kind": "edge", "id": "e_works_at", "memory_ids": ["m_old", "m_live"]},  # T2 closes
        {"kind": "edge", "id": "e_deleted", "memory_ids": ["m_gone"]},            # erase
        {"kind": "edge", "id": "e_about", "memory_ids": []},                      # unbacked
        {"kind": "node", "id": "n_live", "memory_ids": ["m_live"]},              # open
    ])


async def test_dry_run_reports_and_changes_nothing():
    g = _graph()
    report = await graph_reconcile.reconcile(g, commit=False, load_rows=_load)
    n = report.namespaces[0]
    assert (n.ids_checked, n.dead_ids, n.ids_to_erase, n.ids_to_retire) == (3, 2, 1, 1)
    assert n.records_to_close == 2  # e_works_at by T2, e_deleted by T1
    assert n.unbacked_records == 1
    assert g.release_calls == [] and g.state_calls == []


async def test_commit_applies_and_second_run_is_a_no_op():
    g = _graph()
    report = await graph_reconcile.reconcile(g, commit=True, load_rows=_load)
    n = report.namespaces[0]
    assert (["m_gone"], "erase", "deleted", None) in g.release_calls
    assert (["m_old"], "retire", "superseded", "m_new") in g.release_calls
    assert "e_deleted" not in g.records
    assert g.records["e_works_at"]["retired_at"] is not None   # closed by T2
    assert g.records["e_works_at"]["memory_ids"] == ["m_live"]  # live supporter kept
    assert g.records["e_about"].get("retired_at") is None       # unbacked untouched
    assert g.records["n_live"].get("retired_at") is None
    assert n.edges_deleted == 1

    again = (await graph_reconcile.reconcile(g, commit=True, load_rows=_load)).namespaces[0]
    assert (again.dead_ids, again.records_to_close, again.records_to_reopen) == (0, 0, 0)
