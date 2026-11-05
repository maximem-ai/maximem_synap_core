"""graph_reconcile: dry-run reports and changes nothing, --commit releases every
dead id through the lifecycle service (erase for deleted/absent rows, retire
with the successor for superseded ones, copied-text entries included),
unbacked records are left alone, and a second run releases nothing (W6)."""
from __future__ import annotations

from types import SimpleNamespace

from synap.cloud.agents.ingestion import graph_reconcile
from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import SupporterRow


class FakeLifecycle:
    """Records the calls; applies releases to an in-memory graph so a second
    run sees the result. State decisions are the service's, not tested here."""

    def __init__(self, records):
        self.records = {r["id"]: dict(r) for r in records}
        self.release_calls, self.reevaluated = [], []

    async def list_namespaces(self):
        return ["ns1"]

    async def records_in_namespace(self, ns):
        return [dict(r) for r in self.records.values()]

    async def release_in_namespace(self, ns, ids, *, mode, reason, superseded_by=None,
                                   redact=False, scope=None):
        self.release_calls.append((sorted(ids), mode, reason, superseded_by))
        res = SimpleNamespace(edges_retired=0, edges_deleted=0, nodes_retired=0,
                              nodes_deleted=0, legacy_entries_marked=0,
                              records_closed=0, records_reopened=0)
        for rid, r in list(self.records.items()):
            for key in ("memory_ids", "legacy_ids"):
                gone = [i for i in r.get(key, []) if i in ids]
                if not gone:
                    continue
                r[key] = [i for i in r[key] if i not in ids]
                r["released_memory_ids"] = list(r.get("released_memory_ids", [])) + gone
                if key == "legacy_ids":
                    res.legacy_entries_marked += len(gone)
            if not r.get("memory_ids") and not r.get("legacy_ids") and r.get("released_memory_ids"):
                if mode == "erase":
                    del self.records[rid]
                    res.edges_deleted += 1
                elif not r.get("retired_at"):
                    r["retired_at"] = "now"
                    res.edges_retired += 1
                    res.records_closed += 1
        return res

    async def reevaluate(self, ns, ids):
        self.reevaluated.append(sorted(ids))
        return 0


ROWS = {
    "m_old": SupporterRow(id="m_old", superseded_by="m_new"),
    "m_live": SupporterRow(id="m_live"),
    "m_legacy_old": SupporterRow(id="m_legacy_old", superseded_by="m_new2"),
    # m_gone has no row: hard-deleted
}


async def _load(ids, session=None):
    return {i: ROWS[i] for i in ids if i in ROWS}


def _graph():
    return FakeLifecycle([
        {"kind": "edge", "id": "e_works_at", "memory_ids": ["m_old", "m_live"]},
        {"kind": "edge", "id": "e_deleted", "memory_ids": ["m_gone"]},
        {"kind": "edge", "id": "e_about", "memory_ids": []},                    # unbacked
        {"kind": "edge", "id": "e_copies", "memory_ids": [], "legacy_ids": ["m_legacy_old"]},
        {"kind": "node", "id": "n_live", "memory_ids": ["m_live"]},
    ])


async def test_dry_run_reports_and_changes_nothing():
    lc = _graph()
    report = await graph_reconcile.reconcile(None, commit=False, load_rows=_load, lifecycle=lc)
    n = report.namespaces[0]
    assert (n.ids_checked, n.dead_ids, n.ids_to_erase, n.ids_to_retire) == (4, 3, 1, 2)
    assert n.legacy_ids_to_release == 1
    assert n.records_with_dead_support == 3
    assert n.unbacked_records == 1
    assert lc.release_calls == [] and lc.reevaluated == []


async def test_commit_releases_through_the_service_and_second_run_is_a_no_op():
    lc = _graph()
    report = await graph_reconcile.reconcile(None, commit=True, load_rows=_load, lifecycle=lc)
    n = report.namespaces[0]
    assert (["m_gone"], "erase", "deleted", None) in lc.release_calls
    assert (["m_old"], "retire", "superseded", "m_new") in lc.release_calls
    # Copied-text entries go through the same service call.
    assert (["m_legacy_old"], "retire", "superseded", "m_new2") in lc.release_calls
    assert "e_deleted" not in lc.records
    assert lc.records["e_about"].get("retired_at") is None
    assert n.edges_deleted == 1 and n.legacy_entries_marked == 1

    lc.release_calls.clear()
    again = (await graph_reconcile.reconcile(None, commit=True, load_rows=_load,
                                             lifecycle=lc)).namespaces[0]
    assert again.dead_ids == 0
    assert lc.release_calls == []
    # Already-retired records get the rule again (reopen when a live memory backs them).
    assert lc.reevaluated[-1] == ["e_copies"]


async def test_unbacked_only_namespace_changes_nothing():
    lc = FakeLifecycle([{"kind": "edge", "id": "e", "memory_ids": []}])
    n = (await graph_reconcile.reconcile(None, commit=True, load_rows=_load,
                                         lifecycle=lc)).namespaces[0]
    assert n.unbacked_records == 1 and lc.release_calls == [] and lc.reevaluated == []
