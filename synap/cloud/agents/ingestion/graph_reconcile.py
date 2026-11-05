"""Bring the graph in line with Postgres: release memories that are no longer
alive and close (or reopen) graph records by the lifecycle rule.

This is the backfill for graphs written before the graph lifecycle existed,
and the safety net for any release the outbox missed. Dry-run by default:

    python -m synap.cloud.agents.ingestion.graph_reconcile            # report only
    python -m synap.cloud.agents.ingestion.graph_reconcile --commit   # apply
    python -m synap.cloud.agents.ingestion.graph_reconcile --namespace NS [--json]

Per namespace (one per scope that holds graph records):
  1. collect every memory id on edges and nodes (`memory_ids`, and the ids of
     copied-text `memory_contents` entries);
  2. ask Postgres which are alive (batched `id = ANY(...)`);
  3. release dead ids through the graph lifecycle service: `erase` when the
     row is gone or forgotten (`deleted_at`: a deletion whose graph step never
     ran, so the text must go), `retire` with the successor when superseded or
     deprecated (history is kept). The service applies the shared rule
     (`graph_record_state`, T1 and T2) to every record it touches, in both the
     id format and the older copied-text format;
  4. re-apply the rule to records already retired, so one a live memory
     supports reopens.
Records that never had a supporting memory are left alone and counted
(`unbacked_records`): the source they came from cannot be recovered.
Running it twice changes nothing the second time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

BATCH = 1000


@dataclass
class NamespaceReport:
    namespace: str
    records: int = 0
    ids_checked: int = 0
    dead_ids: int = 0
    ids_to_erase: int = 0
    ids_to_retire: int = 0
    legacy_ids_to_release: int = 0
    records_with_dead_support: int = 0
    unbacked_records: int = 0
    # Filled by --commit, from the lifecycle service's own result.
    edges_retired: int = 0
    edges_deleted: int = 0
    nodes_retired: int = 0
    nodes_deleted: int = 0
    legacy_entries_marked: int = 0
    records_closed: int = 0
    records_reopened: int = 0
    states_rewritten: int = 0
    committed: bool = False
    error: Optional[str] = None


@dataclass
class ReconcileReport:
    committed: bool
    namespaces: List[NamespaceReport] = field(default_factory=list)

    def totals(self) -> Dict[str, int]:
        keys = [k for k, v in asdict(NamespaceReport("")).items() if isinstance(v, int)
                and not isinstance(v, bool)]
        return {k: sum(getattr(n, k) for n in self.namespaces) for k in keys}


async def _rows(ids: List[str], load_rows) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for i in range(0, len(ids), BATCH):
        rows.update(await load_rows(ids[i:i + BATCH]))
    return rows


_RESULT_FIELDS = ("edges_retired", "edges_deleted", "nodes_retired", "nodes_deleted",
                  "legacy_entries_marked", "records_closed", "records_reopened")


async def reconcile_namespace(lifecycle, namespace: str, *, commit: bool,
                              load_rows=None) -> NamespaceReport:
    """Report (and with *commit*, repair) one namespace.

    All graph reads and writes, and the decision whether a record is current,
    belong to the lifecycle service (`GraphLifecycle`) and the shared rule it
    applies. This function only decides which ids are dead in Postgres and
    how each must be released: `erase` when the row is gone or forgotten (a
    deletion whose graph step never ran, so the text must go), `retire` with
    its successor when superseded or deprecated.
    """
    from synap.cloud.shared.memory_liveness import is_alive

    if load_rows is None:
        from synap.cloud.shared.memory_liveness import load_supporter_rows as load_rows
    rep = NamespaceReport(namespace=namespace, committed=commit)

    records = await lifecycle.records_in_namespace(namespace)
    rep.records = len(records)
    ids: List[str] = []
    legacy: set = set()
    for r in records:
        mem = [str(i) for i in (r.get("memory_ids") or [])]
        leg = [str(i) for i in (r.get("legacy_ids") or [])]
        if not mem and not leg and not (r.get("released_memory_ids") or []):
            # Never backed by a memory: the source cannot be recovered. Left alone.
            rep.unbacked_records += 1
        ids.extend(mem)
        ids.extend(leg)
        legacy.update(leg)
    ids = list(dict.fromkeys(ids))
    rep.ids_checked = len(ids)
    rows = await _rows(ids, load_rows) if ids else {}

    dead = {i for i in ids if not is_alive(rows.get(i))}
    rep.dead_ids = len(dead)
    rep.legacy_ids_to_release = len(dead & legacy)
    rep.records_with_dead_support = sum(
        1 for r in records
        if dead & {str(i) for i in (r.get("memory_ids") or []) + (r.get("legacy_ids") or [])})

    erase_ids = sorted(i for i in dead
                       if rows.get(i) is None or getattr(rows[i], "deleted_at", None) is not None)
    retire_groups: Dict[Optional[str], List[str]] = {}
    for i in sorted(dead - set(erase_ids)):
        retire_groups.setdefault(getattr(rows[i], "superseded_by", None), []).append(i)
    rep.ids_to_erase = len(erase_ids)
    rep.ids_to_retire = sum(len(v) for v in retire_groups.values())

    if not commit:
        return rep

    def _count(result) -> None:
        for f in _RESULT_FIELDS:
            setattr(rep, f, getattr(rep, f) + int(getattr(result, f, 0) or 0))

    if erase_ids:
        _count(await lifecycle.release_in_namespace(
            namespace, erase_ids, mode="erase", reason="deleted"))
    for successor, group in retire_groups.items():
        _count(await lifecycle.release_in_namespace(
            namespace, group, mode="retire",
            reason="superseded" if successor else "unsupported", superseded_by=successor))
    # Records already retired (by an earlier release or a mistaken one) get the
    # rule again, so one that a live memory supports reopens.
    retired = [str(r.get("id")) for r in records if r.get("retired_at")]
    if retired:
        rep.states_rewritten = int(await lifecycle.reevaluate(namespace, retired) or 0)
    return rep


async def reconcile(graph, *, commit: bool = False,
                    namespaces: Optional[Iterable[str]] = None,
                    load_rows=None, lifecycle=None) -> ReconcileReport:
    if lifecycle is None:
        from synap.cloud.agents.ingestion.storage.graph_lifecycle import GraphLifecycle
        lifecycle = GraphLifecycle(graph, load_supporters=load_rows)
    if namespaces is None:
        namespaces = await lifecycle.list_namespaces()
    report = ReconcileReport(committed=commit)
    for ns in namespaces:
        try:
            report.namespaces.append(
                await reconcile_namespace(lifecycle, ns, commit=commit, load_rows=load_rows))
        except Exception as e:  # noqa: BLE001 - one namespace must not stop the rest
            logger.error("graph_reconcile namespace=%s failed: %s", ns, e, exc_info=True)
            report.namespaces.append(NamespaceReport(namespace=ns, committed=False,
                                                     error=f"{type(e).__name__}: {e}"))
    return report


def _print(report: ReconcileReport, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"committed": report.committed, "totals": report.totals(),
                          "namespaces": [asdict(n) for n in report.namespaces]},
                         indent=2, default=str))
        return
    mode = "COMMIT" if report.committed else "DRY RUN (nothing changed; add --commit)"
    print(f"graph_reconcile: {mode}")
    for n in report.namespaces:
        if n.error:
            print(f"  {n.namespace}: FAILED {n.error}")
            continue
        print(f"  {n.namespace}: records={n.records} ids={n.ids_checked} dead={n.dead_ids} "
              f"erase={n.ids_to_erase} retire={n.ids_to_retire} legacy={n.legacy_ids_to_release} "
              f"records_with_dead_support={n.records_with_dead_support} unbacked={n.unbacked_records}"
              + (f" | edges retired={n.edges_retired} deleted={n.edges_deleted} "
                 f"nodes retired={n.nodes_retired} deleted={n.nodes_deleted} "
                 f"legacy_marked={n.legacy_entries_marked} closed={n.records_closed} "
                 f"reopened={n.records_reopened} reevaluated={n.states_rewritten}"
                 if n.committed else ""))
    print(f"  totals: {report.totals()}")


async def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="graph_reconcile", description=__doc__.split("\n\n")[0])
    parser.add_argument("--commit", action="store_true", help="apply the changes (default: report only)")
    parser.add_argument("--namespace", action="append", help="limit to this namespace (repeatable)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    from synap.cloud.application.shared.providers import get_cached_providers
    graph = get_cached_providers().get("graph")
    if graph is None:
        print("graph_reconcile: no graph store is configured")
        return 2
    report = await reconcile(graph, commit=args.commit, namespaces=args.namespace)
    _print(report, args.json)
    return 1 if any(n.error for n in report.namespaces) else 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(_main()))
