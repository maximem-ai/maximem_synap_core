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
  3. evaluate every backed record with `graph_record_state` (T1 and T2) BEFORE
     releasing anything, because T2 needs the superseded supporter still on
     the record to see who replaced it;
  4. release dead ids: `erase` when the row is gone or forgotten (`deleted_at`:
     a deletion whose graph step never ran, so the text must go), `retire`
     when superseded or deprecated (history is kept);
  5. write the computed states (close, or reopen a record retired by mistake).
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
from datetime import datetime, timezone
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
    legacy_dead_ids: int = 0
    records_to_close: int = 0
    records_to_reopen: int = 0
    unbacked_records: int = 0
    edges_retired: int = 0
    edges_deleted: int = 0
    nodes_retired: int = 0
    nodes_deleted: int = 0
    states_written: int = 0
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


async def _alive_rows(ids: List[str], load_rows) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for i in range(0, len(ids), BATCH):
        rows.update(await load_rows(ids[i:i + BATCH]))
    return rows


def _raw_provider(graph):
    """The synchronous graph adapter (the async wrapper keeps it in _provider)."""
    return getattr(graph, "_provider", graph)


async def reconcile_namespace(provider, namespace: str, *, commit: bool,
                              load_rows=None, now: Optional[str] = None) -> NamespaceReport:
    from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import (
        GraphRecord, graph_record_state, is_alive)

    if load_rows is None:
        from synap.cloud.shared.memory_liveness import load_supporter_rows as load_rows
    now = now or datetime.now(timezone.utc).isoformat()
    rep = NamespaceReport(namespace=namespace, committed=commit)

    records = await asyncio.to_thread(provider.lifecycle_records, namespace)
    rep.records = len(records)
    ids: List[str] = []
    legacy: set = set()
    for r in records:
        ids.extend(str(i) for i in (r.get("memory_ids") or []))
        for i in (r.get("legacy_ids") or []):
            legacy.add(str(i))
            ids.append(str(i))
    ids = list(dict.fromkeys(ids))
    rep.ids_checked = len(ids)
    # Released ids are history, but the rule needs their rows: a record closed
    # by T2 keeps live older supporters, and only the released superseded id
    # says who replaced it. Without it the second run would reopen the record.
    history = list(dict.fromkeys(
        str(i) for r in records for i in (r.get("released_memory_ids") or [])
        if str(i) not in set(ids)))
    rows = await _alive_rows(ids + history, load_rows) if (ids or history) else {}

    dead = [i for i in ids if not is_alive(rows.get(i))]
    rep.dead_ids = len(dead)
    erase_ids = [i for i in dead if i not in legacy
                 and (rows.get(i) is None or getattr(rows[i], "deleted_at", None) is not None)]
    retire_groups: Dict[Optional[str], List[str]] = {}
    for i in dead:
        if i in legacy or i in erase_ids:
            continue
        retire_groups.setdefault(getattr(rows[i], "superseded_by", None), []).append(i)
    rep.ids_to_erase = len(erase_ids)
    rep.ids_to_retire = sum(len(v) for v in retire_groups.values())
    rep.legacy_dead_ids = len([i for i in dead if i in legacy])

    # 3. States, from the records as they are now (before any release).
    states: List[Dict[str, Any]] = []
    for r in records:
        mem = [str(i) for i in (r.get("memory_ids") or [])] or \
              [str(i) for i in (r.get("legacy_ids") or [])]
        released = [str(i) for i in (r.get("released_memory_ids") or [])]
        if not mem and not released:
            rep.unbacked_records += 1
            continue
        if r.get("is_fixed_entity"):
            continue
        state = graph_record_state(
            # TEMPORARY until G1's rule reads released ids itself (lead
            # decision, CONTRACT_GRAPH.md): released ids count only when their
            # memory is not alive, the same as GraphLifecycle._apply. An edited
            # memory (alive, released as "updated") is no longer support.
            GraphRecord(memory_ids=list(dict.fromkeys(
                mem + [i for i in released if i not in mem and not is_alive(rows.get(i))])),
                        released_memory_ids=released,
                        is_fixed_entity=bool(r.get("is_fixed_entity"))),
            rows)
        retired = r.get("retired_at") is not None
        if state.is_open and retired and mem:
            rep.records_to_reopen += 1
            states.append({"kind": r.get("kind"), "id": r.get("id"), "open": True, "now": now,
                           "reason": None, "valid_to": None, "retired_by_memory_id": None})
        elif not state.is_open and not retired:
            rep.records_to_close += 1
            valid_to = getattr(state, "valid_to", None)
            states.append({"kind": r.get("kind"), "id": r.get("id"), "open": False, "now": now,
                           "reason": state.reason,
                           "valid_to": valid_to.isoformat() if valid_to else None,
                           "retired_by_memory_id": getattr(state, "by_memory", None)})

    if not commit:
        return rep

    def _count(res: Dict[str, Any]) -> None:
        rep.edges_retired += int(res.get("edges_retired", 0) or 0)
        rep.edges_deleted += int(res.get("edges_deleted", 0) or 0)
        rep.nodes_retired += int(res.get("nodes_retired", 0) or 0)
        rep.nodes_deleted += int(res.get("nodes_deleted", 0) or 0)

    if erase_ids:
        _count(await asyncio.to_thread(provider.release_memory_ids, namespace, erase_ids,
                                       "erase", now, "deleted", None))
    for successor, group in retire_groups.items():
        reason = "superseded" if successor else "unsupported"
        _count(await asyncio.to_thread(provider.release_memory_ids, namespace, group,
                                       "retire", now, reason, successor))
    if states:
        rep.states_written = int(await asyncio.to_thread(
            provider.set_record_states, namespace, states) or 0)
    return rep


async def reconcile(graph, *, commit: bool = False,
                    namespaces: Optional[Iterable[str]] = None,
                    load_rows=None) -> ReconcileReport:
    provider = _raw_provider(graph)
    if namespaces is None:
        namespaces = await asyncio.to_thread(provider.list_namespaces)
    report = ReconcileReport(committed=commit)
    for ns in namespaces:
        try:
            report.namespaces.append(
                await reconcile_namespace(provider, ns, commit=commit, load_rows=load_rows))
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
              f"erase={n.ids_to_erase} retire={n.ids_to_retire} legacy_dead={n.legacy_dead_ids} "
              f"close={n.records_to_close} reopen={n.records_to_reopen} "
              f"unbacked={n.unbacked_records}"
              + (f" | edges retired={n.edges_retired} deleted={n.edges_deleted} "
                 f"nodes retired={n.nodes_retired} deleted={n.nodes_deleted} "
                 f"states={n.states_written}" if n.committed else ""))
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
