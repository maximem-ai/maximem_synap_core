"""One service for every path that retires or deletes a memory, on the graph side.

A graph record (relationship or entity node) is supported by the memories it
names: their ids in ``memory_ids`` (references mode, the Core default) or their
entries in the JSON ``memory_contents`` (the older copies mode). When a memory
is retired or deleted, the graph must follow, or accurate mode keeps serving a
statement nothing supports any more (issue12_graph.md, sections 1-2).

``GraphLifecycle.release`` does, per graph namespace the memory's scope can
reach:

1. References: one adapter write releases the ids from every edge and node
   (``release_memory_ids``): they move to ``released_memory_ids``; a record
   left with no support is retired, or with ``mode="erase"`` deleted together
   with the nodes it leaves orphaned (never the writer's or customer's node).
2. Copies: each relationship whose ``memory_contents`` names the memory is
   rewritten with the compare-and-set helper (``rewrite_memory_contents``):
   retire marks the entry not current; erase marks it deleted, and ``redact``
   blanks its text and embedding.
3. Every touched record that still exists is evaluated with the shared rule
   (``graph_lifecycle_rule.graph_record_state``) against Postgres, and its
   state is written in one adapter write (``set_record_states``). This is what
   closes "works at Acme" when the memory saying it was superseded while an
   older live memory still names the edge (rule T2).
4. Erase also deletes the Chroma triple and entity vectors of the erased
   relationships and nodes: their text is a copy of the fact.

Graph failures raise: the caller's outbox row stays pending and is retried.
Releasing an id that is no longer present changes nothing, so a retry is safe.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

MODES = ("retire", "erase")
RELEASED_CAP = 500  # same bound as memory_ids (services/node_memories.MAX_PER_NODE)


@dataclass
class ReleaseResult:
    edges_pruned: int = 0
    edges_retired: int = 0
    edges_deleted: int = 0
    nodes_pruned: int = 0
    nodes_retired: int = 0
    nodes_deleted: int = 0
    legacy_entries_marked: int = 0
    records_closed: int = 0
    records_reopened: int = 0
    vectors_deleted: int = 0
    vector_errors: int = 0
    namespaces: List[str] = field(default_factory=list)

    def add(self, other: "ReleaseResult") -> "ReleaseResult":
        for f in fields(self):
            if f.name == "namespaces":
                self.namespaces.extend(n for n in other.namespaces if n not in self.namespaces)
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @property
    def changed(self) -> bool:
        return any(getattr(self, f.name) for f in fields(self)
                   if f.name not in ("namespaces", "vector_errors"))


@dataclass
class ErasedRecord:
    kind: str                      # "relationship" | "entity"
    memory_ids: List[str]          # ids that were on it
    edges_deleted: int = 0
    nodes_deleted: int = 0
    namespace: Optional[str] = None


SupporterLoader = Callable[[Iterable[Any]], Awaitable[Dict[str, Any]]]


def _default_loader() -> SupporterLoader:
    from synap.cloud.shared.memory_liveness import load_supporter_rows
    return load_supporter_rows


def release_namespaces(scope: Any) -> List[str]:
    """Every graph namespace a memory written under *scope* can be on: the
    write namespace, the legacy (pre per-scope) one, and every namespace the
    scope reads. Memory ids are UUIDs, so releasing one where it is absent
    changes nothing."""
    from synap.cloud.shared import scope_key
    out: List[str] = []
    candidates = [scope_key.write_scope_id(scope), scope_key._legacy_graph_id(scope)]
    try:
        candidates += scope_key.readable_scope_ids(scope)
    except Exception as exc:  # noqa: BLE001 - an unusual scope still releases where it was written
        logger.warning("graph_lifecycle readable namespaces not built: %s", exc)
    for ns in candidates:
        if ns and ns != "None" and ns not in out:
            out.append(ns)
    return out


class GraphLifecycle:
    """Retire or erase the graph side of memories. ``graph`` is the
    ``AsyncGraphWrapper`` (anything with ``_provider``) or a provider."""

    def __init__(self, graph: Any, vector: Any = None,
                 load_supporters: Optional[SupporterLoader] = None):
        self._provider = getattr(graph, "_provider", graph)
        self._vector = vector
        self._load = load_supporters

    async def _call(self, name: str, *args, **kwargs):
        return await asyncio.to_thread(getattr(self._provider, name), *args, **kwargs)

    # -- public ------------------------------------------------------------

    async def release(self, scope: Any, memory_ids: Iterable[Any], *, mode: str, reason: str,
                      superseded_by: Optional[str] = None, redact: bool = False) -> ReleaseResult:
        """Release *memory_ids* from the graph of every namespace *scope* reaches."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        ids = [str(i) for i in dict.fromkeys(memory_ids or []) if i]
        total = ReleaseResult()
        if not ids:
            return total
        for ns in release_namespaces(scope):
            total.add(await self.release_in_namespace(
                ns, ids, mode=mode, reason=reason, superseded_by=superseded_by,
                redact=redact, scope=scope))
        logger.info("graph_lifecycle_release mode=%s reason=%s ids=%d result=%s",
                    mode, reason, len(ids), total.as_dict())
        return total

    async def release_in_namespace(self, ns: str, memory_ids: Iterable[Any], *, mode: str, reason: str,
                                   superseded_by: Optional[str] = None, redact: bool = False,
                                   scope: Any = None) -> ReleaseResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        ids = [str(i) for i in dict.fromkeys(memory_ids or []) if i]
        result = ReleaseResult()
        if not ids:
            return result
        now = datetime.now(timezone.utc).isoformat()

        refs = await self._call("release_memory_ids", ns, ids, mode=mode, now=now, reason=reason,
                                superseded_by=superseded_by, cap=RELEASED_CAP)
        for k in ("edges_pruned", "edges_retired", "edges_deleted",
                  "nodes_pruned", "nodes_retired", "nodes_deleted"):
            setattr(result, k, getattr(result, k) + int(refs.get(k, 0)))
        records = [r for r in refs.get("records", []) if not r.get("deleted")]
        deleted_edges = list(refs.get("deleted_edges", []))
        deleted_nodes = list(refs.get("deleted_nodes", []))

        legacy_records, erased = await self._release_copies(ns, ids, mode, reason, superseded_by, redact, now, result)
        records += legacy_records
        deleted_edges += erased.get("deleted_edges", [])
        deleted_nodes += erased.get("deleted_nodes", [])

        if records:
            await self._apply(ns, records, reason, result, now)
        if mode == "erase" and (deleted_edges or deleted_nodes):
            await self._erase_vectors(scope, deleted_edges, deleted_nodes, result)
        if result.changed:
            result.namespaces.append(ns)
        return result

    async def apply_states(self, ns: str, records: List[Dict[str, Any]], reason: str = "unsupported") -> int:
        """Evaluate *records* (``kind, id, memory_ids, released_memory_ids,
        is_fixed_entity``) with the shared rule and write their states.
        Returns how many records were written."""
        result = ReleaseResult()
        return await self._apply(ns, records, reason, result, datetime.now(timezone.utc).isoformat())

    async def erase_record(self, scope: Any, record_id: str, *,
                           namespaces: Optional[List[str]] = None) -> Optional[ErasedRecord]:
        """Delete one relationship or entity by its graph id.

        Looks ONLY in *namespaces*, in that order, and erases in the first one
        that holds the id; a record in any other namespace is neither found nor
        touched. The caller decides where it may write (the SDK route passes
        ``[scope_key.write_scope_id(scope)]``): reading a namespace is not
        permission to delete in it, so there is no default and no widening.
        Raises ValueError without *namespaces*. Returns None when not found."""
        if namespaces is None:
            raise ValueError(
                "erase_record needs explicit namespaces= (the namespaces the caller may write); "
                "it never falls back to the namespaces the caller can read")
        for ns in [str(n) for n in dict.fromkeys(namespaces) if n]:
            out = await self._call("erase_records", ns, [record_id], [record_id])
            recs = out.get("records") or []
            if not recs:
                continue
            first = next((r for r in recs if r["id"] == record_id), recs[0])
            erased = ErasedRecord(
                kind="relationship" if first["kind"] == "edge" else "entity",
                memory_ids=list(first.get("memory_ids") or []),
                edges_deleted=len(out.get("deleted_edges") or []),
                nodes_deleted=len(out.get("deleted_nodes") or []),
                namespace=ns)
            result = ReleaseResult()
            await self._erase_vectors(scope, out.get("deleted_edges") or [],
                                      out.get("deleted_nodes") or [], result)
            return erased
        return None

    async def reevaluate(self, ns: str, ids: Iterable[str]) -> int:
        """Re-apply the rule to the RETIRED records among *ids* (edges or
        nodes of *ns*), after memories were attached to them. This is the only
        way a retired record reopens: when the rule, reading Postgres, finds an
        alive memory that supports it. Records that are not retired are left
        alone. Returns how many states were written."""
        ids = [str(i) for i in dict.fromkeys(ids or []) if i]
        if not ids:
            return 0
        records = [r for r in await self._call("lifecycle_records", ns, ids) if r.get("retired_at")]
        if not records:
            return 0
        return await self.apply_states(ns, records)

    async def list_namespaces(self) -> List[str]:
        return list(await self._call("list_namespaces"))

    async def records_in_namespace(self, ns: str) -> List[Dict[str, Any]]:
        return list(await self._call("lifecycle_records", ns))

    # -- internals -----------------------------------------------------------

    async def _release_copies(self, ns, ids, mode, reason, superseded_by, redact, now, result):
        """Copies mode: mark the entries of *ids* in every ``memory_contents``
        that names one. Returns (records to evaluate, erase output)."""
        try:
            rows = await self._call("edges_with_memory_contents", ns, ids)
        except NotImplementedError:
            return [], {}
        records: List[Dict[str, Any]] = []
        to_erase: List[str] = []
        idset = set(ids)
        for row in rows or []:
            seen: Dict[str, Any] = {}

            def transform(memories, _seen=seen):
                changed = False
                for entry in memories:
                    if not isinstance(entry, dict) or str(entry.get("memory_id")) not in idset:
                        continue
                    if mode == "retire":
                        if entry.get("is_current", True) not in (False, "False", "false") \
                                or (superseded_by and entry.get("superseded_by") != superseded_by):
                            entry["is_current"] = False
                            if superseded_by:
                                entry["superseded_by"] = superseded_by
                            entry.setdefault("retired_at", now)
                            entry["retired_reason"] = reason
                            _seen["marked"] = _seen.get("marked", 0) + 1
                            changed = True
                    else:
                        if entry.get("is_deleted") not in (True, "True", "true"):
                            entry["is_current"] = False
                            entry["is_deleted"] = True
                            entry["deleted_at"] = now
                            _seen["marked"] = _seen.get("marked", 0) + 1
                            changed = True
                        if redact and (entry.get("content") or entry.get("embedding") is not None):
                            entry["content"] = ""
                            entry.pop("embedding", None)
                            changed = True
                _seen["memories"] = memories
                return memories if changed else None

            await self._call("rewrite_memory_contents", ns, row["id"], transform)
            result.legacy_entries_marked += int(seen.get("marked", 0))
            memories = seen.get("memories") or []
            live = [str(m.get("memory_id")) for m in memories
                    if isinstance(m, dict) and m.get("memory_id")
                    and m.get("is_current", True) not in (False, "False", "false")
                    and m.get("is_deleted", False) not in (True, "True", "true")]
            released = [i for i in ids if any(str(m.get("memory_id")) == i for m in memories
                                              if isinstance(m, dict))]
            if mode == "erase" and not live:
                to_erase.append(row["id"])
                continue
            records.append({"kind": "edge", "id": row["id"], "memory_ids": live,
                            "released_memory_ids": released, "is_fixed_entity": False})
        erased: Dict[str, Any] = {}
        if to_erase:
            erased = await self._call("erase_records", ns, to_erase, [])
            result.edges_deleted += len(erased.get("deleted_edges") or [])
            result.nodes_deleted += sum(1 for n in erased.get("deleted_nodes") or [])
        return records, erased

    async def _apply(self, ns, records, reason, result: ReleaseResult, now: str) -> int:
        from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import (
            GraphRecord, REASON_UNSUPPORTED, graph_record_state)
        wanted = set()
        for r in records:
            wanted.update(r.get("memory_ids") or [])
            wanted.update(r.get("released_memory_ids") or [])
        loader = self._load or _default_loader()
        supporters = await loader(sorted(wanted)) if wanted else {}
        states = []
        for r in records:
            # The rule reads memory_ids AND released_memory_ids (lead decision,
            # CONTRACT_GRAPH.md): both are passed as stored.
            record = GraphRecord(memory_ids=tuple(r.get("memory_ids") or ()),
                                 released_memory_ids=tuple(r.get("released_memory_ids") or ()),
                                 is_fixed_entity=bool(r.get("is_fixed_entity")))
            state = graph_record_state(record, supporters)
            if state.is_open:
                states.append({"kind": r["kind"], "id": r["id"], "open": True, "now": now})
                if r.get("retired_at"):
                    result.records_reopened += 1
            else:
                why = state.reason
                if why == REASON_UNSUPPORTED and reason:
                    why = reason
                states.append({"kind": r["kind"], "id": r["id"], "open": False, "now": now,
                               "reason": why, "valid_to": state.valid_to,
                               "retired_by_memory_id": state.by_memory})
                result.records_closed += 1
        if not states:
            return 0
        return int(await self._call("set_record_states", ns, states))

    async def _erase_vectors(self, scope, deleted_edges, deleted_nodes, result: ReleaseResult) -> None:
        """Delete the Chroma triple and entity vectors of erased records. Their
        ids are derived the way Stage 4 wrote them (coordinator triple vectors,
        vector_wrapper.upsert_entities_batch). Best effort: a failure is logged
        and counted, the reconciler and forget's text erase are the backstop."""
        if self._vector is None or scope is None:
            return
        from synap.cloud.agents.ingestion.storage.vector_wrapper import (
            deterministic_entity_id, entity_id_scope)
        client = str(scope.client_id)
        cust = entity_id_scope(scope) if getattr(scope, "customer_id", None) else str(scope.customer_id)

        def det(name, etype):
            return deterministic_entity_id(str(name or ""), str(etype or "unknown"), client, cust)

        triples = sorted({f"{det(e.get('from_vector_name') or e.get('from_name'), e.get('from_type'))}"
                          f"_{e.get('type')}_"
                          f"{det(e.get('to_vector_name') or e.get('to_name'), e.get('to_type'))}"
                          for e in deleted_edges if e.get("from_name") and e.get("to_name")})
        entities = sorted({det(n.get("vector_name") or n.get("name"), n.get("entity_type"))
                           for n in deleted_nodes if n.get("name")})
        for collection, ids in (("triples", triples), ("entities", entities)):
            if not ids:
                continue
            try:
                result.vectors_deleted += int(await self._vector.delete_in(scope, collection, ids) or 0)
            except Exception as exc:  # noqa: BLE001
                result.vector_errors += 1
                logger.warning("graph_lifecycle_vector_delete_failed collection=%s ids=%d error=%s",
                               collection, len(ids), exc)


def lifecycle_enabled(instance_id: Any = None) -> bool:
    """SYNAP_GRAPH_LIFECYCLE (``_INSTANCES`` works): write-side release."""
    from synap.cloud.shared.instance_flags import enabled_for
    return enabled_for("SYNAP_GRAPH_LIFECYCLE", instance_id)

