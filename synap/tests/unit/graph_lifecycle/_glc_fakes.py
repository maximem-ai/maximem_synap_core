"""In-memory graph adapter for the graph lifecycle tests.

It implements the lifecycle primitives of ``GraphStoreInterface`` with the same
meaning as the Neo4j adapter's Cypher (``release_memory_ids``,
``set_record_states``, ``erase_records``, the reopen on ``append_*_list``, and
the copies-mode helpers), so the service can be tested without a database.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _fixed(props: Dict[str, Any]) -> bool:
    return str(props.get("is_writer", "")) in ("True", "true") or \
        str(props.get("is_customer", "")) in ("True", "true")


class FakeGraph:
    def __init__(self):
        # ns -> {node_id: props}; ns -> {edge_id: {"from","to","type","props"}}
        self.nodes: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.edges: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.fail_release = False
        self.calls: List[str] = []

    # -- building fixtures -------------------------------------------------
    def node(self, ns, node_id, name=None, entity_type="person", **props):
        self.nodes.setdefault(ns, {})[node_id] = {"id": node_id, "canonical_name": name or node_id,
                                                  "entity_type": entity_type, **props}
        return node_id

    def edge(self, ns, edge_id, a, b, etype="WORKS_AT", **props):
        self.edges.setdefault(ns, {})[edge_id] = {"from": a, "to": b, "type": etype,
                                                  "props": {"id": edge_id, **props}}
        return edge_id

    def e(self, ns, edge_id):
        return self.edges.get(ns, {}).get(edge_id, {}).get("props")

    def n(self, ns, node_id):
        return self.nodes.get(ns, {}).get(node_id)

    # -- primitives ------------------------------------------------------------
    def _prune(self, props, ids, cap, now, reason, by, fixed=False):
        current = list(props.get("memory_ids") or [])
        kept = [x for x in current if x not in ids]
        gone = [x for x in current if x in ids]
        prev = list(props.get("released_memory_ids") or [])
        rel = prev + [x for x in gone if x not in prev]
        released = rel[len(rel) - cap:] if len(rel) > cap else rel
        unsupported = not kept and not fixed
        newly = unsupported and props.get("retired_at") is None
        props["memory_ids"] = kept
        props["released_memory_ids"] = released
        if newly:
            props["retired_at"] = now
            props["retired_reason"] = reason
            if by:
                props["retired_by_memory_id"] = by
            else:
                props.pop("retired_by_memory_id", None)
        return kept, released, unsupported, newly

    def _orphans(self, ns, candidates):
        out = []
        for nid in sorted(candidates):
            p = self.nodes.get(ns, {}).get(nid)
            if p is None or p.get("memory_ids") or _fixed(p):
                continue
            if any(e["from"] == nid or e["to"] == nid for e in self.edges.get(ns, {}).values()):
                continue
            del self.nodes[ns][nid]
            out.append({"id": nid, "name": p.get("canonical_name"), "entity_type": p.get("entity_type"),
                        "vector_name": p.get("er_vector_name"), "memory_ids": []})
        return out

    def _edge_row(self, ns, eid, e):
        a = self.nodes[ns][e["from"]]
        b = self.nodes[ns][e["to"]]
        return {"id": eid, "type": e["type"], "from_id": e["from"], "from_name": a.get("canonical_name"),
                "from_type": a.get("entity_type"), "from_vector_name": a.get("er_vector_name"),
                "to_id": e["to"], "to_name": b.get("canonical_name"), "to_type": b.get("entity_type"),
                "to_vector_name": b.get("er_vector_name")}

    def release_memory_ids(self, namespace, ids, mode="retire", now=None, reason="superseded",
                           superseded_by=None, cap=500):
        from synap.providers.graph.adapters.neo4j import _release_summary, _empty_release
        self.calls.append(f"release:{namespace}:{mode}")
        if self.fail_release:
            raise RuntimeError("graph down")
        ids = [str(i) for i in dict.fromkeys(ids or []) if i]
        if not ids:
            return _empty_release()
        now = now or "2026-11-05T00:00:00+00:00"
        edges, nodes = [], []
        for eid, e in list(self.edges.get(namespace, {}).items()):
            p = e["props"]
            if not any(x in ids for x in p.get("memory_ids") or []):
                continue
            kept, released, unsupported, newly = self._prune(p, ids, cap, now, reason, superseded_by)
            erased = mode == "erase" and unsupported
            row = {**self._edge_row(namespace, eid, e), "memory_ids": kept, "released_memory_ids": released,
                   "unsupported": unsupported, "newly": newly, "deleted": erased}
            if erased:
                del self.edges[namespace][eid]
            edges.append(row)
        for nid, p in list(self.nodes.get(namespace, {}).items()):
            if not any(x in ids for x in p.get("memory_ids") or []):
                continue
            fixed = _fixed(p)
            kept, released, unsupported, newly = self._prune(p, ids, cap, now, reason, superseded_by, fixed)
            nodes.append({"id": nid, "memory_ids": kept, "released_memory_ids": released,
                          "unsupported": unsupported, "newly": newly, "is_fixed_entity": fixed})
        orphans = []
        if mode == "erase":
            cand = {n["id"] for n in nodes if not n["memory_ids"]}
            for e in edges:
                if e["deleted"]:
                    cand.update((e["from_id"], e["to_id"]))
            orphans = self._orphans(namespace, cand)
        return _release_summary(edges, nodes, orphans)

    def set_record_states(self, namespace, states):
        n = 0
        for s in states or []:
            if s["kind"] == "edge":
                p = self.e(namespace, s["id"])
            else:
                p = self.n(namespace, s["id"])
                if p is not None and not s.get("open") and _fixed(p):
                    continue
            if p is None:
                continue
            if s.get("open"):
                for k in ("retired_at", "retired_reason", "retired_by_memory_id", "valid_to"):
                    p.pop(k, None)
            else:
                p["retired_at"] = p.get("retired_at") or s.get("now") or "2026-11-05T00:00:00+00:00"
                p["retired_reason"] = s.get("reason")
                p["retired_by_memory_id"] = s.get("retired_by_memory_id")
                vt = s.get("valid_to")
                p["valid_to"] = vt.isoformat() if hasattr(vt, "isoformat") else vt
            n += 1
        return n

    def erase_records(self, namespace, edge_ids, node_ids):
        edge_ids, node_ids = set(edge_ids or []), set(node_ids or [])
        nodes = self.nodes.get(namespace, {})
        named_nodes = {i for i in node_ids if i in nodes and not _fixed(nodes[i])}
        edges = []
        for eid, e in list(self.edges.get(namespace, {}).items()):
            if eid in edge_ids or e["from"] in named_nodes or e["to"] in named_nodes:
                edges.append({**self._edge_row(namespace, eid, e),
                              "memory_ids": list(e["props"].get("memory_ids") or [])})
                del self.edges[namespace][eid]
        deleted_nodes = []
        for nid in sorted(named_nodes):
            p = nodes.pop(nid)
            deleted_nodes.append({"id": nid, "name": p.get("canonical_name"), "entity_type": p.get("entity_type"),
                                  "vector_name": p.get("er_vector_name"),
                                  "memory_ids": list(p.get("memory_ids") or [])})
        cand = set()
        for e in edges:
            cand.update((e["from_id"], e["to_id"]))
        orphans = self._orphans(namespace, cand - named_nodes)
        from synap.providers.graph.adapters.neo4j import _edge_names, _node_names
        return {"deleted_edges": [_edge_names(e) for e in edges],
                "deleted_nodes": [_node_names(n) for n in deleted_nodes + orphans],
                "records": ([{"kind": "edge", "id": e["id"], "memory_ids": e["memory_ids"]} for e in edges]
                            + [{"kind": "node", "id": n["id"], "memory_ids": n["memory_ids"]}
                               for n in deleted_nodes])}

    def _append(self, props, key, values, cap):
        current = list(props.get(key) or [])
        merged = current + [v for v in values if v not in current]
        props[key] = merged[len(merged) - cap:] if len(merged) > cap else merged

    def append_to_edge_list(self, namespace, edge_id, key, values, cap):
        self._append(self.e(namespace, edge_id), key, values, cap)

    def append_to_node_list(self, namespace, node_id, key, values, cap):
        self._append(self.n(namespace, node_id), key, values, cap)

    def restore_memory_ids(self, namespace, ids, cap=500):
        ids = [str(i) for i in ids]
        out = []
        items = [("edge", eid, e["props"]) for eid, e in self.edges.get(namespace, {}).items()] + \
                [("node", nid, p) for nid, p in self.nodes.get(namespace, {}).items()]
        for kind, rid, p in items:
            back = [i for i in p.get("released_memory_ids") or [] if i in ids]
            if not back:
                continue
            current = list(p.get("memory_ids") or [])
            merged = current + [i for i in back if i not in current]
            p["memory_ids"] = merged[len(merged) - cap:] if len(merged) > cap else merged
            p["released_memory_ids"] = [i for i in p["released_memory_ids"] if i not in ids]
            out.append({"kind": kind, "id": rid, "memory_ids": list(p["memory_ids"]),
                        "released_memory_ids": list(p["released_memory_ids"]),
                        "is_fixed_entity": kind == "node" and _fixed(p), "retired_at": p.get("retired_at")})
        return out

    def list_namespaces(self):
        return sorted(set(self.nodes) | set(self.edges))

    def lifecycle_records(self, namespace, ids=None):
        out = []
        for eid, e in self.edges.get(namespace, {}).items():
            if ids is not None and eid not in ids:
                continue
            p = e["props"]
            legacy = [m["memory_id"] for m in json.loads(p["memory_contents"])] if p.get("memory_contents") else []
            out.append({"kind": "edge", "id": eid, "memory_ids": list(p.get("memory_ids") or []),
                        "released_memory_ids": list(p.get("released_memory_ids") or []),
                        "legacy_ids": legacy, "is_fixed_entity": False, "retired_at": p.get("retired_at"),
                        "retired_reason": p.get("retired_reason"),
                        "retired_by_memory_id": p.get("retired_by_memory_id")})
        for nid, p in self.nodes.get(namespace, {}).items():
            if ids is not None and nid not in ids:
                continue
            out.append({"kind": "node", "id": nid, "memory_ids": list(p.get("memory_ids") or []),
                        "released_memory_ids": list(p.get("released_memory_ids") or []), "legacy_ids": [],
                        "is_fixed_entity": _fixed(p), "retired_at": p.get("retired_at"),
                        "retired_reason": p.get("retired_reason"),
                        "retired_by_memory_id": p.get("retired_by_memory_id")})
        return out

    def edges_with_memory_contents(self, namespace, ids):
        out = []
        for eid, e in self.edges.get(namespace, {}).items():
            mc = e["props"].get("memory_contents")
            if mc and any(i in mc for i in ids):
                out.append({"id": eid, "memory_contents": mc})
        return out

    def rewrite_memory_contents(self, namespace, edge_id, transform, attempts=5):
        p = self.e(namespace, edge_id)
        if p is None or p.get("memory_contents") is None:
            return 0
        memories = json.loads(p["memory_contents"])
        updated = transform(memories)
        if updated is None:
            return 0
        p["memory_contents"] = json.dumps(updated)
        return max(0, len(memories) - len(updated))


class FakeWrapper:
    """Stands in for AsyncGraphWrapper: the service only needs `_provider`."""

    def __init__(self, provider):
        self._provider = provider


class FakeVector:
    def __init__(self):
        self.deleted: List[tuple] = []

    async def delete_in(self, scope, collection, ids):
        self.deleted.append((collection, sorted(ids)))
        return len(ids)


def supporters_loader(rows: Dict[str, Any]):
    """A `load_supporter_rows` stand-in over a dict of SupporterRow."""
    async def load(ids, session=None):
        return {i: rows[i] for i in ids if i in rows}
    return load
