"""Tests for AsyncGraphWrapper.combined_name_traverse — distinct-return rewrite.

The rewrite dedups nodes/edges in Cypher (collect(DISTINCT ...)) instead of
shipping whole paths and deduping in Python. The accuracy contract: the
node-id and edge-id sets reaching GraphResult must be identical to the
legacy RETURN-path query's (checked with exact
set-equality; kill switch SYNAP_GRAPH_DISTINCT_RETURN=0 reverts).

Unit tests here run against a fake driver (CI has no Neo4j service) and
cover: query-shape selection by the kill switch, the shared Node/Edge
construction rules (property-id dedup, id-less drop, underscore/id property
filtering, node_type label selection, endpoint ids), and old/new parity on
equivalent data. The live-Neo4j parity test is integration-gated by the
conftest TCP probe (same pattern as test_graph.py).
"""

import asyncio

import pytest

from synap.cloud.agents.ingestion.storage.graph_wrapper import AsyncGraphWrapper


# ---------------------------------------------------------------------------
# Fakes mimicking the neo4j driver's graph types (dict(node), .labels, etc.)
# ---------------------------------------------------------------------------

class FakeNode(dict):
    def __init__(self, props, labels=("SynapNode", "Person")):
        super().__init__(props)
        self.labels = list(labels)


class FakeRel(dict):
    def __init__(self, props, start, end, rel_type="RELATES_TO"):
        super().__init__(props)
        self.start_node = start
        self.end_node = end
        self.type = rel_type


class FakePath:
    def __init__(self, nodes, relationships):
        self.nodes = nodes
        self.relationships = relationships


class FakeRecord:
    def __init__(self, data):
        self._data = data

    def get(self, key):
        return self._data.get(key)


class FakeSummary:
    profile = None


class FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None

    def consume(self):
        return FakeSummary()


class FakeSession:
    def __init__(self, records, capture):
        self._records = records
        self._capture = capture

    def run(self, cypher, params=None):
        self._capture["cypher"] = cypher
        self._capture["params"] = params
        return FakeResult(self._records)

    # PR-2: the traverse now runs inside a managed read transaction. execute_read
    # invokes the (unit_of_work-wrapped) transaction function with a tx that has
    # the same .run capture surface as the session.
    def execute_read(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def execute_write(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeProvider:
    _database = "neo4j"

    def __init__(self, records, capture):
        self._records = records
        self._capture = capture

    def _get_driver(self):
        provider = self

        class _D:
            def session(self, database=None):
                return FakeSession(provider._records, provider._capture)

        return _D()


def _sample_graph():
    """Two overlapping 'paths' worth of data: 3 nodes, 2 edges, plus one
    id-less node the legacy code silently drops."""
    a = FakeNode({"id": "n-a", "canonical_name": "A", "_namespace": "ns1",
                  "embedding": [0.1, 0.2]})
    b = FakeNode({"id": "n-b", "canonical_name": "B", "_namespace": "ns1"},
                 labels=("SynapNode",))
    c = FakeNode({"id": "n-c", "canonical_name": "C", "_namespace": "ns1"})
    ghost = FakeNode({"canonical_name": "no-id", "_namespace": "ns1"})
    e1 = FakeRel({"id": "e-1", "memory_contents": "[]", "_namespace": "ns1"}, a, b)
    e2 = FakeRel({"id": "e-2"}, b, c, rel_type="BLOCKS")
    return a, b, c, ghost, e1, e2


def _run(wrapper, **kwargs):
    return asyncio.get_event_loop().run_until_complete(
        wrapper.combined_name_traverse("ns1", ["A"], **kwargs)
    )


def _traverse(records, capture):
    wrapper = AsyncGraphWrapper(FakeProvider(records, capture))
    return asyncio.run(wrapper.combined_name_traverse("ns1", ["A"]))


# ---------------------------------------------------------------------------
# Distinct-return mode (default)
# ---------------------------------------------------------------------------

class TestDistinctReturnMode:
    def test_query_shape_and_consumption(self, monkeypatch):
        monkeypatch.delenv("SYNAP_GRAPH_DISTINCT_RETURN", raising=False)
        a, b, c, ghost, e1, e2 = _sample_graph()
        capture = {}
        record = FakeRecord({"nodes": [a, b, c, ghost], "rels": [e1, e2]})
        res = _traverse([record], capture)

        assert "collect(DISTINCT n) AS nodes" in capture["cypher"]
        assert "RETURN nodes, rels" in capture["cypher"]
        assert "WITH path" in capture["cypher"]  # limit stays on path rows
        assert capture["params"]["per_anchor_limit"] == 50

        node_ids = {n.id for n in res.nodes}
        assert node_ids == {"n-a", "n-b", "n-c"}  # ghost (no id prop) dropped
        by_id = {n.id: n for n in res.nodes}
        assert by_id["n-a"].node_type == "Person"      # first non-SynapNode label
        assert by_id["n-b"].node_type == "Unknown"     # only SynapNode label
        assert "_namespace" not in by_id["n-a"].properties  # underscore filtered
        assert "id" not in by_id["n-a"].properties
        assert by_id["n-a"].properties["embedding"] == [0.1, 0.2]  # embedding kept

        edges = {e.id: e for e in res.edges}
        assert set(edges) == {"e-1", "e-2"}
        assert (edges["e-1"].from_id, edges["e-1"].to_id) == ("n-a", "n-b")
        assert edges["e-2"].edge_type == "BLOCKS"
        assert edges["e-1"].properties["memory_contents"] == "[]"

        assert res.paths == []  # no production reader; empty in distinct mode

    def test_empty_result_row(self, monkeypatch):
        monkeypatch.delenv("SYNAP_GRAPH_DISTINCT_RETURN", raising=False)
        record = FakeRecord({"nodes": [], "rels": []})
        res = _traverse([record], {})
        assert res.nodes == [] and res.edges == [] and res.paths == []

    def test_empty_anchor_names_short_circuits(self, monkeypatch):
        monkeypatch.delenv("SYNAP_GRAPH_DISTINCT_RETURN", raising=False)
        capture = {}
        wrapper = AsyncGraphWrapper(FakeProvider([], capture))
        res = asyncio.run(wrapper.combined_name_traverse("ns1", []))
        assert res.nodes == [] and res.edges == []
        assert "cypher" not in capture  # never hit the driver


# ---------------------------------------------------------------------------
# Kill switch: legacy per-path mode
# ---------------------------------------------------------------------------

class TestLegacyMode:
    def test_kill_switch_selects_return_path_query(self, monkeypatch):
        monkeypatch.setenv("SYNAP_GRAPH_DISTINCT_RETURN", "0")
        a, b, c, ghost, e1, e2 = _sample_graph()
        capture = {}
        records = [
            FakeRecord({"path": FakePath([a, b, ghost], [e1])}),
            FakeRecord({"path": FakePath([b, c], [e2])}),
        ]
        res = _traverse(records, capture)

        assert "RETURN path" in capture["cypher"]
        assert "collect(DISTINCT" not in capture["cypher"]
        assert {n.id for n in res.nodes} == {"n-a", "n-b", "n-c"}
        assert {e.id for e in res.edges} == {"e-1", "e-2"}
        assert res.paths == [["n-a", "n-b"], ["n-b", "n-c"]]  # legacy still builds paths

    def test_modes_produce_identical_sets(self, monkeypatch):
        """Same underlying graph through both consumption branches -> same
        GraphResult nodes/edges (the accuracy contract, unit-level)."""
        a, b, c, ghost, e1, e2 = _sample_graph()

        monkeypatch.setenv("SYNAP_GRAPH_DISTINCT_RETURN", "0")
        legacy = _traverse(
            [FakeRecord({"path": FakePath([a, b, ghost], [e1])}),
             FakeRecord({"path": FakePath([b, c], [e2])}),
             FakeRecord({"path": FakePath([a, b], [e1])})],  # duplicate path
            {},
        )
        monkeypatch.setenv("SYNAP_GRAPH_DISTINCT_RETURN", "1")
        distinct = _traverse(
            [FakeRecord({"nodes": [a, b, c, ghost], "rels": [e1, e2]})], {}
        )

        assert {n.id for n in legacy.nodes} == {n.id for n in distinct.nodes}
        assert {e.id for e in legacy.edges} == {e.id for e in distinct.edges}
        legacy_props = {n.id: n.properties for n in legacy.nodes}
        distinct_props = {n.id: n.properties for n in distinct.nodes}
        assert legacy_props == distinct_props


# ---------------------------------------------------------------------------
# Live-Neo4j parity (integration; auto-skipped without localhost:7687 —
# conftest TCP probe, same as test_graph.py)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLiveParity:
    def test_old_vs_new_sets_identical(self, monkeypatch):
        from synap.providers.graph.registry import get_graph_store, reset_graph_store

        reset_graph_store()
        store = get_graph_store({
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "test",
            "database": "neo4j",
        })
        try:
            ns = "test_ns_distinct_return"
            store._run_write("MATCH (n:SynapNode {_namespace: $ns}) DETACH DELETE n", {"ns": ns})
            store._run_write(
                "CREATE (a:SynapNode {_namespace: $ns, id: 'p-a', canonical_name: 'Alpha'})"
                "-[:KNOWS {id: 'p-e1'}]->"
                "(b:SynapNode {_namespace: $ns, id: 'p-b', canonical_name: 'Beta'})"
                "-[:KNOWS {id: 'p-e2'}]->"
                "(c:SynapNode {_namespace: $ns, id: 'p-c', canonical_name: 'Gamma'})",
                {"ns": ns},
            )
            wrapper = AsyncGraphWrapper(store)

            monkeypatch.setenv("SYNAP_GRAPH_DISTINCT_RETURN", "1")
            new = asyncio.run(wrapper.combined_name_traverse(ns, ["Alpha"]))
            monkeypatch.setenv("SYNAP_GRAPH_DISTINCT_RETURN", "0")
            old = asyncio.run(wrapper.combined_name_traverse(ns, ["Alpha"]))

            assert {n.id for n in new.nodes} == {n.id for n in old.nodes} == {"p-a", "p-b", "p-c"}
            assert {e.id for e in new.edges} == {e.id for e in old.edges} == {"p-e1", "p-e2"}
            for e in new.edges:
                assert e.from_id and e.to_id  # endpoint hydration
        finally:
            store._run_write("MATCH (n:SynapNode {_namespace: $ns}) DETACH DELETE n",
                             {"ns": "test_ns_distinct_return"})
            reset_graph_store()
