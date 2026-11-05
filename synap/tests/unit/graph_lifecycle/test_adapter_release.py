"""The Neo4j adapter's lifecycle primitives (W2).

No database: the adapter's transaction is replaced by a recorder that returns
canned rows, so these tests pin what is sent (one write transaction, the
statements and their parameters) and how the rows are summarised. The meaning
of the Cypher is exercised against the in-memory twin in test_service_*.py and
by the lead's live acceptance run.
"""
from __future__ import annotations

import pytest

from synap.providers.graph.adapters import neo4j as neo
from synap.providers.graph.adapters.neo4j import Neo4jAdapter
from synap.providers.graph.interface import GraphStoreInterface


class _Rec:
    def __init__(self, data):
        self._d = data

    def data(self):
        return dict(self._d)

    def __getitem__(self, k):
        return self._d[k]


class _Result(list):
    def single(self):
        return self[0] if self else None


class _Tx:
    def __init__(self, answers):
        self.answers = answers
        self.sent = []

    def run(self, cypher, params=None):
        self.sent.append((cypher, params))
        for key, rows in self.answers:
            if key in cypher:
                return _Result(_Rec(r) for r in rows)
        return _Result()


def _adapter(answers):
    a = Neo4jAdapter.__new__(Neo4jAdapter)
    tx = _Tx(answers)
    a.transactions = 0

    def run_in_write_tx(work):
        a.transactions += 1
        return work(tx)

    a._run_in_write_tx = run_in_write_tx
    return a, tx


EDGE_ROW = dict(id="e1", type="WORKS_AT", memory_ids=[], released_memory_ids=["m1"], unsupported=True,
                newly=True, deleted=False, from_id="u", from_name="the user", from_type="person",
                from_vector_name=None, to_id="acme", to_name="Acme", to_type="org", to_vector_name=None)
NODE_ROW = dict(id="acme", memory_ids=[], released_memory_ids=["m1"], unsupported=True, newly=True,
                is_fixed_entity=False)


def test_retire_is_one_transaction_and_reports_counts():
    a, tx = _adapter([("[r]->(b:SynapNode)", [EDGE_ROW]), ("MATCH (n:SynapNode {_namespace: $ns})\nWHERE any", [NODE_ROW])])
    out = a.release_memory_ids("ns1", ["m1", "m1"], mode="retire", now="T", reason="superseded",
                               superseded_by="m2", cap=7)
    assert a.transactions == 1
    assert len(tx.sent) == 2, "retire never deletes orphans"
    params = tx.sent[0][1]
    assert params["ids"] == ["m1"] and params["by"] == "m2" and params["cap"] == 7 and params["erase"] is False
    assert out["edges_pruned"] == 1 and out["edges_retired"] == 1 and out["edges_deleted"] == 0
    assert out["nodes_pruned"] == 1 and out["nodes_retired"] == 1 and out["nodes_deleted"] == 0
    assert {r["kind"] for r in out["records"]} == {"edge", "node"}


def test_erase_deletes_orphans_of_deleted_edges_in_the_same_transaction():
    deleted = {**EDGE_ROW, "deleted": True}
    a, tx = _adapter([("[r]->(b:SynapNode)", [deleted]),
                      ("MATCH (n:SynapNode {_namespace: $ns})\nWHERE any", [NODE_ROW]),
                      ("NOT EXISTS { (n)--() }", [dict(id="acme", name="Acme", entity_type="org",
                                                       vector_name=None, memory_ids=[])])])
    out = a.release_memory_ids("ns1", ["m1"], mode="erase", reason="deleted")
    assert a.transactions == 1
    assert len(tx.sent) == 3
    orphan_params = tx.sent[2][1]
    assert set(orphan_params["candidates"]) == {"u", "acme"}
    assert out["edges_deleted"] == 1 and out["edges_retired"] == 0
    assert out["nodes_deleted"] == 1 and out["nodes_retired"] == 0
    assert out["deleted_edges"][0]["to_name"] == "Acme"
    assert out["deleted_nodes"][0]["name"] == "Acme"


def test_empty_ids_touch_nothing():
    a, tx = _adapter([])
    out = a.release_memory_ids("ns1", [], mode="erase")
    assert a.transactions == 0 and out["edges_pruned"] == 0


def test_bad_mode_is_refused():
    a, _ = _adapter([])
    with pytest.raises(ValueError):
        a.release_memory_ids("ns1", ["m1"], mode="drop")


def test_writer_and_customer_are_never_retired_or_deleted_by_the_cypher():
    # The guard is in every statement that can retire or delete a node.
    for stmt in (neo._RELEASE_NODES, neo._DELETE_ORPHANS, neo._SET_NODE_STATES, neo._ERASE_NODES,
                 neo._ERASE_EDGES):
        assert "is_writer" in stmt and "is_customer" in stmt, stmt


def test_release_cypher_keeps_history_and_first_retirement():
    s = neo._RELEASE_EDGES
    assert "released_memory_ids" in s and "$cap" in s
    assert "r.retired_at IS NULL AS newly" in s
    assert "DELETE r" in s and "$erase AND unsupported" in s


def test_set_record_states_is_one_unwind_per_kind():
    a, tx = _adapter([("UNWIND $rows", [dict(n=1)])])
    n = a.set_record_states("ns1", [
        {"kind": "edge", "id": "e1", "open": False, "reason": "superseded", "retired_by_memory_id": "m2",
         "valid_to": None},
        {"kind": "node", "id": "acme", "open": True}])
    assert a.transactions == 1 and len(tx.sent) == 2 and n == 2
    assert tx.sent[0][1]["rows"][0]["by"] == "m2"
    assert tx.sent[0][1]["rows"][0]["now"], "closed rows get a retirement time"


def test_append_of_memory_ids_reopens_but_other_lists_do_not():
    a = Neo4jAdapter.__new__(Neo4jAdapter)
    sent = []
    a._run_write = lambda cypher, params=None: sent.append(cypher)
    a.append_to_edge_list("ns", "e1", "memory_ids", ["m9"], 500)
    a.append_to_node_list("ns", "n1", "memory_ids", ["m9"], 500)
    a.append_to_edge_list("ns", "e1", "instance_ids", ["i"], 500)
    assert "retired_at = CASE WHEN size(merged) > size(current)" in sent[0]
    assert "retired_at = CASE WHEN size(merged) > size(current)" in sent[1]
    assert "retired_at" not in sent[2]


def test_interface_methods_exist_and_refuse_loudly():
    class Bare(GraphStoreInterface):
        pass
    for name in ("release_memory_ids", "set_record_states", "erase_records", "list_namespaces",
                 "lifecycle_records", "edges_with_memory_contents"):
        assert hasattr(GraphStoreInterface, name)
    # Abstract methods unimplemented -> cannot instantiate; the defaults raise.
    with pytest.raises(NotImplementedError):
        GraphStoreInterface.release_memory_ids(object(), "ns", ["m"])
