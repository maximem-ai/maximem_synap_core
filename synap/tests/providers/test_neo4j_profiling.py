"""Unit tests for the opt-in Neo4j query-profiling instrumentation (PR-1).

Pure-function tests — no datastore needed, always run in CI. They guard the
`SYNAP_NEO4J_PROFILE` switch and the DDL guard that keeps PROFILE off schema
statements (PROFILE errors on CREATE INDEX / CONSTRAINT / SHOW).
"""

import pytest

from synap.providers.graph.adapters.neo4j import (
    neo4j_profile_enabled,
    _is_profilable,
    summarize_profile,
)


def test_profile_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SYNAP_NEO4J_PROFILE", raising=False)
    assert neo4j_profile_enabled() is False


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True),
    ("0", False), ("false", False), ("", False), ("no", False),
])
def test_profile_flag_parsing(monkeypatch, val, expected):
    monkeypatch.setenv("SYNAP_NEO4J_PROFILE", val)
    assert neo4j_profile_enabled() is expected


@pytest.mark.parametrize("cypher", [
    "\n  CREATE INDEX foo IF NOT EXISTS FOR (n:SynapNode) ON (n._namespace)",
    "CREATE CONSTRAINT c IF NOT EXISTS FOR (n:SynapNode) REQUIRE (n.x) IS UNIQUE",
    "SHOW INDEXES",
    "SHOW CONSTRAINTS",
    "DROP INDEX foo",
    "CALL db.indexes()",
])
def test_ddl_and_show_are_not_profilable(cypher):
    # PROFILE is rejected by the server on these — the guard must exclude them.
    assert _is_profilable(cypher) is False


@pytest.mark.parametrize("cypher", [
    "MATCH ()-[r]->() RETURN r",
    "\n  UNWIND $names AS name MATCH (a:SynapNode) RETURN a",
    "MERGE (a)-[r:X]->(b) RETURN r.id",
    "MATCH (n:SynapNode {_namespace: $ns, id: $id}) RETURN n",
])
def test_data_queries_are_profilable(cypher):
    assert _is_profilable(cypher) is True


def test_summarize_profile_flattens_plan_with_dbhits():
    plan = {
        "operatorType": "DirectedAllRelationshipsScan",
        "dbHits": 44647,
        "children": [
            {"operatorType": "Filter", "dbHits": 41388, "children": []},
        ],
    }
    out = summarize_profile(plan)
    assert out == "total_db_hits=86035 ops=[DirectedAllRelationshipsScan:44647, Filter:41388]"


def test_summarize_profile_handles_empty():
    assert summarize_profile(None) == "total_db_hits=0 ops=[]"
