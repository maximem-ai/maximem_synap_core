"""Unit tests for the Neo4j schema bootstrap + verification (PR-3).

Mocked ``_run_query`` — no datastore, always run in CI. Guard the global-schema
consolidation: fixed-name index/constraint creation, the run-once guard, the
non-swallowed constraint failure, and shape-based verification.
"""

import logging
from unittest.mock import MagicMock

from synap.providers.graph.adapters.neo4j import Neo4jAdapter


def _adapter():
    return Neo4jAdapter({"uri": "bolt://x", "user": "u", "password": "p"})


def test_ensure_schema_creates_indexes_and_constraint_once():
    a = _adapter()
    a._run_query = MagicMock(return_value=[])
    a.verify_schema = MagicMock(return_value={})  # isolate from SHOW calls
    a._ensure_schema()

    stmts = [c.args[0] for c in a._run_query.call_args_list]
    assert any("CREATE INDEX synap_synapnode_namespace" in s for s in stmts)
    assert any("CREATE INDEX synap_synapnode_id" in s for s in stmts)
    assert any("CREATE CONSTRAINT synap_synapnode_canonical" in s for s in stmts)
    assert a._schema_ensured is True

    # run-once guard: a second call issues no further DDL
    a._run_query.reset_mock()
    a._ensure_schema()
    a._run_query.assert_not_called()


def test_create_namespace_ensures_schema():
    a = _adapter()
    a._ensure_schema = MagicMock()
    a.create_namespace("some-namespace")
    a._ensure_schema.assert_called_once()


def test_constraint_failure_is_logged_not_swallowed(caplog):
    a = _adapter()
    a.verify_schema = MagicMock(return_value={})

    def _rq(cypher, params=None):
        if "CONSTRAINT" in cypher:
            raise RuntimeError("duplicate (canonical_name, _namespace)")
        return []

    a._run_query = MagicMock(side_effect=_rq)
    with caplog.at_level(logging.ERROR):
        a._ensure_schema()  # must NOT raise — ingest proceeds

    assert any("neo4j_constraint_creation_failed" in r.getMessage() for r in caplog.records)
    assert a._schema_ensured is True


def test_a_concurrent_worker_creating_the_same_index_is_not_an_error(caplog):
    # Two workers start together and both run CREATE INDEX ... IF NOT EXISTS; the loser is told an
    # equivalent index already exists. That is the state we wanted, so setup carries on.
    a = _adapter()
    a.verify_schema = MagicMock(return_value={})
    seen = []

    def _rq(cypher, params=None):
        seen.append(cypher)
        raise RuntimeError(
            "{code: Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists} "
            "{message: An equivalent index already exists}")

    a._run_query = MagicMock(side_effect=_rq)
    with caplog.at_level(logging.ERROR):
        a._ensure_schema()  # must not raise

    assert any("CONSTRAINT" in c for c in seen)           # it went on to the constraint
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert a._schema_ensured is True


def test_the_race_is_logged_as_info_without_a_traceback_by_the_query_runner(caplog):
    a = _adapter()
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value.run.side_effect = RuntimeError(
        "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists")
    a._get_driver = MagicMock(return_value=driver)
    a._database = "neo4j"
    with caplog.at_level(logging.INFO, logger="synap.providers.graph.adapters.neo4j"):
        try:
            a._run_query("CREATE INDEX x IF NOT EXISTS FOR (n:SynapNode) ON (n.id)")
        except RuntimeError:
            pass
    failed = [r for r in caplog.records if "neo4j_query_failed" in r.getMessage()]
    assert not failed
    assert any("neo4j_schema_already_exists" in r.getMessage() and r.levelno == logging.INFO for r in caplog.records)


def test_verify_schema_detects_by_shape_not_name():
    a = _adapter()
    # rows named arbitrarily (e.g. old per-namespace names) but matched by shape
    idx_rows = [
        {"labelsOrTypes": ["SynapNode"], "properties": ["_namespace"], "entityType": "NODE"},
        {"labelsOrTypes": ["SynapNode"], "properties": ["id"], "entityType": "NODE"},
        {"labelsOrTypes": ["OtherProduct"], "properties": ["x"], "entityType": "NODE"},
    ]
    con_rows = [
        {"labelsOrTypes": ["SynapNode"], "properties": ["canonical_name", "_namespace"]},
    ]
    a._run_query = MagicMock(side_effect=[idx_rows, con_rows])
    assert a.verify_schema() == {
        "namespace_index": True,
        "id_index": True,
        "canonical_constraint": True,
    }


def test_verify_schema_reports_missing():
    a = _adapter()
    a._run_query = MagicMock(side_effect=[[], []])
    assert a.verify_schema() == {
        "namespace_index": False,
        "id_index": False,
        "canonical_constraint": False,
    }
