"""Unit tests for the Neo4j executor routing (PR-2).

The adapter must route:
  * schema DDL / SHOW  -> auto-commit ``session.run`` (managed txns reject DDL)
  * reads              -> ``session.execute_read``  (retry + tx timeout)
  * writes             -> ``session.execute_write`` (retry + tx timeout)

These are pure routing tests against a mocked session — no datastore, always
run in CI. They guard the split that keeps DDL working while everything else
gets automatic retries.
"""

from unittest.mock import MagicMock

import pytest

from synap.providers.graph.adapters.neo4j import Neo4jAdapter


def _adapter_with_mock_session():
    adapter = Neo4jAdapter({"uri": "bolt://x", "user": "u", "password": "p"})
    adapter._query_timeout = 5.0
    session = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = session
    cm.__exit__.return_value = False
    driver = MagicMock()
    driver.session.return_value = cm
    adapter._driver = driver
    return adapter, session


def test_ddl_uses_autocommit_not_managed_transaction():
    adapter, session = _adapter_with_mock_session()
    session.run.return_value = []  # iterable result
    adapter._run_query(
        "CREATE INDEX foo IF NOT EXISTS FOR (n:SynapNode) ON (n._namespace)"
    )
    session.run.assert_called_once()
    session.execute_read.assert_not_called()
    session.execute_write.assert_not_called()


def test_show_uses_autocommit():
    adapter, session = _adapter_with_mock_session()
    session.run.return_value = []
    adapter._run_query("SHOW INDEXES")
    session.run.assert_called_once()
    session.execute_read.assert_not_called()


def test_read_uses_execute_read():
    adapter, session = _adapter_with_mock_session()
    session.execute_read.return_value = ([], None)  # (records, profile)
    adapter._run_query("MATCH (n:SynapNode {_namespace: $ns}) RETURN n", {"ns": "x"})
    session.execute_read.assert_called_once()
    session.run.assert_not_called()
    session.execute_write.assert_not_called()


def test_write_uses_execute_write():
    adapter, session = _adapter_with_mock_session()
    session.execute_write.return_value = (None, None)  # (single, profile)
    adapter._run_write("MERGE (a)-[r:X]->(b) RETURN r.id")
    session.execute_write.assert_called_once()
    session.run.assert_not_called()
    session.execute_read.assert_not_called()


def test_connect_sets_query_timeout_from_env(monkeypatch):
    monkeypatch.setenv("NEO4J_QUERY_TIMEOUT_S", "7.5")
    adapter = Neo4jAdapter({"uri": "bolt://x", "user": "u", "password": "p"})
    # exercise the config helper directly (avoids opening a real driver)
    assert adapter._cfg_float("query_timeout", "NEO4J_QUERY_TIMEOUT_S", 20.0) == 7.5


def test_config_dict_overrides_env(monkeypatch):
    monkeypatch.setenv("NEO4J_QUERY_TIMEOUT_S", "7.5")
    adapter = Neo4jAdapter(
        {"uri": "bolt://x", "user": "u", "password": "p", "query_timeout": 3.0}
    )
    assert adapter._cfg_float("query_timeout", "NEO4J_QUERY_TIMEOUT_S", 20.0) == 3.0


# --- an expected constraint race is not an ERROR with a traceback (found at the first Docker boot) ---------

class ConstraintError(Exception):
    pass


def test_a_constraint_violation_is_logged_at_info_without_a_traceback(caplog):
    import logging

    adapter, session = _adapter_with_mock_session()
    session.execute_write.side_effect = ConstraintError(
        "{code: Neo.ClientError.Schema.ConstraintValidationFailed} Node(2) already exists")
    with caplog.at_level(logging.INFO):
        with pytest.raises(ConstraintError):
            adapter._run_write("MERGE (n:SynapNode {id: $id}) SET n.canonical_name = $n RETURN n", {"id": 1, "n": "user"})
    failed = [r for r in caplog.records if "neo4j_write" in r.getMessage() and "failed" in r.getMessage()]
    violations = [r for r in caplog.records if "neo4j_write_constraint_violation" in r.getMessage()]
    assert not failed and len(violations) == 1
    assert violations[0].levelno == logging.INFO and violations[0].exc_info is None


def test_any_other_write_failure_is_still_an_error_with_a_traceback(caplog):
    import logging

    adapter, session = _adapter_with_mock_session()
    session.execute_write.side_effect = RuntimeError("connection lost")
    with caplog.at_level(logging.INFO):
        with pytest.raises(RuntimeError):
            adapter._run_write("MERGE (n:SynapNode {id: $id}) RETURN n", {"id": 1})
    failed = [r for r in caplog.records if "neo4j_write_failed" in r.getMessage()]
    assert len(failed) == 1 and failed[0].levelno == logging.ERROR and failed[0].exc_info is not None
