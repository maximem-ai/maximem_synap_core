"""Tests for RDBMS provider."""

from unittest.mock import patch

import pytest
from synap.providers.rdbms import (
    RDBMSInterface,
    SQLQuery,
    SQLResult,
    get_rdbms_store,
    reset_rdbms_store,
    PostgresAdapter,
)


def test_interface_cannot_instantiate():
    """Verify RDBMSInterface is abstract."""
    with pytest.raises(TypeError):
        RDBMSInterface()


def test_adapter_inherits_interface():
    """Verify PostgresAdapter inherits from RDBMSInterface."""
    assert issubclass(PostgresAdapter, RDBMSInterface)


def test_adapter_instantiation():
    """Verify adapter can be instantiated without error."""
    adapter = PostgresAdapter({
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_pass",
        "pool_size": 2
    })
    assert adapter.config["host"] == "localhost"
    assert adapter.config["database"] == "test_db"
    assert adapter._pool is None  # Not connected yet


def test_registry_returns_interface():
    """Verify registry wires up and returns an RDBMSInterface.

    The registry getter eagerly calls connect(); patch it so the contract
    (correct adapter / interface type) is verified without a live database.
    """
    reset_rdbms_store()
    with patch.object(PostgresAdapter, "connect", return_value=None):
        store = get_rdbms_store({
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_pass"
        })
        assert isinstance(store, RDBMSInterface)
        assert isinstance(store, PostgresAdapter)
    reset_rdbms_store()


def test_sql_query_model():
    """Verify SQLQuery model."""
    query = SQLQuery(sql="SELECT * FROM users", params=("user_id",))
    assert query.sql == "SELECT * FROM users"
    assert query.params == ("user_id",)


def test_sql_result_model():
    """Verify SQLResult model."""
    result = SQLResult(rows=[{"id": 1}], row_count=1, last_insert_id=1)
    assert result.rows == [{"id": 1}]
    assert result.row_count == 1
    assert result.last_insert_id == 1
