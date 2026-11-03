"""RDBMS provider abstraction."""

from .interface import RDBMSInterface
from .models import SQLQuery, SQLResult
from .registry import get_rdbms_store, reset_rdbms_store
from .adapters import PostgresAdapter

__all__ = [
    "RDBMSInterface",
    "SQLQuery",
    "SQLResult",
    "get_rdbms_store",
    "reset_rdbms_store",
    "PostgresAdapter",
]
