"""Base interface for relational databases."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import SQLQuery, SQLResult


class RDBMSInterface(ABC):
    """
    Base interface for relational databases.
    Multi-tenancy: columnar (tenant_id column).
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection pool."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection pool."""
        pass

    @abstractmethod
    def execute(self, query: SQLQuery) -> SQLResult:
        """Execute single query."""
        pass

    @abstractmethod
    def execute_many(self, query: SQLQuery, params_list: List[tuple]) -> SQLResult:
        """Execute query with multiple param sets."""
        pass

    @abstractmethod
    def fetch_one(self, query: SQLQuery) -> Optional[Dict[str, Any]]:
        """Fetch single row."""
        pass

    @abstractmethod
    def fetch_all(self, query: SQLQuery) -> List[Dict[str, Any]]:
        """Fetch all rows."""
        pass

    @abstractmethod
    def transaction(self):
        """Context manager for transactions."""
        pass
