"""Standard models for RDBMS operations."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class SQLQuery:
    """Standard SQL query model."""
    sql: str
    params: Tuple = ()


@dataclass
class SQLResult:
    """Standard SQL result model."""
    rows: List[Dict[str, Any]]
    row_count: int
    last_insert_id: Optional[int] = None
