"""Standard models for vector store operations."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class VectorQuery:
    """Standard vector search query."""
    embedding: List[float]
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None  # instance_id for tenant isolation


@dataclass
class VectorDocument:
    """Document to upsert."""
    id: str
    embedding: List[float]
    metadata: Dict[str, Any]
    content: Optional[str] = None


@dataclass
class VectorResult:
    """Single search result."""
    id: str
    score: float
    metadata: Dict[str, Any]
    content: Optional[str] = None


@dataclass
class VectorSearchResult:
    """Search results container."""
    results: List[VectorResult]
    query_time_ms: float
