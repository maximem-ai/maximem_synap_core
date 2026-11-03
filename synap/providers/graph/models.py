"""Standard models for graph store operations."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Node:
    """Graph node."""
    id: str
    node_type: str
    properties: Dict[str, Any]


@dataclass
class Edge:
    """Graph edge."""
    id: str
    from_id: str
    to_id: str
    edge_type: str
    properties: Optional[Dict[str, Any]] = None


@dataclass
class TraversalPattern:
    """Portable traversal definition."""
    edge_types: List[str]           # Filter by edge types
    max_depth: int = 2
    direction: str = "out"          # "out" | "in" | "both"
    node_filters: Optional[Dict[str, Any]] = None
    limit: int = 100


@dataclass
class GraphQuery:
    """Standard graph query."""
    start_node_id: Optional[str] = None
    start_node_type: Optional[str] = None
    pattern: Optional[TraversalPattern] = None
    namespace: Optional[str] = None  # instance_id for tenant isolation


@dataclass
class GraphResult:
    """Graph query result."""
    nodes: List[Node]
    edges: List[Edge]
    paths: Optional[List[List[str]]] = None  # Node ID paths
