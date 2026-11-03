"""Base interface for graph databases."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import Node, Edge, TraversalPattern, GraphQuery, GraphResult


class GraphStoreInterface(ABC):
    """
    Base interface for graph databases.
    Multi-tenancy: namespace/schema per instance_id.
    Traversal-based (not Cypher) for portability.
    """

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    # Namespace management (tenant isolation)
    @abstractmethod
    def create_namespace(self, namespace: str) -> None:
        """Create namespace/schema for instance."""
        pass

    @abstractmethod
    def delete_namespace(self, namespace: str) -> None:
        """Delete namespace and all data."""
        pass

    # Node operations
    @abstractmethod
    def add_node(self, namespace: str, node_type: str, properties: Dict[str, Any]) -> str:
        """Add node. Returns node ID."""
        pass

    @abstractmethod
    def get_node(self, namespace: str, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        pass

    @abstractmethod
    def update_node(self, namespace: str, node_id: str, properties: Dict[str, Any]) -> None:
        """Update node properties."""
        pass

    @abstractmethod
    def delete_node(self, namespace: str, node_id: str) -> None:
        """Delete node and connected edges."""
        pass

    @abstractmethod
    def find_nodes_by_properties(
        self, namespace: str, node_type: str, properties: Dict[str, Any], limit: int = 10
    ) -> List[Node]:
        """Find nodes by property values within a namespace."""
        pass

    @abstractmethod
    def find_nodes_by_properties_any_type(
        self, namespace: str, properties: Dict[str, Any], limit: int = 10
    ) -> List[Node]:
        """Find nodes matching properties within namespace, ignoring node_type label."""
        pass

    # Edge operations
    @abstractmethod
    def add_edge(self, namespace: str, from_id: str, to_id: str, edge_type: str, properties: Dict = None) -> Optional[str]:
        """Add edge. Returns edge ID, or None if either endpoint not found."""
        pass

    @abstractmethod
    def get_edge(self, namespace: str, edge_id: str) -> Optional[Edge]:
        """Get edge by ID."""
        pass

    @abstractmethod
    def update_edge(self, namespace: str, edge_id: str, properties: Dict[str, Any]) -> None:
        """Update edge properties (merge into existing)."""
        pass

    @abstractmethod
    def delete_edge(self, namespace: str, edge_id: str) -> None:
        """Delete edge."""
        pass

    # Traversal operations
    @abstractmethod
    def get_neighbors(self, namespace: str, node_id: str, edge_type: str = None, direction: str = "out") -> List[Node]:
        """Get neighboring nodes."""
        pass

    @abstractmethod
    def traverse(self, namespace: str, start_id: str, pattern: TraversalPattern) -> GraphResult:
        """Traverse graph from start node."""
        pass

    @abstractmethod
    def query(self, query: GraphQuery) -> GraphResult:
        """Execute graph query."""
        pass
