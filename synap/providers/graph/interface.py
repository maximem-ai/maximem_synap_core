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

    # Lifecycle (graph records follow the memories that support them).
    # Not abstract: an adapter that predates them still loads, and a caller
    # learns it cannot release by the NotImplementedError, never by silence.

    def release_memory_ids(
        self,
        namespace: str,
        ids: List[str],
        mode: str = "retire",
        now: Optional[str] = None,
        reason: str = "superseded",
        superseded_by: Optional[str] = None,
        cap: int = 500,
    ) -> Dict[str, Any]:
        """Remove *ids* from ``memory_ids`` on every edge and node of *namespace*,
        in one write transaction.

        Each removed id is appended to ``released_memory_ids`` (newest *cap*
        kept). A record left with no ``memory_ids`` is retired
        (``retired_at``/``retired_reason``/``retired_by_memory_id``, first
        retirement wins). With ``mode="erase"`` such an edge is deleted instead,
        and every node left with no edges and no ``memory_ids`` (a touched node
        or an endpoint of a deleted edge) is DETACH DELETEd, except the writer's
        and the customer's node. Returns counts (``edges_pruned``,
        ``edges_retired``, ``edges_deleted``, ``nodes_pruned``,
        ``nodes_retired``, ``nodes_deleted``) and the touched records
        (``records``: kind, id, memory_ids, released_memory_ids, deleted,
        is_fixed_entity) plus ``deleted_edges``/``deleted_nodes`` with the
        names their vector ids are derived from.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot release memory ids")

    def set_record_states(self, namespace: str, states: List[Dict[str, Any]]) -> int:
        """Write the lifecycle state of edges and nodes in one write transaction.

        Each state is ``{"kind": "edge"|"node", "id", "open": bool, "now",
        "reason", "valid_to", "retired_by_memory_id"}``. Open removes the
        retirement properties; closed sets them (``retired_at`` keeps the first
        value). Returns how many records were written."""
        raise NotImplementedError(f"{type(self).__name__} cannot set record states")

    def erase_records(self, namespace: str, edge_ids: List[str], node_ids: List[str]) -> Dict[str, Any]:
        """Delete the given edges and nodes (DETACH), then every endpoint left
        with no edges and no ``memory_ids``; writer and customer nodes are never
        deleted. One write transaction. Returns ``deleted_edges`` and
        ``deleted_nodes`` like ``release_memory_ids``."""
        raise NotImplementedError(f"{type(self).__name__} cannot erase records")

    def list_namespaces(self) -> List[str]:
        """Every graph namespace that holds a record."""
        raise NotImplementedError(f"{type(self).__name__} cannot list namespaces")

    def lifecycle_records(self, namespace: str) -> List[Dict[str, Any]]:
        """Every edge and node of *namespace* with its lifecycle properties
        (kind, id, memory_ids, released_memory_ids, legacy_ids, is_fixed_entity,
        retired_at)."""
        raise NotImplementedError(f"{type(self).__name__} cannot list lifecycle records")

    def edges_with_memory_contents(self, namespace: str, ids: List[str]) -> List[Dict[str, Any]]:
        """Copies mode: the relationships whose ``memory_contents`` names one of *ids*."""
        raise NotImplementedError(f"{type(self).__name__} cannot search memory_contents")
