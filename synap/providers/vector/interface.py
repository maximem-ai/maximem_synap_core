"""Base interface for vector databases."""

from abc import ABC, abstractmethod
from typing import List, Optional
from .models import VectorQuery, VectorDocument, VectorSearchResult


class VectorStoreInterface(ABC):
    """
    Base interface for vector databases.
    Multi-tenancy: namespace/collection per instance_id.
    """

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    # Namespace management (tenant isolation)
    @abstractmethod
    def create_namespace(self, namespace: str, dimension: int) -> None:
        """Create namespace/collection for instance."""
        pass

    @abstractmethod
    def delete_namespace(self, namespace: str) -> None:
        """Delete namespace and all vectors."""
        pass

    # CRUD operations
    @abstractmethod
    def upsert(self, namespace: str, documents: List[VectorDocument]) -> int:
        """Upsert documents. Returns count."""
        pass

    @abstractmethod
    def delete(self, namespace: str, ids: List[str]) -> int:
        """Delete by IDs. Returns count."""
        pass

    @abstractmethod
    def search(self, query: VectorQuery) -> VectorSearchResult:
        """Similarity search."""
        pass

    @abstractmethod
    def get(self, namespace: str, ids: List[str]) -> List[VectorDocument]:
        """Get documents by ID."""
        pass

    def get_all(
        self,
        namespace: str,
        filters: Optional[dict] = None,
        limit: int = 500,
    ) -> List[VectorDocument]:
        """
        Get all documents from a namespace with optional filters.

        Returns documents WITH embeddings for building adjacency matrices.
        Default implementation returns empty list (override in adapters).
        """
        return []

    def list_namespaces(self) -> List[str]:
        """Every namespace in the store.

        Only maintenance jobs need this. Nothing on a request path should be
        asking the store what exists: a read knows its own namespace, and a
        call that enumerates thousands of collections is how a read-only audit
        script can exhaust the store's memory.

        Implementations must page. Returning names is not enough on its own if
        the store hands back objects to get them from: thousands of collection objects
        at a few MB each is what causes that, even when the caller only
        wanted the names.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot enumerate namespaces"
        )

    # --- Metadata maintenance ---
    #
    # A backfill that adds a key to millions of existing records needs to read
    # metadata without embeddings and write metadata without re-embedding.
    # `get_all` and `upsert` can both do the job and both move the vectors to do
    # it, which on a 1024-dimensional collection is three orders of magnitude
    # more bytes than the change requires.
    #
    # Both default to raising rather than to a no-op. A store that cannot do
    # this should stop a backfill loudly at the first batch, not report success
    # over records it never touched (S-09, S-13).

    def scan_metadata(
        self,
        namespace: str,
        limit: int = 500,
        offset: int = 0,
    ) -> List[dict]:
        """Page through a namespace reading ids and metadata only.

        Returns ``[{"id": ..., "metadata": {...}}, ...]``, shorter than `limit`
        at the end. Ordering must be stable across calls or a resumable job
        cannot resume: an offset into a shifting order re-reads some records and
        skips others, and skipping is silent.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot scan metadata; a metadata backfill "
            "against this store would have to move every vector to read one key"
        )

    def update_metadata(self, namespace: str, updates: dict) -> int:
        """Merge metadata into existing documents. Embeddings are untouched.

        `updates` maps document id to the keys to set. Keys not named are left
        alone, so this adds `node_id` without disturbing anything a record
        already carries. Returns the number of documents changed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot update metadata in place; a backfill "
            "would have to re-upsert every embedding to add one key"
        )
