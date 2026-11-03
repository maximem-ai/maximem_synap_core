"""Tests for vector store provider."""

import pytest
from synap.providers.vector import (
    VectorStoreInterface,
    VectorQuery,
    VectorDocument,
    VectorResult,
    VectorSearchResult,
    get_vector_store,
    reset_vector_store,
    ChromaDBAdapter,
)


def test_interface_cannot_instantiate():
    """Verify VectorStoreInterface is abstract."""
    with pytest.raises(TypeError):
        VectorStoreInterface()


def test_adapter_inherits_interface():
    """Verify ChromaDBAdapter inherits from VectorStoreInterface."""
    assert issubclass(ChromaDBAdapter, VectorStoreInterface)


@pytest.fixture
def vector():
    """Get in-memory vector store for testing."""
    reset_vector_store()
    store = get_vector_store({})  # In-memory for testing
    yield store
    reset_vector_store()


@pytest.fixture
def namespace(vector):
    """Create test namespace."""
    ns = "test_namespace"
    vector.create_namespace(ns, dimension=3)
    yield ns
    vector.delete_namespace(ns)


def test_upsert_and_get(vector, namespace):
    """Test upsert and get operations."""
    doc = VectorDocument(
        id="doc-1",
        embedding=[0.1, 0.2, 0.3],
        metadata={"key": "value"}
    )
    vector.upsert(namespace, [doc])

    result = vector.get(namespace, ["doc-1"])
    assert len(result) == 1
    assert result[0].id == "doc-1"
    assert result[0].metadata == {"key": "value"}


def test_search(vector, namespace):
    """Test similarity search."""
    # Non-empty metadata on every document: chromadb rejects an empty dict
    # outright (ValueError: "Expected metadata to be a non-empty dict"), and
    # every real caller (AsyncVectorWrapper._scope_metadata) always stamps at
    # least client_id/instance_id, so {} here would test a shape production
    # never sends.
    docs = [
        VectorDocument(id="a", embedding=[1.0, 0.0, 0.0], metadata={"label": "a"}),
        VectorDocument(id="b", embedding=[0.0, 1.0, 0.0], metadata={"label": "b"}),
        VectorDocument(id="c", embedding=[0.0, 0.0, 1.0], metadata={"label": "c"}),
    ]
    vector.upsert(namespace, docs)

    results = vector.search(VectorQuery(
        embedding=[1.0, 0.0, 0.0],
        top_k=2,
        namespace=namespace
    ))

    assert len(results.results) == 2
    assert results.results[0].id == "a"  # Most similar


def test_delete(vector, namespace):
    """Test delete operation."""
    # Non-empty metadata: see test_search for why {} is not a realistic
    # (or chromadb-accepted) shape for a document being upserted.
    doc = VectorDocument(
        id="to-delete",
        embedding=[0.1, 0.2, 0.3],
        metadata={"key": "value"}
    )
    vector.upsert(namespace, [doc])
    vector.delete(namespace, ["to-delete"])

    result = vector.get(namespace, ["to-delete"])
    assert len(result) == 0


def test_vector_query_model():
    """Verify VectorQuery model."""
    query = VectorQuery(embedding=[0.1, 0.2], top_k=5, namespace="instance_1")
    assert query.embedding == [0.1, 0.2]
    assert query.top_k == 5
    assert query.namespace == "instance_1"


def test_vector_document_model():
    """Verify VectorDocument model."""
    doc = VectorDocument(
        id="doc1",
        embedding=[0.1, 0.2],
        metadata={"type": "text"},
        content="Test content"
    )
    assert doc.id == "doc1"
    assert doc.embedding == [0.1, 0.2]
    assert doc.metadata == {"type": "text"}
    assert doc.content == "Test content"


def test_vector_result_model():
    """Verify VectorResult model."""
    result = VectorResult(
        id="doc1",
        score=0.95,
        metadata={"type": "text"},
        content="Test"
    )
    assert result.id == "doc1"
    assert result.score == 0.95


def test_vector_search_result_model():
    """Verify VectorSearchResult model."""
    result = VectorSearchResult(
        results=[],
        query_time_ms=10.5
    )
    assert result.results == []
    assert result.query_time_ms == 10.5
