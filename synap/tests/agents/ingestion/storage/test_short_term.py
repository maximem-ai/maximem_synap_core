"""Tests for Short-term Storage."""

from uuid import uuid4
from datetime import datetime, timedelta

import pytest

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.storage.short_term import ShortTermStorage
from synap.providers.rdbms.models import SQLQuery


class MockRDBMSWrapper:
    """Mock RDBMS wrapper."""

    def __init__(self):
        self.executions = []
        self.fetch_all_results = []

    async def execute(self, query):
        self.executions.append(query)

    async def fetch_all(self, query):
        self.executions.append(query)
        # Return mock results
        return self.fetch_all_results


class MockVectorWrapper:
    """Mock Vector wrapper."""

    def __init__(self):
        self.namespaces = []
        self.provider = MockVectorProvider()
        self._provider = self.provider

    async def ensure_namespace(self, scope):
        self.namespaces.append(scope)

    async def delete(self, ids, scope):
        return len(ids)


class MockVectorProvider:
    """Mock Vector provider."""

    def __init__(self):
        self.namespaces = []
        self.upserts = []
        self.deletes = []

    def create_namespace(self, namespace, dimension):
        self.namespaces.append((namespace, dimension))

    def upsert(self, namespace, documents):
        self.upserts.append((namespace, documents))

    def delete(self, namespace, ids):
        self.deletes.append((namespace, ids))


class MockEmbeddingClient:
    """Mock Embedding client."""

    def __init__(self):
        self.calls = []

    async def embed(self, text):
        self.calls.append(text)
        return [0.1] * 1536


@pytest.fixture
def scope():
    """Create a test scope."""
    return Scope(
        client_id=uuid4(),
        instance_id=uuid4(),
        customer_id=uuid4(),
        user_id=uuid4(),
    )


@pytest.fixture
def storage():
    """Create short-term storage with mocks."""
    rdbms = MockRDBMSWrapper()
    vector = MockVectorWrapper()
    embedding = MockEmbeddingClient()
    
    return ShortTermStorage(rdbms, vector, embedding), rdbms, vector, embedding


class TestShortTermPersistence:
    """Tests for short-term context persistence."""

    @pytest.mark.asyncio
    async def test_persist_basic(self, scope, storage):
        """Test basic short-term context persistence."""
        short_term, rdbms, vector, embedding = storage

        memory_id = await short_term.persist(
            content="Test short-term context",
            scope=scope,
            session_id="session-123",
        )

        assert memory_id is not None
        assert len(rdbms.executions) == 1  # RDBMS insert
        assert len(embedding.calls) == 1  # Embedding generation
        assert len(vector.provider.upserts) == 1  # Vector upsert

    @pytest.mark.asyncio
    async def test_persist_with_qualification_result(self, scope, storage):
        """Test persistence with qualification result."""
        short_term, rdbms, _, _ = storage

        qualification_result = {"document_type": "conversation", "should_process": True}

        memory_id = await short_term.persist(
            content="Test context",
            scope=scope,
            session_id="session-123",
            qualification_result=qualification_result,
        )

        assert memory_id is not None
        # Check that qualification result was included in query
        query = rdbms.executions[0]
        assert "qualification_result" in query.sql

    @pytest.mark.asyncio
    async def test_default_ttl(self, scope, storage):
        """Test that default TTL is applied."""
        short_term, rdbms, _, _ = storage

        await short_term.persist(
            content="Test context",
            scope=scope,
            session_id="session-123",
        )

        # Check that expires_at is set
        query = rdbms.executions[0]
        assert "expires_at" in query.sql
        # TTL should be approximately 24 hours from now
        # (exact verification would require parsing the query params)


class TestShortTermRetrieval:
    """Tests for short-term context retrieval."""

    @pytest.mark.asyncio
    async def test_get_by_session(self, scope, storage):
        """Test retrieving context by session."""
        short_term, rdbms, _, _ = storage

        # Mock some results
        rdbms.fetch_all_results = [
            {"id": str(uuid4()), "content": "Context 1", "created_at": datetime.utcnow()},
            {"id": str(uuid4()), "content": "Context 2", "created_at": datetime.utcnow()},
        ]

        results = await short_term.get_by_session(
            session_id="session-123",
            scope=scope,
        )

        assert len(results) == 2
        assert len(rdbms.executions) == 1
        
        # Check that query filters by session and expires_at
        query = rdbms.executions[0]
        assert "session_id" in query.sql
        assert "expires_at" in query.sql

    @pytest.mark.asyncio
    async def test_get_by_session_filters_expired(self, scope, storage):
        """Test that expired context is filtered out."""
        short_term, rdbms, _, _ = storage

        rdbms.fetch_all_results = []

        results = await short_term.get_by_session(
            session_id="session-123",
            scope=scope,
        )

        # Check that query includes expires_at > NOW()
        query = rdbms.executions[0]
        assert "expires_at" in query.sql
        assert ">" in query.sql


class TestShortTermCleanup:
    """Tests for short-term context cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, storage):
        """Test cleanup of expired context."""
        short_term, rdbms, _, _ = storage

        # cleanup_expired first fetches expired rows (for vector cleanup),
        # then issues the DELETE. Provide expired rows so it proceeds past
        # the early-return-on-empty guard.
        rdbms.fetch_all_results = [
            {"id": str(uuid4()), "client_id": str(uuid4()), "customer_id": str(uuid4())},
            {"id": str(uuid4()), "client_id": str(uuid4()), "customer_id": str(uuid4())},
        ]

        count = await short_term.cleanup_expired()

        assert count == 2
        # One SELECT (fetch expired ids) + one DELETE.
        assert len(rdbms.executions) == 2

        # Check that DELETE query targets expired records
        fetch_query, delete_query = rdbms.executions
        assert "SELECT" in fetch_query.sql
        assert "DELETE" in delete_query.sql
        assert "expires_at" in delete_query.sql

    @pytest.mark.asyncio
    async def test_extend_ttl(self, scope, storage):
        """Test extending TTL for a session."""
        short_term, rdbms, _, _ = storage

        await short_term.extend_ttl(
            session_id="session-123",
            scope=scope,
            additional_hours=48,
        )

        assert len(rdbms.executions) == 1
        
        # Check that UPDATE query sets new expires_at
        query = rdbms.executions[0]
        assert "UPDATE" in query.sql
        assert "expires_at" in query.sql
        assert "session_id" in query.sql

    @pytest.mark.asyncio
    async def test_delete_session(self, scope, storage):
        """Test immediate deletion of session context."""
        short_term, rdbms, vector, _ = storage

        # Mock some existing records
        rdbms.fetch_all_results = [
            {"id": str(uuid4()), "content": "Context 1"},
            {"id": str(uuid4()), "content": "Context 2"},
        ]

        count = await short_term.delete_session(
            session_id="session-123",
            scope=scope,
        )

        assert count == 2  # Two records deleted
        # Should have: 1 fetch_all + 1 delete + vector deletion
        assert len(rdbms.executions) >= 2


class TestShortTermEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_session(self, scope, storage):
        """Test retrieving from empty session."""
        short_term, rdbms, _, _ = storage

        rdbms.fetch_all_results = []

        results = await short_term.get_by_session(
            session_id="nonexistent-session",
            scope=scope,
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_delete_empty_session(self, scope, storage):
        """Test deleting empty session."""
        short_term, rdbms, _, _ = storage

        rdbms.fetch_all_results = []

        count = await short_term.delete_session(
            session_id="nonexistent-session",
            scope=scope,
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_persist_scope_without_user_id(self, storage):
        """Test persistence with None user_id."""
        short_term, rdbms, _, _ = storage

        scope = Scope(
            client_id=uuid4(),
            instance_id=uuid4(),
            customer_id=uuid4(),
            user_id=None,  # No user_id
        )

        memory_id = await short_term.persist(
            content="Test context",
            scope=scope,
            session_id="session-123",
        )

        assert memory_id is not None
        # Should still persist successfully


class TestShortTermNamespacing:
    """Tests for namespace handling."""

    @pytest.mark.asyncio
    async def test_separate_namespace_for_shortterm(self, scope, storage):
        """Test that short-term uses separate vector namespace."""
        short_term, _, vector, _ = storage

        await short_term.persist(
            content="Test context",
            scope=scope,
            session_id="session-123",
        )

        # Check that short-term uses its own dedicated namespace suffix,
        # distinct from the main memories collection (now `_st_v2`).
        namespace = vector.provider.upserts[0][0]
        assert namespace.endswith("_st_v2")

    @pytest.mark.asyncio
    async def test_namespace_includes_client_id(self, scope, storage):
        """Test that namespace is tenant-scoped.

        The short-term vector namespace is now keyed by customer_id
        (`synap_{customer_id}_st_v2`), matching the main collections' scoping.
        """
        short_term, _, vector, _ = storage

        await short_term.persist(
            content="Test context",
            scope=scope,
            session_id="session-123",
        )

        namespace = vector.provider.upserts[0][0]
        assert "synap_" in namespace
        assert str(scope.customer_id) in namespace
