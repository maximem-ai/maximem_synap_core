"""
Shared fixtures for MVP user story integration tests.

Reuses patterns from existing conftest files:
- Dashboard conftest: DB session, HTTP client, auth fixtures
- Ingestion conftest: fakeredis, mock storage, MACA config
- Listening/Retrieval conftest: mock vector/graph/cache stores
"""

import asyncio
import os
import sys
import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

# Add SDK to Python path (same pattern as synap/tests/sdk/ conftest)
_SDK_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sdk", "python")
if _SDK_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(_SDK_PATH))

import fakeredis.aioredis


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "gap: test for a story with known implementation gaps")
    config.addinivalue_line("markers", "e2e: full end-to-end flow test")


# ---------------------------------------------------------------------------
# Event loop (session scope)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Database session (AsyncMock — no real DB needed for unit-style E2E)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_session():
    """AsyncMock session suitable for all manager constructors."""
    return AsyncMock()


# ---------------------------------------------------------------------------
# Redis (fakeredis)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def mock_redis():
    """In-memory Redis replacement."""
    redis = fakeredis.aioredis.FakeRedis()
    yield redis
    await redis.aclose()


# ---------------------------------------------------------------------------
# IDs & Common Data
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client_id():
    return "cli_" + uuid4().hex[:16]


@pytest.fixture
def test_instance_id():
    return "inst_" + uuid4().hex[:16]


@pytest.fixture
def test_user_id():
    return "user_" + uuid4().hex[:12]


@pytest.fixture
def test_conversation_id():
    return "conv_" + uuid4().hex[:12]


# ---------------------------------------------------------------------------
# ORM record factories (MagicMock-based, matching existing test patterns)
# ---------------------------------------------------------------------------

@pytest.fixture
def make_client_orm():
    """Factory for mock ClientORM records."""
    def _factory(
        client_id="cli_test1234567890",
        name="Test Corp",
        status=None,
        context=None,
    ):
        record = MagicMock()
        record.client_id = client_id
        record.client_name = name
        record.client_website = "https://test.com"
        record.client_short_description = "A test client"
        record.client_context = context or {"org_id": "org_test"}
        # Use string for status to avoid importing ORM enums
        record.status = status or MagicMock(value="active")
        record.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        record.updated_at = datetime(2025, 1, 2, tzinfo=timezone.utc)
        return record
    return _factory


@pytest.fixture
def make_instance_orm():
    """Factory for mock InstanceORM records."""
    def _factory(
        instance_id="inst_test1234567890",
        client_id="cli_test1234567890",
        name="Test Agent",
        agent_type="sdk",
        status=None,
        metadata=None,
        parent_instance_id=None,
    ):
        record = MagicMock()
        record.instance_id = instance_id
        record.client_id = client_id
        record.instance_name = name
        record.agent_type = agent_type
        record.status = status or MagicMock(value="active")
        record.instance_metadata = metadata or {"description": "A test instance"}
        record.parent_instance_id = parent_instance_id
        record.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        record.updated_at = datetime(2025, 1, 2, tzinfo=timezone.utc)
        return record
    return _factory


# ---------------------------------------------------------------------------
# Mock vector/graph/cache stores (from listening_retrieval conftest pattern)
# ---------------------------------------------------------------------------

class MockVectorStore:
    """In-memory vector store for testing."""

    def __init__(self, data: Optional[List[Dict]] = None):
        self.data = data or []
        self.queries_received: List[Dict] = []

    async def search(self, query: str, top_k: int = 10,
                     filters: Optional[Dict] = None) -> List[Dict]:
        self.queries_received.append({"query": query, "top_k": top_k, "filters": filters})
        results = []
        for i, item in enumerate(self.data[:top_k]):
            results.append({**item, "similarity": 0.95 - (i * 0.05)})
        return results

    async def query(self, query: str, collection: str = None,
                    filter: Optional[Dict] = None, top_k: int = 50) -> List[Dict]:
        self.queries_received.append({"query": query, "collection": collection})
        results = []
        for i, item in enumerate(self.data[:top_k]):
            results.append({
                "id": item.get("id", f"vec_{i}"),
                "content": item.get("content", ""),
                "score": 0.95 - (i * 0.05),
                "metadata": {
                    "context_type": item.get("context_type", "facts"),
                    "confidence": item.get("confidence", 0.9),
                    "scope": item.get("scope", "user"),
                },
            })
        return results


class MockGraphStore:
    """In-memory graph store for testing."""

    def __init__(self, nodes: Optional[List[Dict]] = None,
                 edges: Optional[List[Dict]] = None):
        self.nodes = nodes or []
        self.edges = edges or []
        self.traversals: List = []

    async def traverse(self, start_nodes: List[str],
                       edge_types: Optional[List[str]] = None,
                       max_depth: int = 2) -> List[Dict]:
        self.traversals.append({"start_nodes": start_nodes, "max_depth": max_depth})
        return self.nodes[:5]

    async def get_neighbors(self, node_id: str) -> List[Dict]:
        return [e for e in self.edges if e.get("source_id") == node_id]

    async def get_nodes(self, filter: Optional[Dict] = None) -> List[Dict]:
        return self.nodes

    async def get_edges(self, filter: Optional[Dict] = None) -> List[Dict]:
        return self.edges

    async def count(self, filter: Optional[Dict] = None) -> int:
        return len(self.nodes)


class MockCacheProvider:
    """In-memory cache for testing."""

    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.gets: List[str] = []
        self.sets: List[Dict] = []

    async def get(self, key: str) -> Optional[Any]:
        self.gets.append(key)
        return self.cache.get(key)

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        self.sets.append({"key": key, "ttl": ttl})
        self.cache[key] = value

    async def delete(self, key: str) -> None:
        self.cache.pop(key, None)


class MockPostgresProvider:
    """In-memory postgres provider for short-term context."""

    def __init__(self):
        self.short_term_data: List[Dict] = []
        self.queries: List = []

    async def fetch_short_term(self, user_id: str, instance_id: str,
                               time_window: timedelta) -> List[Dict]:
        self.queries.append({"user_id": user_id, "instance_id": instance_id})
        cutoff = datetime.now(timezone.utc) - time_window
        return [
            item for item in self.short_term_data
            if item.get("created_at", datetime.now(timezone.utc)) > cutoff
        ]

    async def persist_short_term(self, items: List[Dict]) -> None:
        self.short_term_data.extend(items)


class MockStorageCoordinator:
    """Mock multi-store coordinator for ingestion Stage4."""

    def __init__(self):
        self.persisted_memories: List = []
        self.persisted_entities: List = []
        self.persisted_relationships: List = []

    async def persist_memories(self, memories, scope, document_id):
        self.persisted_memories.extend(memories)

    async def persist_entities(self, entities, scope, document_id):
        self.persisted_entities.extend(entities)

    async def persist_relationships(self, relationships, scope, document_id):
        self.persisted_relationships.extend(relationships)

    async def persist_rdbms(self, memories=None, candidates=None, entities=None,
                            scope=None, document_id=None, client_knowledge=None,
                            customer_knowledge=None):
        if memories:
            self.persisted_memories.extend(memories)
        result = MagicMock()
        result.memory_ids = [str(uuid4()) for _ in (memories or [])]
        result.candidate_ids = []
        result.entity_ids = []
        return result

    async def persist_graph(self, entities=None, relationships=None,
                            memory_links=None, scope=None):
        if entities:
            self.persisted_entities.extend(entities)
        if relationships:
            self.persisted_relationships.extend(relationships)
        result = MagicMock()
        result.entity_ids = [str(uuid4()) for _ in (entities or [])]
        result.relationship_ids = [str(uuid4()) for _ in (relationships or [])]
        return result

    async def persist_vector(self, memories=None, entities=None, scope=None):
        result = MagicMock()
        result.vector_ids = [str(uuid4()) for _ in (memories or [])]
        return result


class MockStorageProvider:
    """Mock S3 storage provider for ingestion."""

    def __init__(self):
        self.persist_original_calls: List = []
        self.persist_short_term_calls: List = []

    async def persist_original(self, **kwargs):
        self.persist_original_calls.append(kwargs)

    async def persist_short_term(self, **kwargs):
        self.persist_short_term_calls.append(kwargs)

    async def get_short_term_by_session(self, **kwargs):
        return []


# ---------------------------------------------------------------------------
# Mock store fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_vector_data():
    """Pre-loaded vector store data."""
    return [
        {"id": "vec_1", "content": "User prefers dark mode in all applications",
         "context_type": "preferences", "user_id": "user_456", "confidence": 0.95, "scope": "user"},
        {"id": "vec_2", "content": "User works at Acme Corp as a senior engineer",
         "context_type": "facts", "user_id": "user_456", "confidence": 0.92, "scope": "user"},
        {"id": "vec_3", "content": "User had a meeting about API design on Tuesday",
         "context_type": "episodes", "user_id": "user_456", "confidence": 0.88, "scope": "user"},
    ]


@pytest.fixture
def sample_graph_nodes():
    return [
        {"id": "node_1", "type": "entity", "content": "Acme Corp - software company",
         "context_type": "facts", "confidence": 0.9, "scope": "user"},
        {"id": "node_2", "type": "episode", "content": "User discussed API redesign on Monday",
         "context_type": "episodes", "confidence": 0.85, "scope": "user"},
    ]


@pytest.fixture
def sample_graph_edges():
    return [
        {"source_id": "node_1", "target_id": "node_2", "edge_type": "related_to",
         "is_memory": True, "edge_vector": [0.5] * 384, "target_vector": [0.6] * 384},
    ]


@pytest.fixture
def mock_vector_store(sample_vector_data):
    return MockVectorStore(data=sample_vector_data)


@pytest.fixture
def mock_graph_store(sample_graph_nodes, sample_graph_edges):
    return MockGraphStore(nodes=sample_graph_nodes, edges=sample_graph_edges)


@pytest.fixture
def mock_cache():
    return MockCacheProvider()


@pytest.fixture
def mock_postgres():
    return MockPostgresProvider()


@pytest.fixture
def mock_storage_coordinator():
    return MockStorageCoordinator()


@pytest.fixture
def mock_storage_provider():
    return MockStorageProvider()


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """AsyncMock LLM returning configurable JSON responses."""
    llm = AsyncMock()
    llm.complete.return_value = {
        "content": '{"summary": "test summary", "facts": ["fact1"], "score": 0.9}'
    }
    return llm


# ---------------------------------------------------------------------------
# Scope fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_scope():
    """ScopeHierarchy for testing."""
    from synap.cloud.shared.scope import ScopeHierarchy
    return ScopeHierarchy(
        instance_id="inst_test1234567890",
        client_id="cli_test1234567890",
        customer_id="cust_789",
        user_id="user_456",
        internal_user_id="internal_user_xyz",
        internal_customer_id="internal_cust_def",
    )


# ---------------------------------------------------------------------------
# Sample conversation content
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_conversation():
    return (
        "User: How do I authenticate with the API?\n"
        "Assistant: You can use API keys or OAuth2. API keys go in the Authorization header.\n"
        "User: What about rate limits?\n"
        "Assistant: The default rate limit is 100 requests per minute per API key.\n"
        "User: I prefer using OAuth2 for production. Can you help set that up?\n"
        "Assistant: Sure! First, register your application to get a client_id and client_secret."
    )


@pytest.fixture
def sample_long_conversation():
    """Conversation that exceeds compaction threshold (3000 tokens ~= 20+ messages)."""
    messages = []
    for i in range(25):
        messages.append(f"User: Question {i} about feature implementation and best practices?")
        messages.append(
            f"Assistant: Here is a detailed answer {i} covering architecture patterns, "
            f"code examples, testing strategies, and deployment considerations for this topic."
        )
    return "\n".join(messages)
