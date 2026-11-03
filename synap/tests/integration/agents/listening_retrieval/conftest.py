"""
Shared fixtures for Listening & Retrieval tests.

Provides mock storage providers, sample data, and test utilities.
"""

import pytest
import asyncio
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
import uuid
import numpy as np

from synap.cloud.agents.models.decisions import (
    ContextDecision,
    RetrievalMode,
    TriggerType,
)
from synap.cloud.agents.models.bundles import (
    ContextBundle,
    ContextItem,
    DegradationLevel,
)
from synap.cloud.shared.scope import ScopeHierarchy


# === Mock Storage Providers ===

class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self, data: Optional[List[Dict]] = None):
        self.data = data or []
        self.queries_received = []

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Mock vector search."""
        self.queries_received.append({
            "query": query,
            "top_k": top_k,
            "filters": filters,
        })

        # Return mock results with similarity scores
        results = []
        for i, item in enumerate(self.data[:top_k]):
            results.append({
                **item,
                "similarity": 0.9 - (i * 0.05),  # Decreasing similarity
            })
        return results

    async def query(
        self,
        query: str,
        collection: str = None,
        filter: Optional[Dict] = None,
        top_k: int = 50,
    ) -> List[Dict]:
        """Mock query method for RetrievalManager compatibility."""
        self.queries_received.append({
            "query": query,
            "collection": collection,
            "filter": filter,
            "top_k": top_k,
        })

        # Return mock results in expected format
        results = []
        for i, item in enumerate(self.data[:top_k]):
            results.append({
                "id": item.get("id", f"vec_{i}"),
                "content": item.get("content", ""),
                "score": 0.9 - (i * 0.05),
                "metadata": {
                    "context_type": item.get("context_type", "facts"),
                    "confidence": item.get("confidence", 0.8),
                    "scope": item.get("scope", "user"),
                    "entity_id": item.get("entity_id"),
                },
            })
        return results

    async def embed(self, text: str) -> List[float]:
        """Deterministic embedding from text hash."""
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(384).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class MockGraphStore:
    """Mock graph store for testing."""

    def __init__(self, nodes: Optional[List[Dict]] = None, edges: Optional[List[Dict]] = None):
        self.nodes = nodes or []
        self.edges = edges or []
        self.traversals = []

    async def traverse(
        self,
        start_nodes: List[str],
        edge_types: Optional[List[str]] = None,
        max_depth: int = 2,
    ) -> List[Dict]:
        """Mock graph traversal."""
        self.traversals.append({
            "start_nodes": start_nodes,
            "edge_types": edge_types,
            "max_depth": max_depth,
        })

        # Return connected nodes
        return self.nodes[:5]

    async def get_neighbors(self, node_id: str) -> List[Dict]:
        """Get neighbors of a node."""
        return [e for e in self.edges if e.get("source") == node_id]

    async def get_nodes(self, filter: Optional[Dict] = None) -> List[Dict]:
        """Get nodes with optional filter."""
        return self.nodes

    async def get_edges(self, filter: Optional[Dict] = None) -> List[Dict]:
        """Get edges with optional filter."""
        return self.edges

    async def count(self, filter: Optional[Dict] = None) -> int:
        """Get node count."""
        return len(self.nodes)


class MockCacheProvider:
    """Mock cache provider for testing."""

    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.gets = []
        self.sets = []

    async def get(self, key: str) -> Optional[Any]:
        """Get from cache."""
        self.gets.append(key)
        return self.cache.get(key)

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set in cache."""
        self.sets.append({"key": key, "ttl": ttl})
        self.cache[key] = value

    async def delete(self, key: str) -> None:
        """Delete from cache."""
        self.cache.pop(key, None)


class MockPostgresProvider:
    """Mock Postgres provider for short-term storage."""

    def __init__(self):
        self.short_term_data: List[Dict] = []
        self.queries = []

    async def fetch_short_term(
        self,
        user_id: str,
        instance_id: str,
        time_window: timedelta,
    ) -> List[Dict]:
        """Fetch recent short-term context."""
        self.queries.append({
            "user_id": user_id,
            "instance_id": instance_id,
            "time_window": time_window,
        })

        cutoff = datetime.utcnow() - time_window
        return [
            item for item in self.short_term_data
            if item.get("created_at", datetime.utcnow()) > cutoff
        ]

    async def persist_short_term(self, items: List[Dict]) -> None:
        """Persist items to short-term storage."""
        self.short_term_data.extend(items)

    def fetch_all(self, query) -> List[Dict]:
        """Synchronous fetch for compatibility."""
        # Mock synchronous query execution
        return []


class MockLLMAdapter:
    """Mock LLM adapter for subquery generation."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.calls = []

    async def complete(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """Mock LLM completion."""
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
        })

        # Default subquery response
        content = self.responses.get("subquery", '''{
            "rank_1": "What are the user's preferences?",
            "rank_2": "What relevant facts exist?",
            "rank_3": "What past interactions are relevant?",
            "rank_4": "What customer context applies?"
        }''')

        return {"content": content}


# === Fixtures ===

@pytest.fixture
def sample_context_items() -> List[ContextItem]:
    """Sample context items for testing."""
    return [
        ContextItem(
            item_id="item_1",
            content="User prefers vegetarian food",
            context_type="preferences",
            source="vector",
            similarity_score=0.92,
            confidence=0.85,
        ),
        ContextItem(
            item_id="item_2",
            content="User is allergic to peanuts",
            context_type="facts",
            source="vector",
            similarity_score=0.88,
            confidence=0.95,
        ),
        ContextItem(
            item_id="item_3",
            content="User ordered Thai food last Tuesday",
            context_type="episodes",
            source="graph",
            similarity_score=0.75,
            confidence=0.80,
        ),
        ContextItem(
            item_id="item_4",
            content="User likes spicy food",
            context_type="preferences",
            source="vector",
            similarity_score=0.85,
            confidence=0.78,
        ),
        ContextItem(
            item_id="item_5",
            content="User's usual budget is $15-25",
            context_type="facts",
            source="vector",
            similarity_score=0.82,
            confidence=0.90,
        ),
    ]


@pytest.fixture
def sample_vector_data() -> List[Dict]:
    """Sample vector store data."""
    return [
        {
            "id": "vec_1",
            "content": "User prefers vegetarian food",
            "context_type": "preferences",
            "user_id": "user_456",
            "confidence": 0.85,
            "scope": "user",
        },
        {
            "id": "vec_2",
            "content": "User is allergic to peanuts",
            "context_type": "facts",
            "user_id": "user_456",
            "confidence": 0.95,
            "scope": "user",
        },
        {
            "id": "vec_3",
            "content": "User likes spicy food",
            "context_type": "preferences",
            "user_id": "user_456",
            "confidence": 0.78,
            "scope": "user",
        },
    ]


@pytest.fixture
def sample_graph_nodes() -> List[Dict]:
    """Sample graph nodes."""
    return [
        {
            "id": "node_1",
            "type": "entity",
            "name": "Thai Basil Restaurant",
            "properties": {"cuisine": "Thai"},
            "content": "Thai Basil Restaurant serves Thai cuisine",
            "context_type": "facts",
            "confidence": 0.8,
            "scope": "user",
        },
        {
            "id": "node_2",
            "type": "episode",
            "content": "User ordered Pad Thai on Tuesday",
            "timestamp": datetime.utcnow() - timedelta(days=5),
            "context_type": "episodes",
            "confidence": 0.75,
            "scope": "user",
        },
    ]


@pytest.fixture
def sample_graph_edges() -> List[Dict]:
    """Sample graph edges."""
    return [
        {
            "source_id": "node_1",
            "target_id": "node_2",
            "edge_type": "memory",
            "is_memory": True,
            "edge_vector": [0.5] * 384,
            "target_vector": [0.6] * 384,
        }
    ]


@pytest.fixture
def mock_vector_store(sample_vector_data) -> MockVectorStore:
    """Create mock vector store with sample data."""
    return MockVectorStore(data=sample_vector_data)


@pytest.fixture
def mock_graph_store(sample_graph_nodes, sample_graph_edges) -> MockGraphStore:
    """Create mock graph store with sample data."""
    return MockGraphStore(nodes=sample_graph_nodes, edges=sample_graph_edges)


@pytest.fixture
def mock_cache() -> MockCacheProvider:
    """Create mock cache provider."""
    return MockCacheProvider()


@pytest.fixture
def mock_postgres() -> MockPostgresProvider:
    """Create mock Postgres provider."""
    return MockPostgresProvider()


@pytest.fixture
def mock_llm() -> MockLLMAdapter:
    """Create mock LLM adapter."""
    return MockLLMAdapter()


@pytest.fixture
def sample_scope() -> ScopeHierarchy:
    """Sample resolved scope."""
    return ScopeHierarchy(
        instance_id="inst_123",
        client_id="client_abc",
        customer_id="cust_789",
        user_id="user_456",
        internal_user_id="internal_user_xyz",
        internal_customer_id="internal_cust_def",
    )


@pytest.fixture
def sample_decision() -> ContextDecision:
    """Sample context decision (FAST mode)."""
    return ContextDecision(
        decision_id=str(uuid.uuid4()),
        trigger_type=TriggerType.EXPLICIT_REQUEST,
        instance_id="inst_123",
        user_id="user_456",
        retrieval_mode=RetrievalMode.FAST,
        context_types=["facts", "preferences"],
        search_queries=["food preferences"],
        token_budget=2000,
    )


@pytest.fixture
def accurate_decision() -> ContextDecision:
    """Sample ACCURATE mode decision."""
    return ContextDecision(
        decision_id=str(uuid.uuid4()),
        trigger_type=TriggerType.EXPLICIT_REQUEST,
        instance_id="inst_123",
        user_id="user_456",
        retrieval_mode=RetrievalMode.ACCURATE,
        context_types=["facts", "preferences", "episodes"],
        search_queries=["recommend lunch options"],
        token_budget=3000,
    )


@pytest.fixture
def sample_auth_context() -> Dict[str, Any]:
    """Sample auth context from Gateway."""
    return {
        "instance_id": "inst_123",
        "client_id": "client_abc",
        "user_id": "user_456",
        "permissions": ["read:context", "write:context"],
    }


# === Mock Compaction ===

class MockCompactionStorage:
    """In-memory compaction storage for testing."""

    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.locks: set = set()
        self.messages: Dict[str, List] = {}

    async def get_from_cache(self, conversation_id: str, instance_id: str):
        key = f"{conversation_id}:{instance_id}"
        return self.cache.get(key)

    async def is_compaction_in_progress(self, conversation_id: str) -> bool:
        return conversation_id in self.locks

    async def load_conversation_messages(self, conversation_id: str) -> List:
        return self.messages.get(conversation_id, [])

    async def set_cache(self, conversation_id: str, instance_id: str, value: Any):
        key = f"{conversation_id}:{instance_id}"
        self.cache[key] = value


class MockCompactionManager:
    """Mock compaction manager that records calls but returns None."""

    def __init__(self):
        self.calls: List[Dict] = []

    async def get_or_create(self, **kwargs):
        self.calls.append(kwargs)
        return None


@pytest.fixture
def mock_compaction_storage() -> MockCompactionStorage:
    """Create mock compaction storage."""
    return MockCompactionStorage()


@pytest.fixture
def mock_compaction_manager() -> MockCompactionManager:
    """Create mock compaction manager."""
    return MockCompactionManager()


# === Graph Config for DI ===

@pytest.fixture
def graph_config():
    """Build a RunnableConfig with test dependencies.

    Tests that invoke the graph should pass this config:
        result = await graph.ainvoke(state, config=graph_config)
    """
    return {
        "configurable": {
            "deps": {
                "retrieval_manager": None,
                "redis_client": None,
                "llm_adapter": None,
                "agent": None,
            }
        }
    }
