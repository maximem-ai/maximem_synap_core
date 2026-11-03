"""
Real Database E2E Tests for the Retrieval Pipeline.

Tests RetrievalManager against real ChromaDB, Neo4j, and OpenAI embeddings.
Requires CHROMA_HOST, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY env vars.
Skipped automatically when env vars are absent.

Uses httpx for ChromaDB REST API (the chromadb Python client is broken on Python 3.14).

Usage:
    set -a && source .env.staging && set +a
    python -m pytest synap/tests/integration/agents/listening_retrieval/test_real_db_e2e.py -v
"""

import os
import asyncio
import uuid
import numpy as np
import pytest
import httpx
from typing import Dict, Any, List, Optional

# Load .env.staging if present (for local runs)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..", ".env.staging"
    )
    load_dotenv(_env_path, override=False)
except ImportError:
    pass

# --- Environment checks ---
CHROMA_HOST = os.environ.get("CHROMA_HOST", "")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

requires_real_dbs = pytest.mark.skipif(
    not all([CHROMA_HOST, NEO4J_URI, OPENAI_API_KEY]),
    reason="Real DB integration: CHROMA_HOST, NEO4J_URI, OPENAI_API_KEY required",
)

# Unique namespace per test session to avoid collisions
TEST_INSTANCE_ID = f"inst_test_{uuid.uuid4().hex[:12]}"
COLLECTION_NAME = f"context_{TEST_INSTANCE_ID}"


# ============================================================
# ChromaDB HTTP Client (bypasses broken chromadb Python package)
# ============================================================

class ChromaHTTPClient:
    """Minimal ChromaDB v2 REST API client using httpx."""

    def __init__(self, host: str, port: int):
        self._base = f"http://{host}:{port}/api/v2/tenants/default_tenant/databases/default_database"
        self._client = httpx.Client(timeout=30.0)
        self._col_id_cache: Dict[str, str] = {}

    def create_collection(self, name: str, metadata: Dict = None):
        resp = self._client.post(
            f"{self._base}/collections",
            json={"name": name, "metadata": metadata or {}},
        )
        resp.raise_for_status()
        data = resp.json()
        self._col_id_cache[name] = data["id"]
        return data

    def _get_col_id(self, name: str) -> str:
        if name in self._col_id_cache:
            return self._col_id_cache[name]
        resp = self._client.get(f"{self._base}/collections/{name}")
        resp.raise_for_status()
        col_id = resp.json()["id"]
        self._col_id_cache[name] = col_id
        return col_id

    def delete_collection(self, name: str):
        resp = self._client.delete(f"{self._base}/collections/{name}")
        if resp.status_code != 404:
            resp.raise_for_status()
        self._col_id_cache.pop(name, None)

    def upsert(self, collection_name: str, ids: List[str], embeddings: List[List[float]],
               metadatas: List[Dict], documents: List[str]):
        col_id = self._get_col_id(collection_name)
        resp = self._client.post(
            f"{self._base}/collections/{col_id}/add",
            json={
                "ids": ids,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "documents": documents,
            },
        )
        resp.raise_for_status()

    def query(self, collection_name: str, query_embedding: List[float],
              n_results: int = 10, where: Dict = None) -> Dict:
        col_id = self._get_col_id(collection_name)
        body = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            body["where"] = where
        resp = self._client.post(
            f"{self._base}/collections/{col_id}/query",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()


# ============================================================
# Async Provider Wrappers
# ============================================================

class RealVectorProvider:
    """Async wrapper: OpenAI embeddings + ChromaDB HTTP API → RetrievalManager interface."""

    def __init__(self, chroma_client: ChromaHTTPClient, openai_api_key: str,
                 model: str = "text-embedding-3-small"):
        self._chroma = chroma_client
        self._api_key = openai_api_key
        self._model = model
        self._openai_client = None

    def _get_openai(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._openai_client

    async def embed(self, text: str) -> List[float]:
        """Generate real OpenAI embedding."""
        client = self._get_openai()
        response = await client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    @staticmethod
    def _normalize_where(filter: Optional[Dict]) -> Optional[Dict]:
        """Convert multi-key where dicts to explicit $and (required by ChromaDB v2)."""
        if not filter:
            return filter
        # If multiple top-level keys that aren't operators, wrap in $and
        non_op_keys = [k for k in filter if not k.startswith("$")]
        if len(non_op_keys) > 1:
            return {"$and": [{k: filter[k]} for k in non_op_keys]}
        return filter

    async def query(
        self,
        query: str,
        collection: str = None,
        filter: Optional[Dict] = None,
        top_k: int = 50,
    ) -> List[Dict]:
        """String query → embedding → ChromaDB search → List[Dict]."""
        embedding = await self.embed(query)
        where = self._normalize_where(filter)

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, self._chroma.query, collection, embedding, top_k, where
        )

        results = []
        if raw.get("ids") and raw["ids"][0]:
            ids = raw["ids"][0]
            distances = raw["distances"][0] if raw.get("distances") else [0] * len(ids)
            metadatas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(ids)
            documents = raw["documents"][0] if raw.get("documents") else [""] * len(ids)

            for i, doc_id in enumerate(ids):
                score = 1 - distances[i] if distances[i] else 1.0
                results.append({
                    "id": doc_id,
                    "content": documents[i] or "",
                    "score": score,
                    "metadata": metadatas[i] or {},
                })
        return results


class RealGraphProvider:
    """Async wrapper around Neo4jAdapter matching RetrievalManager's adjacency interface."""

    def __init__(self, neo4j_adapter, namespace: str):
        self._adapter = neo4j_adapter
        self._namespace = namespace

    async def get_nodes(self, filter: Optional[Dict] = None) -> List[Dict]:
        """Fetch all nodes in namespace as flat dicts."""
        loop = asyncio.get_event_loop()
        cypher = """
            MATCH (n:SynapNode {_namespace: $ns})
            RETURN n, labels(n) as labels
        """
        results = await loop.run_in_executor(
            None, self._adapter._run_query, cypher, {"ns": self._namespace}
        )
        nodes = []
        for record in results:
            node_data = dict(record["n"])
            labels = record["labels"]
            node_type = next((l for l in labels if l != "SynapNode"), "Unknown")
            props = {k: v for k, v in node_data.items() if not k.startswith("_")}
            props["node_type"] = node_type
            nodes.append(props)
        return nodes

    async def get_edges(self, filter: Optional[Dict] = None) -> List[Dict]:
        """Fetch all edges in namespace as flat dicts."""
        loop = asyncio.get_event_loop()
        cypher = """
            MATCH (a:SynapNode {_namespace: $ns})-[r]->(b:SynapNode {_namespace: $ns})
            RETURN r, type(r) as edge_type, a.id as source_id, b.id as target_id
        """
        results = await loop.run_in_executor(
            None, self._adapter._run_query, cypher, {"ns": self._namespace}
        )
        edges = []
        for record in results:
            edge_data = dict(record["r"])
            props = {k: v for k, v in edge_data.items() if not k.startswith("_")}
            props["source_id"] = record["source_id"]
            props["target_id"] = record["target_id"]
            props["edge_type"] = record["edge_type"]
            edges.append(props)
        return edges

    async def count(self, filter: Optional[Dict] = None) -> int:
        """Count nodes in namespace."""
        loop = asyncio.get_event_loop()
        cypher = "MATCH (n:SynapNode {_namespace: $ns}) RETURN count(n) as cnt"
        results = await loop.run_in_executor(
            None, self._adapter._run_query, cypher, {"ns": self._namespace}
        )
        return results[0]["cnt"] if results else 0


class StubScopeResolver:
    """Bypass Postgres-backed scope resolution for testing."""

    async def resolve(self, instance_id, user_id=None, conversation_id=None):
        from synap.cloud.shared.scope import ScopeHierarchy
        return ScopeHierarchy(
            instance_id=instance_id,
            client_id="client_test",
            customer_id="cust_test",
            user_id=user_id or "user_test",
            internal_user_id="internal_user_test",
            internal_customer_id="internal_cust_test",
        )


class InMemoryCache:
    """Simple in-memory cache matching RetrievalManager's cache interface."""

    def __init__(self):
        self._store: Dict[str, Any] = {}

    async def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


# ============================================================
# Seed Data
# ============================================================

SEED_DOCUMENTS = [
    {"id": "vec_1", "content": "User is allergic to peanuts", "context_type": "facts", "scope": "user", "confidence": 0.95},
    {"id": "vec_2", "content": "User prefers vegetarian food", "context_type": "preferences", "scope": "user", "confidence": 0.85},
    {"id": "vec_3", "content": "User likes spicy Thai food", "context_type": "preferences", "scope": "user", "confidence": 0.80},
    {"id": "vec_4", "content": "User's usual lunch budget is $15-25", "context_type": "facts", "scope": "user", "confidence": 0.90},
    {"id": "vec_5", "content": "User ordered Pad Thai last Tuesday", "context_type": "episodes", "scope": "user", "confidence": 0.75},
    {"id": "vec_6", "content": "User works at Acme Corp in engineering", "context_type": "facts", "scope": "user", "confidence": 0.88},
    {"id": "vec_7", "content": "User prefers window seats in restaurants", "context_type": "preferences", "scope": "user", "confidence": 0.70},
    {"id": "vec_8", "content": "Customer organization is headquartered in San Francisco", "context_type": "facts", "scope": "customer", "confidence": 0.85},
    {"id": "vec_9", "content": "User had a bad experience with seafood delivery last week", "context_type": "episodes", "scope": "user", "confidence": 0.72},
    {"id": "vec_10", "content": "User's favorite cuisine is Italian", "context_type": "preferences", "scope": "user", "confidence": 0.82},
]

SEED_GRAPH_NODES = [
    {"id": "node_user", "type": "Entity", "content": "User profile entity", "context_type": "facts", "confidence": 0.9, "scope": "user"},
    {"id": "node_allergy", "type": "Fact", "content": "Severe peanut allergy, carries EpiPen", "context_type": "facts", "confidence": 0.95, "scope": "user"},
    {"id": "node_restaurant", "type": "Entity", "content": "Thai Basil Restaurant, favorite spot", "context_type": "facts", "confidence": 0.8, "scope": "user"},
    {"id": "node_order", "type": "Episode", "content": "Ordered Pad Thai at Thai Basil, enjoyed it", "context_type": "episodes", "confidence": 0.75, "scope": "user"},
    {"id": "node_veg", "type": "Preference", "content": "Strictly vegetarian, no exceptions", "context_type": "preferences", "confidence": 0.85, "scope": "user"},
]

SEED_GRAPH_EDGES = [
    {"source": "node_user", "target": "node_allergy", "type": "HAS_FACT", "is_memory": True},
    {"source": "node_user", "target": "node_restaurant", "type": "VISITED", "is_memory": False},
    {"source": "node_restaurant", "target": "node_order", "type": "EPISODE", "is_memory": True},
    {"source": "node_user", "target": "node_veg", "type": "HAS_PREFERENCE", "is_memory": True},
    {"source": "node_allergy", "target": "node_veg", "type": "RELATED_TO", "is_memory": False},
]


# ============================================================
# Embedding Helper
# ============================================================

async def generate_embeddings(texts: List[str], api_key: str, model: str = "text-embedding-3-small") -> List[List[float]]:
    """Batch generate OpenAI embeddings."""
    import openai
    client = openai.AsyncOpenAI(api_key=api_key)
    response = await client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def _run_async(coro):
    """Run async coroutine in a new event loop (for sync fixtures)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================
# Module-Scoped Fixture
# ============================================================

@pytest.fixture(scope="module")
def real_stores():
    """Connect to real ChromaDB + Neo4j, seed data, yield providers, clean up."""
    from synap.providers.graph.adapters.neo4j import Neo4jAdapter

    # 1. Connect ChromaDB via HTTP
    chroma = ChromaHTTPClient(CHROMA_HOST, CHROMA_PORT)
    chroma.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # 2. Connect Neo4j
    neo4j = Neo4jAdapter({
        "uri": NEO4J_URI,
        "user": NEO4J_USER,
        "password": NEO4J_PASSWORD,
        "database": "neo4j",
    })
    neo4j.connect()
    neo4j.create_namespace(TEST_INSTANCE_ID)

    # 3. Generate embeddings for all seed content (single batch call)
    all_texts = [doc["content"] for doc in SEED_DOCUMENTS]
    node_texts = [node["content"] for node in SEED_GRAPH_NODES]
    all_embeddings = _run_async(generate_embeddings(all_texts + node_texts, OPENAI_API_KEY))

    doc_embeddings = all_embeddings[:len(SEED_DOCUMENTS)]
    node_embeddings = all_embeddings[len(SEED_DOCUMENTS):]

    # 4. Seed ChromaDB via HTTP API
    chroma.upsert(
        collection_name=COLLECTION_NAME,
        ids=[doc["id"] for doc in SEED_DOCUMENTS],
        embeddings=doc_embeddings,
        metadatas=[{
            "context_type": doc["context_type"],
            "scope": doc["scope"],
            "confidence": doc["confidence"],
        } for doc in SEED_DOCUMENTS],
        documents=[doc["content"] for doc in SEED_DOCUMENTS],
    )

    # 5. Seed Neo4j nodes
    for node, emb in zip(SEED_GRAPH_NODES, node_embeddings):
        props = {
            "id": node["id"],
            "content": node["content"],
            "context_type": node["context_type"],
            "confidence": node["confidence"],
            "scope": node["scope"],
            "embedding": emb,
            "_namespace": TEST_INSTANCE_ID,
        }
        neo4j._run_write(
            f"CREATE (n:SynapNode:{node['type']} $props) RETURN n.id as id",
            {"props": props},
        )

    # 6. Seed Neo4j edges with edge_vector and target_vector
    node_emb_map = {node["id"]: emb for node, emb in zip(SEED_GRAPH_NODES, node_embeddings)}

    for edge in SEED_GRAPH_EDGES:
        src_emb = np.array(node_emb_map[edge["source"]])
        tgt_emb = np.array(node_emb_map[edge["target"]])
        edge_vec = ((src_emb + tgt_emb) / 2).tolist()

        props = {
            "_namespace": TEST_INSTANCE_ID,
            "id": str(uuid.uuid4()),
            "is_memory": edge["is_memory"],
            "edge_vector": edge_vec,
            "target_vector": tgt_emb.tolist(),
        }

        neo4j._run_write(
            f"""
            MATCH (a:SynapNode {{_namespace: $ns, id: $from_id}})
            MATCH (b:SynapNode {{_namespace: $ns, id: $to_id}})
            CREATE (a)-[r:{edge['type']} $props]->(b)
            RETURN r.id as id
            """,
            {"ns": TEST_INSTANCE_ID, "from_id": edge["source"],
             "to_id": edge["target"], "props": props},
        )

    # 7. Build providers
    vector_provider = RealVectorProvider(chroma, OPENAI_API_KEY)
    graph_provider = RealGraphProvider(neo4j, TEST_INSTANCE_ID)

    yield {
        "vector_provider": vector_provider,
        "graph_provider": graph_provider,
        "instance_id": TEST_INSTANCE_ID,
        "collection": COLLECTION_NAME,
    }

    # Cleanup
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    try:
        neo4j.delete_namespace(TEST_INSTANCE_ID)
    except Exception:
        pass
    chroma.close()
    neo4j.disconnect()


# ============================================================
# Tests
# ============================================================

@requires_real_dbs
class TestRealDBFastRetrieval:
    """Test FAST mode retrieval against real ChromaDB."""

    @pytest.mark.asyncio
    async def test_fast_mode_returns_seeded_vector_data(self, real_stores):
        """Seeded documents are returned via real vector search with meaningful scores."""
        from synap.cloud.application.client_api.retrieval_manager import RetrievalManager, RetrievalConfig
        from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType

        config = RetrievalConfig(fast_timeout_ms=15000, accurate_timeout_ms=30000)
        manager = RetrievalManager(
            scope_resolver=StubScopeResolver(),
            vector_provider=real_stores["vector_provider"],
            cache_provider=InMemoryCache(),
            config=config,
        )

        decision = ContextDecision(
            decision_id=str(uuid.uuid4()),
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id=real_stores["instance_id"],
            user_id="user_test",
            retrieval_mode=RetrievalMode.FAST,
            context_types=["facts", "preferences"],
            search_queries=["food preferences and dietary restrictions"],
            token_budget=2000,
        )

        bundle = await manager.fetch_context(decision)

        assert bundle.item_count > 0, "Should return at least one item from seeded data"
        all_content = [
            item.content
            for items in bundle.items_by_type.values()
            for item in items
        ]
        assert any(
            "peanut" in c.lower() or "vegetarian" in c.lower() or "spicy" in c.lower()
            for c in all_content
        ), f"Expected food-related content, got: {all_content}"


@requires_real_dbs
class TestRealDBAccurateRetrieval:
    """Test ACCURATE mode retrieval with graph traversal."""

    @pytest.mark.asyncio
    async def test_accurate_mode_with_graph_traversal(self, real_stores):
        """ACCURATE mode queries both vector and graph stores."""
        from synap.cloud.application.client_api.retrieval_manager import RetrievalManager, RetrievalConfig
        from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType

        config = RetrievalConfig(fast_timeout_ms=15000, accurate_timeout_ms=30000)
        manager = RetrievalManager(
            scope_resolver=StubScopeResolver(),
            vector_provider=real_stores["vector_provider"],
            graph_provider=real_stores["graph_provider"],
            cache_provider=InMemoryCache(),
            config=config,
        )

        decision = ContextDecision(
            decision_id=str(uuid.uuid4()),
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id=real_stores["instance_id"],
            user_id="user_test",
            retrieval_mode=RetrievalMode.ACCURATE,
            context_types=["facts", "preferences", "episodes"],
            search_queries=["Tell me about the user's food history and preferences"],
            token_budget=3000,
        )

        bundle = await manager.fetch_context(decision)

        assert bundle.item_count > 0, "ACCURATE mode should return results"
        sources = set()
        for items in bundle.items_by_type.values():
            for item in items:
                sources.add(item.source)
        # Vector results should always be present
        assert "vector" in sources, f"Expected vector source, got: {sources}"


@requires_real_dbs
class TestRealDBSemanticRelevance:
    """Test that semantic search returns contextually relevant results."""

    @pytest.mark.asyncio
    async def test_allergy_query_returns_peanut_fact(self, real_stores):
        """Query about food allergies should return the peanut allergy fact."""
        from synap.cloud.application.client_api.retrieval_manager import RetrievalManager, RetrievalConfig
        from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType

        config = RetrievalConfig(fast_timeout_ms=15000, accurate_timeout_ms=30000)
        manager = RetrievalManager(
            scope_resolver=StubScopeResolver(),
            vector_provider=real_stores["vector_provider"],
            cache_provider=InMemoryCache(),
            config=config,
        )

        decision = ContextDecision(
            decision_id=str(uuid.uuid4()),
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id=real_stores["instance_id"],
            user_id="user_test",
            retrieval_mode=RetrievalMode.FAST,
            context_types=["facts"],
            search_queries=["food allergies"],
            token_budget=2000,
        )

        bundle = await manager.fetch_context(decision)

        fact_items = bundle.items_by_type.get("facts", [])
        peanut_items = [i for i in fact_items if "peanut" in i.content.lower()]
        assert len(peanut_items) >= 1, (
            f"Expected peanut allergy fact, got facts: "
            f"{[i.content for i in fact_items]}"
        )
        assert peanut_items[0].similarity_score > 0.3, (
            f"Expected reasonable similarity, got {peanut_items[0].similarity_score}"
        )


@requires_real_dbs
class TestRealDBTypeFiltering:
    """Test that context_type filtering works correctly."""

    @pytest.mark.asyncio
    async def test_preferences_only_filter(self, real_stores):
        """Querying with types=['preferences'] should only return preferences."""
        from synap.cloud.application.client_api.retrieval_manager import RetrievalManager, RetrievalConfig
        from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType

        config = RetrievalConfig(fast_timeout_ms=15000, accurate_timeout_ms=30000)
        manager = RetrievalManager(
            scope_resolver=StubScopeResolver(),
            vector_provider=real_stores["vector_provider"],
            cache_provider=InMemoryCache(),
            config=config,
        )

        decision = ContextDecision(
            decision_id=str(uuid.uuid4()),
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id=real_stores["instance_id"],
            user_id="user_test",
            retrieval_mode=RetrievalMode.FAST,
            context_types=["preferences"],
            search_queries=["what does the user like"],
            token_budget=2000,
        )

        bundle = await manager.fetch_context(decision)

        for ctx_type, items in bundle.items_by_type.items():
            assert ctx_type == "preferences", (
                f"Expected only preferences, got type '{ctx_type}' with items: "
                f"{[i.content for i in items]}"
            )


@requires_real_dbs
class TestRealDBTokenBudget:
    """Test token budget enforcement with real data."""

    @pytest.mark.asyncio
    async def test_hard_budget_truncates_results(self, real_stores):
        """Very small hard budget should truncate results."""
        from synap.cloud.application.client_api.retrieval_manager import RetrievalManager, RetrievalConfig
        from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType

        config = RetrievalConfig(fast_timeout_ms=15000, accurate_timeout_ms=30000)
        manager = RetrievalManager(
            scope_resolver=StubScopeResolver(),
            vector_provider=real_stores["vector_provider"],
            cache_provider=InMemoryCache(),
            config=config,
        )

        decision = ContextDecision(
            decision_id=str(uuid.uuid4()),
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id=real_stores["instance_id"],
            user_id="user_test",
            retrieval_mode=RetrievalMode.FAST,
            context_types=["facts", "preferences", "episodes"],
            search_queries=["everything about the user"],
            token_budget=50,
            budget_mode="hard",
        )

        bundle = await manager.fetch_context(decision)

        assert bundle.total_tokens <= 50, (
            f"Hard budget of 50 tokens exceeded: {bundle.total_tokens}"
        )


@requires_real_dbs
class TestRealDBUnrelatedQuery:
    """Test that unrelated queries return low scores or no results."""

    @pytest.mark.asyncio
    async def test_unrelated_query_low_relevance(self, real_stores):
        """Completely unrelated query should return low-scoring or no results."""
        from synap.cloud.application.client_api.retrieval_manager import RetrievalManager, RetrievalConfig
        from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType

        config = RetrievalConfig(fast_timeout_ms=15000, accurate_timeout_ms=30000)
        manager = RetrievalManager(
            scope_resolver=StubScopeResolver(),
            vector_provider=real_stores["vector_provider"],
            cache_provider=InMemoryCache(),
            config=config,
        )

        decision = ContextDecision(
            decision_id=str(uuid.uuid4()),
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id=real_stores["instance_id"],
            user_id="user_test",
            retrieval_mode=RetrievalMode.FAST,
            context_types=["facts", "preferences", "episodes"],
            search_queries=["quantum physics wave function collapse"],
            token_budget=2000,
        )

        bundle = await manager.fetch_context(decision)

        # Either no results, or all similarity scores should be low
        for items in bundle.items_by_type.values():
            for item in items:
                if item.source == "vector":
                    assert item.similarity_score < 0.7, (
                        f"Unrelated query got high score {item.similarity_score} "
                        f"for: {item.content}"
                    )
