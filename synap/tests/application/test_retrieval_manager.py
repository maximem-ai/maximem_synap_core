"""Tests for RetrievalManager."""

import asyncio
import json
import uuid
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from synap.cloud.agents.models.decisions import ContextDecision, RetrievalMode, TriggerType
from synap.cloud.agents.models.bundles import (
    ContextBundle,
    ContextItem,
    ConversationContext,
    DegradationLevel,
)
import synap.cloud.application.client_api.retrieval_manager as RM
import synap.cloud.application.shared.providers as providers
from synap.cloud.application.client_api.retrieval_manager import (
    RetrievalManager,
    RetrievalConfig,
    ResponseFilter,
    FilterReason,
    VISIBILITY_ALL,
)
from synap.cloud.shared.scope import ScopeHierarchy


# === Fixtures ===


@pytest.fixture(autouse=True)
def _stub_query_decomposition():
    """Stub the spaCy-backed query_decomposition module.

    FAST/ACCURATE retrieval lazily imports
    ``synap.cloud.application.client_api.query_decomposition`` which imports
    spaCy at module load AND calls ``spacy.load("en_core_web_sm")`` at import
    time. spaCy is unimportable under this test env (Python 3.14 + Pydantic v1
    internals it depends on), so the real import explodes with
    ``ConfigError: unable to infer type for attribute "REGEX"`` and the whole
    retrieval path fail-softs to DegradationLevel.FAILED.

    The stub replaces ``decompose_queries`` with its identity behaviour for
    non-compound queries (return queries unchanged) — exactly what the real
    function does for the simple queries these tests use — so the rest of the
    retrieval pipeline runs as in production.
    """
    import sys
    import types

    mod_name = "synap.cloud.application.client_api.query_decomposition"
    saved = sys.modules.get(mod_name)
    stub = types.ModuleType(mod_name)
    stub.decompose_queries = lambda queries: list(queries)
    sys.modules[mod_name] = stub
    try:
        yield
    finally:
        if saved is not None:
            sys.modules[mod_name] = saved
        else:
            sys.modules.pop(mod_name, None)


@pytest.fixture
def mock_cache():
    """Mock cache (Redis) provider."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


@pytest.fixture
def mock_embedding():
    """Mock embedding client.

    Retrieval encodes queries via ``embedding_client.embed_batch(queries,
    is_query=True)`` (see RetrievalManager._batch_encode_queries). Return a
    fixed small vector per query so downstream vector search can run without a
    real embedding model.
    """
    embedding = AsyncMock()

    async def _embed_batch(queries, is_query=False, **kwargs):
        return [[0.1] * 8 for _ in queries]

    embedding.embed_batch = AsyncMock(side_effect=_embed_batch)
    return embedding


def _make_vector_provider(customer_results):
    """Build a mock vector provider matching the current provider interface.

    The retrieval path no longer calls ``vector.query``; it calls
    ``vector.batch_search(embeddings=..., scope=..., top_k=..., filters=...,
    scope_level=...)`` once per scope level (customer + client), each returning
    a list-of-lists (one inner list of result dicts per query embedding). HQE
    discovery additionally calls ``vector.batch_search_questions(...)``.

    *customer_results* is the list of result dicts returned for the
    customer-scope collection; the client-scope collection returns empty.
    """
    vector = AsyncMock()

    async def _batch_search(embeddings, scope, top_k, filters=None, scope_level=None, **kwargs):
        if scope_level == "client":
            return [[] for _ in embeddings]
        return [list(customer_results) for _ in embeddings]

    async def _batch_search_questions(embeddings, scope, top_k, filters=None, scope_level=None, **kwargs):
        return [[] for _ in embeddings]

    vector.batch_search = AsyncMock(side_effect=_batch_search)
    vector.batch_search_questions = AsyncMock(side_effect=_batch_search_questions)
    return vector


@pytest.fixture
def mock_vector():
    """Mock vector provider returning a single facts result."""
    return _make_vector_provider([])


@pytest.fixture
def mock_graph():
    """Mock graph provider."""
    graph = AsyncMock()
    graph.get_nodes = AsyncMock(return_value=[])
    graph.get_edges = AsyncMock(return_value=[])
    graph.count = AsyncMock(return_value=0)
    return graph


@pytest.fixture
def mock_postgres():
    """Mock postgres provider."""
    postgres = MagicMock()
    postgres.fetch_all = MagicMock(return_value=[])
    return postgres


@pytest.fixture
def mock_scope_resolver():
    """Mock scope resolver."""
    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value=ScopeHierarchy(
        instance_id="inst-1",
        client_id="cli-1",
        user_id="user-1",
        internal_user_id="internal-user-1",
    ))
    return resolver


@pytest.fixture
def manager(mock_cache, mock_scope_resolver):
    """RetrievalManager with mocked providers."""
    return RetrievalManager(
        scope_resolver=mock_scope_resolver,
        cache_provider=mock_cache,
    )


@pytest.fixture
def decision():
    """Standard ContextDecision for testing."""
    return ContextDecision(
        decision_id="d-test",
        trigger_type=TriggerType.EXPLICIT_REQUEST,
        instance_id="inst-1",
        user_id="user-1",
        conversation_id="conv-1",
        retrieval_mode=RetrievalMode.FAST,
        context_types=["facts", "preferences"],
        search_queries=["user preferences"],
        token_budget=2000,
    )


def _make_items(n=3, context_type="facts", source="vector"):
    """Helper to create test ContextItems."""
    return [
        ContextItem(
            item_id=f"item-{i}",
            content=f"Test content number {i} with enough words to pass utility",
            context_type=context_type,
            source=source,
            similarity_score=0.9 - (i * 0.1),
            confidence=0.8,
            scope="user",
        )
        for i in range(n)
    ]


# === TestRetrievalManager ===


class TestRetrievalManager:
    """Tests for RetrievalManager core methods."""

    @pytest.mark.asyncio
    async def test_fetch_context(self, manager, decision):
        """Test main context fetch entry point returns a bundle."""
        bundle = await manager.fetch_context(decision)

        assert isinstance(bundle, ContextBundle)
        assert bundle.decision_id == decision.decision_id
        assert bundle.retrieval_mode == "fast"
        assert bundle.retrieval_time_ms >= 0
        # No embedding backend is an explicit degraded retrieval, never a
        # pseudo-random vector search that would be safe to cache.
        assert bundle.cache_key is None

    @pytest.mark.asyncio
    async def test_fetch_context_cache_hit(self, mock_cache, mock_scope_resolver, decision):
        """Test fetch_context returns cached bundle on L1 hit."""
        cached_bundle = ContextBundle(
            bundle_id="b-cached",
            decision_id=decision.decision_id,
            items_by_type={
                "facts": [ContextItem(
                    item_id="f1",
                    content="Cached fact about the user's preferences and settings",
                    context_type="facts",
                    source="cache",
                    confidence=0.9,
                )]
            },
            total_tokens=20,
        )
        mock_cache.get = AsyncMock(return_value=cached_bundle.to_dict())

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            cache_provider=mock_cache,
        )
        bundle = await manager.fetch_context(decision)

        assert bundle.cache_hit is True
        assert bundle.bundle_id == "b-cached"
        # NOTE: scope resolution now happens BEFORE the L1 cache check — the
        # cache key is built from the resolved scope IDs (so it lines up with
        # ingestion-side invalidation keys). The cache hit still short-circuits
        # the expensive fetch/store work; it just doesn't skip scope resolution.

    @pytest.mark.asyncio
    async def test_fetch_context_caches_result(self, mock_cache, mock_scope_resolver, decision):
        """Test fetch_context writes result to L1 cache after retrieval.

        Needs a provider that actually returns something: since 2026-09-19 an
        empty bundle is deliberately not cached (see the sibling test), so a
        no-provider manager would pass this assertion for the wrong reason.
        """
        mock_cache.get = AsyncMock(return_value=None)
        mock_vector = _make_vector_provider([
            {
                "id": "v1",
                "content": "User works at Acme Corp in engineering department",
                "score": 0.92,
                "metadata": {
                    "memory_type": "facts",
                    "context_type": "facts",
                    "confidence": 0.85,
                    "scope": "user",
                    "user_id": "user-1",
                },
            }
        ])

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            cache_provider=mock_cache,
            vector_provider=mock_vector,
        )
        # This unit test exercises cache writes, not policy-store failure.
        # Model the ordinary, successfully resolved default-open client.
        manager._get_visibility_allowed_set = AsyncMock(return_value=VISIBILITY_ALL)
        bundle = await manager.fetch_context(decision)

        assert bundle.item_count > 0, "fixture returned nothing, so this proves nothing"
        # cache.set should be called to cache the result
        assert mock_cache.set.called

    @pytest.mark.asyncio
    async def test_fetch_context_does_not_cache_an_empty_result(
        self, mock_cache, mock_scope_resolver, decision
    ):
        """An empty answer is the one answer never worth remembering.

        The write was once unconditional, so a fetch
        that found nothing (or timed out into a degraded bundle) pinned that
        emptiness to the key for the full 300s TTL.
        """
        mock_cache.get = AsyncMock(return_value=None)

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            cache_provider=mock_cache,
        )
        bundle = await manager.fetch_context(decision)

        assert bundle.item_count == 0
        assert not mock_cache.set.called, "an empty bundle was cached"

    def test_normalize_retrieval_request(self, manager):
        """Test request normalization and validation."""
        request = {
            "conversation_id": "conv-1",
            "user_id": "user-1",
            "searchQuery": "What does user prefer?",
            "types": ["facts", "invalid_type", "preferences"],
            "max_results": 5,
        }
        normalized = manager.normalize_retrieval_request(request)

        assert normalized["conversation_id"] == "conv-1"
        assert normalized["search_queries"] == ["What does user prefer?"]
        assert "invalid_type" not in normalized["context_types"]
        assert "facts" in normalized["context_types"]
        assert "preferences" in normalized["context_types"]
        assert normalized["max_results"] == 5

    def test_normalize_retrieval_request_defaults(self, manager):
        """Test normalization applies defaults for missing fields."""
        normalized = manager.normalize_retrieval_request({})

        assert normalized["search_queries"] == []
        assert normalized["max_results"] == 10
        # Default context_types now includes temporal_events (all are valid types).
        assert set(normalized["context_types"]) == {
            "facts",
            "preferences",
            "episodes",
            "temporal_events",
        }

    def test_normalize_retrieval_request_list_query(self, manager):
        """Test normalization handles list-type searchQuery."""
        request = {"searchQuery": ["q1", "q2"]}
        normalized = manager.normalize_retrieval_request(request)
        assert normalized["search_queries"] == ["q1", "q2"]

    @pytest.mark.asyncio
    async def test_fetch_short_term_context_no_conversation(self, manager):
        """Test short-term fetch returns empty for missing conversation_id."""
        # fetch_short_term_context now returns a tuple of
        # (short-term ContextItems, ConversationContext or None).
        items, conv_ctx = await manager.fetch_short_term_context(
            conversation_id=None,
            instance_id="inst-1",
            user_id="user-1",
        )
        assert items == []
        assert conv_ctx is None

    @pytest.mark.asyncio
    async def test_fetch_short_term_context_l2_cache_hit(self, mock_cache, mock_scope_resolver):
        """Regression (P2): short-term fetch returns the L2-cached ConversationContext.

        The L2 short-term cache no longer stores a flat list of ContextItems;
        it stores a wrapped dict ``{"conversation_context": <serialized
        ConversationContext>}`` under the ``user_context:`` key. On a hit the
        method should return ``([], ConversationContext)`` — short-term vector
        items are returned separately from the coherent conversation block.
        """
        cached_ctx = ConversationContext(
            summary="User has been discussing dark mode preferences",
            key_extractions={
                "preferences": [{"text": "prefers dark mode"}],
            },
            conversation_id="conv-1",
        )

        async def selective_get(key):
            if key.startswith("user_context:"):
                return {"conversation_context": cached_ctx.to_dict()}
            return None

        mock_cache.get = AsyncMock(side_effect=selective_get)

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            cache_provider=mock_cache,
        )
        items, conv_ctx = await manager.fetch_short_term_context(
            conversation_id="conv-1",
            instance_id="inst-1",
            user_id="user-1",
            filters={"types": ["preferences"]},
        )
        # L2 hit returns the conversation block; no atomized vector items.
        assert items == []
        assert conv_ctx is not None
        assert conv_ctx.conversation_id == "conv-1"
        assert conv_ctx.summary == "User has been discussing dark mode preferences"

    @pytest.mark.asyncio
    async def test_fetch_long_term_context_fast_mode(self, mock_embedding, mock_scope_resolver):
        """Test long-term fetch in FAST mode returns vector-sourced items.

        The vector provider interface is now batch_search(embeddings=...,
        scope=..., top_k=..., filters=..., scope_level=...) returning a
        list-of-lists (per-query result dicts). Scope post-filter requires
        result metadata to carry user_id matching the scope.
        """
        mock_vector = _make_vector_provider([
            {
                "id": "v1",
                "content": "User works at Acme Corp in engineering department",
                "score": 0.92,
                "metadata": {
                    "memory_type": "facts",
                    "context_type": "facts",
                    "confidence": 0.85,
                    "scope": "user",
                    "user_id": "user-1",
                },
            }
        ])

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            vector_provider=mock_vector,
            embedding_client=mock_embedding,
        )
        scope = ScopeHierarchy(
            instance_id="inst-1",
            client_id="cli-1",
            user_id="user-1",
        )
        items = await manager.fetch_long_term_context(
            scope=scope,
            queries=["user info"],
            filters={"types": ["facts"]},
            mode="fast",
        )
        assert len(items) == 1
        assert items[0].source == "vector"
        assert items[0].similarity_score == 0.92

    @pytest.mark.asyncio
    async def test_fetch_long_term_context_publishes_one_tripwire_event(
        self, mock_embedding, mock_scope_resolver
    ):
        """F4b/FIX 1: the tripwire producer is hoisted to fire once, right
        after query_vectors materialises in fetch_long_term_context, so it
        covers every vector-search arm below it (quick memory search, direct
        similarity, questions) from a single chokepoint — instead of the old
        wiring where only the questions arm (gated behind
        ENABLE_QUESTION_EMBEDDINGS) published at all.

        FIX 2: ``asyncio.create_task(...)`` with no held reference can be
        garbage-collected mid-flight, silently dropping the publish. This
        asserts the task is strong-ref'd in a module-level set while
        in-flight and discarded once it completes — not just that the queue
        entry appears.
        """
        mock_vector = _make_vector_provider([])
        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            vector_provider=mock_vector,
            embedding_client=mock_embedding,
        )
        scope = ScopeHierarchy(instance_id="inst-1", client_id="cli-1", user_id="user-1")

        fake_redis = MagicMock()
        fake_redis.lpush = AsyncMock()
        fake_redis.ltrim = AsyncMock()

        before = set(RM._TRIPWIRE_PUBLISH_TASKS)
        with patch.object(providers, "get_cached_providers", return_value={"redis": fake_redis}):
            await manager.fetch_long_term_context(
                scope=scope,
                queries=["user info"],
                filters={"types": ["facts"]},
                mode="fast",
            )
            # The publish is fire-and-forget: give the event loop a few ticks
            # to run it to completion (and fire its done-callback) before
            # asserting on it.
            for _ in range(10):
                await asyncio.sleep(0)

        # Exactly one publish for the whole fetch — not one per arm.
        fake_redis.lpush.assert_awaited_once()
        key, payload = fake_redis.lpush.await_args.args
        assert key == "synap:consolidation:tripwire:queue"
        body = json.loads(payload)
        assert body["client_id"] == "cli-1"
        assert body["query"] == "user info"
        assert isinstance(body["embedding"], list) and body["embedding"]
        assert isinstance(body["ts"], float)

        # FIX 2: strong-ref'd while pending, discarded once done — no leak.
        assert set(RM._TRIPWIRE_PUBLISH_TASKS) == before

    @pytest.mark.asyncio
    async def test_fast_reuses_precomputed_query_vectors(self, mock_scope_resolver):
        """FAST #1: reuse the caller's encoded vectors, encode only new subqueries.

        When subquery generation expands the query set, the caller's already-
        encoded vectors (``precomputed_query_vectors``, as ``_fetch_fast`` passes
        them) must be reused verbatim for the matching strings; only the newly
        added template strings get a fresh encode. This removes the second full
        embed of the caller's queries on every FAST request and is result-neutral
        (identical strings map to identical vectors).
        """
        manager = RetrievalManager(scope_resolver=mock_scope_resolver)
        scope = ScopeHierarchy(instance_id="inst-1", client_id="cli-1", user_id="user-1")

        # Caller pre-encoded these two queries upstream (as _fetch_fast does).
        queries = ["q original zero", "q original one"]
        pre_vecs = [
            np.array([0.11] * 8, dtype=np.float32),
            np.array([0.22] * 8, dtype=np.float32),
        ]

        # Subquery generation keeps q0 and adds two new template strings.
        async def _fake_subqueries(*args, **kwargs):
            return {
                "queries": ["q original zero", "template alpha", "template beta"],
                "relationship_types": [],
            }

        # Spy: record exactly which strings hit a fresh encode.
        encode_calls: list = []

        async def _fake_encode(strings):
            encode_calls.append(list(strings))
            return [np.array([0.99] * 8, dtype=np.float32) for _ in strings]

        # Capture the query_vectors + all_queries that flow into the vector arm.
        captured: dict = {}

        async def _fake_similarity(scope, all_queries, query_vectors, context_types, top_k=None, **kw):
            captured["vectors"] = query_vectors
            captured["all_queries"] = all_queries
            return []

        async def _fake_questions(scope, queries, query_vectors, top_k=None, **kw):
            return []

        manager._generate_retrieval_subqueries = AsyncMock(side_effect=_fake_subqueries)
        manager._batch_encode_queries = AsyncMock(side_effect=_fake_encode)
        manager._direct_similarity_search = AsyncMock(side_effect=_fake_similarity)
        manager._search_questions = AsyncMock(side_effect=_fake_questions)

        await manager.fetch_long_term_context(
            scope=scope,
            queries=queries,
            filters={"types": ["facts"]},
            mode="fast",
            skip_subquery_gen=False,
            precomputed_query_vectors=pre_vecs,
        )

        # 1. Only the NEW strings were encoded — the originals were reused.
        assert encode_calls == [["template alpha", "template beta"]], (
            f"expected a single encode of only the new subqueries, got {encode_calls}"
        )

        # 2. all_queries order: original-zero, two templates, then merged original-one.
        assert captured["all_queries"] == [
            "q original zero", "template alpha", "template beta", "q original one",
        ]

        # 3. Reused positions are the SAME vector objects the caller precomputed.
        qv = captured["vectors"]
        assert qv[0] is pre_vecs[0], "original q0 vector must be reused, not re-encoded"
        assert qv[3] is pre_vecs[1], "original q1 vector must be reused, not re-encoded"

    @pytest.mark.asyncio
    async def test_fetch_long_term_context_accurate_mode(self, mock_embedding, mock_graph, mock_scope_resolver):
        """Test long-term fetch in ACCURATE mode uses vector + graph."""
        mock_vector = _make_vector_provider([
            {
                "id": "v1",
                "content": "User works at Acme Corp engineering department",
                "score": 0.9,
                "metadata": {
                    "memory_type": "facts",
                    "context_type": "facts",
                    "confidence": 0.8,
                    "scope": "user",
                    "user_id": "user-1",
                    "entity_id": "ent-1",
                },
            }
        ])

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            vector_provider=mock_vector,
            graph_provider=mock_graph,
            embedding_client=mock_embedding,
        )
        scope = ScopeHierarchy(
            instance_id="inst-1",
            client_id="cli-1",
            user_id="user-1",
            internal_user_id="iuser-1",
        )
        items = await manager.fetch_long_term_context(
            scope=scope,
            queries=["user info"],
            filters={"types": ["facts"]},
            mode="accurate",
        )
        # Should have vector results; graph is called but returns empty
        assert len(items) >= 1
        assert items[0].source == "vector"

    @pytest.mark.asyncio
    async def test_accurate_mode_does_not_run_the_unified_graph_walk(
        self, mock_embedding, mock_graph, mock_scope_resolver
    ):
        """The walk over the Chroma adjacency is not run, in any mode.

        Adjacency and anchor selection stay, because Neo4j discovery starts from
        the anchors; only the walk, whose items were labelled "graph", is gone.
        """
        from unittest.mock import AsyncMock

        mock_vector = _make_vector_provider([
            {
                "id": "v1",
                "content": "User works at Acme Corp engineering department",
                "score": 0.9,
                "metadata": {
                    "memory_type": "facts", "context_type": "facts",
                    "confidence": 0.8, "scope": "user", "user_id": "user-1",
                    "entity_id": "ent-1",
                },
            }
        ])
        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            vector_provider=mock_vector,
            graph_provider=mock_graph,
            embedding_client=mock_embedding,
        )
        walk = AsyncMock(side_effect=AssertionError("the unified walk must not run"))
        manager._unified_graph_traversal = walk
        scope = ScopeHierarchy(
            instance_id="inst-1", client_id="cli-1", user_id="user-1",
            internal_user_id="iuser-1",
        )
        items = await manager.fetch_long_term_context(
            scope=scope, queries=["user info"],
            filters={"types": ["facts"]}, mode="accurate",
        )
        walk.assert_not_called()
        assert items and all(i.source != "graph" for i in items)

    def test_assemble_context_bundle(self, manager, decision):
        """Test context bundle assembly with filtering and categorization."""
        items = [
            ContextItem(
                item_id="f1",
                content="User works at Acme Corporation in the engineering department",
                context_type="facts",
                source="vector",
                similarity_score=0.9,
                confidence=0.8,
            ),
            ContextItem(
                item_id="p1",
                content="User prefers concise technical answers when discussing code",
                context_type="preferences",
                source="vector",
                similarity_score=0.85,
                confidence=0.9,
            ),
            ContextItem(
                item_id="e1",
                content="User discussed database migration strategies last week in detail",
                context_type="episodes",
                source="cache",
                similarity_score=0.7,
                confidence=0.6,
            ),
        ]
        bundle = ContextBundle(
            bundle_id="b-asm",
            decision_id=decision.decision_id,
        )
        result = manager.assemble_context_bundle(items, decision, bundle)

        assert isinstance(result, ContextBundle)
        # Items should be categorized
        assert "facts" in result.items_by_type or "preferences" in result.items_by_type

    @pytest.mark.asyncio
    async def test_fetch_context_with_filters(self, mock_cache, mock_scope_resolver):
        """Test fetch_context respects context type filters."""
        mock_cache.get = AsyncMock(return_value=None)

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            cache_provider=mock_cache,
        )
        decision = ContextDecision(
            decision_id="d-filter",
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id="inst-1",
            user_id="user-1",
            context_types=["emotions"],
            search_queries=["how is user feeling"],
        )
        bundle = await manager.fetch_context(decision)

        assert isinstance(bundle, ContextBundle)
        assert bundle.decision_id == "d-filter"

    @pytest.mark.asyncio
    async def test_fetch_context_empty_results(self, mock_cache, mock_scope_resolver):
        """Test fetch_context returns empty bundle when no results found."""
        mock_cache.get = AsyncMock(return_value=None)

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            cache_provider=mock_cache,
        )
        decision = ContextDecision(
            decision_id="d-empty",
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id="inst-1",
            context_types=["facts"],
            search_queries=["nonexistent topic xyz"],
        )
        bundle = await manager.fetch_context(decision)

        assert isinstance(bundle, ContextBundle)
        assert bundle.item_count == 0

    @pytest.mark.asyncio
    async def test_fetch_context_conversation_scope(self, mock_cache, mock_scope_resolver):
        """Test fetch_context scoped to conversation."""
        mock_cache.get = AsyncMock(return_value=None)

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            cache_provider=mock_cache,
        )
        decision = ContextDecision(
            decision_id="d-conv",
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id="inst-1",
            conversation_id="conv-specific",
            context_types=["facts"],
            search_queries=["test"],
        )
        bundle = await manager.fetch_context(decision)

        assert isinstance(bundle, ContextBundle)
        # Scope resolver should be called with the conversation_id
        mock_scope_resolver.resolve.assert_called_once()
        call_kwargs = mock_scope_resolver.resolve.call_args.kwargs
        assert call_kwargs["conversation_id"] == "conv-specific"

    @pytest.mark.asyncio
    async def test_fetch_context_user_scope(self, mock_cache, mock_scope_resolver):
        """Test fetch_context scoped to user."""
        mock_cache.get = AsyncMock(return_value=None)

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            cache_provider=mock_cache,
        )
        decision = ContextDecision(
            decision_id="d-user",
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id="inst-1",
            user_id="user-specific",
            context_types=["preferences"],
            search_queries=["user settings"],
        )
        bundle = await manager.fetch_context(decision)

        assert isinstance(bundle, ContextBundle)
        call_kwargs = mock_scope_resolver.resolve.call_args.kwargs
        assert call_kwargs["user_id"] == "user-specific"

    @pytest.mark.asyncio
    async def test_fetch_context_customer_scope(self, mock_cache, mock_scope_resolver):
        """Test fetch_context scoped to customer via scope resolver."""
        resolved_scope = ScopeHierarchy(
            instance_id="inst-1",
            client_id="cli-1",
            customer_id="cust-1",
            user_id="user-1",
        )
        mock_scope_resolver.resolve = AsyncMock(return_value=resolved_scope)
        mock_cache.get = AsyncMock(return_value=None)

        manager = RetrievalManager(
            scope_resolver=mock_scope_resolver,
            cache_provider=mock_cache,
        )
        decision = ContextDecision(
            decision_id="d-cust",
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id="inst-1",
            user_id="user-1",
            customer_id="cust-1",
            context_types=["facts"],
            search_queries=["customer info"],
        )
        bundle = await manager.fetch_context(decision)

        assert isinstance(bundle, ContextBundle)
        mock_scope_resolver.resolve.assert_called_once()


# === TestCheckCache ===


class TestCheckCache:
    """Tests for RetrievalManager._check_cache with deserialization."""

    @pytest.mark.asyncio
    async def test_check_cache_returns_none_no_provider(self):
        """Test _check_cache returns None when no cache provider."""
        manager = RetrievalManager(cache_provider=None)
        result = await manager._check_cache("some:key")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_cache_miss(self, mock_cache):
        """Test _check_cache returns None on cache miss."""
        mock_cache.get = AsyncMock(return_value=None)
        manager = RetrievalManager(cache_provider=mock_cache)
        result = await manager._check_cache("missing:key")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_cache_hit_dict(self, mock_cache):
        """Test _check_cache deserializes when cache returns a dict."""
        bundle_dict = {
            "bundle_id": "b-dict",
            "decision_id": "d-dict",
            "items_by_type": {
                "facts": [{
                    "item_id": "f1",
                    "content": "User works at Acme Corp",
                    "context_type": "facts",
                    "source": "vector",
                }]
            },
            "total_tokens": 10,
            "degradation_level": "none",
            "created_at": datetime(2026, 1, 1).isoformat(),
        }
        mock_cache.get = AsyncMock(return_value=bundle_dict)
        manager = RetrievalManager(cache_provider=mock_cache)

        result = await manager._check_cache("hit:key")
        assert isinstance(result, ContextBundle)
        assert result.bundle_id == "b-dict"
        assert len(result.items_by_type["facts"]) == 1
        assert result.items_by_type["facts"][0].content == "User works at Acme Corp"

    @pytest.mark.asyncio
    async def test_check_cache_hit_json_string(self, mock_cache):
        """Test _check_cache deserializes when cache returns a JSON string."""
        bundle_dict = {
            "bundle_id": "b-json",
            "decision_id": "d-json",
            "items_by_type": {},
            "total_tokens": 0,
            "degradation_level": "none",
            "created_at": datetime(2026, 1, 1).isoformat(),
        }
        mock_cache.get = AsyncMock(return_value=json.dumps(bundle_dict))
        manager = RetrievalManager(cache_provider=mock_cache)

        result = await manager._check_cache("hit:json:key")
        assert isinstance(result, ContextBundle)
        assert result.bundle_id == "b-json"

    @pytest.mark.asyncio
    async def test_check_cache_error_returns_none(self, mock_cache):
        """Test _check_cache returns None on deserialization error."""
        mock_cache.get = AsyncMock(return_value="not-valid-json{{{")
        manager = RetrievalManager(cache_provider=mock_cache)

        result = await manager._check_cache("bad:key")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_cache_redis_error_returns_none(self, mock_cache):
        """Test _check_cache returns None on Redis connection error."""
        mock_cache.get = AsyncMock(side_effect=ConnectionError("Redis unreachable"))
        manager = RetrievalManager(cache_provider=mock_cache)

        result = await manager._check_cache("error:key")
        assert result is None


# === TestResponseFilter ===


class TestResponseFilter:
    """Tests for the 3-check ResponseFilter (utility, echo, Jaccard dedup)."""

    def test_utility_filter_removes_short_content(self):
        """Test utility check removes items with content < 10 chars."""
        rf = ResponseFilter()
        items = [
            ContextItem(item_id="short", content="hi", context_type="facts", source="vector"),
            ContextItem(item_id="long", content="User prefers dark mode for all applications", context_type="facts", source="vector", similarity_score=0.9),
        ]
        result = rf.filter_items(items, "user preferences")
        assert all(item.item_id != "short" for item in result)

    def test_echo_filter_removes_query_repeats(self):
        """Test echo check removes items that just repeat the query."""
        rf = ResponseFilter()
        items = [
            ContextItem(item_id="echo", content="hire interns for summer", context_type="facts", source="vector", similarity_score=0.9),
            ContextItem(item_id="real", content="User is a senior engineering manager at large tech company", context_type="facts", source="vector", similarity_score=0.8),
        ]
        result = rf.filter_items(items, "hire interns for summer")
        assert all(item.item_id != "echo" for item in result)

    def test_action_items_are_retained(self):
        """Test action/todo items survive the filter.

        The ResponseFilter no longer performs a state-vs-action check (it is now
        a 3-check filter: utility, echo, Jaccard dedup — see the class docstring
        in retrieval_manager.py). Action/todo phrasing that is otherwise useful,
        non-echoing, and non-duplicate is kept rather than dropped.
        """
        rf = ResponseFilter()
        items = [
            ContextItem(item_id="action", content="need to schedule the quarterly budget review next week", context_type="facts", source="vector", similarity_score=0.9),
            ContextItem(item_id="state", content="User prefers morning meetings and concise agendas always", context_type="preferences", source="vector", similarity_score=0.8),
        ]
        result = rf.filter_items(items, "meeting context")
        ids = [item.item_id for item in result]
        assert "action" in ids
        assert "state" in ids

    def test_deduplication_removes_similar(self):
        """Test Jaccard deduplication removes near-duplicate items."""
        rf = ResponseFilter(similarity_threshold=0.65)
        items = [
            ContextItem(item_id="orig", content="User works at Acme Corporation in engineering department", context_type="facts", source="vector", similarity_score=0.9),
            ContextItem(item_id="dupe", content="User works at Acme Corporation in engineering department doing backend", context_type="facts", source="graph", similarity_score=0.85),
            ContextItem(item_id="diff", content="User prefers Python and Go for backend development tasks", context_type="preferences", source="vector", similarity_score=0.8),
        ]
        result = rf.filter_items(items, "user info")
        # Should keep orig and diff, remove dupe
        assert len(result) <= 3
        # At minimum the different item should be present
        ids = [item.item_id for item in result]
        assert "diff" in ids


# === TestTokenBudget ===


class TestTokenBudget:
    """Tests for token budget enforcement."""

    def test_under_budget_keeps_all(self, manager, decision):
        """Test items under budget are all kept."""
        items = _make_items(2)
        bundle = ContextBundle(bundle_id="b-1", decision_id="d-1")
        bundle = manager.assemble_context_bundle(items, decision, bundle)
        result = manager._apply_token_budget(bundle, decision)

        assert result.budget_exceeded is False

    def test_hard_budget_truncates(self, manager):
        """Test hard budget mode truncates items."""
        decision = ContextDecision(
            decision_id="d-budget",
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id="inst-1",
            token_budget=10,  # Very small budget
            budget_mode="hard",
            search_queries=["test"],
        )
        bundle = ContextBundle(
            bundle_id="b-1",
            decision_id="d-1",
            items_by_type={
                "facts": _make_items(5, "facts"),
            },
        )
        result = manager._apply_token_budget(bundle, decision)
        # With a 10-token budget, not all items can fit
        assert result.total_tokens <= 10

    def test_soft_budget_warns(self, manager):
        """Test soft budget mode warns but keeps all items."""
        decision = ContextDecision(
            decision_id="d-soft",
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id="inst-1",
            token_budget=10,  # Very small
            budget_mode="soft",
            search_queries=["test"],
        )
        bundle = ContextBundle(
            bundle_id="b-1",
            decision_id="d-1",
            items_by_type={
                "facts": _make_items(5, "facts"),
            },
        )
        result = manager._apply_token_budget(bundle, decision)
        assert result.budget_exceeded is True
        assert any("exceeded" in w.lower() for w in result.warnings)


class TestTheLLMFilterKillSwitch:
    """The switch itself, which nothing covered.

    That gap is why six tests in the two classes below could keep passing
    their own assertions while the default flipped underneath them: with the
    filter off, `filter_items_with_llm` returns every item and never reaches
    the code those tests describe. A switch with no test is a switch that can
    change meaning silently, so both positions are pinned here.
    """

    def _items(self, n):
        return [
            ContextItem(
                item_id=f"k-{i}",
                content=f"Item {i} with enough words to be a real content string",
                context_type="facts",
                source="vector",
                similarity_score=1.0 - i * 0.001,
                confidence=0.8,
                scope="user",
            )
            for i in range(n)
        ]

    @pytest.mark.asyncio
    async def test_off_keeps_everything_and_makes_no_model_call(self, monkeypatch):
        monkeypatch.setattr(
            "synap.cloud.application.client_api.retrieval_manager."
            "_LLM_FILTER_ENABLED",
            False,
        )
        llm = MagicMock()
        llm.complete = AsyncMock(
            side_effect=AssertionError("the model was called with the filter off")
        )
        m = RetrievalManager(llm_client=llm)

        items = self._items(5)
        out = await m.filter_items_with_llm(items, query="topic")

        assert [i.item_id for i in out] == [f"k-{i}" for i in range(5)]
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_actually_filters(self, monkeypatch):
        monkeypatch.setattr(
            "synap.cloud.application.client_api.retrieval_manager."
            "_LLM_FILTER_ENABLED",
            True,
        )

        async def _complete(**kwargs):
            return {"content": "TOPIC: things\nKEEP: 2,4"}

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=_complete)
        m = RetrievalManager(llm_client=llm)

        out = await m.filter_items_with_llm(self._items(5), query="topic")

        assert [i.item_id for i in out] == ["k-1", "k-3"]
        llm.complete.assert_called()


class TestLLMFilterConcurrency:
    """A5: filter_items_with_llm bounds its batch fan-out without changing output."""

    @pytest.fixture(autouse=True)
    def _filter_on(self, monkeypatch):
        """Turn the LLM filter ON for this class.

        ⚠ `_LLM_FILTER_ENABLED` now defaults to OFF. That is a deliberate
        latency change: on a reasoning model the filter was keeping every
        candidate anyway, so off produces the same output without the round
        trip. These tests exist to pin what the filter does when it IS on, and
        without this fixture they short-circuit at the kill switch and assert
        against a code path they never reach. Six of them were silently
        asserting nothing.
        """
        monkeypatch.setattr(
            "synap.cloud.application.client_api.retrieval_manager."
            "_LLM_FILTER_ENABLED",
            True,
        )

    def _items(self, n):
        # Distinct, descending similarity scores so the sort order == creation
        # order, making the kept positions identifiable.
        return [
            ContextItem(
                item_id=f"f-{i}",
                content=f"Item {i} with enough words to be a real content string",
                context_type="facts",
                source="vector",
                similarity_score=1.0 - i * 0.001,
                confidence=0.8,
                scope="user",
            )
            for i in range(n)
        ]

    @pytest.mark.asyncio
    async def test_fan_out_is_bounded_and_output_identical(self):
        """Cap concurrency at the config limit; kept set matches an unbounded run."""
        # 150 items -> 5 batches of 30. Each stubbed batch keeps its first item.
        items = self._items(150)

        def _make_manager(limit):
            cfg = RetrievalConfig()
            cfg.llm_filter_max_concurrency = limit
            m = RetrievalManager(config=cfg)

            concurrency = {"cur": 0, "max": 0}

            async def _fake_batch(batch_items, batch_index, global_offset, **kw):
                concurrency["cur"] += 1
                concurrency["max"] = max(concurrency["max"], concurrency["cur"])
                # Yield so other scheduled batches can interleave up to the cap.
                await asyncio.sleep(0.01)
                concurrency["cur"] -= 1
                # Keep the first (highest-scored) item of this batch.
                return [global_offset]

            m._filter_batch_with_llm = AsyncMock(side_effect=_fake_batch)
            return m, concurrency

        # Bounded run at limit=2.
        m2, conc2 = _make_manager(2)
        bounded = await m2.filter_items_with_llm(items, query="topic")

        # Effectively-unbounded run (limit far above the 5 batches).
        m_hi, conc_hi = _make_manager(100)
        unbounded = await m_hi.filter_items_with_llm(items, query="topic")

        # Concurrency was actually capped at the limit (5 batches, all sleeping).
        assert conc2["max"] == 2, f"expected peak concurrency 2, got {conc2['max']}"
        assert conc_hi["max"] == 5, f"unbounded run should overlap all 5 batches, got {conc_hi['max']}"

        # Same kept items, same order — the cap changes only scheduling.
        assert [i.item_id for i in bounded] == [i.item_id for i in unbounded]
        # And the kept items are exactly the batch heads: positions 0,30,60,90,120.
        assert [i.item_id for i in bounded] == ["f-0", "f-30", "f-60", "f-90", "f-120"]

    @pytest.mark.asyncio
    async def test_limit_below_one_coerced_to_one(self):
        """A misconfigured non-positive limit must not deadlock the gather."""
        items = self._items(60)  # 2 batches
        cfg = RetrievalConfig()
        cfg.llm_filter_max_concurrency = 0
        m = RetrievalManager(config=cfg)

        async def _fake_batch(batch_items, batch_index, global_offset, **kw):
            return [global_offset]

        m._filter_batch_with_llm = AsyncMock(side_effect=_fake_batch)
        result = await m.filter_items_with_llm(items, query="topic")
        assert [i.item_id for i in result] == ["f-0", "f-30"]


class TestLLMFilterFailsOpen:
    """A failing/unparseable batch must KEEP its items, never delete them.

    The old ``return []`` meant one Groq timeout silently dropped up to 30
    candidates for that turn — and because the OTHER batches still returned
    results, the whole-call top-K fallback never fired, so the loss was
    invisible. Only an explicit ``KEEP: NONE`` may drop a batch.
    """

    @pytest.fixture(autouse=True)
    def _filter_on(self, monkeypatch):
        """Turn the LLM filter ON for this class.

        ⚠ `_LLM_FILTER_ENABLED` now defaults to OFF. That is a deliberate
        latency change: on a reasoning model the filter was keeping every
        candidate anyway, so off produces the same output without the round
        trip. These tests exist to pin what the filter does when it IS on, and
        without this fixture they short-circuit at the kill switch and assert
        against a code path they never reach. Six of them were silently
        asserting nothing.
        """
        monkeypatch.setattr(
            "synap.cloud.application.client_api.retrieval_manager."
            "_LLM_FILTER_ENABLED",
            True,
        )

    def _items(self, n):
        return [
            ContextItem(
                item_id=f"f-{i}",
                content=f"Item {i} with enough words to be a real content string",
                context_type="facts",
                source="vector",
                similarity_score=1.0 - i * 0.001,
                confidence=0.8,
                scope="user",
            )
            for i in range(n)
        ]

    def _manager(self, complete_impl):
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=complete_impl)
        return RetrievalManager(llm_client=llm)

    @pytest.mark.asyncio
    async def test_batch_timeout_keeps_that_batch(self):
        # 45 items -> 2 batches. Batch 0 answers normally, batch 1 times out.
        items = self._items(45)

        async def _complete(**kwargs):
            user_msg = kwargs["messages"][-1]["content"]
            # Batch 1's prompt contains the 31st item; batch 0's does not.
            if "Item 30 " in user_msg:
                raise asyncio.TimeoutError("groq timeout")
            return {"content": "TOPIC: things\nKEEP: 1,2"}

        m = self._manager(_complete)
        out = await m.filter_items_with_llm(items, query="topic")
        ids = [i.item_id for i in out]

        # Batch 0: only its two picks survive (real filtering still applies).
        assert ids[:2] == ["f-0", "f-1"]
        assert "f-2" not in ids
        # Batch 1: every item kept — the timeout deleted nothing.
        assert [f"f-{i}" for i in range(30, 45)] == ids[2:]

    @pytest.mark.asyncio
    async def test_normal_filtering_unchanged(self):
        items = self._items(5)

        async def _complete(**kwargs):
            return {"content": "TOPIC: things\nKEEP: 2,4"}

        m = self._manager(_complete)
        out = await m.filter_items_with_llm(items, query="topic")
        assert [i.item_id for i in out] == ["f-1", "f-3"]

    @pytest.mark.asyncio
    async def test_explicit_none_still_drops_the_batch(self):
        # NONE is a judgement, not a failure: the batch is dropped, and with
        # every batch dropped the whole-call top-K fallback takes over.
        items = self._items(5)

        async def _complete(**kwargs):
            return {"content": "TOPIC: things\nKEEP: NONE"}

        m = self._manager(_complete)
        out = await m.filter_items_with_llm(items, query="topic")
        # Fallback = top-20 by similarity, i.e. all 5 here (not "kept by parse").
        assert [i.item_id for i in out] == [f"f-{i}" for i in range(5)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("content", ["", "TOPIC: things", "KEEP: none of them"])
    async def test_unparseable_response_fails_open(self, content):
        # Model returned text but no usable indices -> parse failure, not a
        # judgement. Previously these silently discarded the whole batch.
        items = self._items(3)

        async def _complete(**kwargs):
            return {"content": content}

        m = self._manager(_complete)
        out = await m.filter_items_with_llm(items, query="topic")
        assert [i.item_id for i in out] == ["f-0", "f-1", "f-2"]

    @pytest.mark.asyncio
    async def test_batch_helper_exception_fails_open(self):
        # An exception escaping the batch helper (gather return_exceptions
        # path) gets the same treatment as an in-helper failure.
        items = self._items(60)  # 2 batches
        cfg = RetrievalConfig()
        m = RetrievalManager(config=cfg)

        async def _fake_batch(batch_items, batch_index, global_offset, **kw):
            if batch_index == 1:
                raise RuntimeError("boom")
            return [global_offset]

        m._filter_batch_with_llm = AsyncMock(side_effect=_fake_batch)
        out = await m.filter_items_with_llm(items, query="topic")
        ids = [i.item_id for i in out]
        assert ids[0] == "f-0"  # batch 0 filtered normally
        assert ids[1:] == [f"f-{i}" for i in range(30, 60)]  # batch 1 fully kept

    @pytest.mark.asyncio
    async def test_all_batches_fail_falls_back_to_top_k(self):
        # Every batch failing keeps everything; the top-K fallback is unchanged
        # for the whole-call failure path (see test below).
        items = self._items(30)

        async def _complete(**kwargs):
            raise asyncio.TimeoutError("groq timeout")

        m = self._manager(_complete)
        out = await m.filter_items_with_llm(items, query="topic")
        assert [i.item_id for i in out] == [f"f-{i}" for i in range(30)]

    @pytest.mark.asyncio
    async def test_whole_call_failure_still_uses_top_k_fallback(self):
        items = self._items(50)
        m = RetrievalManager()

        def _explode(*a, **kw):
            raise RuntimeError("fan-out setup blew up")

        # Fail before any batch coroutine exists -> the outer handler runs.
        with patch("asyncio.Semaphore", new=_explode):
            out = await m.filter_items_with_llm(items, query="topic")
        assert [i.item_id for i in out] == [f"f-{i}" for i in range(20)]


class TestAdjacencySingleFlight:
    """1.3: concurrent cold adjacency requests for one scope build exactly once."""

    def _scope(self):
        return ScopeHierarchy(
            instance_id="inst-sf",
            client_id="cli-sf",
            customer_id="cust-sf",
            user_id="user-sf",
        )

    @pytest.mark.asyncio
    async def test_concurrent_cold_calls_build_once(self):
        """Two concurrent cold calls share one build; result identical to a single call."""
        manager = RetrievalManager(cache_provider=None)  # no L3 -> straight to build

        build_count = {"n": 0}
        adjacency = {"entity_ids": ["e1"], "entity_matrix": None, "nodes": {}}

        async def _fake_build(scope):
            build_count["n"] += 1
            await asyncio.sleep(0.02)  # hold the leader so the second call coalesces
            return adjacency

        manager._build_adjacency_matrix = AsyncMock(side_effect=_fake_build)
        scope = self._scope()

        r1, r2 = await asyncio.gather(
            manager._get_adjacency_matrix(scope),
            manager._get_adjacency_matrix(scope),
        )

        assert build_count["n"] == 1, "two concurrent cold calls must build only once"
        assert r1 is r2 is adjacency, "both callers must receive the leader's result"

        # A later call hits the now-populated L2 — still only one build total.
        r3 = await manager._get_adjacency_matrix(scope)
        assert r3 is adjacency
        assert build_count["n"] == 1

    @pytest.mark.asyncio
    async def test_inflight_map_cleared_after_completion(self):
        """The in-flight future is removed once the build finishes (no leak)."""
        manager = RetrievalManager(cache_provider=None)
        adjacency = {"entity_ids": ["e1"], "entity_matrix": None, "nodes": {}}
        manager._build_adjacency_matrix = AsyncMock(return_value=adjacency)

        await manager._get_adjacency_matrix(self._scope())
        await asyncio.sleep(0.01)  # let the done-callback run
        assert manager._adjacency_inflight == {}, "inflight future must be cleared after build"

    @pytest.mark.asyncio
    async def test_build_error_shared_by_waiters(self):
        """A leader build error propagates to all coalesced waiters (they degrade the same)."""
        manager = RetrievalManager(cache_provider=None)

        build_count = {"n": 0}

        async def _boom(scope):
            build_count["n"] += 1
            await asyncio.sleep(0.02)
            raise RuntimeError("chroma down")

        manager._build_adjacency_matrix = AsyncMock(side_effect=_boom)
        scope = self._scope()

        results = await asyncio.gather(
            manager._get_adjacency_matrix(scope),
            manager._get_adjacency_matrix(scope),
            return_exceptions=True,
        )
        assert build_count["n"] == 1, "the failed build must still run only once"
        assert all(isinstance(r, RuntimeError) for r in results)
        # Map is cleared so the next request can retry from scratch.
        await asyncio.sleep(0.01)
        assert manager._adjacency_inflight == {}

    @pytest.mark.asyncio
    async def test_dirty_flag_during_inflight_build_is_not_swallowed(self):
        """A stage4 invalidation landing mid-build must not be coalesced away.

        Sequence: a leader starts building (pre-ingestion data); ingestion
        completes and stage4 sets the module-level dirty flag; a new request
        arrives. That request observes the invalidation, so it must NOT join
        the pre-invalidation flight — it drops the in-flight entry and leads
        a fresh build. The already-running leader keeps its own (old-world)
        result, exactly as pre-single-flight behaviour.
        """
        from synap.cloud.application.client_api import retrieval_manager as rm_mod

        manager = RetrievalManager(cache_provider=None)
        scope = self._scope()
        user_key = scope.user_id or scope.internal_user_id or "global"
        l2_key = f"{scope.client_id}:{scope.customer_id}:{user_key}"

        stale = {"entity_ids": ["pre-ingestion"], "entity_matrix": None, "nodes": {}}
        fresh = {"entity_ids": ["post-ingestion"], "entity_matrix": None, "nodes": {}}
        build_started = asyncio.Event()
        release_leader = asyncio.Event()
        build_count = {"n": 0}

        async def _fake_build(scope):
            build_count["n"] += 1
            if build_count["n"] == 1:
                build_started.set()
                await release_leader.wait()  # hold the leader mid-build
                return stale
            return fresh

        manager._build_adjacency_matrix = AsyncMock(side_effect=_fake_build)

        leader_task = asyncio.create_task(manager._get_adjacency_matrix(scope))
        await build_started.wait()  # leader is mid-build on pre-ingestion data

        # Ingestion completes: stage4 flags the scope dirty.
        rm_mod._l2_dirty_keys.add(l2_key)

        # The invalidation-observing request rebuilds fresh instead of joining.
        # (wait_for: with the bug this coalesces onto the blocked leader and
        # would deadlock — surface that as a fast, clean failure instead.)
        r_follower = await asyncio.wait_for(
            manager._get_adjacency_matrix(scope), timeout=2.0
        )

        release_leader.set()
        r_leader = await leader_task

        assert r_follower is fresh, "post-invalidation request must get a fresh build"
        assert r_leader is stale, "pre-invalidation request keeps its own result"
        assert build_count["n"] == 2, "the invalidation must force a second build"
        assert l2_key not in rm_mod._l2_dirty_keys, "the dirty flag was consumed"
        await asyncio.sleep(0.01)
        assert manager._adjacency_inflight == {}



class TestTheL2ConversationContextIsNotServedOnceANewerCompactionExists:
    """The post-compaction invalidation cannot rebuild the L2 key (it has external
    names, the key has internal UUIDs), so the read checks the cached entry against
    the newest compaction itself."""

    def _cached(self, compaction_id, summary="old summary"):
        return ConversationContext(
            summary=summary, compaction_id=compaction_id, conversation_id="conv-1",
        )

    def _manager(self, mock_cache, mock_scope_resolver, cached_ctx, newest_id):
        async def selective_get(key):
            if key.startswith("user_context:"):
                return {"conversation_context": cached_ctx.to_dict()}
            return None

        mock_cache.get = AsyncMock(side_effect=selective_get)
        storage = AsyncMock()  # every store method is awaitable, as the rebuild path calls several
        if newest_id is None:
            storage.get_from_cache = AsyncMock(return_value=None)
        else:
            newest = MagicMock()
            newest.metadata.compaction_id = newest_id
            storage.get_from_cache = AsyncMock(return_value=newest)
        return RetrievalManager(
            scope_resolver=mock_scope_resolver, cache_provider=mock_cache,
            compaction_storage=storage,
        ), storage

    @pytest.mark.asyncio
    async def test_a_cached_context_from_an_older_compaction_is_not_served(
        self, mock_cache, mock_scope_resolver
    ):
        manager, _ = self._manager(mock_cache, mock_scope_resolver, self._cached("c1"), "c3")
        assert await manager._l2_context_is_older_than_newest_compaction(self._cached("c1"), "conv-1")

    @pytest.mark.asyncio
    async def test_a_cached_context_from_the_newest_compaction_is_served(
        self, mock_cache, mock_scope_resolver
    ):
        manager, _ = self._manager(mock_cache, mock_scope_resolver, self._cached("c3"), "c3")
        assert not await manager._l2_context_is_older_than_newest_compaction(self._cached("c3"), "conv-1")

    @pytest.mark.asyncio
    async def test_a_context_cached_before_any_compaction_is_replaced_by_the_first(
        self, mock_cache, mock_scope_resolver
    ):
        manager, _ = self._manager(mock_cache, mock_scope_resolver, self._cached(None), "c1")
        assert await manager._l2_context_is_older_than_newest_compaction(self._cached(None), "conv-1")

    @pytest.mark.asyncio
    async def test_no_compaction_yet_serves_the_cache(self, mock_cache, mock_scope_resolver):
        manager, _ = self._manager(mock_cache, mock_scope_resolver, self._cached(None), None)
        assert not await manager._l2_context_is_older_than_newest_compaction(self._cached(None), "conv-1")

    @pytest.mark.asyncio
    async def test_a_failing_freshness_check_serves_the_cache(self, mock_cache, mock_scope_resolver):
        manager, storage = self._manager(mock_cache, mock_scope_resolver, self._cached("c1"), "c3")
        storage.get_from_cache = AsyncMock(side_effect=RuntimeError("redis down"))
        assert not await manager._l2_context_is_older_than_newest_compaction(self._cached("c1"), "conv-1")

    @pytest.mark.asyncio
    async def test_no_compaction_storage_serves_the_cache(self, mock_cache, mock_scope_resolver):
        manager = RetrievalManager(scope_resolver=mock_scope_resolver, cache_provider=mock_cache)
        assert not await manager._l2_context_is_older_than_newest_compaction(self._cached("c1"), "conv-1")

    @pytest.mark.asyncio
    async def test_the_fetch_rebuilds_instead_of_serving_the_stale_entry(
        self, mock_cache, mock_scope_resolver
    ):
        manager, _ = self._manager(mock_cache, mock_scope_resolver, self._cached("c1", "STALE"), "c3")
        served = {}

        async def rebuilt(*a, **k):
            served["rebuilt"] = True
            return None

        manager._get_compacted_context = rebuilt
        items, conv_ctx = await manager.fetch_short_term_context(
            conversation_id="conv-1", instance_id="inst-1", user_id="user-1",
            filters={"types": ["preferences"]},
        )
        assert served.get("rebuilt"), "the stale cached entry was served instead of rebuilding"
        assert not (conv_ctx and conv_ctx.summary == "STALE")

    @pytest.mark.asyncio
    async def test_the_fetch_still_serves_a_current_entry(self, mock_cache, mock_scope_resolver):
        manager, _ = self._manager(mock_cache, mock_scope_resolver, self._cached("c3", "CURRENT"), "c3")
        items, conv_ctx = await manager.fetch_short_term_context(
            conversation_id="conv-1", instance_id="inst-1", user_id="user-1",
            filters={"types": ["preferences"]},
        )
        assert conv_ctx is not None and conv_ctx.summary == "CURRENT"
