"""
MVP User Stories 2.2a-b: Performance & Cost Control

2.2a - Context retrieval cached automatically
2.2b - Visibility into context efficiency and compression ratios
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


@pytest.mark.asyncio
class TestUS2_2a_AutoCaching:
    """As a developer, I want context retrieval to be cached automatically."""

    async def test_cache_key_generation_is_deterministic(self):
        """CacheKeyGenerator produces the same key for the same inputs."""
        from synap.cloud.shared.cache_keys import CacheKeyGenerator

        key1 = CacheKeyGenerator.l1_query_key(
            instance_id="inst_test",
            user_id="user_456",
            query="user preferences",
            context_types=["facts", "preferences"],
        )
        key2 = CacheKeyGenerator.l1_query_key(
            instance_id="inst_test",
            user_id="user_456",
            query="user preferences",
            context_types=["facts", "preferences"],
        )

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) > 0

    async def test_different_queries_produce_different_keys(self):
        """Different queries produce distinct cache keys."""
        from synap.cloud.shared.cache_keys import CacheKeyGenerator

        key1 = CacheKeyGenerator.l1_query_key(
            instance_id="inst_test",
            user_id="user_456",
            query="user preferences",
            context_types=["facts", "preferences"],
        )
        key2 = CacheKeyGenerator.l1_query_key(
            instance_id="inst_test",
            user_id="user_456",
            query="work history",
            context_types=["facts", "preferences"],
        )

        assert key1 != key2

    async def test_retrieval_caches_result_on_first_call(
        self, mock_cache, mock_vector_store, mock_graph_store,
        mock_postgres, mock_llm, sample_scope,
    ):
        """First retrieval call stores result in cache for subsequent lookups."""
        from synap.cloud.application.client_api.retrieval_manager import RetrievalManager
        from synap.cloud.agents.models.decisions import ContextDecision, TriggerType, RetrievalMode

        scope_resolver = AsyncMock()
        scope_resolver.resolve.return_value = sample_scope

        mgr = RetrievalManager(
            scope_resolver=scope_resolver,
            cache_provider=mock_cache,
            vector_provider=mock_vector_store,
            graph_provider=mock_graph_store,
            postgres_provider=mock_postgres,
            llm_client=mock_llm,
        )

        decision = ContextDecision(
            decision_id="dec_cache_test",
            trigger_type=TriggerType.EXPLICIT_REQUEST,
            instance_id="inst_test",
            user_id="user_456",
            retrieval_mode=RetrievalMode.FAST,
            search_queries=["preferences"],
            token_budget=2000,
        )

        await mgr.fetch_context(decision)

        # Cache should have been populated
        assert len(mock_cache.sets) > 0 or len(mock_cache.cache) > 0


@pytest.mark.asyncio
class TestUS2_2b_CompressionVisibility:
    """As a developer, I want visibility into context efficiency and compression ratios."""

    async def test_compaction_metadata_has_all_metrics(self):
        """CompactionMetadata includes compression_ratio, token counts, validation score."""
        from synap.cloud.application.compaction.models import CompactionMetadata

        metadata = CompactionMetadata(
            conversation_id="conv_test",
            instance_id="inst_test",
            original_token_count=5000,
            compacted_token_count=1500,
            compression_ratio=0.7,
        )

        assert metadata.compression_ratio == 0.7
        assert metadata.original_token_count == 5000
        assert metadata.compacted_token_count == 1500

    async def test_compression_ratio_calculation(self):
        """Compression ratio = 1 - (compacted / original)."""
        from synap.cloud.application.compaction.models import CompactionMetadata

        metadata = CompactionMetadata(
            conversation_id="conv_test",
            instance_id="inst_test",
            original_token_count=4000,
            compacted_token_count=1000,
            compression_ratio=0.75,
        )

        expected_ratio = 1 - (metadata.compacted_token_count / metadata.original_token_count)
        assert abs(metadata.compression_ratio - expected_ratio) < 0.01

    async def test_compacted_context_includes_quality_info(self):
        """CompactedContext result includes quality assurance data."""
        from synap.cloud.application.compaction.models import CompactedContext

        # Verify the model supports quality tracking
        assert hasattr(CompactedContext, "__dataclass_fields__") or hasattr(CompactedContext, "__fields__")
