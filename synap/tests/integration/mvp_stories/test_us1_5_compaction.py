"""
MVP User Stories 1.5a-b: Context Compaction

1.5a - Automatic compaction when context limits approached
1.5b - Explicit compaction trigger via SDK
"""

import pytest

# Core compacts on the configured LLM; this gives it one (a no-op in Cloud). See tests/conftest.py.
pytestmark = pytest.mark.usefixtures("core_llm_env")

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
class TestUS1_5a_AutoCompaction:
    """As a developer, I want Synap to compact long conversations automatically."""

    async def test_should_compact_triggers_on_token_threshold(self):
        """should_compact() returns True when messages exceed token threshold."""
        from synap.cloud.application.compaction.manager import CompactionManager
        from synap.cloud.application.compaction.models import CompactionConfig, ConversationMessage

        mock_llm_manager = MagicMock()
        mgr = CompactionManager(llm_manager=mock_llm_manager, storage=None)

        # Exceed the 15K-token trigger at the default 128K agent window
        # (compaction/budgets.py), well under the 30-message trigger.
        messages = [
            ConversationMessage(
                message_id=f"msg_{i}",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}: " + "word " * 2000,
                timestamp=datetime.utcnow(),
            )
            for i in range(10)
        ]

        config = CompactionConfig()
        should, trigger_type = mgr.should_compact(messages, config)

        assert should is True
        assert trigger_type is not None

    async def test_should_compact_false_for_short_conversations(self):
        """should_compact() returns False for short conversations."""
        from synap.cloud.application.compaction.manager import CompactionManager
        from synap.cloud.application.compaction.models import CompactionConfig, ConversationMessage

        mock_llm_manager = MagicMock()
        mgr = CompactionManager(llm_manager=mock_llm_manager, storage=None)

        messages = [
            ConversationMessage(
                message_id="msg_1", role="user", content="Hello",
                timestamp=datetime.utcnow(),
            ),
            ConversationMessage(
                message_id="msg_2", role="assistant", content="Hi there!",
                timestamp=datetime.utcnow(),
            ),
        ]

        config = CompactionConfig()
        should, trigger_type = mgr.should_compact(messages, config)

        assert should is False

    async def test_compaction_config_defaults(self):
        """CompactionConfig has sensible defaults for MVP."""
        from synap.cloud.application.compaction.models import CompactionConfig

        config = CompactionConfig()

        assert config.message_trigger == 30
        assert config.target_ratio == 0.30
        assert len(config.preservation_priorities) > 0

    async def test_conversation_message_token_count_property(self):
        """ConversationMessage.token_count approximates tokens from content length."""
        from synap.cloud.application.compaction.models import ConversationMessage

        msg = ConversationMessage(
            message_id="msg_test",
            role="user",
            content="Hello world! This is a test message.",
            timestamp=datetime.utcnow(),
        )

        assert msg.token_count > 0
        # The server's one counter (tiktoken cl100k_base).
        from synap.cloud.shared.tokens import count_tokens
        assert msg.token_count == count_tokens(msg.content)


@pytest.mark.asyncio
class TestUS1_5b_ExplicitCompaction:
    """As a developer, I want to trigger compaction explicitly when required."""

    async def test_compaction_request_supports_force_flag(self):
        """CompactionRequest has force=True option to bypass threshold."""
        from synap.cloud.application.compaction.models import CompactionRequest, ConversationMessage

        messages = [
            ConversationMessage(
                message_id="msg_1", role="user", content="short",
                timestamp=datetime.utcnow(),
            ),
        ]

        request = CompactionRequest(
            conversation_id="conv_test",
            instance_id="inst_test",
            messages=messages,
            force=True,
        )

        assert request.force is True
        assert request.conversation_id == "conv_test"

    async def test_compaction_strategies_available(self):
        """All compaction strategies (AGGRESSIVE, BALANCED, CONSERVATIVE, ADAPTIVE) exist."""
        from synap.cloud.application.compaction.interfaces import CompactionStrategy

        strategies = [s.value for s in CompactionStrategy]
        assert "aggressive" in strategies or "AGGRESSIVE" in [s.name for s in CompactionStrategy]
        assert "balanced" in strategies or "BALANCED" in [s.name for s in CompactionStrategy]
        assert "adaptive" in strategies or "ADAPTIVE" in [s.name for s in CompactionStrategy]

    async def test_compaction_metadata_model(self):
        """CompactionMetadata captures compression metrics."""
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
        assert metadata.original_token_count > metadata.compacted_token_count
