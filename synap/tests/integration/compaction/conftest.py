"""Shared fixtures for compaction integration tests.

Mocks LLM and storage, runs real CompactionManager pipeline logic.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from synap.cloud.application.compaction.models import (
    ConversationMessage,
    CompactionConfig,
    PassResult,
)
from synap.cloud.application.compaction.interfaces import CompactionStrategy


@pytest.fixture
def mock_llm_manager():
    """Mock LLMConfigManager that returns structured LLM responses."""
    manager = MagicMock()
    manager.get_provider = MagicMock(return_value=MagicMock())

    # Mock the completion call
    mock_response = MagicMock()
    mock_response.content = '{"summary": "test", "facts": [], "score": 0.9}'
    mock_response.usage = MagicMock(total_tokens=100)

    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value=mock_response)
    mock_provider.acompletion = AsyncMock(return_value=mock_response)
    manager.get_provider.return_value = mock_provider

    return manager


@pytest.fixture
def mock_compaction_storage():
    """Mock CompactionStorage for tests."""
    storage = AsyncMock()
    storage.save = AsyncMock()
    storage.get_latest = AsyncMock(return_value=None)
    storage.get_version = AsyncMock(return_value=None)
    storage.get_from_cache = AsyncMock(return_value=None)
    storage.set_in_cache = AsyncMock()
    storage.invalidate_cache = AsyncMock()
    storage.is_compaction_in_progress = AsyncMock(return_value=False)
    storage.set_compaction_in_progress = AsyncMock()
    storage.clear_compaction_in_progress = AsyncMock()
    storage.load_conversation_messages = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def sample_messages():
    """List of ConversationMessage objects for testing."""
    now = datetime.now(timezone.utc)
    return [
        ConversationMessage(
            message_id=f"msg-{i}",
            role="user" if i % 2 == 0 else "assistant",
            # ~1K tokens each, ~20K in all: over the 15K trigger at the 128K default.
            content=f"This is message number {i} with some detailed content about topic {i % 3} and various subjects. " * 50,
            timestamp=now,
        )
        for i in range(20)
    ]


@pytest.fixture
def sample_config():
    """CompactionConfig for testing."""
    return CompactionConfig(
        strategy=CompactionStrategy.BALANCED,
        validation_threshold=0.8,
    )


@pytest.fixture
def short_messages():
    """Short message list (below threshold)."""
    now = datetime.now(timezone.utc)
    return [
        ConversationMessage(
            message_id="msg-1",
            role="user",
            content="Hello",
            timestamp=now,
        ),
        ConversationMessage(
            message_id="msg-2",
            role="assistant",
            content="Hi there!",
            timestamp=now,
        ),
    ]
