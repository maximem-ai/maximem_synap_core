"""
Pytest fixtures for compaction tests.

Provides mocked LLMConfigManager and related fixtures.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

from synap.cloud.infrastructure.llm import (
    LLMConfigManager,
    LLMResponse,
    LLMProvider
)
from synap.cloud.application.compaction import (
    CompactionManager,
    CompactionWorker,
    CompactionStorage
)


@pytest.fixture
def mock_llm_manager():
    """
    Create a mocked LLMConfigManager for testing.
    
    Returns a mock that simulates successful Groq completions.
    """
    manager = MagicMock(spec=LLMConfigManager)
    
    async def mock_complete(**kwargs) -> LLMResponse:
        """Mock completion that returns valid JSON responses."""
        # Default response - can be customized in tests
        return LLMResponse(
            content='{"result": "test", "success": true}',
            model="gpt-oss-20b",
            provider=LLMProvider.GROQ,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=250.0,
            finish_reason="stop"
        )
    
    manager.complete = AsyncMock(side_effect=mock_complete)
    return manager


@pytest.fixture
def mock_llm_manager_with_responses(mock_llm_manager):
    """
    Create a mocked LLMConfigManager that can return custom responses.
    
    Usage in tests:
        mock_llm_manager_with_responses.set_response('{"custom": "response"}')
    """
    responses = []
    
    async def mock_complete_with_queue(**kwargs) -> LLMResponse:
        """Return responses from queue or default."""
        if responses:
            content = responses.pop(0)
        else:
            content = '{"result": "default"}'
            
        return LLMResponse(
            content=content,
            model="gpt-oss-20b",
            provider=LLMProvider.GROQ,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=250.0,
            finish_reason="stop"
        )
    
    mock_llm_manager.complete = AsyncMock(side_effect=mock_complete_with_queue)
    mock_llm_manager.set_response = lambda resp: responses.append(resp)
    mock_llm_manager.set_responses = lambda resps: responses.extend(resps)
    
    return mock_llm_manager


@pytest.fixture
def compaction_manager(mock_llm_manager):
    """
    Create a CompactionManager instance with mocked LLM manager.
    
    Uses in-memory storage for testing.
    """
    storage = CompactionStorage()
    return CompactionManager(
        llm_manager=mock_llm_manager,
        storage=storage
    )


@pytest.fixture
def compaction_worker(mock_llm_manager):
    """
    Create a CompactionWorker instance with mocked LLM manager.
    
    Configured with short intervals for testing.
    """
    return CompactionWorker(
        llm_manager=mock_llm_manager,
        check_interval_seconds=1,  # Short interval for tests
        idle_threshold_minutes=1   # Short threshold for tests
    )


@pytest.fixture
def sample_messages():
    """
    Sample conversation messages for testing.
    
    Returns a list of ConversationMessage objects.
    """
    from synap.cloud.application.compaction import ConversationMessage
    from datetime import datetime
    
    return [
        ConversationMessage(
            message_id="msg_1",
            role="user",
            content="I need help debugging my API integration.",
            timestamp=datetime.utcnow(),
            token_count=10
        ),
        ConversationMessage(
            message_id="msg_2",
            role="assistant",
            content="I'd be happy to help you debug your API integration. Can you provide more details about the error you're seeing?",
            timestamp=datetime.utcnow(),
            token_count=25
        ),
        ConversationMessage(
            message_id="msg_3",
            role="user",
            content="I'm getting a 401 Unauthorized error when trying to authenticate.",
            timestamp=datetime.utcnow(),
            token_count=15
        ),
        ConversationMessage(
            message_id="msg_4",
            role="assistant",
            content="A 401 error typically indicates an authentication problem. Let's check your API key configuration.",
            timestamp=datetime.utcnow(),
            token_count=20
        ),
    ]


@pytest.fixture
def sample_compaction_config():
    """
    Sample CompactionConfig for testing.
    """
    from synap.cloud.application.compaction import CompactionConfig, CompactionStrategy

    return CompactionConfig(
        strategy=CompactionStrategy.BALANCED,
        target_ratio=0.30,
        message_trigger=30,
        validation_threshold=0.8
    )


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    import fakeredis.aioredis
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    from unittest.mock import MagicMock
    return MagicMock()


# Pass-specific fixtures for testing individual passes

@pytest.fixture
def pass0_density_response():
    """Sample response for Pass 0 (Density Scoring)."""
    return '''{
        "segments": [
            {
                "segment_id": 0,
                "density_score": 0.8,
                "redundancy_score": 0.2,
                "criticality_score": 0.9,
                "density_level": "high",
                "compression_rate": 0.35
            }
        ]
    }'''


@pytest.fixture
def pass1_classification_response():
    """Sample response for Pass 1 (Classification)."""
    return '''{
        "conversation_type": "technical_debugging",
        "confidence": 0.9,
        "primary_objective": "Debug API authentication issue",
        "secondary_objectives": ["Identify error cause", "Fix configuration"],
        "reasoning": "User is troubleshooting a 401 authentication error"
    }'''


@pytest.fixture
def pass2_extraction_response():
    """Sample response for Pass 2 (Extraction)."""
    return '''{
        "facts": [
            {
                "fact_id": "fact_1",
                "content": "User is experiencing 401 Unauthorized error",
                "confidence": "high",
                "message_ids": ["msg_3"],
                "category": "explicit_facts"
            }
        ],
        "code_artifacts": [],
        "decisions": []
    }'''


@pytest.fixture
def pass3_summarization_response():
    """Sample response for Pass 3 (Summarization)."""
    return '''{
        "summaries": [
            {
                "segment_id": 0,
                "summary": "User requested help debugging API integration, specifically a 401 authentication error.",
                "compression_ratio": 0.4,
                "preserved_message_ids": ["msg_1", "msg_3"]
            }
        ]
    }'''


@pytest.fixture
def pass4_validation_response():
    """Sample response for Pass 4 (Validation)."""
    return '''{
        "validation_questions": [
            {
                "question": "What error is the user experiencing?",
                "expected_answer": "401 Unauthorized",
                "actual_answer": "401 Unauthorized",
                "correct": true
            }
        ],
        "overall_score": 0.95,
        "passed": true
    }'''


@pytest.fixture
def pass5_merge_response():
    """Sample response for Pass 5 (Merge)."""
    return '''{
        "merged_context": "User is debugging a 401 API authentication error.",
        "narrative_summary": {
            "overview": "API authentication troubleshooting session",
            "chronological_narrative": "User reported 401 error and requested debugging help.",
            "current_state": "Investigating API key configuration"
        },
        "total_facts": 1,
        "total_decisions": 0
    }'''
