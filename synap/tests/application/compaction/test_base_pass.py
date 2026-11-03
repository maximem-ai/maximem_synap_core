"""
Tests for BaseCompactionPass and LLM integration.

Verifies that compaction passes correctly use LLMConfigManager.
"""
import pytest


# Core compacts on the configured LLM; this gives it one (a no-op in Cloud). See tests/conftest.py.
pytestmark = pytest.mark.usefixtures("core_llm_env")

from synap.cloud.application.compaction.passes.base import GROQ_PROFILE
from synap.cloud.application.compaction.passes import BaseCompactionPass
from synap.cloud.infrastructure.llm import LLMProvider


class TestBaseCompactionPass:
    """Tests for BaseCompactionPass with LLMConfigManager."""
    
    def test_base_pass_initialization(self, mock_llm_manager):
        """Test that BaseCompactionPass initializes with LLMConfigManager."""
        # Create a concrete implementation for testing
        class TestPass(BaseCompactionPass):
            @property
            def pass_name(self):
                return "test_pass"
            
            @property
            def pass_number(self):
                return 99
            
            @property
            def temperature(self):
                return 0.0
            
            async def _execute_impl(self, input_data, config, previous_results):
                return {"test": "result"}
        
        pass_instance = TestPass(mock_llm_manager)
        
        assert pass_instance._llm_manager is mock_llm_manager
        assert pass_instance._tokens_consumed == 0
        assert pass_instance._retry_count == 0
    
    def test_compaction_provider_constants(self):
        """Test that compaction provider constants are set correctly."""
        assert BaseCompactionPass.COMPACTION_PROVIDER == LLMProvider.GROQ
        assert BaseCompactionPass.COMPACTION_MODEL == "openai/gpt-oss-120b"
    
    @pytest.mark.asyncio
    async def test_call_llm_uses_forced_provider(self, mock_llm_manager):
        """Test that _call_llm forces Groq provider and model."""
        class TestPass(BaseCompactionPass):
            @property
            def pass_name(self):
                return "test_pass"
            
            @property
            def pass_number(self):
                return 99
            
            @property
            def temperature(self):
                return 0.0
            
            async def _execute_impl(self, input_data, config, previous_results):
                return await self._call_llm(
                    system_prompt="Test system",
                    user_prompt="Test user"
                )
        
        # Groq is chosen explicitly: the default profile is gpt-4o when an OpenAI
        # key is present (test_compaction_profile.py pins that).
        pass_instance = TestPass(mock_llm_manager, GROQ_PROFILE)
        result = await pass_instance._call_llm(
            system_prompt="Test system prompt",
            user_prompt="Test user prompt",
            max_tokens=2000,
            json_mode=True
        )
        
        # Verify LLM was called
        mock_llm_manager.complete.assert_called_once()
        
        # Verify call arguments
        call_kwargs = mock_llm_manager.complete.call_args.kwargs
        assert call_kwargs["force_provider"] == LLMProvider.GROQ
        assert call_kwargs["force_model"] == "openai/gpt-oss-120b"
        assert call_kwargs["temperature"] == 0.0
        # Requested max_tokens is propagated with the reasoning-token overhead added
        # (Groq reasoning models consume reasoning tokens from the max_tokens budget).
        assert call_kwargs["max_tokens"] == 2000 + BaseCompactionPass.REASONING_TOKEN_OVERHEAD
        assert call_kwargs["json_mode"] is True
        
        # Verify messages structure
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Test system prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test user prompt"
        
        # Verify tokens were tracked
        assert pass_instance._tokens_consumed == 150  # From mock response
    
    @pytest.mark.asyncio
    async def test_execute_wraps_implementation(self, mock_llm_manager, sample_compaction_config):
        """Test that execute wraps _execute_impl with timing and error handling."""
        class TestPass(BaseCompactionPass):
            @property
            def pass_name(self):
                return "test_pass"
            
            @property
            def pass_number(self):
                return 99
            
            @property
            def temperature(self):
                return 0.0
            
            async def _execute_impl(self, input_data, config, previous_results):
                # Take a measurable amount of time so the wrapper records a
                # non-zero latency (a trivial sync return can complete within
                # the timer's resolution and round to 0.0ms).
                import asyncio
                await asyncio.sleep(0.001)
                return {"success": True, "data": "test"}

        pass_instance = TestPass(mock_llm_manager)
        result = await pass_instance.execute(
            input_data={"test": "data"},
            config=sample_compaction_config,
            previous_results={}
        )
        
        assert result.success is True
        assert result.pass_name == "test_pass"
        assert result.pass_number == 99
        assert result.output == {"success": True, "data": "test"}
        assert result.latency_ms > 0
        assert result.error is None
