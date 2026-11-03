"""
Tests for LLM Config Manager.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from synap.cloud.infrastructure.llm import (
    LLMConfigManager,
    LLMProvider,
    LLMResponse,
    LLMError,
    RateLimitError,
    ProviderError,
    SelectionCriteria,
)


class TestLLMConfigManager:
    """Test the main config manager."""

    @pytest.fixture
    async def manager(self, redis_client, test_config_path):
        """Create a manager instance.

        No provider key is set under test, so `_initialize_providers` creates
        nothing, and the selector (which now offers only initialised
        providers) would have no candidates. Stand in a client for each
        registry-enabled provider; the tests below patch `get_provider`
        anyway, so these stand-ins are never called.
        """
        manager = LLMConfigManager(redis_client, test_config_path)
        for provider in manager.get_registry().get_enabled_providers():
            manager._providers[provider] = MagicMock(name=f"stand-in {provider.value}")
        return manager

    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager is not None
        assert manager.get_registry() is not None
        assert manager.get_tracker() is not None

    @pytest.mark.asyncio
    async def test_get_provider(self, manager):
        """Test getting provider instances."""
        # Should have at least one provider available
        providers = manager.get_registry().get_enabled_providers()
        assert len(providers) > 0

    def test_get_provider_not_available(self, manager):
        """Test error when provider not available."""
        # Anthropic is disabled in the registry, so it was never initialised.
        with pytest.raises(ProviderError):
            manager.get_provider(LLMProvider.ANTHROPIC)

    @pytest.mark.asyncio
    async def test_selector_offers_only_initialised_providers(self, manager):
        """The registry may enable a provider this process has no key for.
        The selector must never offer it. Groq is enabled in the registry;
        take its client away and it must vanish from every selection."""
        manager._providers.pop(LLMProvider.GROQ, None)
        registry = manager.get_registry()
        assert LLMProvider.GROQ in registry.get_enabled_providers()  # still "enabled"

        for _ in range(3):
            picked = await manager._selector.select_model(
                criteria=SelectionCriteria(speed=10, accuracy=1, reasoning=1, tool_usage=1)
            )
            assert picked is not None
            assert picked.provider != LLMProvider.GROQ
        fallback = await manager._selector.get_fallback_model(
            original_provider=None, original_model="gpt-4o"
        )
        assert fallback is None or fallback.provider != LLMProvider.GROQ

    @pytest.mark.asyncio
    async def test_complete_with_agent_type(self, manager):
        """Test completion with agent type."""
        # Mock the provider
        mock_response = LLMResponse(
            content="Test response",
            model="gpt-4o-mini",
            provider=LLMProvider.OPENAI,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30
        )
        
        with patch.object(manager, 'get_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_get_provider.return_value = mock_provider
            
            response = await manager.complete(
                messages=[{"role": "user", "content": "Hello"}],
                agent_type="listening_retrieval_agent"
            )
            
            assert response is not None
            assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_complete_with_criteria(self, manager):
        """Test completion with custom criteria."""
        mock_response = LLMResponse(
            content="Test response",
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30
        )
        
        criteria = SelectionCriteria(
            speed=10,
            accuracy=5,
            reasoning=3,
            tool_usage=8
        )
        
        with patch.object(manager, 'get_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_get_provider.return_value = mock_provider
            
            response = await manager.complete(
                messages=[{"role": "user", "content": "Hello"}],
                criteria=criteria
            )
            
            assert response is not None

    @pytest.mark.asyncio
    async def test_complete_with_force_model(self, manager):
        """Test completion with forced model."""
        mock_response = LLMResponse(
            content="Test response",
            model="openai/gpt-oss-120b",
            provider=LLMProvider.GROQ,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30
        )

        with patch.object(manager, 'get_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_get_provider.return_value = mock_provider

            response = await manager.complete(
                messages=[{"role": "user", "content": "Hello"}],
                force_provider=LLMProvider.GROQ,
                force_model="openai/gpt-oss-120b"
            )
            
            assert response is not None
            assert response.provider == LLMProvider.GROQ

    @pytest.mark.asyncio
    async def test_complete_retry_on_rate_limit(self, manager):
        """Test retry logic on rate limit error."""
        mock_success_response = LLMResponse(
            content="Success",
            model="gpt-4o-mini",
            provider=LLMProvider.OPENAI,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30
        )
        
        with patch.object(manager, 'get_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            # First call fails with rate limit, second succeeds
            mock_provider.complete = AsyncMock(
                side_effect=[
                    RateLimitError(provider="openai", retry_after=1),
                    mock_success_response
                ]
            )
            mock_get_provider.return_value = mock_provider
            
            response = await manager.complete(
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            # Should eventually succeed with fallback
            assert response is not None

    @pytest.mark.asyncio
    async def test_complete_all_retries_exhausted(self, manager):
        """Test error when all retries exhausted."""
        with patch.object(manager, 'get_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(
                side_effect=ProviderError("Server error", provider="openai", status_code=500)
            )
            mock_get_provider.return_value = mock_provider
            
            with pytest.raises(LLMError) as exc_info:
                await manager.complete(
                    messages=[{"role": "user", "content": "Hello"}]
                )
            # Every attempt is named. "Last error: <fallback's error>" hid the
            # primary model's real rejection for an afternoon.
            msg = str(exc_info.value)
            assert msg.startswith("All models failed.")
            assert msg.count("Server error") >= 2, msg

    @pytest.mark.asyncio
    async def test_provider_error_leaves_the_forced_provider_by_default(self, manager):
        """The behaviour ingestion had: a gpt-4o error was answered by Groq.

        Retrieval depends on this in the other direction (Groq down, gpt-4o
        answers), so it stays the default.
        """
        asked = []

        async def complete(model, **kwargs):
            asked.append(model)
            raise ProviderError("Server error", provider="openai", status_code=500)

        with patch.object(manager, 'get_provider') as mock_get_provider:
            mock_get_provider.return_value = MagicMock(complete=complete)
            with pytest.raises(LLMError):
                await manager.complete(
                    messages=[{"role": "user", "content": "Hello"}],
                    force_provider=LLMProvider.OPENAI,
                    force_model="gpt-4o",
                )

        assert asked[0] == "gpt-4o"
        assert "openai/gpt-oss-120b" in asked, asked

    @pytest.mark.asyncio
    async def test_provider_error_stays_on_a_pinned_provider(self, manager):
        """With the fallback pinned, no other provider's model is ever tried."""
        asked = []

        async def complete(model, **kwargs):
            asked.append(model)
            raise ProviderError("Server error", provider="openai", status_code=500)

        with patch.object(manager, 'get_provider') as mock_get_provider:
            mock_get_provider.return_value = MagicMock(complete=complete)
            with pytest.raises(LLMError):
                await manager.complete(
                    messages=[{"role": "user", "content": "Hello"}],
                    force_provider=LLMProvider.OPENAI,
                    force_model="gpt-4o",
                    allow_cross_provider_fallback=False,
                )

        openai_models = {
            m.name for m in manager.get_registry().get_models_by_provider(LLMProvider.OPENAI)
        }
        assert asked == ["gpt-4o", "gpt-4o-mini"], asked
        assert set(asked) <= openai_models

    @pytest.mark.asyncio
    async def test_pinned_provider_with_no_other_model_does_not_leave(self, manager):
        """The selector goes cross-provider when the pinned one has nothing
        left to offer; pinning has to hold there too."""
        asked = []

        async def complete(model, **kwargs):
            asked.append(model)
            raise ProviderError("Server error", provider="openai", status_code=500)

        only_gpt4o = [
            m for m in manager.get_registry().get_models_by_provider(LLMProvider.OPENAI)
            if m.name == "gpt-4o"
        ]
        real = manager.get_registry().get_models_by_provider

        def one_openai_model(provider):
            return only_gpt4o if provider == LLMProvider.OPENAI else real(provider)

        with patch.object(manager, 'get_provider') as mock_get_provider, \
                patch.object(manager.get_registry(), 'get_models_by_provider', one_openai_model):
            mock_get_provider.return_value = MagicMock(complete=complete)
            with pytest.raises(LLMError):
                await manager.complete(
                    messages=[{"role": "user", "content": "Hello"}],
                    force_provider=LLMProvider.OPENAI,
                    force_model="gpt-4o",
                    allow_cross_provider_fallback=False,
                )

        assert asked == ["gpt-4o"], asked

    @pytest.mark.asyncio
    async def test_complete_tracks_usage(self, manager):
        """Test that successful requests track usage."""
        mock_response = LLMResponse(
            content="Test",
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150
        )
        
        with patch.object(manager, 'get_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_get_provider.return_value = mock_provider
            
            await manager.complete(
                messages=[{"role": "user", "content": "Hello"}],
                force_provider=LLMProvider.OPENAI,
                force_model="gpt-4o"
            )
            
            # Check usage was recorded
            status = await manager.get_model_status(LLMProvider.OPENAI, "gpt-4o")
            assert status["rpm_usage"] > 0

    @pytest.mark.asyncio
    async def test_get_model_status(self, manager):
        """Test getting model status."""
        status = await manager.get_model_status(LLMProvider.OPENAI, "gpt-4o")
        assert "provider" in status
        assert "model" in status
        assert "is_available" in status
        assert "availability_factor" in status

    @pytest.mark.asyncio
    async def test_get_model_status_not_found(self, manager):
        """Test status for non-existent model."""
        status = await manager.get_model_status(LLMProvider.OPENAI, "nonexistent")
        assert "error" in status

    @pytest.mark.asyncio
    async def test_stream(self, manager):
        """Test streaming completion."""
        async def mock_stream(*args, **kwargs):
            for chunk in ["Hello", " ", "world"]:
                yield chunk
        
        with patch.object(manager, 'get_provider') as mock_get_provider:
            mock_provider = AsyncMock()
            mock_provider.stream = mock_stream
            mock_get_provider.return_value = mock_provider
            
            chunks = []
            async for chunk in manager.stream(
                messages=[{"role": "user", "content": "Hello"}]
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert "".join(chunks) == "Hello world"

    @pytest.mark.asyncio
    async def test_complete_no_models_available(self, manager):
        """Test error when no models available."""
        # Mark all enabled models unavailable. Driven off the registry's
        # enabled-provider set so it covers gemini (now enabled/default) and
        # skips anthropic (now disabled); a hardcoded list would leave the
        # gemini models available and the selector would still succeed.
        tracker = manager.get_tracker()
        registry = manager.get_registry()
        for provider in registry.get_enabled_providers():
            for model in registry.get_models_by_provider(provider):
                await tracker.mark_unavailable(provider, model.name, 60)
        
        with pytest.raises(LLMError, match="No models available"):
            await manager.complete(
                messages=[{"role": "user", "content": "Hello"}]
            )
