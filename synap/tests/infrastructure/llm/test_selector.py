"""
Tests for LLM Model Selector.
"""

import pytest
from synap.cloud.infrastructure.llm import (
    LLMProvider,
    SelectionCriteria,
)


class TestLLMModelSelector:
    """Test model selection logic."""

    @pytest.mark.asyncio
    async def test_select_with_balanced_criteria(self, selector):
        """Test selection with balanced criteria."""
        criteria = SelectionCriteria(
            speed=5,
            accuracy=5,
            reasoning=5,
            tool_usage=5
        )
        
        model = await selector.select_model(criteria=criteria)
        assert model is not None
        # Enabled providers are openai + groq. Gemini is disabled in the
        # registry (no deployment has a key) and must never be picked.
        assert model.provider in [
            LLMProvider.OPENAI,
            LLMProvider.GROQ,
        ]

    @pytest.mark.asyncio
    async def test_select_with_speed_priority(self, selector):
        """Test selection prioritizing speed."""
        criteria = SelectionCriteria(
            speed=10,
            accuracy=1,
            reasoning=1,
            tool_usage=1
        )
        
        model = await selector.select_model(criteria=criteria)
        assert model is not None
        # Should select a fast model (score 9-10)
        assert model.scores.speed >= 9

    @pytest.mark.asyncio
    async def test_select_with_accuracy_priority(self, selector):
        """Test selection prioritizing accuracy."""
        criteria = SelectionCriteria(
            speed=1,
            accuracy=10,
            reasoning=10,
            tool_usage=1
        )
        
        model = await selector.select_model(criteria=criteria)
        assert model is not None
        # Should select high-accuracy model (Claude or GPT-4o)
        assert model.scores.accuracy >= 9

    @pytest.mark.asyncio
    async def test_select_with_agent_type(self, selector):
        """Test selection using agent type defaults."""
        model = await selector.select_model(agent_type="listening_retrieval_agent")
        assert model is not None
        # listening_retrieval_agent prioritizes speed
        assert model.scores.speed >= 8

    @pytest.mark.asyncio
    async def test_select_with_force_provider(self, selector):
        """Test forcing specific provider."""
        model = await selector.select_model(
            force_provider=LLMProvider.GROQ
        )
        assert model is not None
        assert model.provider == LLMProvider.GROQ

    @pytest.mark.asyncio
    async def test_select_with_exclude_models(self, selector):
        """Test excluding specific models."""
        model = await selector.select_model(
            force_provider=LLMProvider.OPENAI,
            exclude_models=["gpt-4o"]
        )
        assert model is not None
        assert model.name != "gpt-4o"
        assert model.provider == LLMProvider.OPENAI

    @pytest.mark.asyncio
    async def test_fallback_same_provider(self, selector):
        """Test fallback to different model in same provider."""
        fallback = await selector.get_fallback_model(
            original_provider=LLMProvider.OPENAI,
            original_model="gpt-4o"
        )
        assert fallback is not None
        assert fallback.provider == LLMProvider.OPENAI
        assert fallback.name != "gpt-4o"

    @pytest.mark.asyncio
    async def test_fallback_different_provider(self, selector):
        """Test fallback to different provider."""
        fallback = await selector.get_fallback_model(
            original_provider=None,
            original_model="gpt-4o"
        )
        assert fallback is not None
        assert fallback.name != "gpt-4o"

    @pytest.mark.asyncio
    async def test_fallback_returns_default(self, selector, tracker, registry):
        """Test fallback returns default when all else fails."""
        default_provider = registry.get_default_provider()
        default_model = registry.get_default_model()

        # Mark every enabled model unavailable so the only path left is the
        # registry-default last resort (which is returned regardless of its
        # availability flag).
        for provider in registry.get_enabled_providers():
            for model_info in registry.get_models_by_provider(provider):
                await tracker.mark_unavailable(provider, model_info.name, 60)

        fallback = await selector.get_fallback_model(
            original_provider=None,
            original_model="nonexistent"
        )
        assert fallback is not None
        assert fallback.name == default_model  # the registry default (openai/gpt-4o)
        assert fallback.provider == default_provider

    @pytest.mark.asyncio
    async def test_select_respects_availability(self, selector, tracker):
        """Test that unavailable models are not selected."""
        # Mark a model unavailable
        await tracker.mark_unavailable(LLMProvider.OPENAI, "gpt-4o", 60)
        
        # Try to select with high criteria that would favor gpt-4o
        criteria = SelectionCriteria(
            speed=5,
            accuracy=10,
            reasoning=10,
            tool_usage=10
        )
        
        model = await selector.select_model(
            force_provider=LLMProvider.OPENAI,
            criteria=criteria
        )
        
        # Should get a different model since gpt-4o is unavailable
        assert model is not None
        assert model.name != "gpt-4o"

    @pytest.mark.asyncio
    async def test_select_no_models_available(self, selector, tracker, registry):
        """Test selection when no models are available."""
        # Mark every enabled model unavailable (driven off the registry so it
        # tracks model/provider renames, e.g. openai/gpt-oss-120b and the
        # gemini models).
        for provider in registry.get_enabled_providers():
            for model_info in registry.get_models_by_provider(provider):
                await tracker.mark_unavailable(provider, model_info.name, 60)

        model = await selector.select_model()
        assert model is None


# ---------------------------------------------------------------------------
# The selector may only offer providers this process can actually call
# ---------------------------------------------------------------------------

# The old registry shape: gemini enabled with
# the best paper scores AND named as the default, while no deployment had a
# Google key. Kept here as a fixture so the guard reproduces the real failure.
OLD_PROD_REGISTRY = """
defaults:
  provider: "gemini"
  model: "gemini-2.5-flash"
providers:
  gemini:
    enabled: true
    rate_limits: {rpm: 2000, tpm: 4000000}
    models:
      gemini-2.5-flash:
        type: "flagship"
        max_tokens: 8192
        context_window: 1000000
        scores: {speed: 10, accuracy: 9, reasoning: 9, tool_usage: 9}
  openai:
    enabled: true
    rate_limits: {rpm: 10000, tpm: 2000000}
    models:
      gpt-4o:
        type: "flagship"
        max_tokens: 4096
        context_window: 128000
        scores: {speed: 8, accuracy: 9, reasoning: 9, tool_usage: 10}
      gpt-4o-mini:
        type: "economy"
        max_tokens: 16384
        context_window: 128000
        scores: {speed: 10, accuracy: 7, reasoning: 6, tool_usage: 8}
"""


class TestNeverOffersAProviderThatIsNotRunning:
    """⚠ Written after repeated "Provider gemini not available" failures.

    Two lists disagreed. The registry said which providers MAY be used; the
    manager's initialised providers said which CAN be (a key was present at
    startup). The selector read only the first, so it kept offering
    gemini-2.5-flash as first pick, as fallback and as last-resort default, and
    every one of those calls failed before reaching a model. These tests feed
    the selector the old registry and a process that only has
    OpenAI, and require that gemini is never returned by any path.
    """

    @pytest.fixture
    def old_prod_registry(self, tmp_path):
        from synap.cloud.infrastructure.llm import LLMRegistry
        path = tmp_path / "llm_registry.yaml"
        path.write_text(OLD_PROD_REGISTRY)
        return LLMRegistry(str(path))

    @pytest.fixture
    def openai_only_selector(self, old_prod_registry, tracker):
        from synap.cloud.infrastructure.llm import LLMModelSelector
        return LLMModelSelector(
            old_prod_registry, tracker,
            available_providers=lambda: [LLMProvider.OPENAI],
        )

    @pytest.mark.asyncio
    async def test_the_old_selector_really_did_pick_gemini(self, old_prod_registry, tracker):
        # The bug, reproduced: without the availability list the registry's
        # paper scores win and a provider with no client is selected.
        from synap.cloud.infrastructure.llm import LLMModelSelector
        unguarded = LLMModelSelector(old_prod_registry, tracker)
        assert (await unguarded.select_model()).provider == LLMProvider.GEMINI

    @pytest.mark.asyncio
    async def test_first_pick_skips_a_registry_enabled_provider_with_no_client(self, openai_only_selector):
        model = await openai_only_selector.select_model()
        assert model is not None
        assert model.provider == LLMProvider.OPENAI

    @pytest.mark.asyncio
    async def test_fallback_never_lands_on_the_missing_provider(self, openai_only_selector):
        fallback = await openai_only_selector.get_fallback_model(
            original_provider=None, original_model="gpt-4o",
        )
        assert fallback is not None
        assert fallback.provider == LLMProvider.OPENAI
        assert fallback.name == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_last_resort_default_is_withheld_when_its_provider_is_missing(
        self, openai_only_selector, old_prod_registry, tracker
    ):
        # Every callable model is out; the registry default is gemini, which
        # this process cannot call. The honest answer is None, not a model
        # that fails with a misleading name.
        for model_info in old_prod_registry.get_models_by_provider(LLMProvider.OPENAI):
            await tracker.mark_unavailable(LLMProvider.OPENAI, model_info.name, 60)
        assert await openai_only_selector.get_fallback_model(
            original_provider=None, original_model="nonexistent",
        ) is None

    @pytest.mark.asyncio
    async def test_last_resort_default_is_withheld_when_it_is_the_model_that_failed(
        self, old_prod_registry, tracker
    ):
        from synap.cloud.infrastructure.llm import LLMModelSelector
        gemini_only = LLMModelSelector(
            old_prod_registry, tracker,
            available_providers=lambda: [LLMProvider.GEMINI],
        )
        await tracker.mark_unavailable(LLMProvider.GEMINI, "gemini-2.5-flash", 60)
        # The default IS the model that just failed; handing it back again is
        # a retry loop wearing a fallback's name.
        assert await gemini_only.get_fallback_model(
            original_provider=None, original_model="gemini-2.5-flash",
        ) is None
