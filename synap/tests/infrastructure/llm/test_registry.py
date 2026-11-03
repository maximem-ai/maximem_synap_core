"""
Tests for LLM Registry.
"""

import pytest
from synap.cloud.infrastructure.llm import LLMRegistry, LLMProvider, ModelType


class TestLLMRegistry:
    """Test registry configuration loading and model lookup."""

    def test_load_config(self, registry):
        """Test that config loads successfully."""
        assert registry is not None
        providers = registry.get_enabled_providers()
        assert len(providers) > 0

    def test_get_enabled_providers(self, registry):
        """Test getting list of enabled providers."""
        # Enabled set is openai + groq. Anthropic is disabled in the file.
        # Gemini is disabled on purpose: no deployment carries a Google key,
        # and an enabled-but-keyless provider gets offered and then fails.
        providers = registry.get_enabled_providers()
        assert LLMProvider.OPENAI in providers
        assert LLMProvider.GROQ in providers
        assert LLMProvider.GEMINI not in providers
        assert LLMProvider.ANTHROPIC not in providers

    def test_get_default_provider(self, registry):
        """Test getting default provider."""
        default_provider = registry.get_default_provider()
        assert default_provider == LLMProvider.OPENAI

    def test_get_default_model(self, registry):
        """Test getting default model."""
        default_model = registry.get_default_model()
        assert default_model == "gpt-4o"

    def test_gemini_is_never_a_default(self, registry):
        """Gemini is never our default, anywhere a default is named.

        ⚠ It was, for the registry: `defaults.provider: gemini` plus
        `gemini.enabled: true` with the best paper scores, and no deployment
        with a Google key. Every cross-provider fallback
        therefore went to gemini-2.5-flash and died with "Provider gemini not
        available", which also hid the primary
        model's real error. This test refuses that shape in the registry AND
        in the per-agent defaults.
        """
        from synap.cloud.infrastructure.llm.models import AGENT_DEFAULTS

        assert registry.get_default_provider() != LLMProvider.GEMINI
        assert not registry.get_default_model().lower().startswith("gemini")
        assert LLMProvider.GEMINI not in registry.get_enabled_providers()
        for agent_type, cfg in AGENT_DEFAULTS.items():
            assert cfg.get("provider_override") != LLMProvider.GEMINI, agent_type
            for model in cfg.get("preferred_models", []):
                assert not model.lower().startswith("gemini"), (agent_type, model)

    def test_get_model_info_success(self, registry):
        """Test retrieving model info for valid model."""
        model_info = registry.get_model_info(LLMProvider.OPENAI, "gpt-4o")
        assert model_info is not None
        assert model_info.name == "gpt-4o"
        assert model_info.provider == LLMProvider.OPENAI
        assert model_info.model_type == ModelType.FLAGSHIP
        assert model_info.scores.speed == 8
        assert model_info.scores.accuracy == 9
        assert model_info.rate_limits.rpm == 5000
        assert model_info.rate_limits.tpm == 800000

    def test_get_model_info_not_found(self, registry):
        """Test retrieving model info for invalid model."""
        model_info = registry.get_model_info(LLMProvider.OPENAI, "nonexistent-model")
        assert model_info is None

    def test_get_models_by_provider(self, registry):
        """Test getting all models for a provider."""
        models = registry.get_models_by_provider(LLMProvider.OPENAI)
        assert len(models) >= 2
        model_names = [m.name for m in models]
        assert "gpt-4o" in model_names
        assert "gpt-4o-mini" in model_names

    def test_get_all_models(self, registry):
        """Test getting all models across providers."""
        all_models = registry.get_all_models()
        # Enabled providers: openai (2) + groq (2). Gemini and anthropic are
        # disabled so their models are not loaded at all.
        assert len(all_models) >= 4
        providers = set(m.provider for m in all_models)
        assert LLMProvider.OPENAI in providers
        assert LLMProvider.GROQ in providers
        assert LLMProvider.GEMINI not in providers

    def test_model_scores_loaded(self, registry):
        """Test that model scores are loaded correctly."""
        # gpt-4o carries the flagship accuracy/reasoning scores from the YAML.
        model_info = registry.get_model_info(LLMProvider.OPENAI, "gpt-4o")
        assert model_info is not None
        assert model_info.scores.accuracy == 9
        assert model_info.scores.reasoning == 9

    def test_rate_limits_loaded(self, registry):
        """Test that rate limits are loaded correctly."""
        # Groq's llama/qwen/mixtral IDs were decommissioned upstream; the
        # registry now carries only the gpt-oss pair.
        model_info = registry.get_model_info(LLMProvider.GROQ, "openai/gpt-oss-120b")
        assert model_info is not None
        assert model_info.rate_limits.rpm == 1000
        assert model_info.rate_limits.tpm == 1000000

    def test_reload(self, registry):
        """Test reloading configuration."""
        original_models = registry.get_all_models()
        registry.reload()
        reloaded_models = registry.get_all_models()
        assert len(original_models) == len(reloaded_models)
