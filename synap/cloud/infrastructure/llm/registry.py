"""
LLM Model Registry - Loads and manages model configurations from YAML.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    LLMProvider,
    ModelInfo,
    ModelScores,
    ModelType,
    RateLimits,
)


class LLMRegistry:
    """
    Manages the registry of available LLM providers and models.
    Loads configuration from llm_registry.yaml.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize registry from YAML configuration.

        Args:
            config_path: Path to llm_registry.yaml. If None, uses default location.
        """
        self._config_path = config_path or self._get_default_config_path()
        self._providers: Dict[LLMProvider, Dict[str, ModelInfo]] = {}
        self._defaults: Dict[str, str] = {}
        # Models a caller can ask for by name but the selector must never pick
        # (``selectable: false`` in the YAML).
        self._not_selectable: set = set()
        self._load_config()

    def _get_default_config_path(self) -> str:
        """Get default config path relative to this file."""
        # Path: synap/cloud/infrastructure/llm/ -> synap/providers/llm/llm_registry.yaml
        current_file = Path(__file__)
        synap_root = current_file.parent.parent.parent.parent
        return str(synap_root / "providers" / "llm" / "llm_registry.yaml")

    def _load_config(self) -> None:
        """Load and parse the YAML configuration."""
        with open(self._config_path, 'r') as f:
            config = yaml.safe_load(f)

        self._defaults = config.get('defaults', {
            'provider': 'openai',
            'model': 'gpt-4o'
        })

        providers_config = config.get('providers', {})

        for provider_name, provider_data in providers_config.items():
            try:
                provider_enum = LLMProvider(provider_name)
            except ValueError:
                continue  # Skip unknown providers

            if not provider_data.get('enabled', False):
                continue

            # Get provider-level rate limits as defaults
            provider_limits = provider_data.get('rate_limits', {})
            default_rpm = provider_limits.get('rpm', 1000)
            default_tpm = provider_limits.get('tpm', 100000)

            models = {}
            for model_name, model_data in provider_data.get('models', {}).items():
                scores_data = model_data.get('scores', {})
                scores = ModelScores(
                    speed=scores_data.get('speed', 5),
                    accuracy=scores_data.get('accuracy', 5),
                    reasoning=scores_data.get('reasoning', 5),
                    tool_usage=scores_data.get('tool_usage', 5),
                )

                # Get model-specific rate limits, fall back to provider defaults
                model_limits = model_data.get('rate_limits', model_data.get('limits', {}))
                rate_limits = RateLimits(
                    rpm=model_limits.get('rpm', default_rpm),
                    tpm=model_limits.get('tpm', default_tpm),
                )

                # Determine model type
                model_type_str = model_data.get('type', 'balanced')
                try:
                    model_type = ModelType(model_type_str)
                except ValueError:
                    model_type = ModelType.BALANCED

                if not model_data.get('selectable', True):
                    self._not_selectable.add((provider_enum, model_name))

                models[model_name] = ModelInfo(
                    name=model_name,
                    provider=provider_enum,
                    model_type=model_type,
                    scores=scores,
                    rate_limits=rate_limits,
                    max_output_tokens=model_data.get('max_tokens', 4096),
                    context_window=model_data.get('context_window', 128000),
                )

            self._providers[provider_enum] = models

    def get_model_info(self, provider: LLMProvider, model: str) -> Optional[ModelInfo]:
        """
        Get information for a specific model.

        Args:
            provider: The provider enum
            model: The model name

        Returns:
            ModelInfo if found, None otherwise. Includes models marked
            ``selectable: false``: a caller that names one gets it.
        """
        provider_models = self._providers.get(provider)
        if not provider_models:
            return None
        return provider_models.get(model)

    def register_model(self, provider: LLMProvider, model: str) -> None:
        """Accept a model the YAML does not list, for a developer who picked their own.

        Without this a call that names it fails with "model not found". The entry is generic:
        nothing scores or ranks it, because with one configured model nothing is selecting.
        """
        models = self._providers.setdefault(provider, {})
        if model not in models:
            models[model] = ModelInfo(
                name=model, provider=provider, model_type=ModelType.BALANCED,
                scores=ModelScores(), rate_limits=RateLimits(),
                max_output_tokens=16384, description="Configured by the developer",
            )

    def get_all_models(self) -> List[ModelInfo]:
        """Get all available models across all providers."""
        models = []
        for provider, provider_models in self._providers.items():
            models.extend(m for name, m in provider_models.items()
                          if (provider, name) not in self._not_selectable)
        return models

    def get_models_by_provider(self, provider: LLMProvider) -> List[ModelInfo]:
        """Get all models for a specific provider."""
        provider_models = self._providers.get(provider)
        if not provider_models:
            return []
        return [m for name, m in provider_models.items()
                if (provider, name) not in self._not_selectable]

    def get_enabled_providers(self) -> List[LLMProvider]:
        """Get list of enabled providers."""
        return list(self._providers.keys())

    def get_default_provider(self) -> LLMProvider:
        """Get the default provider."""
        return LLMProvider(self._defaults.get('provider', 'openai'))

    def get_default_model(self) -> str:
        """Get the default model name."""
        return self._defaults.get('model', 'gpt-4o')

    def reload(self) -> None:
        """Reload configuration from file."""
        self._providers.clear()
        self._not_selectable.clear()
        self._load_config()
