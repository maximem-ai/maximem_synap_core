"""LLM configuration manager for model selection."""

import time
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

import yaml

from .tracker import UsageTracker


@dataclass
class ModelSelection:
    """Result of model selection process."""
    provider: str
    model: str
    config: Dict


class LLMUnavailableError(Exception):
    """Raised when no LLM models are available."""
    pass


# Availability penalty thresholds (aligned with cloud layer)
_HEAVY_LOAD = 0.8
_MODERATE_LOAD = 0.5


class LLMConfigManager:
    """Selects best LLM based on task criteria and availability.

    Coordinates model selection by:
    - Matching task criteria (speed, accuracy, reasoning, tool usage) to model scores
    - Checking availability via UsageTracker (rate limits, 429 responses)
    - Respecting BYOK (Bring Your Own Key) provider preferences
    - Penalizing busy models to distribute load

    Model scores and limits are defined in llm_registry.yaml.
    """

    def __init__(self, registry_path: str, tracker: UsageTracker = None):
        """Initialize LLM config manager.

        Args:
            registry_path: Path to llm_registry.yaml
            tracker: Optional UsageTracker for rate limit coordination
        """
        self.registry_path = registry_path
        self.tracker = tracker
        self._unavailable: Dict[str, float] = {}  # "provider:model" -> expiry timestamp

        with open(Path(registry_path)) as f:
            self.registry = yaml.safe_load(f)

    def get_model(self, criteria: Dict[str, int], byok_provider: Optional[str] = None) -> ModelSelection:
        """Select best model based on criteria and availability.

        Args:
            criteria: Dict with keys "speed", "accuracy", "reasoning", "tool_usage"
                     Values are 0-10 weights indicating importance
            byok_provider: Optional provider name if user brings own key

        Returns:
            ModelSelection with provider, model, and config

        Raises:
            LLMUnavailableError: If no models are available

        Algorithm:
        1. Filter to enabled models (or just byok_provider if specified)
        2. Remove models marked unavailable
        3. Score each model: weighted dot product of criteria and model scores
        4. Penalize busy models (high RPM/TPM usage) by reducing score
        5. Return highest scoring model
        """
        now = time.time()
        providers = self.registry.get("providers", {})

        best_score = -1.0
        best_selection: Optional[ModelSelection] = None

        for provider_name, provider_cfg in providers.items():
            if not provider_cfg.get("enabled", False):
                continue
            if byok_provider and provider_name != byok_provider:
                continue

            models = provider_cfg.get("models", {})
            for model_name, model_cfg in models.items():
                # Check unavailability
                unavail_key = f"{provider_name}:{model_name}"
                expiry = self._unavailable.get(unavail_key)
                if expiry is not None:
                    if now < expiry:
                        continue
                    else:
                        del self._unavailable[unavail_key]

                # Compute weighted dot product
                scores = model_cfg.get("scores", {})
                base_score = sum(
                    criteria.get(factor, 0) * scores.get(factor, 5)
                    for factor in ("speed", "accuracy", "reasoning", "tool_usage")
                )

                # Apply availability penalty from tracker
                penalty = 1.0
                if self.tracker:
                    usage = self.tracker.get_usage(provider_name, model_name)
                    max_usage = max(usage.get("rpm_usage", 0), usage.get("tpm_usage", 0))
                    if max_usage >= 1.0:
                        penalty = 0.0
                    elif max_usage > _HEAVY_LOAD:
                        penalty = 0.5
                    elif max_usage > _MODERATE_LOAD:
                        penalty = 0.9

                final_score = base_score * penalty

                if final_score > best_score:
                    best_score = final_score
                    best_selection = ModelSelection(
                        provider=provider_name,
                        model=model_name,
                        config=model_cfg,
                    )

        if best_selection is None:
            raise LLMUnavailableError("No LLM models are currently available")

        return best_selection

    def mark_unavailable(self, provider: str, model: str, duration_seconds: int) -> None:
        """Mark model as temporarily unavailable.

        Call this after receiving 429 (rate limit) or 503 (service unavailable)
        responses. The model will be excluded from selection for the specified duration.

        Args:
            provider: Provider name
            model: Model identifier
            duration_seconds: How long to mark unavailable
        """
        key = f"{provider}:{model}"
        self._unavailable[key] = time.time() + duration_seconds
