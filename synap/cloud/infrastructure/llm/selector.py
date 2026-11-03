"""
LLM Model Selector - Intelligent model selection based on criteria and availability.
"""

from typing import Callable, Iterable, List, Optional
from dataclasses import dataclass

from .models import (
    LLMProvider,
    ModelInfo,
    SelectionCriteria,
    AGENT_DEFAULTS,
)
from .registry import LLMRegistry
from .tracker import LLMUsageTracker


@dataclass
class ModelCandidate:
    """A model candidate with its calculated score."""
    model_info: ModelInfo
    base_score: float
    availability_factor: float
    final_score: float


class LLMModelSelector:
    """
    Selects the best model for a given task based on criteria and availability.

    Selection algorithm:
    1. Filter to enabled providers (and forced provider if specified)
    2. Calculate base score from criteria weights × model scores
    3. Apply availability factor (load balancing)
    4. Return highest scoring available model
    """

    def __init__(
        self,
        registry: LLMRegistry,
        tracker: LLMUsageTracker,
        available_providers: Optional[Callable[[], Iterable[LLMProvider]]] = None,
    ):
        """
        Initialize selector with registry and tracker.

        Args:
            registry: Model registry for model information
            tracker: Usage tracker for availability data
            available_providers: Returns the providers that actually have a
                client in this process (a key was present at startup). The
                registry says what MAY be used; this says what CAN be. When
                given, no model from any other provider is ever offered, not as
                a first pick, not as a fallback, not as the last-resort default.
                Without it the selector trusted the registry alone, and it
                kept handing out gemini-2.5-flash, a provider no
                deployment carries a key for, so every fallback died with
                "Provider gemini not available".
        """
        self._registry = registry
        self._tracker = tracker
        self._available_providers = available_providers

    def _usable_providers(self, providers: List[LLMProvider]) -> List[LLMProvider]:
        """Drop providers the process cannot call, whatever the registry says."""
        if self._available_providers is None:
            return providers
        available = set(self._available_providers())
        return [p for p in providers if p in available]

    async def select_model(
        self,
        criteria: Optional[SelectionCriteria] = None,
        agent_type: Optional[str] = None,
        force_provider: Optional[LLMProvider] = None,
        exclude_models: Optional[List[str]] = None,
    ) -> Optional[ModelInfo]:
        """
        Select the best model for the given criteria.

        Args:
            criteria: Selection criteria with weights. If None, uses agent_type defaults.
            agent_type: Agent type key to get default criteria from AGENT_DEFAULTS.
            force_provider: Force selection from a specific provider (for BYOK).
            exclude_models: List of model names to exclude from selection.

        Returns:
            Best matching ModelInfo, or None if no models available.
        """
        # Get criteria and provider_override from agent type defaults
        provider_override: Optional[LLMProvider] = None
        if criteria is None:
            if agent_type and agent_type in AGENT_DEFAULTS:
                agent_config = AGENT_DEFAULTS[agent_type]
                criteria = agent_config.get("criteria", SelectionCriteria(
                    speed=5,
                    accuracy=5,
                    reasoning=5,
                    tool_usage=5
                ))
                provider_override = agent_config.get("provider_override")
            else:
                # Default balanced criteria
                criteria = SelectionCriteria(
                    speed=5,
                    accuracy=5,
                    reasoning=5,
                    tool_usage=5
                )

        exclude_models = exclude_models or []

        # Determine effective provider: explicit force_provider wins,
        # then agent_type provider_override, then None (all providers).
        effective_provider = force_provider or provider_override

        # Get candidate models
        candidates = await self._get_candidates(
            criteria=criteria,
            force_provider=effective_provider,
            exclude_models=exclude_models,
        )

        # If provider_override yielded no candidates, fall back to all providers
        if not candidates and effective_provider and not force_provider:
            candidates = await self._get_candidates(
                criteria=criteria,
                force_provider=None,
                exclude_models=exclude_models,
            )

        if not candidates:
            return None

        # Sort by final score (descending) and return best
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates[0].model_info

    async def _get_candidates(
        self,
        criteria: SelectionCriteria,
        force_provider: Optional[LLMProvider] = None,
        exclude_models: Optional[List[str]] = None,
    ) -> List[ModelCandidate]:
        """
        Get all candidate models with their scores.

        Args:
            criteria: Selection criteria
            force_provider: Limit to specific provider
            exclude_models: Models to exclude

        Returns:
            List of ModelCandidate objects
        """
        candidates = []
        exclude_models = exclude_models or []

        # Determine which providers to consider
        if force_provider:
            providers = [force_provider]
        else:
            providers = self._registry.get_enabled_providers()
        providers = self._usable_providers(providers)

        for provider in providers:
            models = self._registry.get_models_by_provider(provider)

            for model_info in models:
                # Skip excluded models
                if model_info.name in exclude_models:
                    continue

                # Get availability
                availability = await self._tracker.get_availability(
                    provider=provider,
                    model=model_info.name,
                    rpm_limit=model_info.rate_limits.rpm,
                    tpm_limit=model_info.rate_limits.tpm,
                )

                # Skip unavailable models
                if not availability.is_available:
                    continue

                # Calculate scores
                base_score = self._calculate_base_score(model_info, criteria)
                final_score = base_score * availability.availability_factor

                candidates.append(ModelCandidate(
                    model_info=model_info,
                    base_score=base_score,
                    availability_factor=availability.availability_factor,
                    final_score=final_score,
                ))

        return candidates

    def _calculate_base_score(
        self,
        model_info: ModelInfo,
        criteria: SelectionCriteria
    ) -> float:
        """
        Calculate base score using weighted sum.

        Score = Σ(criterion_weight × model_score)

        Args:
            model_info: Model to score
            criteria: Weights for each criterion

        Returns:
            Weighted score (0-100 range typically)
        """
        scores = model_info.scores

        total = (
            criteria.speed * scores.speed +
            criteria.accuracy * scores.accuracy +
            criteria.reasoning * scores.reasoning +
            criteria.tool_usage * scores.tool_usage
        )

        return total

    async def get_fallback_model(
        self,
        original_provider: Optional[LLMProvider] = None,
        original_model: Optional[str] = None,
    ) -> Optional[ModelInfo]:
        """
        Get a fallback model when the primary fails.

        Fallback strategy:
        1. If original provider specified, try another model from same provider
        2. Try models from other providers
        3. Return registry default as last resort

        Args:
            original_provider: The provider that failed
            original_model: The model that failed

        Returns:
            Fallback ModelInfo or None
        """
        exclude_models = [original_model] if original_model else []

        # First try same provider, different model
        if original_provider:
            fallback = await self.select_model(
                force_provider=original_provider,
                exclude_models=exclude_models,
            )
            if fallback:
                return fallback

        # Try other providers (balanced criteria)
        fallback = await self.select_model(
            exclude_models=exclude_models,
        )
        if fallback:
            return fallback

        # Last resort: registry default. Only if this process can actually
        # call it and it is not the model that just failed; returning a model
        # nobody can run is not a fallback, it is a second failure with a
        # misleading name.
        default_provider = self._registry.get_default_provider()
        default_model = self._registry.get_default_model()
        if default_model in exclude_models:
            return None
        if not self._usable_providers([default_provider]):
            return None
        return self._registry.get_model_info(default_provider, default_model)
