"""Retrieval agent configurator.

Configures ranking, anticipation, and context budgets.
"""

import logging
from typing import Any

from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
from .base import BaseConfigurator

logger = logging.getLogger(__name__)


class RetrievalConfigurator(BaseConfigurator):
    """
    Configures retrieval behavior based on YAML config.

    Responsibilities:
    - Configure retrieval modes (fast/accurate/comprehensive)
    - Set ranking weights
    - Configure anticipation patterns
    - Set context budget
    - Store agent hints for retrieval
    """

    def __init__(self, config_store: Any):
        self.config_store = config_store

    async def apply(
        self,
        instance_id: str,
        config: MemoryArchitectureConfig
    ) -> dict[str, Any]:
        """Apply retrieval configuration.

        Orchestrates modes, ranking, anticipation, budget, and agent hints.
        """
        rc = config.retrieval_config
        results: dict[str, Any] = {}
        results["modes"] = await self._configure_modes(instance_id, rc.modes)
        results["ranking"] = await self._configure_ranking(instance_id, rc.ranking)
        results["anticipation"] = await self._configure_anticipation(
            instance_id, rc.anticipation
        )
        results["context_budget"] = await self._configure_context_budget(
            instance_id, rc.context_budget
        )
        results["agent_hints"] = await self._store_agent_hints(
            instance_id, config.agent_hints.retrieval_agent
        )

        logger.info(f"Retrieval configuration applied for instance {instance_id}")
        return {"success": True, "default_mode": rc.default_mode, **results}

    async def validate(
        self,
        instance_id: str,
        config: MemoryArchitectureConfig
    ) -> list[str]:
        """Validate retrieval config (schema + business rules)."""
        errors: list[str] = []
        rc = config.retrieval_config

        # At least one mode must be enabled (if modes are defined)
        if rc.modes:
            enabled_modes = [name for name, m in rc.modes.items() if m.enabled]
            if not enabled_modes:
                errors.append("At least one retrieval mode must be enabled")

            # default_mode must reference an enabled mode
            if rc.default_mode and rc.default_mode not in enabled_modes:
                errors.append(
                    f"default_mode '{rc.default_mode}' is not an enabled mode. "
                    f"Enabled: {enabled_modes}"
                )

            # Each mode's max_latency_ms must be positive
            for name, mode in rc.modes.items():
                if mode.enabled and mode.max_latency_ms <= 0:
                    errors.append(f"Mode '{name}' max_latency_ms must be > 0")

        # Ranking signal weights must be in [0.0, 1.0]
        for signal, weight in rc.ranking.primary_signals.items():
            if not (0.0 <= weight <= 1.0):
                errors.append(
                    f"Ranking signal '{signal}' weight must be in [0.0, 1.0], "
                    f"got {weight}"
                )

        # Context budget checks
        budget = rc.context_budget
        if budget.max_tokens <= 0:
            errors.append("context_budget max_tokens must be > 0")

        total_reserves = (
            budget.reserve_for_conversation
            + budget.reserve_for_memories
            + budget.reserve_for_org_context
        )
        if total_reserves > budget.max_tokens:
            errors.append(
                f"Context budget reserves ({total_reserves}) exceed "
                f"max_tokens ({budget.max_tokens})"
            )

        # Anticipation checks
        if rc.anticipation.max_prefetch_items < 0:
            errors.append("anticipation max_prefetch_items must be >= 0")

        return errors

    async def rollback(
        self,
        instance_id: str,
        from_config: MemoryArchitectureConfig,
        to_config: MemoryArchitectureConfig
    ) -> bool:
        """Rollback retrieval configuration by re-applying the previous config."""
        result = await self.apply(instance_id, to_config)
        return result.get("success", False)

    # ---- Private helpers ----

    async def _configure_modes(
        self,
        instance_id: str,
        modes: dict
    ) -> dict:
        """Configure retrieval modes."""
        serialized = {name: mode.model_dump() for name, mode in modes.items()}
        await self.config_store.put(instance_id, "retrieval/modes", serialized)
        enabled = [n for n, m in modes.items() if m.enabled]
        return {"status": "configured", "enabled_modes": enabled}

    async def _configure_ranking(
        self,
        instance_id: str,
        ranking_config: Any
    ) -> dict:
        """Configure ranking weights."""
        await self.config_store.put(
            instance_id, "retrieval/ranking", ranking_config.model_dump()
        )
        return {
            "status": "configured",
            "signals": list(ranking_config.primary_signals.keys()),
        }

    async def _configure_anticipation(
        self,
        instance_id: str,
        anticipation_config: Any
    ) -> dict:
        """Configure anticipation patterns."""
        await self.config_store.put(
            instance_id, "retrieval/anticipation", anticipation_config.model_dump()
        )
        return {"status": "configured", "enabled": anticipation_config.enabled}

    async def _configure_context_budget(
        self,
        instance_id: str,
        budget_config: Any
    ) -> dict:
        """Configure context budget."""
        await self.config_store.put(
            instance_id, "retrieval/context_budget", budget_config.model_dump()
        )
        return {"status": "configured", "max_tokens": budget_config.max_tokens}

    async def _store_agent_hints(
        self,
        instance_id: str,
        hints: Any
    ) -> dict:
        """Store retrieval agent hints."""
        await self.config_store.put(
            instance_id, "retrieval/agent_hints", hints.model_dump()
        )
        return {"status": "stored"}
