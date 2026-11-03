"""Ingestion agent configurator.

Configures extraction rules, chunking, and agent hints.
"""

import logging
from typing import Any

from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig
from .base import BaseConfigurator

logger = logging.getLogger(__name__)


class IngestionConfigurator(BaseConfigurator):
    """
    Configures ingestion behavior based on YAML config.

    Responsibilities:
    - Set memory categories (enabled/disabled)
    - Configure extraction triggers
    - Set PII handling rules
    - Configure chunking strategies
    - Store domain hints
    - Store agent hints for ingestion
    """

    def __init__(self, config_store: Any):
        self.config_store = config_store

    async def apply(
        self,
        instance_id: str,
        config: MemoryArchitectureConfig
    ) -> dict[str, Any]:
        """Apply ingestion configuration.

        Orchestrates extraction, chunking, and agent hints setup.
        """
        results: dict[str, Any] = {}
        results["extraction"] = await self._configure_extraction(
            instance_id, config.memory_extraction
        )
        results["chunking"] = await self._configure_chunking(
            instance_id, config.chunking_config
        )
        results["agent_hints"] = await self._store_agent_hints(
            instance_id, config.agent_hints.ingestion_agent
        )

        logger.info(f"Ingestion configuration applied for instance {instance_id}")
        return {"success": True, **results}

    async def validate(
        self,
        instance_id: str,
        config: MemoryArchitectureConfig
    ) -> list[str]:
        """Validate ingestion config (schema + business rules)."""
        errors: list[str] = []
        cats = config.memory_categories

        # At least one base memory category must be enabled
        base_cats = [
            cats.facts, cats.preferences, cats.episodes,
            cats.emotions, cats.temporal_events,
        ]
        if not any(c.enabled for c in base_cats):
            errors.append("At least one base memory category must be enabled")

        # Extraction filter thresholds
        filters = config.memory_extraction.filters
        if not (0.0 <= filters.min_confidence_threshold <= 1.0):
            errors.append(
                f"min_confidence_threshold must be in [0.0, 1.0], "
                f"got {filters.min_confidence_threshold}"
            )

        return errors

    async def rollback(
        self,
        instance_id: str,
        from_config: MemoryArchitectureConfig,
        to_config: MemoryArchitectureConfig
    ) -> bool:
        """Rollback ingestion configuration by re-applying the previous config."""
        result = await self.apply(instance_id, to_config)
        return result.get("success", False)

    # ---- Private helpers ----

    async def _configure_extraction(
        self,
        instance_id: str,
        extraction_config: Any
    ) -> dict:
        """Configure extraction settings."""
        await self.config_store.put(
            instance_id, "ingestion/extraction", extraction_config.model_dump()
        )
        return {"status": "configured"}

    async def _configure_chunking(
        self,
        instance_id: str,
        chunking_config: Any
    ) -> dict:
        """Configure chunking strategies."""
        await self.config_store.put(
            instance_id, "ingestion/chunking", chunking_config.model_dump()
        )
        return {"status": "configured", "strategy": str(chunking_config.default_strategy.value)}

    async def _store_agent_hints(
        self,
        instance_id: str,
        hints: Any
    ) -> dict:
        """Store ingestion agent hints."""
        await self.config_store.put(
            instance_id, "ingestion/agent_hints", hints.model_dump()
        )
        return {"status": "stored"}
