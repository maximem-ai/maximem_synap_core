"""Base configurator interface."""

from abc import ABC, abstractmethod
from typing import Any

from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig


class BaseConfigurator(ABC):
    """
    Base class for system configurators.

    Each configurator is responsible for applying a portion
    of the memory architecture config to a specific system.
    """

    @abstractmethod
    async def apply(
        self,
        instance_id: str,
        config: MemoryArchitectureConfig
    ) -> dict[str, Any]:
        """
        Apply configuration to the system.

        Args:
            instance_id: Instance to configure
            config: Full memory architecture config

        Returns:
            Dict with apply results (success, changes made, etc.)
        """
        pass

    @abstractmethod
    async def validate(
        self,
        instance_id: str,
        config: MemoryArchitectureConfig
    ) -> list[str]:
        """
        Validate config can be applied (dry run).

        Args:
            instance_id: Instance to configure
            config: Full memory architecture config

        Returns:
            List of validation errors (empty if valid)
        """
        pass

    @abstractmethod
    async def rollback(
        self,
        instance_id: str,
        from_config: MemoryArchitectureConfig,
        to_config: MemoryArchitectureConfig
    ) -> bool:
        """
        Rollback configuration changes.

        Args:
            instance_id: Instance to rollback
            from_config: Current config to undo
            to_config: Previous config to restore

        Returns:
            True if rollback successful
        """
        pass
