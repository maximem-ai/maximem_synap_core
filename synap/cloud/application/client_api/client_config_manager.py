"""Client Config Manager for client-side configuration."""

from typing import Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.database.repositories.client_config_repo import ClientConfigRepository
from synap.cloud.application.models.errors import NotFoundError, ValidationError


class ClientConfigManager:
    """Handles client-side configuration (SDK-accessible).

    Provides CRUD operations for key-value config stored in the
    client_config table. Supports both client-level (instance_id=None)
    and instance-level config. When retrieving config with an instance_id,
    client-level defaults are merged with instance-level overrides.
    """

    def __init__(self, session: AsyncSession):
        self._repo = ClientConfigRepository(session)

    async def get_config(
        self, client_id: str, instance_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get configuration for client/instance.

        When instance_id is provided, returns client-level defaults
        merged with instance-level overrides (instance wins on conflict).

        Args:
            client_id: Client identifier.
            instance_id: Optional instance identifier.

        Returns:
            Flat dict of {config_key: config_value}.
        """
        if not client_id:
            raise ValidationError("client_id is required")

        client_rows = await self._repo.get_all_for_client(client_id)
        config = {row.config_key: row.config_value for row in client_rows}

        if instance_id is not None:
            instance_rows = await self._repo.get_all_for_instance(
                client_id, instance_id
            )
            for row in instance_rows:
                config[row.config_key] = row.config_value

        return config

    async def update_config(
        self,
        client_id: str,
        config: Dict[str, Any],
        instance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update configuration via flat key-value upsert.

        Each key in the dict is upserted independently. Keys not present
        in the dict are left untouched. Versions auto-increment per key.

        Args:
            client_id: Client identifier.
            config: Dict of {config_key: config_value} to upsert.
            instance_id: Optional instance identifier.

        Returns:
            Full merged config after upsert.
        """
        if not client_id:
            raise ValidationError("client_id is required")
        if not config:
            raise ValidationError("config must be a non-empty dict")

        for key, value in config.items():
            await self._repo.upsert(
                client_id=client_id,
                config_key=key,
                config_value=value,
                instance_id=instance_id,
            )

        return await self.get_config(client_id, instance_id)

    async def store_config(
        self,
        client_id: str,
        key: str,
        value: Any,
        instance_id: Optional[str] = None,
    ) -> None:
        """Store a specific config key.

        Inserts or updates a single key-value pair. Version auto-increments.

        Args:
            client_id: Client identifier.
            key: Configuration key.
            value: Configuration value (stored as JSONB).
            instance_id: Optional instance identifier.
        """
        if not client_id:
            raise ValidationError("client_id is required")
        if not key:
            raise ValidationError("key is required")

        await self._repo.upsert(
            client_id=client_id,
            config_key=key,
            config_value=value,
            instance_id=instance_id,
        )

    async def delete_config(
        self,
        client_id: str,
        key: str,
        instance_id: Optional[str] = None,
    ) -> None:
        """Delete a specific config key.

        Args:
            client_id: Client identifier.
            key: Configuration key.
            instance_id: Optional instance identifier.

        Raises:
            NotFoundError: If the config key does not exist.
        """
        if not client_id:
            raise ValidationError("client_id is required")
        if not key:
            raise ValidationError("key is required")

        deleted = await self._repo.delete_by_key(
            client_id=client_id,
            config_key=key,
            instance_id=instance_id,
        )
        if not deleted:
            scope = f"instance '{instance_id}'" if instance_id else f"client '{client_id}'"
            raise NotFoundError(
                f"Config key '{key}' not found for {scope}"
            )
