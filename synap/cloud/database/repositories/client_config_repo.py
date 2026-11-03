"""Repository for client configuration operations."""

from datetime import datetime, timezone
from typing import Optional, List, Any

from sqlalchemy import select, delete, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.database.repositories.base import BaseRepository
from synap.cloud.database.core_orm import ClientConfigORM


class ClientConfigRepository(BaseRepository[ClientConfigORM]):
    """Repository for client_config CRUD operations."""

    model = ClientConfigORM

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_by_key(
        self,
        client_id: str,
        config_key: str,
        instance_id: Optional[str] = None,
    ) -> Optional[ClientConfigORM]:
        """Get a single config entry by composite key.

        Returns the most recently updated row if more than one matches. That
        can only be a client-level key written before migration 162: until
        then nothing stopped two rows with a NULL instance_id, and a read that
        raised on them made the key unreadable for good.
        """
        stmt = select(ClientConfigORM).where(
            ClientConfigORM.client_id == client_id,
            ClientConfigORM.config_key == config_key,
        )
        if instance_id is None:
            stmt = stmt.where(ClientConfigORM.instance_id.is_(None))
        else:
            stmt = stmt.where(ClientConfigORM.instance_id == instance_id)
        stmt = stmt.order_by(ClientConfigORM.updated_at.desc()).limit(1)

        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_all_for_client(self, client_id: str) -> List[ClientConfigORM]:
        """Get all client-level config rows (instance_id IS NULL)."""
        result = await self.session.execute(
            select(ClientConfigORM).where(
                ClientConfigORM.client_id == client_id,
                ClientConfigORM.instance_id.is_(None),
            )
        )
        return list(result.scalars().all())

    async def get_all_for_instance(
        self, client_id: str, instance_id: str
    ) -> List[ClientConfigORM]:
        """Get all instance-level config rows."""
        result = await self.session.execute(
            select(ClientConfigORM).where(
                ClientConfigORM.client_id == client_id,
                ClientConfigORM.instance_id == instance_id,
            )
        )
        return list(result.scalars().all())

    async def upsert(
        self,
        client_id: str,
        config_key: str,
        config_value: Any,
        instance_id: Optional[str] = None,
    ) -> ClientConfigORM:
        """Insert or update a config entry.

        Instance-level keys use PostgreSQL ON CONFLICT ... DO UPDATE on the
        (client_id, instance_id, config_key) unique constraint.

        Client-level keys (instance_id IS NULL) cannot: a unique constraint
        treats two NULLs as different values, so ON CONFLICT never fires for
        them and every write used to add a row. They are written as an update
        of the existing row, or an insert when there is none. Migration 162
        adds a partial unique index for these keys; it is what stops two
        concurrent first writes both inserting, and the insert below falls
        back to an update when it hits that index.
        """
        existing = await self.get_by_key(client_id, config_key, instance_id)
        next_version = str(int(existing.version) + 1) if existing else "1"
        now = datetime.now(timezone.utc)

        if instance_id is None:
            if existing is None:
                try:
                    async with self.session.begin_nested():
                        await self.session.execute(
                            insert(ClientConfigORM).values(
                                client_id=client_id,
                                instance_id=None,
                                config_key=config_key,
                                config_value=config_value,
                                version=next_version,
                                updated_at=now,
                            )
                        )
                except IntegrityError:
                    # Another writer inserted the key first. Update theirs.
                    existing = await self.get_by_key(client_id, config_key, None)
                    if existing is None:
                        raise
                    next_version = str(int(existing.version) + 1)
            if existing is not None:
                await self.session.execute(
                    update(ClientConfigORM)
                    .where(
                        ClientConfigORM.client_id == client_id,
                        ClientConfigORM.config_key == config_key,
                        ClientConfigORM.instance_id.is_(None),
                    )
                    .values(
                        config_value=config_value,
                        version=next_version,
                        updated_at=now,
                    )
                )
            await self.session.flush()
            return await self.get_by_key(client_id, config_key, None)

        stmt = insert(ClientConfigORM).values(
            client_id=client_id,
            instance_id=instance_id,
            config_key=config_key,
            config_value=config_value,
            version=next_version,
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_client_config_key",
            set_={
                "config_value": stmt.excluded.config_value,
                "version": next_version,
                "updated_at": now,
            },
        )
        await self.session.execute(stmt)
        await self.session.flush()

        return await self.get_by_key(client_id, config_key, instance_id)

    async def delete_by_key(
        self,
        client_id: str,
        config_key: str,
        instance_id: Optional[str] = None,
    ) -> bool:
        """Delete a config entry by composite key. Returns True if deleted."""
        stmt = delete(ClientConfigORM).where(
            ClientConfigORM.client_id == client_id,
            ClientConfigORM.config_key == config_key,
        )
        if instance_id is None:
            stmt = stmt.where(ClientConfigORM.instance_id.is_(None))
        else:
            stmt = stmt.where(ClientConfigORM.instance_id == instance_id)

        result = await self.session.execute(stmt)
        return result.rowcount > 0
