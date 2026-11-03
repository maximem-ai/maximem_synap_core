"""Repository for instance operations."""

from typing import Any, Optional, List, Tuple
from uuid import uuid5, NAMESPACE_URL

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.database.repositories.base import BaseRepository
from synap.cloud.database.core_orm import InstanceORM


class InstanceRepository(BaseRepository[InstanceORM]):
    """Repository for instance CRUD operations."""

    model = InstanceORM

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_by_instance_id(self, instance_id: str) -> Optional[InstanceORM]:
        """Get an instance by its instance_id (primary key)."""
        result = await self.session.execute(
            select(InstanceORM).where(InstanceORM.instance_id == instance_id)
        )
        return result.scalar_one_or_none()

    async def get_for_client(self, client_id: str) -> List[InstanceORM]:
        """Get all instances belonging to a client."""
        result = await self.session.execute(
            select(InstanceORM)
            .where(InstanceORM.client_id == client_id)
            .where(InstanceORM.instance_metadata["deleted_at"].astext.is_(None))
        )
        return list(result.scalars().all())

    async def get_by_client_and_name(
        self, client_id: str, instance_name: str
    ) -> Optional[InstanceORM]:
        """Get a non-deleted instance by client + human-readable name."""
        result = await self.session.execute(
            select(InstanceORM)
            .where(InstanceORM.client_id == client_id)
            .where(InstanceORM.instance_name == instance_name)
            .where(InstanceORM.instance_metadata["deleted_at"].astext.is_(None))
        )
        return result.scalar_one_or_none()

    async def create_instance(
        self,
        instance_id: str,
        client_id: str,
        instance_name: str,
        instance_metadata: dict,
        agent_type: str = "sdk",
        *,
        enforce_limit: bool = True,
        onboarding_session_key: Optional[str] = None,
    ) -> InstanceORM:
        """Create a new instance record.

        Every live instance is created here, so the workspace/session/rate limit
        check lives here too (shared.instance_limits). ``enforce_limit=False``
        is for the system playground agents only. Raises InstanceLimitReached
        (after rolling the transaction back) when a limit is hit.
        """
        if enforce_limit:
            from synap.cloud.shared.instance_limits import (
                assert_can_create_instance,
            )

            await assert_can_create_instance(
                self.session, client_id, onboarding_session_key=onboarding_session_key
            )
        instance = InstanceORM(
            uuid=uuid5(NAMESPACE_URL, instance_id),
            instance_id=instance_id,
            client_id=client_id,
            instance_name=instance_name,
            instance_metadata=instance_metadata,
            agent_type=agent_type,
        )
        self.session.add(instance)
        await self.session.flush()
        await self.session.refresh(instance)
        return instance

    async def update_instance(
        self, instance_id: str, **kwargs: Any
    ) -> Optional[InstanceORM]:
        """Update an instance by instance_id.

        Uses instance_id (PK) instead of BaseRepository.update which expects model.id.
        """
        await self.session.execute(
            update(InstanceORM)
            .where(InstanceORM.instance_id == instance_id)
            .values(**kwargs)
        )
        await self.session.flush()
        return await self.get_by_instance_id(instance_id)

    async def delete_by_instance_id(self, instance_id: str) -> bool:
        """Delete an instance row by instance_id."""
        result = await self.session.execute(
            delete(InstanceORM).where(InstanceORM.instance_id == instance_id)
        )
        await self.session.flush()
        return bool(result.rowcount and result.rowcount > 0)

    async def list_filtered(
        self,
        client_id: str,
        limit: int = 100,
        offset: int = 0,
        status_filter: Optional[str] = None,
    ) -> Tuple[List[InstanceORM], int]:
        """List instances for a client with optional status filter and total count."""
        not_deleted_predicate = InstanceORM.instance_metadata["deleted_at"].astext.is_(None)
        not_playground_predicate = or_(
            InstanceORM.instance_metadata["is_playground"].astext.is_(None),
            InstanceORM.instance_metadata["is_playground"].astext != "true",
        )
        query = (
            select(InstanceORM)
            .where(InstanceORM.client_id == client_id)
            .where(not_deleted_predicate)
            .where(not_playground_predicate)
        )
        count_query = (
            select(func.count())
            .select_from(InstanceORM)
            .where(InstanceORM.client_id == client_id)
            .where(not_deleted_predicate)
            .where(not_playground_predicate)
        )

        if status_filter:
            query = query.where(InstanceORM.status == status_filter)
            count_query = count_query.where(InstanceORM.status == status_filter)

        query = query.order_by(InstanceORM.created_at.desc()).limit(limit).offset(offset)

        # Imported here, not at load: the observability package's __init__ pulls in
        # the rest of that folder, which Core does not ship.
        from synap.cloud.observability.tracing import (
            set_current_span_attributes,
            start_span,
        )

        async with start_span(
            "dashboard.instances.db.instance_page_rows",
            {
                "synap.dashboard.db.system": "postgresql",
                "synap.dashboard.db.table": "instance",
                "synap.dashboard.db.operation": "SELECT",
                "synap.dashboard.query_kind": "instance_page_rows",
                "synap.client_id": client_id,
                "synap.dashboard.limit": limit,
                "synap.dashboard.offset": offset,
                "synap.dashboard.status_filter": status_filter or "",
            },
        ):
            result = await self.session.execute(query)
            instances = list(result.scalars().all())
            set_current_span_attributes(
                {"synap.dashboard.db.rows_returned": len(instances)}
            )

        async with start_span(
            "dashboard.instances.db.instance_page_count",
            {
                "synap.dashboard.db.system": "postgresql",
                "synap.dashboard.db.table": "instance",
                "synap.dashboard.db.operation": "SELECT",
                "synap.dashboard.query_kind": "instance_page_count",
                "synap.client_id": client_id,
                "synap.dashboard.status_filter": status_filter or "",
            },
        ):
            count_result = await self.session.execute(count_query)
            total = count_result.scalar() or 0
            set_current_span_attributes(
                {"synap.dashboard.db.scalar_result": int(total)}
            )

        return instances, total
