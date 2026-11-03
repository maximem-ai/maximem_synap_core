"""Repository for client operations."""

from typing import Any, Optional, List, Tuple
from uuid import uuid5, NAMESPACE_URL

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.database.repositories.base import BaseRepository
from synap.cloud.database.core_orm import ClientORM, EntityStatus


class ClientRepository(BaseRepository[ClientORM]):
    """Repository for client CRUD operations."""

    model = ClientORM

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_by_client_id(self, client_id: str) -> Optional[ClientORM]:
        """Get a client by its client_id (primary key)."""
        result = await self.session.execute(
            select(ClientORM).where(ClientORM.client_id == client_id)
        )
        return result.scalar_one_or_none()

    async def create_client(
        self,
        client_id: str,
        client_name: str,
        client_short_description: str,
        client_context: dict,
        client_website: Optional[str] = None,
    ) -> ClientORM:
        """Create a new client record.

        Side effect: when ``SYNAP_CREDITS_ENABLED=true``, also creates a
        zero-balance credit wallet and drops the free-tier allowance
        grant. The hook imports lazily so analytics / test harnesses
        that don't exercise the credit module don't pay the import cost.
        """
        # Cloud only: the region module does not ship in Core, where the client
        # has no home region.
        try:
            from synap.cloud.shared.region import owning_region
        except ModuleNotFoundError as exc:
            if exc.name != "synap.cloud.shared.region":
                raise
            home_region = None
        else:
            home_region = owning_region()

        client = ClientORM(
            uuid=uuid5(NAMESPACE_URL, client_id),
            client_id=client_id,
            client_name=client_name,
            client_website=client_website,
            client_short_description=client_short_description,
            client_context=client_context,
            # The region creating the client owns its scheduled credit work
            # from here on, so the trial grant and the bucket expiry happen
            # once rather than once per region (migration 145).
            home_region=home_region,
        )
        self.session.add(client)
        await self.session.flush()
        await self.session.refresh(client)

        # Credit system hook — ensure wallet, Trial subscription, and signup
        # free tier are all in place. ensure_trial_subscription puts the
        # client on the 'free' plan with a synthetic 30-day cycle so the
        # plan_monthly bucket logic (migration 057) has dates to anchor on.
        try:
            from synap.cloud.application.credits.config import get_config
            from synap.cloud.application.credits.allowance_worker import (
                grant_signup_free_tier,
            )
            from synap.cloud.application.credits.variant_service import (
                ensure_trial_subscription,
            )
            from synap.cloud.database.repositories.credit_wallet_repo import (
                CreditWalletRepository,
            )

            if get_config().enabled:
                wallet_repo = CreditWalletRepository(self.session)
                await wallet_repo.ensure_wallet(client_id)
                await ensure_trial_subscription(
                    self.session, client_id=client_id,
                )
                await grant_signup_free_tier(
                    session=self.session, client_id=client_id,
                )
        except Exception as exc:
            import logging as _logging
            if (
                isinstance(exc, ModuleNotFoundError)
                and (
                    (exc.name or "").startswith("synap.cloud.application.credits")
                    or exc.name == "synap.cloud.database.repositories.credit_wallet_repo"
                )
            ):
                # Core ships no credit system: nothing to do, nothing to warn about.
                _logging.getLogger(__name__).debug(
                    "credits not present, skipping signup hook client_id=%s", client_id
                )
            else:
                # Never let a credit-system failure block client creation.
                _logging.getLogger(__name__).warning(
                    "post_signup_credit_hook_failed client_id=%s",
                    client_id, exc_info=True,
                )

        # SHIP-17. Give the client the default scope ladder.
        #
        # This is the ONE seam all three creation callers pass through
        # (client_manager.create_client, dashboard_manager.create_client_for_admin
        # and the admin route), so seeding here covers all of them; seeding at
        # any one of the three would leave the other two making ladderless
        # clients.
        #
        # Without it, `resolve_legacy_chain` returns None for this client
        # forever and nested scoping is silently off, which is the state every
        # client created after migration 095 was in.
        #
        # `client.uuid` is READ off the row just written rather than recomputed
        # from `client_id` (N-14). It happens to be a uuid5 of the client_id for
        # a client created here, but some existing clients
        # carry a v4 random uuid instead, so the habit of reading it is what
        # keeps the backfill and this path agreeing.
        #
        # Runs in its own transaction, like the credit hook above it, and never
        # raises. There is no foreign key from the ladder tables to `client`, so
        # the un-flushed client row is not a problem, and an orphaned ladder if
        # this signup later rolls back is inert.
        try:
            try:
                from synap.cloud.shared.ladder_api import ensure_client_ladder
            except ModuleNotFoundError as exc:
                if exc.name != "synap.cloud.shared.ladder_api":
                    raise
                # Core has no scope ladder (the ladder modules do not ship).
                ensure_client_ladder = None

            if ensure_client_ladder is not None:
                await ensure_client_ladder(client.uuid, client_id)
        except Exception:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "default_ladder_seed_failed client_id=%s", client_id, exc_info=True,
            )
        return client

    async def update_client(
        self, client_id: str, **kwargs: Any
    ) -> Optional[ClientORM]:
        """Update a client by client_id.

        Uses client_id (PK) instead of BaseRepository.update which expects model.id.
        """
        await self.session.execute(
            update(ClientORM)
            .where(ClientORM.client_id == client_id)
            .values(**kwargs)
        )
        await self.session.flush()
        return await self.get_by_client_id(client_id)

    async def get_all_active(self, limit: int = 100, offset: int = 0) -> List[ClientORM]:
        """Get all active clients with pagination."""
        result = await self.session.execute(
            select(ClientORM)
            .where(ClientORM.status == EntityStatus.ACTIVE)
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def list_filtered(
        self,
        limit: int = 100,
        offset: int = 0,
        status_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[ClientORM], int]:
        """List clients with optional status filter, name search, and total count."""
        query = select(ClientORM)
        count_query = select(func.count()).select_from(ClientORM)

        if status_filter:
            query = query.where(ClientORM.status == status_filter)
            count_query = count_query.where(ClientORM.status == status_filter)

        if search:
            pattern = f"%{search}%"
            query = query.where(ClientORM.client_name.ilike(pattern))
            count_query = count_query.where(ClientORM.client_name.ilike(pattern))

        query = query.order_by(ClientORM.created_at.desc()).limit(limit).offset(offset)

        result = await self.session.execute(query)
        clients = list(result.scalars().all())

        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0

        return clients, total
