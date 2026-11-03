"""Centralized ID resolution with mapping persistence.

Maps external IDs (user_id, customer_id) to deterministic uuid5 hashes
and stores the mapping in PostgreSQL for reverse lookup. Client/instance
IDs are looked up from existing tables and rejected if not found.
"""

import uuid as _uuid
import logging
from typing import Optional, Dict, Tuple

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from synap.cloud.shared.scope import NAMESPACE_URL, customer_deterministic_uuid, user_deterministic_uuid

logger = logging.getLogger(__name__)

# LRU-style bounded dict
_MAX_CACHE_SIZE = 10_000


def _looks_like_uuid(value: str) -> bool:
    """Cheap gate so the extra lookup below only runs on uuid-shaped ids.

    Deliberately shape-only. Which uuid VERSION it is says nothing useful: our
    own ids are uuid5, but a client is free to send a uuid5 of their own, and
    the question that matters is whether this exact value is already one of
    this client's customer uuids. That is a lookup, not a guess.
    """
    try:
        _uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


class IdMapper:
    """Centralized ID resolution with mapping persistence."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory
        # Forward cache: (client_id, entity_type, external_id) -> uuid5_hash
        self._cache: Dict[Tuple[str, str, str], _uuid.UUID] = {}
        # Reverse cache: (client_id, entity_type, uuid5_hash_str) -> external_id
        self._reverse_cache: Dict[Tuple[str, str, str], str] = {}

    def _cache_put(
        self,
        client_id: str,
        entity_type: str,
        external_id: str,
        uuid5_hash: _uuid.UUID,
    ) -> None:
        """Add to both forward and reverse caches, evicting oldest if full."""
        if len(self._cache) >= _MAX_CACHE_SIZE:
            # Evict ~10% oldest entries
            keys = list(self._cache.keys())
            for k in keys[: _MAX_CACHE_SIZE // 10]:
                h = self._cache.pop(k, None)
                if h is not None:
                    self._reverse_cache.pop((k[0], k[1], str(h)), None)

        fwd_key = (client_id, entity_type, external_id)
        self._cache[fwd_key] = uuid5_hash
        self._reverse_cache[(client_id, entity_type, str(uuid5_hash))] = external_id

    # ------------------------------------------------------------------
    # User resolution
    # ------------------------------------------------------------------

    async def resolve_user(
        self,
        client_id: str,
        external_user_id: str,
        external_customer_id: Optional[str] = None,
        *,
        customer_uuid: Optional[_uuid.UUID] = None,
        is_b2c: bool = False,
    ) -> _uuid.UUID:
        """Resolve external user_id to uuid5 hash. Auto-creates mapping if new.

        Args:
            client_id: The client's string ID (from auth).
            external_user_id: The user_id string sent by the client SDK.
            external_customer_id: Optional customer_id for FK.
            customer_uuid: Pre-resolved customer UUID for scoped hashing.
            is_b2c: If True, returns customer_uuid directly (user == customer).

        Returns:
            The uuid5 hash used in ChromaDB/Neo4j/PostgreSQL memories.
        """
        if is_b2c and customer_uuid is not None:
            return customer_uuid

        fwd_key = (client_id, "user", external_user_id)
        cached = self._cache.get(fwd_key)
        if cached is not None:
            return cached

        if customer_uuid is not None:
            uuid5_hash = user_deterministic_uuid(customer_uuid, external_user_id)
        else:
            uuid5_hash = _uuid.uuid5(NAMESPACE_URL, external_user_id)

        async with self._session_factory() as session:
            row = await session.execute(
                text(
                    'SELECT uuid5_hash FROM "user" '
                    "WHERE client_id = :cid AND user_id = :uid"
                ),
                {"cid": client_id, "uid": external_user_id},
            )
            existing = row.scalar_one_or_none()

            if existing is not None:
                result = _uuid.UUID(str(existing))
                self._cache_put(client_id, "user", external_user_id, result)
                return result

            # Auto-create mapping. Same race as for a customer: a second, simultaneous insert for a new
            # user meets the first one's row through another unique index. Read the row back.
            try:
                await session.execute(
                    text(
                        'INSERT INTO "user" (user_id, client_id, customer_id, uuid5_hash) '
                        "VALUES (:uid, :cid, :custid, :hash) "
                        "ON CONFLICT (client_id, user_id) DO UPDATE SET uuid5_hash = :hash"
                    ),
                    {
                        "uid": external_user_id,
                        "cid": client_id,
                        "custid": external_customer_id,
                        "hash": uuid5_hash,
                    },
                )
                await session.commit()
            except IntegrityError:
                await session.rollback()
                row = await session.execute(
                    text('SELECT uuid5_hash FROM "user" WHERE client_id = :cid AND user_id = :uid'),
                    {"cid": client_id, "uid": external_user_id},
                )
                existing = row.scalar_one_or_none()
                if existing is None:
                    raise
                result = _uuid.UUID(str(existing))
                self._cache_put(client_id, "user", external_user_id, result)
                return result
            logger.info(
                "Created user mapping: %s -> %s (client=%s)",
                external_user_id,
                uuid5_hash,
                client_id,
            )

        self._cache_put(client_id, "user", external_user_id, uuid5_hash)
        return uuid5_hash

    # ------------------------------------------------------------------
    # Customer resolution
    # ------------------------------------------------------------------

    async def resolve_customer(
        self, client_id: str, external_customer_id: str,
        *, client_uuid: Optional[_uuid.UUID] = None,
    ) -> _uuid.UUID:
        """Resolve external customer_id to uuid5 hash. Auto-creates if new.

        Args:
            client_id: The client's string ID (from auth).
            external_customer_id: The customer_id string sent by the client SDK.
            client_uuid: Pre-resolved client UUID for scoped hashing.
        """
        fwd_key = (client_id, "customer", external_customer_id)
        cached = self._cache.get(fwd_key)
        if cached is not None:
            return cached

        if client_uuid is not None:
            uuid5_hash = customer_deterministic_uuid(client_uuid, external_customer_id)
        else:
            uuid5_hash = _uuid.uuid5(NAMESPACE_URL, external_customer_id)

        async with self._session_factory() as session:
            row = await session.execute(
                text(
                    "SELECT uuid5_hash FROM customer "
                    "WHERE client_id = :cid AND customer_id = :custid"
                ),
                {"cid": client_id, "custid": external_customer_id},
            )
            existing = row.scalar_one_or_none()

            if existing is not None:
                result = _uuid.UUID(str(existing))
                self._cache_put(client_id, "customer", external_customer_id, result)
                return result

            # ⚠ Before creating anything: is this OUR OWN id coming back?
            #
            # This function maps an external id to an internal uuid, and it
            # auto-creates on first sight. Hand it an id it already produced
            # and it cheerfully makes a second customer for the first one, with
            # an external id no client will ever send. This produced stray rows
            # from more than one call site.
            #
            # The property that was missing is idempotence. `resolve` should be
            # able to see its own output and give the same answer, and here it
            # can: our ids are uuid5 and we can look one up. So a value that IS
            # one of this client's customer uuids resolves to that customer,
            # which is what the caller meant, and no row is written.
            #
            # Narrow on purpose. A uuid that is NOT one of ours is a perfectly
            # ordinary external id and is created as normal: a client
            # may use a uuid4 per customer,
            # and breaking them to fix this would be much worse than the bug.
            #
            # Fixed at the mapper rather than at the two call sites that were
            # found, because a third exists and has not been found yet. Every
            # source passes through here.
            if _looks_like_uuid(external_customer_id):
                own = await session.execute(
                    text(
                        "SELECT customer_id FROM customer "
                        "WHERE client_id = :cid AND uuid5_hash = CAST(:val AS uuid)"
                    ),
                    {"cid": client_id, "val": external_customer_id},
                )
                owner = own.scalar_one_or_none()
                if owner is not None:
                    result = _uuid.UUID(external_customer_id)
                    # Error, not warning. This is a caller bug with a real
                    # consequence: the same wrong id also picks the vector
                    # collection, so whatever this request goes on to do is
                    # aimed at a namespace that does not exist. The stack is
                    # the point, because the remaining call site is unknown.
                    logger.error(
                        "resolve_customer was handed OUR OWN internal uuid %s "
                        "for client=%s (it belongs to customer %r). Resolving "
                        "it to itself and creating nothing. The caller is "
                        "double-mapping a stored id: see "
                        "test_no_stored_id_is_remapped.py.",
                        external_customer_id, client_id, owner,
                        stack_info=True,
                    )
                    self._cache_put(client_id, "customer", external_customer_id, result)
                    return result

            # Auto-create mapping. Two requests for a customer that is new can arrive together: the
            # second insert meets the first one's row through the (client_id, uuid5_hash) index, which
            # ON CONFLICT (client_id, customer_id) does not cover. That is not a failure: the row exists
            # with the same hash, so read it back.
            try:
                await session.execute(
                    text(
                        "INSERT INTO customer (customer_id, client_id, uuid5_hash) "
                        "VALUES (:custid, :cid, :hash) "
                        "ON CONFLICT (client_id, customer_id) DO UPDATE SET uuid5_hash = :hash"
                    ),
                    {
                        "custid": external_customer_id,
                        "cid": client_id,
                        "hash": uuid5_hash,
                    },
                )
                await session.commit()
            except IntegrityError:
                await session.rollback()
                row = await session.execute(
                    text("SELECT uuid5_hash FROM customer WHERE client_id = :cid AND customer_id = :custid"),
                    {"cid": client_id, "custid": external_customer_id},
                )
                existing = row.scalar_one_or_none()
                if existing is None:
                    raise
                result = _uuid.UUID(str(existing))
                self._cache_put(client_id, "customer", external_customer_id, result)
                return result
            logger.info(
                "Created customer mapping: %s -> %s (client=%s)",
                external_customer_id,
                uuid5_hash,
                client_id,
            )

        self._cache_put(client_id, "customer", external_customer_id, uuid5_hash)
        return uuid5_hash

    # ------------------------------------------------------------------
    # Client / Instance resolution (lookup only, reject if missing)
    # ------------------------------------------------------------------

    async def resolve_client(self, client_id: str) -> _uuid.UUID:
        """Lookup client UUID. Rejects if client_id not found."""
        fwd_key = (client_id, "client", client_id)
        cached = self._cache.get(fwd_key)
        if cached is not None:
            return cached

        async with self._session_factory() as session:
            row = await session.execute(
                text("SELECT uuid FROM client WHERE client_id = :cid"),
                {"cid": client_id},
            )
            result = row.scalar_one_or_none()
            if result is None:
                raise ValueError(f"Unknown client_id: {client_id}")

            uuid_val = _uuid.UUID(str(result))
            self._cache_put(client_id, "client", client_id, uuid_val)
            return uuid_val

    async def resolve_instance(self, instance_id: str) -> _uuid.UUID:
        """Lookup instance UUID. Rejects if instance_id not found."""
        fwd_key = ("", "instance", instance_id)
        cached = self._cache.get(fwd_key)
        if cached is not None:
            return cached

        async with self._session_factory() as session:
            row = await session.execute(
                text("SELECT uuid FROM instance WHERE instance_id = :iid"),
                {"iid": instance_id},
            )
            result = row.scalar_one_or_none()
            if result is None:
                raise ValueError(f"Unknown instance_id: {instance_id}")

            uuid_val = _uuid.UUID(str(result))
            self._cache_put("", "instance", instance_id, uuid_val)
            return uuid_val

    # ------------------------------------------------------------------
    # Reverse lookups
    # ------------------------------------------------------------------

    async def reverse_lookup_user(
        self, client_id: str, uuid5_hash: _uuid.UUID
    ) -> Optional[str]:
        """Reverse: uuid5_hash -> external_user_id."""
        rev_key = (client_id, "user", str(uuid5_hash))
        cached = self._reverse_cache.get(rev_key)
        if cached is not None:
            return cached

        async with self._session_factory() as session:
            row = await session.execute(
                text(
                    'SELECT user_id FROM "user" '
                    "WHERE client_id = :cid AND uuid5_hash = :hash"
                ),
                {"cid": client_id, "hash": uuid5_hash},
            )
            result = row.scalar_one_or_none()
            if result is not None:
                self._cache_put(client_id, "user", result, uuid5_hash)
            return result

    async def reverse_lookup_customer(
        self, client_id: str, uuid5_hash: _uuid.UUID
    ) -> Optional[str]:
        """Reverse: uuid5_hash -> external_customer_id."""
        rev_key = (client_id, "customer", str(uuid5_hash))
        cached = self._reverse_cache.get(rev_key)
        if cached is not None:
            return cached

        async with self._session_factory() as session:
            row = await session.execute(
                text(
                    "SELECT customer_id FROM customer "
                    "WHERE client_id = :cid AND uuid5_hash = :hash"
                ),
                {"cid": client_id, "hash": uuid5_hash},
            )
            result = row.scalar_one_or_none()
            if result is not None:
                self._cache_put(client_id, "customer", result, uuid5_hash)
            return result


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_id_mapper: Optional[IdMapper] = None


def get_id_mapper() -> IdMapper:
    """Get or create the singleton IdMapper instance.

    Raises RuntimeError if the database connection is not configured.
    """
    global _id_mapper
    if _id_mapper is not None:
        return _id_mapper

    from synap.cloud.database.connection import get_session_factory

    session_factory = get_session_factory()
    _id_mapper = IdMapper(session_factory)
    logger.info("IdMapper initialized")
    return _id_mapper
