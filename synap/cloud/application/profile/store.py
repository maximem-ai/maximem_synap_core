"""ProfileStore — the single read/write path for user-profile documents.

The profile document lives under the ``"profile"`` key of the owning row's
context JSONB (``"user".user_context`` for B2B/strict, or
``customer.customer_context`` for B2C/equals_customer — spec §5.2). Under
B2C isolation IdMapper returns the customer UUID and creates NO user row
(shared/id_mapper.py resolve_user early-return), so owner resolution mirrors
the dashboard entity-detail fallback (dashboard_manager.py:3770-3777):
user row if it exists, else customer row.

Tenancy (adversarial-review finding 8): the legacy user-hash fallback
``uuid5(NAMESPACE_URL, external_user_id)`` is NOT client-scoped, so matching
a row on ``uuid5_hash`` alone can cross tenants. Every uuid5-based lookup
here therefore first maps the scope's client UUID back to the client_id
string (``client.uuid`` unique index) and constrains
``WHERE client_id = :cid AND uuid5_hash = :hash`` — never the hash alone.

Deadlock posture: every write transaction locks exactly ONE row
(``SELECT ... FOR UPDATE`` on the owning user/customer row) and touches no
other row while the lock is held. There are no multi-row transactions, so
there is no lock-ordering hazard; concurrent writers for the same owner
simply serialize on the row lock.
"""

import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from sqlalchemy import text

logger = logging.getLogger(__name__)

# owner kind -> (table, primary-key column, context JSONB column)
_OWNER_TABLES: Dict[str, Tuple[str, str, str]] = {
    "user": ('"user"', "internal_user_id", "user_context"),
    "customer": ("customer", "internal_customer_id", "customer_context"),
}

# An owner handle: ("user" | "customer", internal row PK as str).
Owner = Tuple[str, str]



def _strip_nuls(value: Any) -> Any:
    """Remove NUL characters from every string in a structure, recursively.

    Postgres ``jsonb`` cannot store U+0000. It is the one code point JSON allows and
    jsonb does not, and asyncpg surfaces the refusal as
    ``UntranslatableCharacterError: unsupported Unicode escape sequence``, which names
    neither the column nor the character and reads like an encoding bug in our own
    code rather than one byte of customer text.

    Stripping is done on the Python objects rather than on the serialized JSON. A
    string that genuinely contains the six characters backslash-u-0-0-0-0 serializes
    to ``\\u0000``, so a textual replace of ``\u0000`` on the dumped output would
    corrupt it into a stray backslash. Here there is no ambiguity: a NUL is a NUL.

    One profile write failing is not the cost. The updater retries, so a single
    unlucky character stopped that user's profile updating at all, for as long as the
    document held it.
    """
    if isinstance(value, str):
        return value.replace("\x00", "") if "\x00" in value else value
    if isinstance(value, dict):
        return {_strip_nuls(k): _strip_nuls(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_nuls(v) for v in value]
    return value


class ProfileStore:
    """Async accessor for profile documents over the app session factory."""

    def __init__(self, session_factory=None):
        # An injected factory (tests, dashboard DI) keeps the session path;
        # the default production path reads via the shared AUTOCOMMIT engine
        # so single-statement reads skip the BEGIN/COMMIT wire round trips.
        self._injected_factory = session_factory is not None
        if session_factory is None:
            from synap.cloud.database.connection import get_session_factory

            session_factory = get_session_factory()
        self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _fetch_one(self, sql: str, params: dict):
        if self._injected_factory:
            async with self._session_factory() as session:
                result = await session.execute(text(sql), params)
                return result.fetchone()

        from synap.cloud.database.connection import get_autocommit_engine

        async with get_autocommit_engine().connect() as conn:
            result = await conn.execute(text(sql), params)
            return result.fetchone()

    @staticmethod
    def _as_dict(value: Any) -> Optional[dict]:
        """JSONB defensively: asyncpg+SQLAlchemy returns dict, raw drivers str."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else None
            except (ValueError, TypeError):
                return None
        return None

    # ------------------------------------------------------------------
    # Owner resolution
    # ------------------------------------------------------------------

    async def resolve_client_id_string(self, client_uuid) -> Optional[str]:
        """Map a scope client UUID (``client.uuid``) back to the client_id string."""
        if not client_uuid:
            return None
        row = await self._fetch_one(
            "SELECT client_id FROM client WHERE uuid = CAST(:u AS uuid)",
            {"u": str(client_uuid)},
        )
        return str(row[0]) if row else None

    async def resolve_owner(
        self,
        client_id: str,
        external_user_id: str,
        customer_uuid_hint=None,
        *,
        external_customer_id: Optional[str] = None,
    ) -> Optional[Owner]:
        """Resolve the owning row from external ids (endpoint / SDK path).

        Fallback order mirrors the dashboard's B2C entity-detail fallback:
        1. user row by (client_id, user_id);
        2. customer row by (client_id, customer_id = external_user_id) —
           B2C stores the caller under the customer table with the same
           external id (``_resolve_scope`` normalizes user_id == customer_id);
        3. customer row by (client_id, customer_id = external_customer_id)
           when the caller passed an explicit customer id;
        4. customer row by (client_id, uuid5_hash = customer_uuid_hint).

        Returns ("user" | "customer", internal row key) or None.
        """
        row = await self._fetch_one(
            'SELECT internal_user_id FROM "user" '
            "WHERE client_id = :cid AND user_id = :ext",
            {"cid": client_id, "ext": external_user_id},
        )
        if row:
            return ("user", str(row[0]))

        candidates = [external_user_id]
        if external_customer_id and external_customer_id != external_user_id:
            candidates.append(external_customer_id)
        for ext in candidates:
            row = await self._fetch_one(
                "SELECT internal_customer_id FROM customer "
                "WHERE client_id = :cid AND customer_id = :ext",
                {"cid": client_id, "ext": ext},
            )
            if row:
                return ("customer", str(row[0]))

        if customer_uuid_hint:
            row = await self._fetch_one(
                "SELECT internal_customer_id FROM customer "
                "WHERE client_id = :cid AND uuid5_hash = CAST(:h AS uuid)",
                {"cid": client_id, "h": str(customer_uuid_hint)},
            )
            if row:
                return ("customer", str(row[0]))
        return None

    async def resolve_owner_by_scope_uuids(
        self,
        client_uuid,
        user_uuid,
        customer_uuid=None,
    ) -> Optional[Owner]:
        """Resolve the owning row from ingestion-scope UUIDs (worker path).

        Stage-4 holds only uuid5 hashes (ingestion Scope carries UUIDs, not
        external ids). Resolution is client-scoped end to end: the client
        UUID is mapped to the client_id string first, then user/customer
        rows are matched on (client_id, uuid5_hash) — never uuid5_hash
        alone (review finding 8: the legacy NAMESPACE_URL user-hash
        fallback is not client-scoped, so a bare-hash match can hit
        another tenant).
        """
        client_id = await self.resolve_client_id_string(client_uuid)
        if not client_id:
            return None

        if user_uuid:
            row = await self._fetch_one(
                'SELECT internal_user_id FROM "user" '
                "WHERE client_id = :cid AND uuid5_hash = CAST(:h AS uuid)",
                {"cid": client_id, "h": str(user_uuid)},
            )
            if row:
                return ("user", str(row[0]))

        # B2C: no user row; the user UUID *is* the customer UUID.
        seen = set()
        for cand in (customer_uuid, user_uuid):
            if not cand or str(cand) in seen:
                continue
            seen.add(str(cand))
            row = await self._fetch_one(
                "SELECT internal_customer_id FROM customer "
                "WHERE client_id = :cid AND uuid5_hash = CAST(:h AS uuid)",
                {"cid": client_id, "h": str(cand)},
            )
            if row:
                return ("customer", str(row[0]))
        return None

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    async def read(self, owner: Owner) -> Optional[dict]:
        """Return the ``profile`` key of the owning row's context JSONB."""
        kind, key = owner
        table, pk, ctx_col = _OWNER_TABLES[kind]
        row = await self._fetch_one(
            f"SELECT {ctx_col} FROM {table} WHERE {pk} = CAST(:k AS uuid)",  # noqa: S608 — identifiers from fixed map
            {"k": key},
        )
        if row is None:
            return None
        ctx = self._as_dict(row[0]) or {}
        profile = ctx.get("profile")
        return profile if isinstance(profile, dict) else None

    # One statement expressing resolve_owner + read: each branch returns a row
    # iff the OWNING ROW exists (profile may be SQL NULL), and pri encodes the
    # resolve_owner fallback order — user row, customer row by external user
    # id, customer row by explicit customer id, customer row by uuid5 hint.
    # ORDER BY pri LIMIT 1 therefore picks the same owner resolve_owner would,
    # and returns that row's profile (possibly NULL), in ONE round trip
    # instead of up to four sequential sessions. Every branch is client-scoped
    # (tenancy guard — adversarial-review finding 8 applies here unchanged).
    _READ_PROFILE_SQL = """
        SELECT profile FROM (
            SELECT 1 AS pri, user_context->'profile' AS profile
            FROM "user" WHERE client_id = :cid AND user_id = :ext
            UNION ALL
            SELECT 2, customer_context->'profile'
            FROM customer WHERE client_id = :cid AND customer_id = :ext
            UNION ALL
            SELECT 3, customer_context->'profile'
            FROM customer WHERE client_id = :cid AND customer_id = :extc
            UNION ALL
            SELECT 4, customer_context->'profile'
            FROM customer WHERE client_id = :cid AND uuid5_hash = CAST(:h AS uuid)
        ) owner_rows
        ORDER BY pri
        LIMIT 1
    """

    async def read_profile(
        self,
        client_id: str,
        external_user_id: str,
        customer_uuid_hint=None,
        *,
        external_customer_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Resolve the owner from external ids and read its profile — one query.

        The read surface for the GET endpoint / SDK getter (PR-D) and the
        conversation-summary fetch. Semantically identical to
        ``resolve_owner(...) -> read(owner)`` (row-existence precedence: a
        user row with no profile still wins over a customer row with one),
        collapsed into a single statement because this sits on the
        conversation-start hot path and each extra session costs a full DB
        round trip.
        """
        extc = (
            external_customer_id
            if external_customer_id and external_customer_id != external_user_id
            else None
        )
        row = await self._fetch_one(
            self._READ_PROFILE_SQL,
            {
                "cid": client_id,
                "ext": external_user_id,
                "extc": extc,
                "h": str(customer_uuid_hint) if customer_uuid_hint else None,
            },
        )
        if row is None:
            return None
        profile = self._as_dict(row[0])
        return profile if isinstance(profile, dict) else None

    async def write(
        self,
        owner: Owner,
        merge: Callable[[Optional[dict]], dict],
    ) -> Optional[dict]:
        """Merge-write the profile document under a single-row lock.

        BEGIN -> ``SELECT ... FOR UPDATE`` on the owning row -> re-read the
        FRESH profile under the lock -> ``merge(fresh) -> new`` (caller
        provided; must be cheap/synchronous — any LLM work happens BEFORE
        this call, outside the transaction, per review finding 9) -> one
        ``UPDATE ... SET ctx = jsonb_set(COALESCE(ctx,'{}'),'{profile}',new)``.

        Because ``merge`` receives the fresh row (not a pre-lock snapshot),
        a writer that lost a race can reconcile attribute-wise on top of the
        winner instead of clobbering it. ``merge`` may raise to abort the
        transaction (nothing is written).

        Returns the stored document, or None if the owning row vanished.
        """
        kind, key = owner
        table, pk, ctx_col = _OWNER_TABLES[kind]
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    text(
                        f"SELECT {ctx_col} FROM {table} "  # noqa: S608
                        f"WHERE {pk} = CAST(:k AS uuid) FOR UPDATE"
                    ),
                    {"k": key},
                )
                row = result.fetchone()
                if row is None:
                    logger.warning(
                        "profile_store_write_missing_row kind=%s key=%s", kind, key
                    )
                    return None
                ctx = self._as_dict(row[0]) or {}
                fresh = ctx.get("profile")
                new_doc = merge(fresh if isinstance(fresh, dict) else None)
                await session.execute(
                    text(
                        f"UPDATE {table} "  # noqa: S608
                        f"SET {ctx_col} = jsonb_set("
                        f"COALESCE({ctx_col}, '{{}}'::jsonb), "
                        f"'{{profile}}', CAST(:doc AS jsonb)), "
                        f"updated_at = NOW() "
                        f"WHERE {pk} = CAST(:k AS uuid)"
                    ),
                    {"doc": json.dumps(_strip_nuls(new_doc)), "k": key},
                )
        return new_doc
