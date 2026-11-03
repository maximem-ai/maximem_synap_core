"""ProfileStore — owner resolution, read/write, tenancy guard.

Sessions are mocked with a tiny in-memory table set; SQL is dispatched on
substrings (the store's statements are fixed strings over a fixed table
map). The critical test is the cross-tenant guard: a user row whose
uuid5_hash collides across tenants (the legacy NAMESPACE_URL hash is not
client-scoped) must NEVER be matched without the client_id constraint.
"""

import uuid

import pytest

from synap.cloud.application.profile.store import ProfileStore

CLIENT_UUID_A = uuid.uuid4()
CLIENT_UUID_B = uuid.uuid4()
LEGACY_HASH = uuid.uuid5(uuid.NAMESPACE_URL, "+15550001111")  # client-unscoped


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _NoopBegin:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return _NoopBegin()

    async def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        params = params or {}
        self._db.executed.append((sql, dict(params)))
        return _FakeResult(self._db.dispatch(sql, params))


class _FakeDB:
    """Mini row store keyed the way the real tables are."""

    def __init__(self):
        self.clients = {}    # uuid(str) -> client_id string
        self.users = []      # dicts: client_id, user_id, uuid5_hash, internal_user_id, user_context
        self.customers = []  # dicts: client_id, customer_id, uuid5_hash, internal_customer_id, customer_context
        self.executed = []

    def session_factory(self):
        return _FakeSession(self)

    def _find(self, rows, **conds):
        for row in rows:
            if all(str(row.get(k)) == str(v) for k, v in conds.items()):
                return row
        return None

    def dispatch(self, sql, params):
        import json

        is_user_table = '"user"' in sql
        table = self.users if is_user_table else self.customers
        pk = "internal_user_id" if is_user_table else "internal_customer_id"
        ctx_col = "user_context" if is_user_table else "customer_context"

        if "FROM client" in sql:
            cid = self.clients.get(str(params["u"]))
            return (cid,) if cid else None

        # Context reads by internal PK (plain read + FOR UPDATE inside write).
        if sql.startswith(f"SELECT {ctx_col}"):
            row = self._find(table, **{pk: params["k"]})
            return (row[ctx_col],) if row else None

        if sql.startswith("UPDATE"):
            row = self._find(table, **{pk: params["k"]})
            if row is not None:
                ctx = row.get(ctx_col) or {}
                ctx["profile"] = json.loads(params["doc"])
                row[ctx_col] = ctx
            return None

        # read_profile's single-statement path: emulate the UNION's
        # row-existence precedence (owner row wins even with a NULL profile).
        if "UNION ALL" in sql:
            assert sql.count("client_id = :cid") == 4, (
                f"tenancy guard: every UNION branch must be client-scoped, got: {sql}"
            )
            candidates = []
            u = self._find(self.users, client_id=params["cid"], user_id=params["ext"])
            if u is not None:
                candidates.append((1, u.get("user_context")))
            c = self._find(self.customers, client_id=params["cid"], customer_id=params["ext"])
            if c is not None:
                candidates.append((2, c.get("customer_context")))
            if params.get("extc"):
                c2 = self._find(self.customers, client_id=params["cid"], customer_id=params["extc"])
                if c2 is not None:
                    candidates.append((3, c2.get("customer_context")))
            if params.get("h"):
                c3 = self._find(self.customers, client_id=params["cid"], uuid5_hash=params["h"])
                if c3 is not None:
                    candidates.append((4, c3.get("customer_context")))
            if not candidates:
                return None
            ctx = sorted(candidates)[0][1] or {}
            return (ctx.get("profile"),)

        # Owner-resolution lookups: MUST be client-scoped (review finding 8).
        if 'FROM "user"' in sql or "FROM customer" in sql:
            assert "client_id = :cid" in sql, (
                f"tenancy guard: lookups must be client-scoped, got: {sql}"
            )
            if "uuid5_hash" in sql:
                row = self._find(table, client_id=params["cid"], uuid5_hash=params["h"])
            else:
                ext_col = "user_id" if is_user_table else "customer_id"
                row = self._find(table, client_id=params["cid"], **{ext_col: params["ext"]})
            return (row[pk],) if row else None

        raise AssertionError(f"unexpected SQL: {sql}")


@pytest.fixture()
def db():
    fake = _FakeDB()
    fake.clients[str(CLIENT_UUID_A)] = "cli_a"
    fake.clients[str(CLIENT_UUID_B)] = "cli_b"
    return fake


@pytest.fixture()
def store(db):
    return ProfileStore(session_factory=db.session_factory)


USER_KEY = str(uuid.uuid4())
CUST_KEY = str(uuid.uuid4())


def _add_user(db, client_id="cli_a", user_id="u-1", uuid5_hash=None, ctx=None, key=USER_KEY):
    db.users.append({
        "client_id": client_id,
        "user_id": user_id,
        "uuid5_hash": str(uuid5_hash or uuid.uuid4()),
        "internal_user_id": key,
        "user_context": ctx,
    })


def _add_customer(db, client_id="cli_a", customer_id="u-1", uuid5_hash=None, ctx=None, key=CUST_KEY):
    db.customers.append({
        "client_id": client_id,
        "customer_id": customer_id,
        "uuid5_hash": str(uuid5_hash or uuid.uuid4()),
        "internal_customer_id": key,
        "customer_context": ctx,
    })


class TestResolveOwnerExternalIds:
    async def test_user_row_wins(self, db, store):
        _add_user(db)
        _add_customer(db)  # same external id also present as customer
        assert await store.resolve_owner("cli_a", "u-1") == ("user", USER_KEY)

    async def test_b2c_falls_back_to_customer_row(self, db, store):
        # B2C: IdMapper creates NO user row; caller lives in customer table
        # under the same external id (dashboard fallback semantics).
        _add_customer(db)
        assert await store.resolve_owner("cli_a", "u-1") == ("customer", CUST_KEY)

    async def test_explicit_customer_id_fallback(self, db, store):
        _add_customer(db, customer_id="acme")
        owner = await store.resolve_owner(
            "cli_a", "someone", external_customer_id="acme"
        )
        assert owner == ("customer", CUST_KEY)

    async def test_absent_everywhere_is_none(self, db, store):
        assert await store.resolve_owner("cli_a", "ghost") is None

    async def test_other_tenants_rows_invisible(self, db, store):
        _add_user(db, client_id="cli_b")
        _add_customer(db, client_id="cli_b")
        assert await store.resolve_owner("cli_a", "u-1") is None


class TestResolveOwnerByScopeUuids:
    async def test_user_row_by_client_scoped_hash(self, db, store):
        _add_user(db, uuid5_hash=LEGACY_HASH)
        owner = await store.resolve_owner_by_scope_uuids(CLIENT_UUID_A, LEGACY_HASH)
        assert owner == ("user", USER_KEY)

    async def test_b2c_customer_fallback_same_uuid(self, db, store):
        # B2C: user UUID == customer UUID, no user row.
        h = uuid.uuid4()
        _add_customer(db, uuid5_hash=h)
        owner = await store.resolve_owner_by_scope_uuids(CLIENT_UUID_A, h, customer_uuid=h)
        assert owner == ("customer", CUST_KEY)

    async def test_cross_tenant_hash_collision_never_matches(self, db, store):
        # Review finding 8: legacy uuid5(NAMESPACE_URL, external_user_id) is
        # NOT client-scoped — the same phone number ingested by two tenants
        # yields the SAME uuid5_hash. Tenant B's row must never resolve for
        # tenant A's scope, and every lookup must carry the client filter.
        _add_user(db, client_id="cli_b", uuid5_hash=LEGACY_HASH)
        _add_customer(db, client_id="cli_b", uuid5_hash=LEGACY_HASH)
        owner = await store.resolve_owner_by_scope_uuids(
            CLIENT_UUID_A, LEGACY_HASH, customer_uuid=LEGACY_HASH
        )
        assert owner is None
        # And the same hash resolves fine for the tenant that owns it.
        owner_b = await store.resolve_owner_by_scope_uuids(CLIENT_UUID_B, LEGACY_HASH)
        assert owner_b == ("user", USER_KEY)
        # Belt and braces: no user/customer lookup ran without client_id.
        for sql, params in db.executed:
            if 'FROM "user"' in sql or "FROM customer" in sql:
                assert "client_id = :cid" in sql
                assert params.get("cid")

    async def test_unknown_client_uuid_is_none(self, db, store):
        _add_user(db, uuid5_hash=LEGACY_HASH)
        assert (
            await store.resolve_owner_by_scope_uuids(uuid.uuid4(), LEGACY_HASH) is None
        )


class TestReadWrite:
    async def test_read_returns_profile_key(self, db, store):
        _add_user(db, ctx={"profile": {"overview": "hi", "_meta": {"version": 3}}})
        assert (await store.read(("user", USER_KEY)))["overview"] == "hi"

    async def test_read_null_context_is_none(self, db, store):
        _add_user(db, ctx=None)
        assert await store.read(("user", USER_KEY)) is None

    async def test_read_profile_convenience_b2c(self, db, store):
        _add_customer(db, ctx={"profile": {"overview": "cust"}})
        prof = await store.read_profile("cli_a", "u-1")
        assert prof == {"overview": "cust"}

    async def test_read_profile_is_single_query(self, db, store):
        # The whole point of the collapsed statement: hot path = 1 round trip.
        _add_customer(db, ctx={"profile": {"overview": "cust"}})
        await store.read_profile("cli_a", "u-1")
        assert len(db.executed) == 1


class TestReadProfileParity:
    """read_profile must keep resolve_owner->read semantics exactly."""

    async def test_user_row_without_profile_beats_customer_with_one(self, db, store):
        # Row-existence precedence, not profile-presence: the user row is the
        # owner even when its profile is absent, so the result is None.
        _add_user(db, ctx=None)
        _add_customer(db, ctx={"profile": {"overview": "cust"}})
        assert await store.read_profile("cli_a", "u-1") is None

    async def test_user_row_profile_wins(self, db, store):
        _add_user(db, ctx={"profile": {"overview": "user"}})
        _add_customer(db, ctx={"profile": {"overview": "cust"}})
        assert (await store.read_profile("cli_a", "u-1"))["overview"] == "user"

    async def test_explicit_customer_id_branch(self, db, store):
        _add_customer(db, customer_id="acme", ctx={"profile": {"overview": "org"}})
        prof = await store.read_profile(
            "cli_a", "someone", external_customer_id="acme"
        )
        assert prof == {"overview": "org"}

    async def test_uuid5_hint_branch(self, db, store):
        _add_customer(db, customer_id="other-ext", uuid5_hash=LEGACY_HASH,
                      ctx={"profile": {"overview": "hinted"}})
        prof = await store.read_profile("cli_a", "ghost", customer_uuid_hint=LEGACY_HASH)
        assert prof == {"overview": "hinted"}

    async def test_absent_everywhere_is_none(self, db, store):
        assert await store.read_profile("cli_a", "ghost") is None

    async def test_cross_tenant_invisible(self, db, store):
        _add_customer(db, client_id="cli_b", ctx={"profile": {"overview": "b"}})
        assert await store.read_profile("cli_a", "u-1") is None

    async def test_write_locks_and_merges_fresh(self, db, store):
        _add_user(db, ctx={"profile": {"overview": "old", "_meta": {"version": 1}}, "other_key": "kept"})
        seen = {}

        def merge(fresh):
            seen["fresh"] = fresh
            return {"overview": "new", "_meta": {"version": 2}}

        written = await store.write(("user", USER_KEY), merge)
        assert written["overview"] == "new"
        assert seen["fresh"]["overview"] == "old"  # merge saw the fresh row
        assert db.users[0]["user_context"]["profile"]["overview"] == "new"
        assert db.users[0]["user_context"]["other_key"] == "kept"  # jsonb_set semantics
        select_sql = next(s for s, _ in db.executed if "FOR UPDATE" in s)
        assert "FOR UPDATE" in select_sql

    async def test_write_missing_row_returns_none(self, db, store):
        assert await store.write(("user", str(uuid.uuid4())), lambda f: {"x": 1}) is None

    async def test_write_customer_row(self, db, store):
        _add_customer(db, ctx=None)
        written = await store.write(("customer", CUST_KEY), lambda f: {"overview": "c"})
        assert written == {"overview": "c"}
        assert db.customers[0]["customer_context"]["profile"] == {"overview": "c"}
