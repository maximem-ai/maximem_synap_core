"""Handing `resolve_customer` an id it produced must not invent a customer.

WHAT WENT WRONG

`resolve_customer` turns a client's external id into an internal uuid and
auto-creates a `customer` row on first sight. Hand it back an id it produced
and it made a SECOND customer, whose external id is a uuid string no client
will ever send. The same wrong id then picks the vector collection, so the
request that caused it was also aimed at a namespace that does not exist.

WHAT THE NUMBERS ACTUALLY ARE, BECAUSE THE FIRST COUNT WAS WRONG

The audit reported phantom customer rows and memories behind them,
using the rule "the external id looks like a uuid". That rule is not sound. A
client may use uuids as their own external ids, and a client may do
exactly that for nearly all of their customers.

Split by what the uuid actually is:

uuid4   client-generated. Legitimate. All addressable, because the client
sends the same id back.
uuid7   client-generated. Legitimate.
uuid5   OURS. Every one is a value of `customer.uuid5_hash` for that same
client. These are the real defect, and they hold very few memories.

So the damage is small, and still
worth closing, because the same call that creates one is also deleting from,
or writing to, the wrong collection.

WHY THE FIX IS HERE AND NOT AT THE CALL SITES

Two call sites were found and fixed (`delete_memory`, `update_memory`). A third
exists and has NOT been identified: 21 of the 73 belong to one client whose
rows appear alongside `memories/create` traffic with no delete anywhere near
them. Every source passes through this function, so this is where it closes.

The property that was missing is idempotence. `resolve` could not recognise its
own output. Now it can, and it answers with the customer that id already
belongs to, which is what the caller meant.
"""
from __future__ import annotations

import uuid as _uuid

import pytest

from synap.cloud.shared.id_mapper import IdMapper

CLIENT = "cli_test"
CLIENT_UUID = _uuid.uuid4()


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _Session:
    """Answers the two SELECTs from a dict and records any INSERT.

    A fake rather than Postgres on purpose: this behaviour has nothing to do
    with the schema, and 96 tests in this directory already skip when no DSN is
    present, which is how a guard quietly stops running.

    Rows are keyed by (client_id, external_id) and every query honours `:cid`.
    The first version keyed by external id alone, and a mutation that dropped
    `WHERE client_id = :cid` from the lookup passed against it, so the fake was
    hiding a cross-client read. A fake that is looser than the database cannot
    see the bug the database would have.
    """

    def __init__(self, customers):
        # (client_id, external_id) -> uuid5_hash
        self.customers = dict(customers)
        self.inserted = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        params = params or {}
        if sql.startswith("INSERT INTO customer"):
            self.inserted.append(params)
            return _Result(None)
        if "customer_id = :custid" in sql:
            return _Result(self.customers.get((params["cid"], params["custid"])))
        if "uuid5_hash = CAST(:val AS uuid)" in sql:
            scoped = "client_id = :cid" in sql
            for (cid, ext), h in self.customers.items():
                if str(h) != str(params["val"]):
                    continue
                if scoped and cid != params["cid"]:
                    continue
                return _Result(ext)
            return _Result(None)
        raise AssertionError(f"unexpected sql: {sql}")

    async def commit(self):
        pass


def _mapper(customers):
    session = _Session(customers)

    class _Factory:
        def __call__(self):
            return session

    m = IdMapper(_Factory())
    return m, session


REAL = _uuid.uuid5(CLIENT_UUID, "acme")


@pytest.mark.asyncio
async def test_our_own_uuid_resolves_to_itself_and_creates_nothing():
    """The bug, in one call. Before this, the second line created a row."""
    m, session = _mapper({(CLIENT, "acme"): REAL})
    first = await m.resolve_customer(CLIENT, "acme", client_uuid=CLIENT_UUID)
    again = await m.resolve_customer(CLIENT, str(first), client_uuid=CLIENT_UUID)
    assert again == first, "the mapper does not recognise its own output"
    assert session.inserted == [], (
        "a phantom customer row was created for an id we produced ourselves"
    )


@pytest.mark.asyncio
async def test_a_clients_own_uuid_is_still_an_ordinary_external_id():
    """The half that must not break, and the reason the check is a lookup
    rather than a uuid-version test.

    A client may use a uuid4 per customer.
    Refusing uuid-shaped ids would have broken them to fix a few rows.
    """
    theirs = str(_uuid.uuid4())
    m, session = _mapper({})
    got = await m.resolve_customer(CLIENT, theirs, client_uuid=CLIENT_UUID)
    assert got == _uuid.uuid5(CLIENT_UUID, theirs)
    assert len(session.inserted) == 1
    assert session.inserted[0]["custid"] == theirs


@pytest.mark.asyncio
async def test_a_uuid5_that_is_not_ours_is_still_an_ordinary_external_id():
    """Version is not the test. A client may send a uuid5 of their own, and it
    is theirs unless it is a value we hold."""
    theirs = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "their-own-namespace"))
    m, session = _mapper({(CLIENT, "acme"): REAL})
    got = await m.resolve_customer(CLIENT, theirs, client_uuid=CLIENT_UUID)
    assert got == _uuid.uuid5(CLIENT_UUID, theirs)
    assert len(session.inserted) == 1


@pytest.mark.asyncio
async def test_a_non_uuid_never_pays_for_the_extra_lookup():
    """Ordinary external ids are the overwhelming majority of traffic, and they
    must not gain a query. `_Session` raises on any sql it did not expect, so
    an extra SELECT here fails the test rather than slipping through."""
    m, session = _mapper({})
    await m.resolve_customer(CLIENT, "acme-corp", client_uuid=CLIENT_UUID)
    assert len(session.inserted) == 1


@pytest.mark.asyncio
async def test_another_clients_uuid_is_not_borrowed():
    """The lookup is scoped by client, and this is the test that says so.

    Another client's customer holds this uuid. If the `WHERE client_id = :cid`
    is ever dropped, resolution reaches across a client boundary and returns
    somebody else's customer, which is a far worse bug than the one being
    fixed. A mutation that removed that clause passed until this test existed.
    """
    theirs = _uuid.uuid5(_uuid.uuid4(), "someone-elses-customer")
    m, session = _mapper({("cli_someone_else", "their-customer"): theirs})
    got = await m.resolve_customer(CLIENT, str(theirs), client_uuid=CLIENT_UUID)
    assert got == _uuid.uuid5(CLIENT_UUID, str(theirs)), (
        "another client's customer uuid was treated as ours"
    )
    assert len(session.inserted) == 1


@pytest.mark.asyncio
async def test_the_gate_is_the_shape_of_a_uuid_not_its_version():
    """Why `_looks_like_uuid` asks about shape and the lookup does the rest.

    Our customer uuids are uuid5 today, so narrowing the gate to version 5
    would behave identically right now and would be wrong the moment a stored
    uuid is made any other way. The question that matters is never which
    version it is, it is whether this exact value is already one of this
    client's customer uuids, and that is a lookup.
    """
    stored_v4 = _uuid.uuid4()
    m, session = _mapper({(CLIENT, "acme"): stored_v4})
    got = await m.resolve_customer(CLIENT, str(stored_v4), client_uuid=CLIENT_UUID)
    assert got == stored_v4
    assert session.inserted == [], (
        "a stored uuid was not recognised as ours because of its version"
    )
