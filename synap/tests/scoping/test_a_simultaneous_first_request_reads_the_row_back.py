"""Two requests for a new customer or user at once must both resolve (found at the first Docker boot).

The second INSERT meets the first one's row through the (client_id, uuid5_hash) unique index, which the
statement's ON CONFLICT target does not cover, so Postgres raised and the scope code logged a warning and fell
back to derived ids. The row exists with the same hash, so the mapper now reads it back.
"""
import uuid as _uuid

import pytest
from sqlalchemy.exc import IntegrityError

from synap.cloud.shared.id_mapper import IdMapper

CLIENT = "cli_test"
EXISTING = _uuid.uuid4()


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _Session:
    """No row at the first look, an IntegrityError on the INSERT, the other request's row afterwards."""

    def __init__(self, row_after):
        self.row_after = row_after
        self.looked = 0
        self.rolled_back = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        if sql.startswith("INSERT INTO"):
            raise IntegrityError("INSERT", {}, Exception("duplicate key value violates unique constraint"))
        if "uuid5_hash = CAST(:val AS uuid)" in sql:
            return _Result(None)
        if sql.startswith("SELECT uuid5_hash"):
            self.looked += 1
            return _Result(None if self.looked == 1 else self.row_after)
        raise AssertionError(f"unexpected sql: {sql}")

    async def commit(self):
        pass

    async def rollback(self):
        self.rolled_back += 1


def _mapper(row_after):
    session = _Session(row_after)
    return IdMapper(lambda: session), session


async def test_a_new_customer_inserted_by_the_other_request_is_read_back():
    mapper, session = _mapper(EXISTING)
    assert await mapper.resolve_customer(CLIENT, "user_jev") == EXISTING
    assert session.rolled_back == 1


async def test_a_new_user_inserted_by_the_other_request_is_read_back():
    mapper, session = _mapper(EXISTING)
    assert await mapper.resolve_user(CLIENT, "user_jev", "user_jev") == EXISTING
    assert session.rolled_back == 1


async def test_an_integrity_error_with_no_row_to_read_still_raises():
    mapper, _ = _mapper(None)
    with pytest.raises(IntegrityError):
        await mapper.resolve_customer(CLIENT, "user_jev")
