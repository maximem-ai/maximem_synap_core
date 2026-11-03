"""
ClientConfigRepository.upsert for client-level keys.

The table's unique constraint includes instance_id, and NULLs never conflict,
so ON CONFLICT against it does nothing for a client-level key. These tests pin
that such a key is written as an update or a plain insert, never through the
constraint. The behaviour against a real database (one row after repeated and
concurrent writes) was checked by hand when migration 162 was written; no
database is available to this suite.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError

from synap.cloud.database.repositories.client_config_repo import ClientConfigRepository


class _Session:
    """Records the SQL it is asked to run and serves canned reads."""

    def __init__(self, existing=None, insert_conflicts=False):
        self.existing = existing
        self.insert_conflicts = insert_conflicts
        self.statements = []

    async def execute(self, statement):
        sql = str(statement.compile(dialect=postgresql.dialect())).strip()
        self.statements.append(sql)
        if sql.startswith("INSERT") and self.insert_conflicts:
            # Someone else's row now exists.
            self.insert_conflicts = False
            self.existing = SimpleNamespace(version="1", config_value={"v": "theirs"})
            raise IntegrityError("insert", {}, Exception("duplicate key"))
        scalars = SimpleNamespace(first=lambda: self.existing)
        return SimpleNamespace(scalars=lambda: scalars, rowcount=1)

    @asynccontextmanager
    async def begin_nested(self):
        yield

    async def flush(self):
        pass

    def writes(self):
        return [s for s in self.statements if not s.startswith("SELECT")]


@pytest.mark.asyncio
async def test_a_first_client_level_write_is_a_plain_insert():
    session = _Session(existing=None)
    await ClientConfigRepository(session).upsert("cli_1", "k", {"v": 1})

    (write,) = session.writes()
    assert write.startswith("INSERT INTO client_config")
    assert "ON CONFLICT" not in write


@pytest.mark.asyncio
async def test_a_later_client_level_write_updates_the_row_and_bumps_the_version():
    session = _Session(existing=SimpleNamespace(version="4", config_value={"v": 1}))
    await ClientConfigRepository(session).upsert("cli_1", "k", {"v": 2})

    (write,) = session.writes()
    assert write.startswith("UPDATE client_config")
    assert "instance_id IS NULL" in write


@pytest.mark.asyncio
async def test_losing_a_race_on_the_first_write_falls_back_to_an_update():
    session = _Session(existing=None, insert_conflicts=True)
    await ClientConfigRepository(session).upsert("cli_1", "k", {"v": 2})

    writes = session.writes()
    assert writes[0].startswith("INSERT INTO client_config")
    assert writes[1].startswith("UPDATE client_config")


@pytest.mark.asyncio
async def test_an_instance_level_write_still_uses_the_constraint():
    session = _Session(existing=None)
    await ClientConfigRepository(session).upsert(
        "cli_1", "k", {"v": 1}, instance_id="inst_1"
    )

    (write,) = session.writes()
    assert "ON CONFLICT ON CONSTRAINT uq_client_config_key" in write


@pytest.mark.asyncio
async def test_a_read_takes_the_newest_row_rather_than_raising_on_two():
    session = _Session(existing=SimpleNamespace(version="1", config_value={}))
    await ClientConfigRepository(session).get_by_key("cli_1", "k")

    (read,) = session.statements
    assert "ORDER BY client_config.updated_at DESC" in read
    assert "LIMIT" in read
