"""A real Postgres for the erasure tests that touch the scope registry.

The behaviour under test is an ltree subtree query, a trigger-maintained
`scope_path`, and a join between the registry and the two identity tables that
are keyed by the PUBLIC client id rather than the client uuid. None of that
exists in a fake, and the identity-table join is exactly the shape that has
produced five separate defects in this codebase by being reasoned about instead
of run.

Set SYNAP_TEST_DATABASE_URL to a throwaway database. These tests create their
own schema from the real migration files and clean up after themselves, but
they do write.

Deliberately not shared with `synap/tests/scoping/conftest.py`, which applies a
different set of migrations and resets the scope singletons on every test. This
directory needs the pii tables and the identity tables and does not touch the
singletons, and a fixture that grows to serve two directories is a fixture that
starts skipping for reasons neither directory can see.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

TEST_DSN = os.environ.get("SYNAP_TEST_DATABASE_URL")

requires_pg = pytest.mark.skipif(
    not TEST_DSN, reason="SYNAP_TEST_DATABASE_URL is not set"
)

_MIGRATIONS = Path(__file__).resolve().parents[3] / "db" / "migrations"

# Applied in order and for real. A test schema written by hand is a test schema
# that drifts, and then the tests pass against a database production does not
# have.
_FILES = (
    "001_enums.sql",
    "002_setup_entities.sql",
    # `client.uuid`, `customer.uuid5_hash`, `"user".uuid5_hash`: the whole
    # bridge between the client's own ids and the ids the data plane files
    # records under.
    "032_identity_mapping_alignment.sql",
    "012_ingestion_storage_tables.sql",
    "087_pii_floor_audit.sql",
    "088_pii_aliases.sql",
    # `actor`, `actor_type`, `reason`, `target_count` on pii_floor_audit: the
    # columns the erasure writes its audit row into before it deletes anything.
    "089_pii_reveal_and_audit.sql",
    "095_nested_scoping_registry.sql",
    "096_scope_level_unique_within.sql",
    "097_scope_level_ladder_fields.sql",
    "098_scope_grants.sql",
    "101_scope_grant_client_integrity.sql",
    "103_ancestor_backfill_owed.sql",
)

_applied = False


def _async_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn
    return dsn.replace("postgresql://", "postgresql+asyncpg://", 1).replace(
        "postgres://", "postgresql+asyncpg://", 1
    )


_SENTINELS = ("client", "customer", "user", "memories", "pii_alias", "scope_node")


async def _apply_migrations() -> None:
    """Through raw asyncpg, not the SQLAlchemy engine: its asyncpg dialect
    sends everything as a prepared statement and Postgres refuses more than one
    command in one of those. A migration file is many commands by definition.

    Skipped entirely when the schema is already there. Several of these files
    predate `IF NOT EXISTS` and would fail on a second run, which would turn a
    developer's second `pytest` of the day into a wall of DuplicateObject
    errors that say nothing about the code.
    """
    global _applied
    if _applied:
        return
    import asyncpg

    raw = TEST_DSN.replace("postgresql+asyncpg://", "postgresql://").replace(
        "+asyncpg", ""
    )
    conn = await asyncpg.connect(raw)
    try:
        present = {
            r["tablename"]
            for r in await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
        }
        if not set(_SENTINELS).issubset(present):
            for name in _FILES:
                await conn.execute((_MIGRATIONS / name).read_text())
    finally:
        await conn.close()
    _applied = True


@pytest.fixture
async def pg_session_factory():
    await _apply_migrations()
    engine = create_async_engine(_async_dsn(TEST_DSN))
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()
