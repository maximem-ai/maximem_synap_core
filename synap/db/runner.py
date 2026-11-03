"""Apply the database schema at startup.

An empty database gets ``baseline.sql``. After that, every numbered ``.sql`` file in the
migrations folder that has not been applied yet is applied in order. What was applied is
recorded in ``schema_migrations`` with a checksum, so a file edited after it ran is an
error, not a silent difference between two installs.

    python -m synap.db.runner            # DATABASE_URL from the environment
    python -m synap.db.runner --status

Nothing here runs by itself: the Core compose file runs it before the server starts.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import sys
from pathlib import Path

import asyncpg

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline.sql"
MIGRATIONS = HERE / "core_migrations"
BASELINE_VERSION = "baseline"
# Two processes starting together must not both apply the baseline.
_LOCK_KEY = 0x53594E41  # "SYNA"

_NUMBER = re.compile(r"^(\d+)([a-z]?)_.*\.sql$")

_CREATE = """
create table if not exists schema_migrations (
    version    text primary key,
    checksum   text not null,
    applied_at timestamptz not null default now()
)"""


class MigrationError(Exception):
    pass


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def migration_files(folder: Path) -> list[Path]:
    """The numbered files, in order: by number, then by name (the rule for numbers used twice)."""
    if not folder.is_dir():
        return []
    found = []
    for p in folder.iterdir():
        if p.suffix != ".sql":
            continue
        m = _NUMBER.match(p.name)
        if not m:
            raise MigrationError(f"{p.name}: a migration file name starts with its number, e.g. 001_name.sql")
        found.append((int(m.group(1)), m.group(2), p.name, p))
    return [p for *_, p in sorted(found)]


async def _is_empty(conn: asyncpg.Connection) -> bool:
    n = await conn.fetchval(
        "select count(*) from pg_tables where schemaname = 'public' and tablename <> 'schema_migrations'")
    return n == 0


async def _apply(conn: asyncpg.Connection, version: str, sql: str) -> None:
    async with conn.transaction():
        await conn.execute(sql)
        # a pg_dump file empties search_path for the session; later files must see public again
        await conn.execute("select pg_catalog.set_config('search_path', 'public', false)")
        await conn.execute(
            "insert into schema_migrations (version, checksum) values ($1, $2)", version, _checksum(sql))


async def migrate(
    dsn: str, *, baseline: Path = BASELINE, migrations: Path = MIGRATIONS, log=print
) -> list[str]:
    """Bring the database up to date. Returns the versions applied by this call."""
    files = migration_files(migrations)
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("select pg_advisory_lock($1)", _LOCK_KEY)
        await conn.execute(_CREATE)
        done = {r["version"]: r["checksum"] for r in await conn.fetch("select version, checksum from schema_migrations")}
        applied: list[str] = []

        if BASELINE_VERSION not in done:
            if not await _is_empty(conn):
                raise MigrationError(
                    "this database already has tables but no record of the baseline. "
                    "Point DATABASE_URL at an empty database; the runner does not change a database it did not create."
                )
            if not baseline.is_file():
                raise MigrationError(f"{baseline} is missing; it ships with the server image")
            sql = baseline.read_text()
            await _apply(conn, BASELINE_VERSION, sql)
            done[BASELINE_VERSION] = _checksum(sql)
            applied.append(BASELINE_VERSION)
            log("schema: applied the baseline")
        elif baseline.is_file() and done[BASELINE_VERSION] != _checksum(baseline.read_text()):
            raise MigrationError("baseline.sql differs from the one this database was built from; it must never change after a release")

        for path in files:
            sql = path.read_text()
            if path.name in done:
                if done[path.name] != _checksum(sql):
                    raise MigrationError(f"{path.name} was edited after it was applied; add a new numbered file instead")
                continue
            await _apply(conn, path.name, sql)
            applied.append(path.name)
            log(f"schema: applied {path.name}")
        if not applied:
            log("schema: up to date")
        return applied
    finally:
        await conn.close()


async def status(dsn: str, *, migrations: Path = MIGRATIONS) -> dict:
    conn = await asyncpg.connect(dsn)
    try:
        has = await conn.fetchval("select to_regclass('public.schema_migrations') is not null")
        done = [r["version"] for r in await conn.fetch("select version from schema_migrations order by applied_at")] if has else []
    finally:
        await conn.close()
    pending = [p.name for p in migration_files(migrations) if p.name not in done]
    return {"applied": done, "pending": pending, "baseline_applied": BASELINE_VERSION in done}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="show what is applied and what is pending; change nothing")
    args = ap.parse_args(argv)
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set", file=sys.stderr)
        return 2
    try:
        if args.status:
            s = asyncio.run(status(dsn))
            print(f"baseline applied: {s['baseline_applied']}; applied: {len(s['applied'])}; pending: {len(s['pending'])}")
            for n in s["pending"]:
                print(f"  pending {n}")
            return 0
        asyncio.run(migrate(dsn))
        return 0
    except MigrationError as e:
        print(f"schema: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
