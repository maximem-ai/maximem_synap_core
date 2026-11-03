"""The schema runner. The database tests need a Postgres:

    SYNAP_TEST_PG_URL=postgresql://user:pass@127.0.0.1:5432/postgres pytest synap/tests/db

Each test makes its own database and drops it.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

asyncpg = pytest.importorskip("asyncpg")

from synap.db import runner  # noqa: E402

PG_URL = os.environ.get("SYNAP_TEST_PG_URL")
needs_pg = pytest.mark.skipif(not PG_URL, reason="set SYNAP_TEST_PG_URL to a Postgres to run this")


def _url(dbname: str) -> str:
    p = urlsplit(PG_URL)
    return urlunsplit((p.scheme, p.netloc, "/" + dbname, p.query, ""))


@pytest.fixture
async def dsn():
    name = "runner_" + secrets.token_hex(4)
    admin = await asyncpg.connect(_url("postgres"))
    await admin.execute(f'create database "{name}"')
    try:
        yield _url(name)
    finally:
        await admin.execute(f'drop database if exists "{name}" with (force)')
        await admin.close()


@pytest.fixture
def files(tmp_path):
    base = tmp_path / "baseline.sql"
    base.write_text("create table thing (id int primary key, name text);\n")
    mig = tmp_path / "migrations"
    mig.mkdir()
    return base, mig


async def _tables(dsn):
    c = await asyncpg.connect(dsn)
    try:
        return sorted(r[0] for r in await c.fetch("select tablename from pg_tables where schemaname = 'public'"))
    finally:
        await c.close()


# --- no database needed ----------------------------------------------------------

def test_files_run_in_number_order_then_name(tmp_path):
    for n in ["010_b.sql", "002_a.sql", "002_a2.sql", "048b_x.sql", "048_y.sql", "100_z.sql"]:
        (tmp_path / n).write_text("select 1;")
    assert [p.name for p in runner.migration_files(tmp_path)] == [
        "002_a.sql", "002_a2.sql", "010_b.sql", "048_y.sql", "048b_x.sql", "100_z.sql"]


def test_a_file_without_a_number_is_refused(tmp_path):
    (tmp_path / "add_column.sql").write_text("select 1;")
    with pytest.raises(runner.MigrationError, match="starts with its number"):
        runner.migration_files(tmp_path)


def test_a_missing_folder_means_no_files(tmp_path):
    assert runner.migration_files(tmp_path / "nope") == []


# --- against a Postgres ----------------------------------------------------------

@needs_pg
async def test_empty_database_gets_the_baseline(dsn, files):
    base, mig = files
    assert await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None) == ["baseline"]
    assert await _tables(dsn) == ["schema_migrations", "thing"]


@needs_pg
async def test_second_run_changes_nothing(dsn, files):
    base, mig = files
    await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)
    assert await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None) == []


@needs_pg
async def test_later_files_apply_once_in_order(dsn, files):
    base, mig = files
    await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)
    (mig / "001_add_col.sql").write_text("alter table thing add column extra text;")
    (mig / "002_more.sql").write_text("alter table thing add column extra2 text;")
    assert await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None) == ["001_add_col.sql", "002_more.sql"]
    assert await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None) == []


@needs_pg
async def test_a_failing_file_leaves_no_trace(dsn, files):
    base, mig = files
    await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)
    (mig / "001_bad.sql").write_text("alter table thing add column ok text; select 1/0;")
    with pytest.raises(asyncpg.PostgresError):
        await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)
    c = await asyncpg.connect(dsn)
    try:
        cols = {r[0] for r in await c.fetch("select column_name from information_schema.columns where table_name = 'thing'")}
        assert "ok" not in cols
        assert await c.fetchval("select count(*) from schema_migrations where version = '001_bad.sql'") == 0
    finally:
        await c.close()


@needs_pg
async def test_a_file_edited_after_it_ran_is_an_error(dsn, files):
    base, mig = files
    await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)
    f = mig / "001_x.sql"
    f.write_text("alter table thing add column a text;")
    await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)
    f.write_text("alter table thing add column b text;")
    with pytest.raises(runner.MigrationError, match="edited after it was applied"):
        await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)


@needs_pg
async def test_a_changed_baseline_is_an_error(dsn, files):
    base, mig = files
    await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)
    base.write_text("create table other (id int);")
    with pytest.raises(runner.MigrationError, match="baseline.sql differs"):
        await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)


@needs_pg
async def test_a_database_that_already_has_tables_is_left_alone(dsn, files):
    base, mig = files
    c = await asyncpg.connect(dsn)
    await c.execute("create table somebody_elses (id int)")
    await c.close()
    with pytest.raises(runner.MigrationError, match="already has tables"):
        await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)
    assert await _tables(dsn) == ["schema_migrations", "somebody_elses"]


@needs_pg
async def test_status_reports_pending_without_changing_anything(dsn, files):
    base, mig = files
    await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None)
    (mig / "001_x.sql").write_text("select 1;")
    s = await runner.status(dsn, migrations=mig)
    assert s == {"applied": ["baseline"], "pending": ["001_x.sql"], "baseline_applied": True}
    assert (await runner.status(dsn, migrations=mig))["pending"] == ["001_x.sql"]


@needs_pg
async def test_a_baseline_that_empties_search_path_does_not_break_later_files(dsn, files):
    base, mig = files
    base.write_text("select pg_catalog.set_config('search_path', '', false);\ncreate table public.thing (id int);\n")
    (mig / "001_x.sql").write_text("alter table thing add column a text;")
    assert await runner.migrate(dsn, baseline=base, migrations=mig, log=lambda *_: None) == ["baseline", "001_x.sql"]
