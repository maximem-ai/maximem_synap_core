"""Retention expires memories older than guardrails.retention.max_memory_age_days.

The sweep is tested through fakes: a dict of memories stands in for Postgres, and
a fake delete-by-id stands in for the route. The real delete wiring is checked
separately, against the route function itself.
"""
from __future__ import annotations

import ast
import contextlib
import logging
import pathlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from synap.cloud.application.retention import sweep
from synap.cloud.memory_config.config_schema import (
    GuardrailsConfig, MemoryArchitectureConfig, RetentionConfig)
from synap.cloud.application.retention.sweep import (
    Candidate, CleanupLedger, Deps, DeleteOutcome, run_cycle)

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
SECRET = "TOP-SECRET-ALLERGY-TEXT"


def cfg(days):
    """A config as the loaders build it: from a dict, so the record of what was
    written is real."""
    return MemoryArchitectureConfig(
        metadata={}, guardrails={"retention": {"max_memory_age_days": days}})


def forced(value):
    """A config whose limit is set to *value* even where validation would refuse it."""
    retention = RetentionConfig.model_construct(
        _fields_set={"max_memory_age_days"}, max_memory_age_days=value)
    guardrails = GuardrailsConfig.model_construct(_fields_set={"retention"}, retention=retention)
    return MemoryArchitectureConfig.model_construct(
        _fields_set={"guardrails"}, guardrails=guardrails)


class World:
    """Postgres stand-in: id -> row."""

    def __init__(self):
        self.rows = {}
        self.delete_calls = []
        self.fail = {}      # memory id -> how many more times the delete raises
        self.partial = {}   # memory id -> stores reported as still holding it
        self.retry_calls = []
        self.retry_ok = True

    def add(self, mid, instance, age_days, deleted=False):
        self.rows[mid] = dict(instance=instance, created=NOW - timedelta(days=age_days),
                              deleted=deleted, content=SECRET)

    def ids(self, instance=None):
        return {m for m, r in self.rows.items() if instance in (None, r["instance"])}

    async def find_expired(self, instance, cutoff, limit, after):
        out = sorted((r["created"], m) for m, r in self.rows.items()
                     if r["instance"] == instance and not r["deleted"] and r["created"] < cutoff)
        if after is not None:
            out = [x for x in out if x > after]
        return [Candidate(m, c, "client", "cust", "user") for c, m in out[:limit]]

    async def delete_by_id(self, instance, cand):
        self.delete_calls.append(cand.memory_id)
        if self.fail.get(cand.memory_id, 0) > 0:
            self.fail[cand.memory_id] -= 1
            raise RuntimeError(f"db down {SECRET}")
        if cand.memory_id not in self.rows:
            return DeleteOutcome(gone=True)
        del self.rows[cand.memory_id]
        return DeleteOutcome(still_present_in=self.partial.pop(cand.memory_id, []))

    async def retry_cleanup(self, item):
        self.retry_calls.append((item.candidate.memory_id, list(item.stores)))
        return [] if self.retry_ok else list(item.stores)


def make_deps(world, configs, enabled=lambda iid: True):
    async def list_instances():
        return list(configs)

    async def get_config(iid):
        return configs[iid]

    return Deps(list_instances=list_instances, get_config=get_config,
                find_expired=world.find_expired, delete_by_id=world.delete_by_id,
                retry_cleanup=world.retry_cleanup, flag_enabled=enabled, now=lambda: NOW)


# --- the switch ---------------------------------------------------------------


def test_switch_off_means_no_worker(monkeypatch):
    monkeypatch.delenv("SYNAP_GUARDRAIL_RETENTION", raising=False)
    monkeypatch.delenv("SYNAP_GUARDRAIL_RETENTION_INSTANCES", raising=False)
    assert sweep.retention_wanted() is False


def test_switch_on_starts_the_worker(monkeypatch):
    monkeypatch.setenv("SYNAP_GUARDRAIL_RETENTION", "true")
    assert sweep.retention_wanted() is True


def test_switch_named_for_some_instances_starts_the_worker(monkeypatch):
    monkeypatch.delenv("SYNAP_GUARDRAIL_RETENTION", raising=False)
    monkeypatch.setenv("SYNAP_GUARDRAIL_RETENTION_INSTANCES", "inst_a")
    assert sweep.retention_wanted() is True


def test_main_starts_the_worker_only_behind_the_switch():
    src = pathlib.Path(sweep.__file__).parents[1].joinpath("main.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "retention_wanted" in ast.unparse(node.test):
            assert "RetentionWorker()" in ast.unparse(node)
            return
    pytest.fail("main.py does not gate the retention worker on retention_wanted()")


async def test_an_instance_the_switch_is_off_for_is_not_touched():
    w = World()
    w.add("a1", "inst_a", 100)
    w.add("b1", "inst_b", 100)
    deps = make_deps(w, {"inst_a": cfg(30), "inst_b": cfg(30)},
                     enabled=lambda iid: iid == "inst_a")
    await run_cycle(deps, CleanupLedger())
    assert w.ids() == {"b1"}


# --- what expires ---------------------------------------------------------------


async def test_expires_only_memories_older_than_the_cutoff():
    w = World()
    w.add("old", "inst_a", 31)
    w.add("edge_young", "inst_a", 29)
    w.add("new", "inst_a", 1)
    out = await run_cycle(make_deps(w, {"inst_a": cfg(30)}), CleanupLedger())
    assert w.ids() == {"edge_young", "new"}
    assert out["inst_a"]["expired"] == 1


async def test_already_deleted_memories_are_skipped():
    w = World()
    w.add("gone", "inst_a", 400, deleted=True)
    w.add("live", "inst_a", 400)
    await run_cycle(make_deps(w, {"inst_a": cfg(30)}), CleanupLedger())
    assert w.delete_calls == ["live"]


@pytest.mark.parametrize("value", [None, 0, -5, True, "30"])
async def test_no_usable_age_limit_expires_nothing(value):
    w = World()
    w.add("m", "inst_a", 4000)
    out = await run_cycle(make_deps(w, {"inst_a": forced(value)}), CleanupLedger())
    assert w.ids() == {"m"} and w.delete_calls == []
    assert out["inst_a"]["skipped"] == "no_age_limit"


async def test_a_config_without_the_block_expires_nothing():
    w = World()
    w.add("m", "inst_a", 4000)
    await run_cycle(make_deps(w, {"inst_a": SimpleNamespace()}), CleanupLedger())
    assert w.ids() == {"m"}


async def test_running_twice_removes_nothing_extra():
    w = World()
    for i in range(5):
        w.add(f"o{i}", "inst_a", 90)
    w.add("young", "inst_a", 2)
    deps, ledger = make_deps(w, {"inst_a": cfg(30)}), CleanupLedger()
    await run_cycle(deps, ledger)
    calls = len(w.delete_calls)
    second = await run_cycle(deps, ledger)
    assert w.ids() == {"young"}
    assert len(w.delete_calls) == calls and second["inst_a"]["expired"] == 0


async def test_instances_do_not_share_an_age_limit():
    w = World()
    w.add("a1", "inst_a", 100)
    w.add("b1", "inst_b", 100)
    await run_cycle(make_deps(w, {"inst_a": cfg(30), "inst_b": cfg(365)}), CleanupLedger())
    assert w.ids() == {"b1"}


# --- only an explicit setting acts ---------------------------------------------------


def from_dict(guardrails=None):
    raw = {"metadata": {}}
    if guardrails is not None:
        raw["guardrails"] = guardrails
    return MemoryArchitectureConfig(**raw)


@pytest.mark.parametrize("config", [
    from_dict(),
    from_dict({}),
    from_dict({"retention": {}}),
    from_dict({"retention": {"episode_retention_days": 10}}),
    from_dict({"compliance": {"frameworks": ["hipaa"]}}),
    MemoryArchitectureConfig(metadata={}),
], ids=["no-block", "empty-guardrails", "empty-retention", "retention-without-age",
        "other-guardrails-block", "defaults"])
async def test_an_unset_limit_expires_nothing_even_though_the_schema_says_365(config):
    assert config.guardrails.retention.max_memory_age_days == 365   # the trap
    w = World()
    w.add("ancient", "inst_a", 5000)
    out = await run_cycle(make_deps(w, {"inst_a": config}), CleanupLedger())
    assert w.ids() == {"ancient"} and w.delete_calls == []
    assert out["inst_a"]["skipped"] == "no_age_limit"


async def test_an_explicit_365_acts_at_365():
    w = World()
    w.add("old", "inst_a", 366)
    w.add("young", "inst_a", 364)
    await run_cycle(make_deps(w, {"inst_a": cfg(365)}), CleanupLedger())
    assert w.ids() == {"young"}


async def test_an_explicit_30_acts_at_30():
    w = World()
    w.add("old", "inst_a", 31)
    w.add("young", "inst_a", 29)
    await run_cycle(make_deps(w, {"inst_a": cfg(30)}), CleanupLedger())
    assert w.ids() == {"young"}


@pytest.mark.parametrize("days", [0, -1, -365])
async def test_an_explicit_zero_or_negative_expires_nothing(days):
    w = World()
    w.add("ancient", "inst_a", 5000)
    await run_cycle(make_deps(w, {"inst_a": cfg(days)}), CleanupLedger())
    assert w.ids() == {"ancient"}


async def test_a_config_that_carries_no_record_of_what_was_set_expires_nothing():
    plain = SimpleNamespace(guardrails=SimpleNamespace(
        retention=SimpleNamespace(max_memory_age_days=30)))
    assert sweep.max_age_days(plain) is None


def test_every_level_of_the_path_must_be_set_not_only_the_leaf():
    leaf = RetentionConfig.model_construct(
        _fields_set={"max_memory_age_days"}, max_memory_age_days=30)
    no_retention_mark = MemoryArchitectureConfig.model_construct(
        _fields_set={"guardrails"},
        guardrails=GuardrailsConfig.model_construct(_fields_set=set(), retention=leaf))
    no_guardrails_mark = MemoryArchitectureConfig.model_construct(
        _fields_set=set(),
        guardrails=GuardrailsConfig.model_construct(_fields_set={"retention"}, retention=leaf))
    assert sweep.max_age_days(no_retention_mark) is None
    assert sweep.max_age_days(no_guardrails_mark) is None


# --- both loader paths keep the record of what was written ---------------------------


def _file_config(tmp_path, text):
    from synap.cloud.memory_config.file_loader import FileConfigLoader

    path = tmp_path / "memory.yml"
    path.write_text(text)
    return FileConfigLoader(str(path))


@pytest.mark.parametrize("yaml_text, expected", [
    ("", None),
    ("guardrails:\n  compliance:\n    frameworks: [hipaa]\n", None),
    ("guardrails:\n  retention:\n    episode_retention_days: 10\n", None),
    ("guardrails:\n  retention:\n    max_memory_age_days: 365\n", 365),
    ("guardrails:\n  retention:\n    max_memory_age_days: 30\n", 30),
    ("guardrails:\n  retention:\n    max_memory_age_days: 0\n", None),
])
async def test_the_core_file_loader_keeps_what_was_written(tmp_path, yaml_text, expected):
    loader = _file_config(tmp_path, yaml_text)
    assert sweep.max_age_days(await loader.get_config("any")) == expected


async def test_the_core_loader_with_no_file_expires_nothing(tmp_path):
    from synap.cloud.memory_config.file_loader import FileConfigLoader

    loader = FileConfigLoader(str(tmp_path / "missing.yml"))
    assert sweep.max_age_days(await loader.get_config("any")) is None


class _FakeFactory:
    def __call__(self):
        return self

    async def __aenter__(self):
        return "SESSION"

    async def __aexit__(self, *exc):
        return False


def _db_loader(monkeypatch, stored):
    from synap.cloud.application.ingestion import startup

    class Repo:
        def __init__(self, session):
            pass

        async def get_active_for_instance(self, instance_id):
            return None if stored is None else SimpleNamespace(config_content=stored)

    monkeypatch.setattr("synap.cloud.database.connection.get_session_factory",
                        lambda: _FakeFactory())
    monkeypatch.setattr(
        "synap.cloud.database.repositories.maca_config_repo.MACAConfigRepository", Repo)
    startup._ingestion_maca_cache.clear()
    return startup._DBConfigLoader()


@pytest.mark.parametrize("stored, expected", [
    ({"metadata": {}}, None),
    ({"metadata": {}, "guardrails": {"retention": {"episode_retention_days": 5}}}, None),
    ({"metadata": {}, "guardrails": {"retention": {"max_memory_age_days": 365}}}, 365),
    ({"metadata": {}, "guardrails": {"retention": {"max_memory_age_days": 30}}}, 30),
    ({"metadata": {}, "guardrails": {"retention": {"max_memory_age_days": -4}}}, None),
])
async def test_the_cloud_db_loader_keeps_what_was_stored(monkeypatch, stored, expected):
    loader = _db_loader(monkeypatch, stored)
    assert sweep.max_age_days(await loader.get_config("inst_x")) == expected


async def test_the_cloud_db_loader_with_no_stored_config_expires_nothing(monkeypatch):
    loader = _db_loader(monkeypatch, None)
    # no stored config also tries to resolve the external id; give it none
    monkeypatch.setattr(
        "synap.cloud.application.ingestion.startup._resolve_external_instance_id",
        lambda session, iid: _none())
    assert sweep.max_age_days(await loader.get_config("inst_x")) is None


async def _none():
    return None


# --- limits ---------------------------------------------------------------------


async def test_a_cycle_does_at_most_batch_size_times_max_batches():
    w = World()
    for i in range(100):
        w.add(f"m{i:03d}", "inst_a", 100 + i)
    deps, ledger = make_deps(w, {"inst_a": cfg(30)}), CleanupLedger()
    out = await run_cycle(deps, ledger, size=4)
    cap = 4 * sweep.MAX_BATCHES_PER_CYCLE
    assert len(w.delete_calls) == cap and out["inst_a"]["capped"] is True
    await run_cycle(deps, ledger, size=4)
    assert len(w.delete_calls) == 2 * cap


async def test_batch_size_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("SYNAP_RETENTION_BATCH_SIZE", "7")
    assert sweep.batch_size() == 7
    monkeypatch.setenv("SYNAP_RETENTION_BATCH_SIZE", "nope")
    assert sweep.batch_size() == 200
    monkeypatch.setenv("SYNAP_RETENTION_INTERVAL_SECONDS", "60")
    assert sweep.interval_seconds() == 60
    monkeypatch.delenv("SYNAP_RETENTION_INTERVAL_SECONDS")
    assert sweep.interval_seconds() == 3600


# --- failures -------------------------------------------------------------------


async def test_one_failure_does_not_stop_the_batch_and_is_retried_next_cycle():
    w = World()
    for m in ("a", "b", "c"):
        w.add(m, "inst_a", 90)
    w.fail["b"] = 1
    deps, ledger = make_deps(w, {"inst_a": cfg(30)}), CleanupLedger()
    out = await run_cycle(deps, ledger)
    assert w.ids() == {"b"} and out["inst_a"]["failed"] == 1
    out = await run_cycle(deps, ledger)
    assert w.ids() == set() and out["inst_a"]["failed"] == 0


async def test_a_failure_is_not_picked_up_again_in_the_same_cycle():
    w = World()
    for m in ("a", "b", "c"):
        w.add(m, "inst_a", 90)
    w.fail["a"] = 99
    await run_cycle(make_deps(w, {"inst_a": cfg(30)}), CleanupLedger(), size=1)
    assert w.delete_calls.count("a") == 1


async def test_a_leftover_copy_is_retried_on_a_later_cycle_until_it_is_gone():
    w = World()
    w.add("m", "inst_a", 90)
    w.partial["m"] = ["graph", "source text"]
    w.retry_ok = False
    deps, ledger = make_deps(w, {"inst_a": cfg(30)}), CleanupLedger()
    out = await run_cycle(deps, ledger)
    assert out["inst_a"]["partial"] == 1 and "m" in ledger.items
    await run_cycle(deps, ledger)
    assert w.retry_calls == [("m", ["graph"])]       # source text is not retried
    w.retry_ok = True
    out = await run_cycle(deps, ledger)
    assert out["inst_a"]["retried_ok"] == 1 and ledger.items == {}
    await run_cycle(deps, ledger)
    assert len(w.retry_calls) == 2                    # nothing left to retry


async def test_a_leftover_copy_is_given_up_after_the_attempt_limit(caplog):
    w = World()
    w.add("m", "inst_a", 90)
    w.partial["m"] = ["search index"]
    w.retry_ok = False
    deps, ledger = make_deps(w, {"inst_a": cfg(30)}), CleanupLedger()
    await run_cycle(deps, ledger)
    with caplog.at_level(logging.ERROR):
        for _ in range(sweep.MAX_CLEANUP_ATTEMPTS):
            await run_cycle(deps, ledger)
    assert ledger.items == {}
    assert "retention_cleanup_given_up memory_id=m" in caplog.text


async def test_a_retry_is_kept_apart_per_instance():
    w = World()
    w.add("m", "inst_a", 90)
    w.partial["m"] = ["graph"]
    ledger = CleanupLedger()
    await run_cycle(make_deps(w, {"inst_a": cfg(30)}), ledger)
    await run_cycle(make_deps(w, {"inst_b": cfg(30)}), ledger)
    assert w.retry_calls == [] and "m" in ledger.items


async def test_a_broken_instance_does_not_stop_the_others():
    w = World()
    w.add("b1", "inst_b", 90)
    configs = {"inst_a": cfg(30), "inst_b": cfg(30)}
    deps = make_deps(w, configs)
    real = deps.get_config

    async def flaky(iid):
        if iid == "inst_a":
            raise RuntimeError("config store down")
        return await real(iid)

    deps.get_config = flaky
    out = await run_cycle(deps, CleanupLedger())
    assert w.ids() == set() and out["inst_a"] == {"error": True}


# --- logging --------------------------------------------------------------------


async def test_the_summary_line_carries_counts_and_no_content(caplog):
    w = World()
    w.add("old", "inst_a", 90)
    w.add("bad", "inst_a", 91)
    w.fail["bad"] = 1
    with caplog.at_level(logging.DEBUG, logger=sweep.logger.name):
        await run_cycle(make_deps(w, {"inst_a": cfg(30), "inst_b": forced(None)}), CleanupLedger())
    summaries = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("retention_sweep ")]
    assert len(summaries) == 2          # one line per instance
    assert "instance=inst_a" in summaries[0] and "expired=1" in summaries[0]
    assert "failed=1" in summaries[0]
    assert SECRET not in caplog.text    # not in the summary, nor in the failure line


# --- the delete path ----------------------------------------------------------------


@contextlib.asynccontextmanager
async def _fake_session():
    yield "SESSION"


async def test_expiry_calls_the_delete_by_id_route_function(monkeypatch):
    from synap.cloud.application.sdk_api import sdk_memories

    seen = {}

    async def fake_delete(memory_id, auth, session):
        seen.update(memory_id=memory_id, auth=auth, session=session)
        return SimpleNamespace(still_present_in=["graph"])

    monkeypatch.setattr(sdk_memories, "delete_memory", fake_delete)
    monkeypatch.setattr("synap.cloud.database.connection.get_db_session", _fake_session)
    deps = sweep._real_deps()
    cid = "11111111-1111-1111-1111-111111111111"
    outcome = await deps.delete_by_id(
        "inst_a", Candidate("22222222-2222-2222-2222-222222222222", NOW, cid, None, None))
    assert seen["memory_id"] == "22222222-2222-2222-2222-222222222222"
    assert seen["auth"].client_id == cid and seen["auth"].instance_id == "inst_a"
    assert seen["session"] == "SESSION"
    assert outcome.still_present_in == ["graph"]


async def test_a_memory_the_route_cannot_find_counts_as_already_gone(monkeypatch):
    from fastapi import HTTPException
    from synap.cloud.application.sdk_api import sdk_memories

    async def missing(memory_id, auth, session):
        raise HTTPException(status_code=404, detail="Memory not found")

    monkeypatch.setattr(sdk_memories, "delete_memory", missing)
    monkeypatch.setattr("synap.cloud.database.connection.get_db_session", _fake_session)
    outcome = await sweep._real_deps().delete_by_id(
        "inst_a", Candidate("m", NOW, "11111111-1111-1111-1111-111111111111", None, None))
    assert outcome.gone is True


async def test_any_other_route_error_is_a_failure_to_retry(monkeypatch):
    from fastapi import HTTPException
    from synap.cloud.application.sdk_api import sdk_memories

    async def broken(memory_id, auth, session):
        raise HTTPException(status_code=500, detail="x")

    monkeypatch.setattr(sdk_memories, "delete_memory", broken)
    monkeypatch.setattr("synap.cloud.database.connection.get_db_session", _fake_session)
    with pytest.raises(HTTPException):
        await sweep._real_deps().delete_by_id(
            "inst_a", Candidate("m", NOW, "11111111-1111-1111-1111-111111111111", None, None))


def test_retention_has_no_deletion_sql_of_its_own():
    folder = pathlib.Path(sweep.__file__).parent
    for f in folder.glob("*.py"):
        text = f.read_text().upper()
        assert "DELETE FROM" not in text and "UPDATE MEMORIES" not in text, f.name


def test_the_query_leaves_out_deleted_rows_and_pages_by_cursor():
    src = " ".join(pathlib.Path(sweep.__file__).read_text().split())
    assert "AND deleted_at IS NULL" in src
    assert "(created_at, id) > (" in src
