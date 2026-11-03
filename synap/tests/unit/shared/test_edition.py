"""The edition module: the setting, the Core environment, and the hooks."""

from __future__ import annotations

import asyncio

import pytest

from synap.cloud.shared import edition as ed


@pytest.fixture(autouse=True)
def _clean_hooks():
    ed.reset_hooks()
    yield
    # Leave the registry as a fresh process has it: empty, and filled by Cloud on first use.
    ed.reset_hooks()
    ed.load_cloud_hooks_for_test()


def test_unset_and_empty_mean_cloud():
    assert ed.resolve({}).name == "cloud"
    assert ed.resolve({"SYNAP_EDITION": ""}).name == "cloud"
    assert not ed.resolve({}).is_core


def test_core_value_means_core():
    e = ed.resolve({"SYNAP_EDITION": "core"})
    assert e.name == "core" and e.is_core


def test_unknown_value_raises():
    with pytest.raises(ValueError, match="SYNAP_EDITION"):
        ed.resolve({"SYNAP_EDITION": "enterprise"})


def test_unknown_value_raises_at_import(monkeypatch):
    import importlib
    monkeypatch.setenv("SYNAP_EDITION", "bogus")
    with pytest.raises(ValueError):
        importlib.reload(ed)
    monkeypatch.delenv("SYNAP_EDITION")
    importlib.reload(ed)


def test_cloud_choices_are_todays_behaviour():
    c = ed.resolve({})
    assert c.compaction_profile is None
    assert c.request_shape_policy is None
    assert c.scope_levels is None
    assert c.retrieval_llm is True
    assert c.config_source == "table" and c.file_store == "s3"


def test_cloud_writes_no_environment():
    env = {"SYNAP_EDITION": "cloud", "OTHER": "x"}
    ed.apply_environment(ed.resolve(env), env)
    assert env == {"SYNAP_EDITION": "cloud", "OTHER": "x"}


def test_core_writes_exactly_the_expected_set():
    env = {"OTHER": "x"}
    ed.apply_environment(ed._CORE, env)
    assert set(env) == set(ed.CORE_ENV) | {"OTHER"}
    assert env["STAGE3_EXTRACTION_MODE"] == "hybrid_graph"
    assert env["SYNAP_ER_RESOLUTION_MODE"] == "adjudicated"
    assert env["SYNAP_ER_HOLD_UNSURE"] == "false"
    assert env["SYNAP_B2C_ID_POLICY"] == "enforce"
    assert env["SYNAP_MACA_EXTRACTION_INSTANCES"] == "*"
    assert env["SYNAP_MACA_CHUNKING_INSTANCES"] == "*"
    assert len(ed.CORE_ENV) == 28  # 22, four SYNAP_GUARDRAIL_* switches, and two more
    for flag in ("INGEST_CAP", "PROHIBITED_CONTENT", "RETRIEVAL_RATE", "RETENTION", "ENTITY_GROUNDING"):
        assert env[f"SYNAP_GUARDRAIL_{flag}"] == "true"
    assert env["SYNAP_CUSTOM_CATEGORIES"] == "true"


def test_cloud_leaves_the_guardrail_switches_unset():
    env = {}
    ed.apply_environment(ed._CLOUD, env)
    assert env == {}


def test_conflicting_value_is_refused_naming_variable_and_values():
    env = {"SYNAP_ER_RESOLUTION_MODE": "legacy"}
    with pytest.raises(ed.EditionConflict) as exc:
        ed.apply_environment(ed._CORE, env)
    msg = str(exc.value)
    assert "SYNAP_ER_RESOLUTION_MODE" in msg and "'legacy'" in msg and "'adjudicated'" in msg
    assert env == {"SYNAP_ER_RESOLUTION_MODE": "legacy"}  # nothing was written


def test_equal_value_is_accepted():
    env = {"SYNAP_ER_HOLD_UNSURE": "false", "SYNAP_B2C_ID_POLICY": "Enforce"}
    ed.apply_environment(ed._CORE, env)
    assert env["SYNAP_ER_HOLD_UNSURE"] == "false"


def test_undeclared_hook_raises_on_register_and_call():
    with pytest.raises(ed.UndeclaredHook):
        ed.register("not_a_hook", lambda: None)
    with pytest.raises(ed.UndeclaredHook):
        ed.call("not_a_hook")


def test_declared_but_unregistered_hook_does_nothing():
    assert ed.call("billing_charge", 1, a=2) is None
    assert asyncio.run(ed.acall("billing_charge", 1)) is None


def test_registered_hook_is_called_with_its_arguments():
    seen = []
    ed.register("analytics", lambda *a, **k: seen.append((a, k)) or "done")
    assert ed.call("analytics", 1, x=2) == "done"
    assert seen == [((1,), {"x": 2})]


def test_acall_awaits_async_and_plain_hooks():
    async def fn(x):
        return x + 1

    ed.register("billing_charge", fn)
    ed.register("analytics", lambda x: x * 2)
    assert asyncio.run(ed.acall("billing_charge", 1)) == 2
    assert asyncio.run(ed.acall("analytics", 3)) == 6


def test_every_hook_name_in_the_cards_is_declared():
    assert len(set(ed.HOOK_NAMES)) == len(ed.HOOK_NAMES)
    # The features that are hooks. New names may be added; none may be removed.
    for name in ("pii_ingestion", "billing_charge", "analytics", "consolidation_links",
                 "replication_log_write", "grpc_push", "review_queue_write", "usage_tracking",
                 "instance_readiness_check", "maca_build_supervisor", "llm_trace_sink",
                 "anticipation_miss_record", "daily_scope_check", "forget_dashboard_queue_write"):
        assert name in ed.HOOK_NAMES


def test_health_body_carries_the_edition():
    from synap.cloud.application.main import health
    assert asyncio.run(health()) == {"status": "ok", "edition": ed.name}


def test_health_waits_for_the_first_start_downloads_in_core_only(monkeypatch, tmp_path):
    import json
    from synap.boot import first_start
    from synap.cloud.application.main import health
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    # Cloud ships its models in the image: the marker is not looked at
    monkeypatch.setattr(ed, "waits_for_first_start", False)
    assert asyncio.run(health()) == {"status": "ok", "edition": ed.name}
    # Core before the downloads finish: 503, not ready, with a reason
    monkeypatch.setattr(ed, "waits_for_first_start", True)
    r = asyncio.run(health())
    assert r.status_code == 503 and json.loads(r.body)["status"] == "not_ready"
    # a marker for some other revision is still not ready
    (tmp_path / first_start.MARKER).write_text("old-revision old-hash\n")
    assert asyncio.run(health()).status_code == 503
    # after the downloads: ok
    (tmp_path / first_start.MARKER).write_text(first_start.marker_text() + "\n")
    assert asyncio.run(health()) == {"status": "ok", "edition": ed.name}


def test_core_health_is_not_ready_while_an_ingestion_worker_is_down(monkeypatch):
    import json
    from synap.cloud.application.ingestion import startup
    from synap.cloud.application.main import health
    monkeypatch.setattr(ed, "waits_for_first_start", False)
    monkeypatch.setattr(startup, "_start_failures", ["Stage23Worker"])
    monkeypatch.setattr(startup, "_running", True)
    # Cloud answers ok whatever the workers did
    monkeypatch.setattr(ed, "health_needs_ingestion", False)
    assert asyncio.run(health()) == {"status": "ok", "edition": ed.name}
    # Core says which worker, with 503
    monkeypatch.setattr(ed, "health_needs_ingestion", True)
    r = asyncio.run(health())
    body = json.loads(r.body)
    assert r.status_code == 503 and body["status"] == "not_ready" and "Stage23Worker" in body["reason"]
    # all started: ok
    monkeypatch.setattr(startup, "_start_failures", [])
    assert asyncio.run(health()) == {"status": "ok", "edition": ed.name}
    # never started at all: not ready too
    monkeypatch.setattr(startup, "_running", False)
    assert asyncio.run(health()).status_code == 503


def test_only_core_waits_for_the_first_start():
    assert ed.resolve({}).waits_for_first_start is False
    assert ed.resolve({"SYNAP_EDITION": "core"}).waits_for_first_start is True
    assert ed.resolve({}).health_needs_ingestion is False
    assert ed.resolve({"SYNAP_EDITION": "core"}).health_needs_ingestion is True

