"""The Core wiring in main.py and sdk_api/router.py (S3 and S5 hooked up).

Behavioural tests first; the source-order test at the end is only a supplement.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from synap.cloud.application.sdk_api import core_identity, core_seed
from synap.cloud.application.sdk_api.sdk_auth import get_sdk_auth
from synap.cloud.shared import edition

REPO = Path(__file__).resolve().parents[3]

_WALK = textwrap.dedent(
    """
    import json, sys
    from fastapi import FastAPI
    from synap.cloud.application.sdk_api.router import sdk_router

    def walk(routes, prefix=""):
        for r in routes:
            if hasattr(r, "original_router"):
                yield from walk(r.original_router.routes, prefix + (r.include_context.prefix or ""))
            elif hasattr(r, "methods"):
                for m in sorted(r.methods or []):
                    yield [m, prefix + r.path]

    app = FastAPI()
    app.include_router(sdk_router)
    print("ROUTES" + json.dumps(list(walk(app.routes))))
    """
)


def _routes_in_subprocess(edition_name: str) -> list[tuple[str, str]]:
    env = dict(os.environ, SYNAP_EDITION=edition_name, PYTHONPATH=str(REPO))
    out = subprocess.run(
        [sys.executable, "-c", _WALK], env=env, cwd=REPO, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith("ROUTES"))
    return [tuple(x) for x in json.loads(line[len("ROUTES"):])]


@pytest.fixture(scope="module")
def core_routes():
    return _routes_in_subprocess("core")


def test_core_sdk_router_has_the_s3_routes_exactly_once(core_routes):
    expected = [
        ("GET", "/api/v1/auth/whoami"),
        ("GET", "/v1/credits/balance"),
        ("GET", "/v1/credits/ledger"),
        ("POST", "/v1/credits/estimate"),
        ("POST", "/v1/credits/redeem"),
        ("POST", "/v1/events/batch"),
        ("POST", "/v1/telemetry/batch"),
    ]
    for item in expected:
        assert core_routes.count(item) == 1, item


def test_core_sdk_router_has_no_duplicate_method_and_path(core_routes):
    dupes = {r for r in core_routes if core_routes.count(r) > 1}
    assert not dupes, dupes


def test_core_sdk_router_has_no_cloud_only_routes(core_routes):
    paths = {p for _, p in core_routes}
    for p in paths:
        assert not p.startswith(("/admin", "/api/v1/admin", "/api/v1/clients", "/api/v1/billing")), p
    assert not any("/admin" in p for p in paths)


def test_sdk_router_source_does_not_use_the_cloud_credits_events_telemetry():
    src = (REPO / "synap/cloud/application/sdk_api/router.py").read_text()
    for name in ("sdk_credits", "sdk_events", "sdk_telemetry"):
        assert name not in src


# --- Cloud: every startup call does nothing --------------------------------

def test_cloud_install_edition_identity_does_nothing(monkeypatch):
    monkeypatch.setattr(edition, "identity", "key")
    app = FastAPI()
    assert core_identity.install_edition_identity(app) is False
    assert app.dependency_overrides == {}


def test_cloud_seed_on_startup_does_nothing_and_never_reads_config(monkeypatch):
    monkeypatch.setattr(edition, "identity", "key")

    async def boom():
        raise AssertionError("Cloud must not read the config here")

    monkeypatch.setattr(core_seed, "_yaml_isolation", boom)
    assert asyncio.run(core_seed.seed_on_startup()) is None


def test_cloud_check_config_at_startup_does_nothing(monkeypatch):
    from synap.cloud.memory_config import loader

    monkeypatch.setattr(loader._edition_module, "config_source", "table")
    assert loader.check_config_at_startup() is None


# --- Core: where the scoping mode comes from -------------------------------

def test_core_seed_reads_the_mode_from_the_yaml_when_none_is_given(monkeypatch):
    monkeypatch.setattr(edition, "identity", "fixed")

    class _Loader:
        async def get_raw_config(self, instance_id=None):
            return {"storage_architecture": {"scoping": {"user_context_isolation": "equals_customer"}}}

    import synap.cloud.memory_config.loader as loader
    monkeypatch.setattr(loader, "get_config_loader", lambda: _Loader())
    assert asyncio.run(core_seed._yaml_isolation()) == "equals_customer"

    seen = {}

    async def fake_seed(session, *, isolation=None):
        seen["isolation"] = isolation
        return "seeded"

    class _Ctx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False

    import synap.cloud.database.connection as conn
    monkeypatch.setattr(conn, "get_db_session", lambda: _Ctx())
    monkeypatch.setattr(core_seed, "seed_default_identity", fake_seed)
    assert asyncio.run(core_seed.seed_on_startup()) == "seeded"
    assert seen["isolation"] == "equals_customer"


def test_core_seed_with_no_scoping_in_the_yaml_passes_none(monkeypatch):
    class _Loader:
        async def get_raw_config(self, instance_id=None):
            return {"metadata": {}}

    import synap.cloud.memory_config.loader as loader
    monkeypatch.setattr(loader, "get_config_loader", lambda: _Loader())
    assert asyncio.run(core_seed._yaml_isolation()) is None


def test_core_seed_rejects_maca_suggests(monkeypatch):
    with pytest.raises(ValueError, match="not supported by Synap Core"):
        core_seed.relationship_for_mode("maca_suggests")


def test_core_install_edition_identity_overrides_the_auth_dependency(monkeypatch):
    monkeypatch.setattr(edition, "identity", "fixed")
    app = FastAPI()
    assert core_identity.install_edition_identity(app) is True
    assert app.dependency_overrides[get_sdk_auth] is core_identity.get_core_sdk_auth


# --- main.py order (supplement; the behaviour is tested above) -------------

def test_main_py_calls_are_in_the_documented_order():
    src = (REPO / "synap/cloud/application/main.py").read_text()

    def at(pattern):
        m = re.search(pattern, src, re.M)
        assert m, pattern
        return m.start()

    app_created = at(r"^app = FastAPI\(")
    install = at(r"^install_edition_identity\(app\)")
    routers = at(r"edition\.call\(\"cloud_routers\", app\)")
    init_db = at(r"^    await init_db\(\)")
    check = at(r"^    check_config_at_startup\(\)")
    seed = at(r"^    await seed_on_startup\(\)")
    workers = at(r"^        await start_ingestion_workers\(\)")
    assert app_created < install < routers
    assert init_db < check < seed < workers
    # the config check is not inside a try block: its line is indented as the
    # lifespan body (4 spaces), not as a try body (8)
    assert re.search(r"^    check_config_at_startup\(\)", src, re.M)
