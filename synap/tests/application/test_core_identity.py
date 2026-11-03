"""Core's fixed identity, whoami, and the three "not available" routes.

Run with FastAPI's TestClient and fakes. No Postgres is involved: the seeding
tests use in-memory stand-ins for the two repositories.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from synap.cloud.application.sdk_api import (
    core_identity,
    core_routes,
    core_seed,
    core_whoami,
)
from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.shared import edition


@pytest.fixture(autouse=True)
def _no_token(monkeypatch):
    monkeypatch.delenv(core_identity.API_TOKEN_ENV, raising=False)


def _core_app(monkeypatch, mode="equals_customer") -> FastAPI:
    monkeypatch.setattr(edition, "identity", "fixed")

    async def _mode(instance_id=core_identity.DEFAULT_INSTANCE_ID):
        return mode

    monkeypatch.setattr(core_identity, "load_scoping_mode", _mode)
    monkeypatch.setattr(core_whoami, "load_scoping_mode", _mode)

    app = FastAPI()
    app.include_router(core_whoami.router)
    app.include_router(core_routes.router)

    @app.post("/echo")
    async def echo(body: dict, auth: SDKAuthContext = Depends(get_sdk_auth)):
        return {"client_id": auth.client_id}

    assert core_identity.install_edition_identity(app) is True
    return app


# ---------------------------------------------------------------- selection

def test_cloud_installs_nothing(monkeypatch):
    monkeypatch.setattr(edition, "identity", "key")
    app = FastAPI()
    assert core_identity.install_edition_identity(app) is False
    assert app.dependency_overrides == {}


def test_core_installs_the_override(monkeypatch):
    app = _core_app(monkeypatch)
    assert app.dependency_overrides[get_sdk_auth] is core_identity.get_core_sdk_auth


def test_edition_value_follows_the_edition():
    # Under the test run SYNAP_EDITION is unset: Cloud.
    assert edition.identity == ("fixed" if edition.is_core else "key")


# ---------------------------------------------------------------- the dependency

def test_dependency_returns_the_same_shape_with_fixed_values(monkeypatch):
    app = _core_app(monkeypatch)
    seen = {}

    @app.get("/who")
    async def who(auth: SDKAuthContext = Depends(get_sdk_auth)):
        seen["auth"] = auth
        return {}

    r = TestClient(app).get("/who", headers={"X-Correlation-ID": "corr-1"})
    assert r.status_code == 200
    auth = seen["auth"]
    assert isinstance(auth, SDKAuthContext)
    assert auth.client_id == core_identity.DEFAULT_CLIENT_ID
    assert auth.instance_id == core_identity.DEFAULT_INSTANCE_ID
    assert auth.credential_id == core_identity.DEFAULT_CREDENTIAL_ID
    assert auth.pii_reveal_grant == "full"
    assert auth.correlation_id == "corr-1"
    assert auth.client_ip == "testclient"


def test_no_key_is_needed_when_no_token_is_set(monkeypatch):
    c = TestClient(_core_app(monkeypatch))
    assert c.get("/api/v1/auth/whoami").status_code == 200


# ---------------------------------------------------------------- the token

def test_token_required_when_set(monkeypatch):
    monkeypatch.setenv(core_identity.API_TOKEN_ENV, "s3cret")
    c = TestClient(_core_app(monkeypatch))
    assert c.get("/api/v1/auth/whoami").status_code == 401
    assert c.get("/api/v1/auth/whoami", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/api/v1/auth/whoami", headers={"Authorization": "Basic s3cret"}).status_code == 401
    r = c.get("/api/v1/auth/whoami", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


def test_token_also_guards_the_not_available_routes(monkeypatch):
    monkeypatch.setenv(core_identity.API_TOKEN_ENV, "s3cret")
    c = TestClient(_core_app(monkeypatch))
    assert c.get("/v1/credits/balance").status_code == 401


def test_warning_when_unset_and_not_when_set(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger=core_identity.logger.name):
        assert core_identity.warn_if_no_api_token() is True
    assert "NO AUTHENTICATION" in caplog.text
    monkeypatch.setenv(core_identity.API_TOKEN_ENV, "x")
    assert core_identity.warn_if_no_api_token() is False


def test_install_warns_in_core(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger=core_identity.logger.name):
        _core_app(monkeypatch)
    assert "NO AUTHENTICATION" in caplog.text


# ---------------------------------------------------------------- scope_path

def test_scope_path_in_a_json_body_is_refused(monkeypatch):
    c = TestClient(_core_app(monkeypatch))
    r = c.post("/echo", json={"user_id": "u", "scope": {"customer": "acme"}})
    assert r.status_code == 400
    assert "scope_path is not supported" in r.json()["detail"]


def test_empty_or_absent_scope_is_fine(monkeypatch):
    c = TestClient(_core_app(monkeypatch))
    assert c.post("/echo", json={"user_id": "u"}).status_code == 200
    assert c.post("/echo", json={"user_id": "u", "scope": None}).status_code == 200
    assert c.post("/echo", json={"user_id": "u", "scope": {}}).status_code == 200


def test_scope_path_in_a_multipart_form_is_refused(monkeypatch):
    from fastapi import Form

    app = _core_app(monkeypatch)

    @app.post("/upload")
    async def upload(
        user_id: str = Form(...),
        scope: str = Form(None),
        auth: SDKAuthContext = Depends(get_sdk_auth),
    ):
        return {"ok": True}

    c = TestClient(app)
    assert c.post("/upload", data={"user_id": "u"}).status_code == 200
    r = c.post("/upload", data={"user_id": "u", "scope": '{"customer":"acme"}'})
    assert r.status_code == 400


def test_scope_in_the_query_string_is_refused(monkeypatch):
    c = TestClient(_core_app(monkeypatch))
    assert c.get("/api/v1/auth/whoami?scope=%7B%22a%22%3A%22b%22%7D").status_code == 400
    assert c.get("/api/v1/auth/whoami?scope_path=x").status_code == 400
    assert c.get("/api/v1/auth/whoami?scope=").status_code == 200


# ---------------------------------------------------------------- whoami

@pytest.mark.parametrize("mode", ["equals_customer", "strict", None])
def test_whoami_answers_from_fixed_identity_and_mode(monkeypatch, mode):
    c = TestClient(_core_app(monkeypatch, mode=mode))
    r = c.get("/api/v1/auth/whoami")
    assert r.status_code == 200
    assert r.json() == {
        "client_id": core_identity.DEFAULT_CLIENT_ID,
        "instance_id": core_identity.DEFAULT_INSTANCE_ID,
        "credential_id": str(core_identity.DEFAULT_CREDENTIAL_ID),
        "user_context_isolation": mode,
    }


def test_whoami_has_the_same_fields_as_clouds():
    keys = pytest.importorskip("synap.cloud.application.admin_api.routes.keys")
    Cloud = keys.WhoamiResponse

    assert set(core_whoami.WhoamiResponse.model_fields) == set(Cloud.model_fields)


async def test_load_scoping_mode_is_none_when_unreadable(monkeypatch):
    import synap.cloud.shared.b2c_contract as b2c

    async def boom(_):
        raise RuntimeError("no db")

    monkeypatch.setattr(b2c, "load_isolation", boom)
    assert await core_identity.load_scoping_mode() is None


# ---------------------------------------------------------------- the three routes

@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/v1/credits/balance"),
        ("get", "/v1/credits/ledger"),
        ("post", "/v1/credits/estimate"),
        ("post", "/v1/credits/redeem"),
    ],
)
def test_credits_answers_not_available(monkeypatch, method, path):
    r = getattr(TestClient(_core_app(monkeypatch)), method)(path)
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "not_available_in_open_source"
    assert "not available in the open-source server" in detail["message"]


@pytest.mark.parametrize(
    "method,path,words",
    [
        ("post", "/v1/data/delete", "not available in the open-source server"),
        ("get", "/v1/data/delete/del_1", "not available in the open-source server"),
        ("get", "/v1/migrations", "not available yet"),
        ("post", "/v1/migrations/anything/at/all", "not available yet"),
        ("delete", "/v1/migrations/m1", "not available yet"),
    ],
)
def test_delete_by_scope_and_vendor_import_answer_not_available(monkeypatch, method, path, words):
    r = getattr(TestClient(_core_app(monkeypatch)), method)(path)
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "not_available_in_open_source"
    assert words in detail["message"]


@pytest.mark.parametrize("method", ["get", "post", "put", "patch", "delete"])
@pytest.mark.parametrize(
    "path",
    [
        "/v1/credits", "/v1/credits/balance", "/v1/credits/other", "/v1/credits/a/b/c",
        "/v1/events", "/v1/events/batch", "/v1/events/stream", "/v1/events/a/b",
        "/v1/telemetry", "/v1/telemetry/batch", "/v1/telemetry/status", "/v1/telemetry/a/b",
    ],
)
def test_every_method_on_every_cloud_only_path_gets_the_clear_message(monkeypatch, method, path):
    """Except the two routes that accept and discard (POST events/batch, POST telemetry/batch) and
    the one credits GET that already answers, which are checked on their own."""
    if (method, path) in {("post", "/v1/events/batch"), ("post", "/v1/telemetry/batch")}:
        pytest.skip("accepted and discarded; covered by their own tests")
    r = getattr(TestClient(_core_app(monkeypatch)), method)(path)
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "not_available_in_open_source"
    assert "not available in the open-source server" in detail["message"]


def test_the_credits_catch_all_names_credits(monkeypatch):
    r = TestClient(_core_app(monkeypatch)).post("/v1/credits/anything")
    assert r.json()["detail"]["message"] == core_routes.CREDITS_MESSAGE


def test_delete_by_scope_is_not_available_even_though_it_has_a_scope_argument(monkeypatch):
    """data.delete(scope="user", ...) sends a `scope` that is the delete scope, not a scope path (found at the final test run)."""
    c = TestClient(_core_app(monkeypatch))
    body = {"scope": "user", "user_id": "u1", "confirm": "u1", "dry_run": True}
    r = c.post("/v1/data/delete", json=body)
    assert r.status_code == 404
    assert "not available in the open-source server" in r.json()["detail"]["message"]
    for query in ("scope=user", "scope_path=x"):  # the exemption is for the body of /v1/data only
        assert c.get(f"/api/v1/auth/whoami?{query}").status_code == 400


def test_events_are_accepted_and_discarded(monkeypatch):
    c = TestClient(_core_app(monkeypatch))
    r = c.post("/v1/events/batch", json={"events": [
        {"event_type": "user_message", "user_id": "u", "event_id": "e1", "extra": 1},
        {"event_type": "assistant_message"},
    ]})
    assert r.status_code == 200
    body = r.json()
    assert (body["accepted"], body["duplicate"], body["rejected"]) == (2, 0, 0)
    assert [x["event_id"] for x in body["results"]] == ["e1", ""]
    assert all(x["status"] == "accepted" for x in body["results"])


def test_events_over_the_cap_are_refused_like_cloud(monkeypatch):
    c = TestClient(_core_app(monkeypatch))
    r = c.post("/v1/events/batch", json={"events": [{}] * (core_routes.MAX_EVENTS_PER_BATCH + 1)})
    assert r.status_code == 422


def test_telemetry_is_accepted_and_discarded(monkeypatch):
    c = TestClient(_core_app(monkeypatch))
    r = c.post("/v1/telemetry/batch", json={
        "events": [{"event_type": "x", "anything": "goes"}] * 3,
        "sdk_version": "0.6.0", "batch_id": "b1", "created_at": "2026-10-30T00:00:00Z",
    })
    assert r.status_code == 202
    assert r.json() == {"status": "accepted", "batch_id": "b1", "events_received": 3}


def test_core_routes_have_the_same_paths_as_clouds():
    probe = FastAPI()
    probe.include_router(core_routes.router)
    core_paths = {
        (m.upper(), path)
        for path, ops in probe.openapi()["paths"].items()
        for m in ops
    }
    expected = {
        ("GET", "/v1/credits/balance"), ("GET", "/v1/credits/ledger"),
        ("POST", "/v1/credits/estimate"), ("POST", "/v1/credits/redeem"),
        ("POST", "/v1/events/batch"), ("POST", "/v1/telemetry/batch"),
        ("POST", "/v1/data/delete"), ("GET", "/v1/data/delete/{deletion_id}"),
    }  # the /v1/migrations catch-all is left out of the schema on purpose
    assert core_paths == expected



def test_core_routes_match_clouds_source_when_it_is_present():
    """Compares with the Cloud files only where they exist (they do not ship)."""
    from pathlib import Path

    base = Path(core_routes.__file__).parent
    if not all((base / f).exists() for f in ("sdk_credits.py", "sdk_events.py", "sdk_telemetry.py")):
        pytest.skip("the Cloud route files are not in this tree")
    for fname, snippets in {
        "sdk_credits.py": ['"/balance"', '"/ledger"', '"/estimate"', '"/redeem"', 'prefix="/v1/credits"'],
        "sdk_events.py": ['"/batch"', 'prefix="/v1/events"'],
        "sdk_telemetry.py": ['"/batch"', 'prefix="/v1/telemetry"'],
    }.items():
        text = (base / fname).read_text()
        for s in snippets:
            assert s in text, (fname, s)


# ---------------------------------------------------------------- seeding (fakes)

class _Nested:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, has_memories=False):
        self.clients, self.instances = {}, {}
        self.has_memories = has_memories

    def begin_nested(self):
        return _Nested()

    async def execute(self, stmt, params=None):
        assert "FROM memories" in str(stmt)
        return SimpleNamespace(first=lambda: (1,) if self.has_memories else None)

    async def flush(self):
        pass


@pytest.fixture
def fakes(monkeypatch):
    import synap.cloud.database.repositories.client_repo as cr
    import synap.cloud.database.repositories.instance_repo as ir

    calls = SimpleNamespace(create_client=0, create_instance=0, instance_kwargs=None)

    class FakeClients:
        def __init__(self, s):
            self.s = s

        async def get_by_client_id(self, cid):
            return self.s.clients.get(cid)

        async def create_client(self, **kw):
            calls.create_client += 1
            self.s.clients[kw["client_id"]] = SimpleNamespace(**kw)

    class FakeInstances:
        def __init__(self, s):
            self.s = s

        async def get_by_instance_id(self, iid):
            return self.s.instances.get(iid)

        async def create_instance(self, **kw):
            calls.create_instance += 1
            calls.instance_kwargs = kw
            assert kw["client_id"] in self.s.clients, "foreign key"
            self.s.instances[kw["instance_id"]] = SimpleNamespace(**kw)

    monkeypatch.setattr(cr, "ClientRepository", FakeClients)
    monkeypatch.setattr(ir, "InstanceRepository", FakeInstances)
    return calls


async def test_seed_creates_both_once(fakes):
    s = _FakeSession()
    first = await core_seed.seed_default_identity(s, isolation="equals_customer")
    second = await core_seed.seed_default_identity(s, isolation="equals_customer")
    assert (first.client_created, first.instance_created) == (True, True)
    assert (second.client_created, second.instance_created) == (False, False)
    assert (fakes.create_client, fakes.create_instance) == (1, 1)
    kw = fakes.instance_kwargs
    assert kw["instance_id"] == core_identity.DEFAULT_INSTANCE_ID
    assert kw["client_id"] == core_identity.DEFAULT_CLIENT_ID
    assert kw["instance_metadata"]["user_relationship"] == "b2c"
    assert "enforce_limit" not in kw  # the instance cap check stays on


async def test_seed_strict_maps_to_b2b(fakes):
    await core_seed.seed_default_identity(_FakeSession(), isolation="strict")
    assert fakes.instance_kwargs["instance_metadata"]["user_relationship"] == "b2b"


async def test_seed_refuses_maca_suggests(fakes):
    with pytest.raises(ValueError, match="strict"):
        await core_seed.seed_default_identity(_FakeSession(), isolation="maca_suggests")
    assert fakes.create_client == 0


async def test_the_mode_cannot_change_once_memories_exist(fakes):
    s = _FakeSession()
    await core_seed.seed_default_identity(s, isolation="strict")
    s.has_memories = True
    with pytest.raises(core_seed.ScopingModeChanged, match="not supported") as err:
        await core_seed.seed_default_identity(s, isolation="equals_customer")
    assert "'strict'" in str(err.value) and "down -v" in str(err.value)
    assert s.instances[core_identity.DEFAULT_INSTANCE_ID].instance_metadata["user_relationship"] == "b2b"


async def test_the_mode_can_still_be_chosen_while_nothing_is_stored(fakes, caplog):
    s = _FakeSession()
    await core_seed.seed_default_identity(s, isolation="equals_customer")
    with caplog.at_level("INFO", logger=core_seed.logger.name):
        r = await core_seed.seed_default_identity(s, isolation="strict")
    assert r.instance_created is False
    assert s.instances[core_identity.DEFAULT_INSTANCE_ID].instance_metadata["user_relationship"] == "b2b"
    assert "before any memory was stored" in caplog.text


async def test_the_same_mode_with_memories_is_fine(fakes):
    s = _FakeSession(has_memories=True)
    await core_seed.seed_default_identity(s, isolation="strict")
    await core_seed.seed_default_identity(s, isolation="strict")


async def test_seeding_pins_the_mode_so_a_live_edit_is_ignored(fakes, tmp_path, caplog):
    from synap.cloud.memory_config import file_loader

    await core_seed.seed_default_identity(_FakeSession(), isolation="equals_customer")
    path = tmp_path / "memory.yml"
    path.write_text("storage_architecture:\n  scoping:\n    user_context_isolation: equals_customer\n")
    loader = file_loader.FileConfigLoader(str(path))
    loader.refresh(strict=True)
    path.write_text("storage_architecture:\n  scoping:\n    user_context_isolation: strict\n")
    import os
    os.utime(path, (path.stat().st_atime + 5, path.stat().st_mtime + 5))
    with caplog.at_level("ERROR"):
        loader.refresh()
    raw = await loader.get_raw_config()
    assert raw["storage_architecture"]["scoping"]["user_context_isolation"] == "equals_customer"
    assert "not supported" in caplog.text and "equals_customer" in caplog.text
    file_loader.pin_scoping_mode(None)


async def test_seed_lets_a_cap_refusal_through(fakes, monkeypatch):
    import synap.cloud.database.repositories.instance_repo as ir

    class Full(ir.InstanceRepository):
        async def get_by_instance_id(self, iid):
            return None

        async def create_instance(self, **kw):
            raise RuntimeError("instance_limit_reached")

    monkeypatch.setattr(ir, "InstanceRepository", Full)
    with pytest.raises(RuntimeError, match="instance_limit_reached"):
        await core_seed.seed_default_identity(_FakeSession())


async def test_seed_on_startup_is_a_noop_in_cloud(monkeypatch):
    monkeypatch.setattr(edition, "identity", "key")
    assert await core_seed.seed_on_startup() is None


# ---------------------------------------------------------------- imports in a subprocess

_BLOCKED = [
    "synap.cloud.auth",
    "synap.cloud.database.repositories.credential_repo",
    "synap.cloud.observability",
    "synap.cloud.privacy",
    "synap.cloud.application.credits",
    "synap.cloud.application.grpc",
    "synap.cloud.application.analytics",
    "synap.cloud.application.lifecycle",
]

# No stand-ins: a blocked name raises ImportError, nothing else is replaced.
_BLOCKER = textwrap.dedent(
    """
    import sys, importlib.abc

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            for p in {blocked!r}:
                if name == p or name.startswith(p + "."):
                    raise ImportError("blocked: " + name)

    sys.meta_path.insert(0, Block())
    import {module}
    bad = [m for m in sys.modules if any(m == p or m.startswith(p + ".") for p in {blocked!r})]
    assert not bad, bad
    print("ok")
    """
)


def _import_with_blocked(module: str, blocked: list[str]) -> subprocess.CompletedProcess:
    code = _BLOCKER.format(blocked=blocked, module=module)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180,
    )


def test_sdk_auth_has_no_load_time_import_of_the_privacy_folder():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(core_identity.__file__).with_name("sdk_auth.py").read_text())
    top_level = [
        n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    for node in top_level:
        names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
        assert not any(n.startswith("synap.cloud.privacy") for n in names), names


@pytest.mark.parametrize(
    "module",
    [
        "synap.cloud.application.sdk_api.sdk_auth",
        "synap.cloud.application.sdk_api.core_routes",
        "synap.cloud.application.sdk_api.core_whoami",
        "synap.cloud.application.sdk_api.core_identity",
        "synap.cloud.application.sdk_api.core_seed",
    ],
)
def test_modules_import_with_every_cut_folder_blocked(module):
    r = _import_with_blocked(module, _BLOCKED)
    assert r.returncode == 0, r.stderr[-2500:]
    assert "ok" in r.stdout


# --- A config that leaves the mode out runs under the default, and that is pinned too ---

async def test_a_config_without_the_mode_pins_the_default_so_a_live_edit_is_ignored(fakes, tmp_path, caplog):
    from synap.cloud.memory_config import file_loader

    file_loader.pin_scoping_mode(None)
    await core_seed.seed_default_identity(_FakeSession())
    assert file_loader.pinned_scoping_mode() == "strict"
    path = tmp_path / "memory.yml"
    path.write_text("metadata: {}\n")
    loader = file_loader.FileConfigLoader(str(path))
    loader.refresh(strict=True)
    path.write_text("storage_architecture:\n  scoping:\n    user_context_isolation: equals_customer\n")
    import os
    os.utime(path, (path.stat().st_atime + 5, path.stat().st_mtime + 5))
    with caplog.at_level("ERROR"):
        loader.refresh()
    raw = await loader.get_raw_config()
    assert "storage_architecture" not in raw
    assert "not supported" in caplog.text
    file_loader.pin_scoping_mode(None)


async def test_taking_the_mode_out_of_a_pinned_config_is_ignored(fakes, tmp_path):
    from synap.cloud.memory_config import file_loader

    await core_seed.seed_default_identity(_FakeSession(), isolation="equals_customer")
    path = tmp_path / "memory.yml"
    path.write_text("storage_architecture:\n  scoping:\n    user_context_isolation: equals_customer\n")
    loader = file_loader.FileConfigLoader(str(path))
    loader.refresh(strict=True)
    path.write_text("metadata: {}\n")
    import os
    os.utime(path, (path.stat().st_atime + 5, path.stat().st_mtime + 5))
    loader.refresh()
    raw = await loader.get_raw_config()
    assert raw["storage_architecture"]["scoping"]["user_context_isolation"] == "equals_customer"
    file_loader.pin_scoping_mode(None)


async def test_a_config_without_the_mode_blocks_a_later_change_once_memories_exist(fakes):
    s = _FakeSession()
    await core_seed.seed_default_identity(s)
    assert s.instances[core_identity.DEFAULT_INSTANCE_ID].instance_metadata["user_relationship"] == "b2b"
    s.has_memories = True
    with pytest.raises(core_seed.ScopingModeChanged, match="not supported"):
        await core_seed.seed_default_identity(s, isolation="equals_customer")


async def test_an_instance_seeded_without_a_recorded_mode_is_held_to_the_default(fakes):
    s = _FakeSession(has_memories=True)
    await core_seed.seed_default_identity(s)
    s.instances[core_identity.DEFAULT_INSTANCE_ID].instance_metadata.pop("user_relationship")
    with pytest.raises(core_seed.ScopingModeChanged):
        await core_seed.seed_default_identity(s, isolation="equals_customer")
    await core_seed.seed_default_identity(s, isolation="strict")


async def test_a_config_without_the_mode_while_nothing_is_stored_can_still_choose_later(fakes):
    s = _FakeSession()
    await core_seed.seed_default_identity(s)
    await core_seed.seed_default_identity(s, isolation="equals_customer")
    assert s.instances[core_identity.DEFAULT_INSTANCE_ID].instance_metadata["user_relationship"] == "b2c"
