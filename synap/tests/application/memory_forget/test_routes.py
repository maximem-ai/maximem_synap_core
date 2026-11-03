"""The two route modules, loaded without the rest of the server.

Importing anything under ``admin_api.routes`` runs the package ``__init__``, which
imports every router and with them stripe, torch, neo4j and the MACA stack. These
tests only need the two new modules, so the package and the helpers they import are
replaced by small stand-ins for the duration of each test and the real files are
loaded on top. What is under test is the route code: mode dispatch, the refusals,
and the status codes.
"""

import importlib.util
import sys
import types
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from synap.cloud.application.memory_forget import service
from synap.cloud.application.memory_forget.judge import JudgeNotConfigured, JudgeUnavailable
from synap.cloud.application.memory_forget.models import Candidate, ForgetOutcome
from synap.cloud.shared import edition

PKG = "synap.cloud.application.admin_api.routes"
ROUTES_DIR = (Path(__file__).resolve().parents[3]
              / "cloud" / "application" / "admin_api" / "routes")
# The SDK routes moved out of admin_api.
SDK_PKG = "synap.cloud.application.sdk_api"
SDK_DIR = (Path(__file__).resolve().parents[3] / "cloud" / "application" / "sdk_api")

CLIENT_UUID = str(uuid.uuid4())


@dataclass
class _Auth:
    client_id: str = "cli_test"
    instance_id: str = "inst_test"
    credential_id: uuid.UUID = uuid.UUID(int=7)
    correlation_id: str = "corr-1"


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


class _Scope:
    def __init__(self, level="user", user=True):
        self.scope_level = level
        self.user_id = uuid.uuid4() if user else None
        self.customer_id = uuid.uuid4()
        self.client_id = uuid.uuid4()
        self.instance_id = uuid.uuid4()


def _load(monkeypatch, name, stubs, pkg_name=PKG, pkg_dir=ROUTES_DIR):
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(pkg_dir)]
    monkeypatch.setitem(sys.modules, pkg_name, pkg)
    for modname, mod in stubs.items():
        monkeypatch.setitem(sys.modules, modname, mod)
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", pkg_dir / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, f"{pkg_name}.{name}", mod)
    spec.loader.exec_module(mod)
    return mod


async def _fake_session():
    yield object()


# ------------------------------------------------------------------ SDK route

@pytest.fixture
def sdk(monkeypatch):
    state = {"scope": _Scope(), "resolve_error": None, "b2c": []}

    async def get_sdk_auth(authorization: str = Header("Bearer synap_x")):
        return _Auth()

    async def b2c_check(auth, customer_id, *, where, user_id=None):
        state["b2c"].append(where)

    async def resolve_scope(auth, user_id, customer_id, scope_path=None):
        if state["resolve_error"]:
            raise ValueError(state["resolve_error"])
        return state["scope"]

    mod = _load(monkeypatch, "sdk_forget", {
        f"{SDK_PKG}.sdk_auth": _stub(f"{SDK_PKG}.sdk_auth", SDKAuthContext=_Auth,
                                     get_sdk_auth=get_sdk_auth),
        f"{SDK_PKG}.sdk_memories": _stub(f"{SDK_PKG}.sdk_memories", _b2c_check=b2c_check,
                                         _resolve_scope=resolve_scope),
    }, pkg_name=SDK_PKG, pkg_dir=SDK_DIR)
    calls = {}
    monkeypatch.setattr(service, "build_deps", lambda session, **kw: object())
    # These tests name Cloud's default (preview); the Core tests below set their own.
    monkeypatch.setattr(edition, "forget_default_mode", "preview")

    app = FastAPI()
    app.include_router(mod.router)
    from synap.cloud.database.connection import get_async_session
    app.dependency_overrides[get_async_session] = _fake_session
    return types.SimpleNamespace(client=TestClient(app), state=state, calls=calls,
                                 monkeypatch=monkeypatch)


def _outcome(**kw):
    return ForgetOutcome(**{"status": "preview", **kw})


def test_preview_is_the_default_mode_and_returns_the_review_list(sdk):
    seen = {}

    async def preview(session, *, scope, description, deps, requested_by):
        seen.update(description=description, requested_by=requested_by)
        return _outcome(request_id="req-1",
                        candidates=[Candidate("m1", "number is 512-555-0142", "fact"),
                                    Candidate("c1", "I said it is 512-555-0142",
                                              "source text", "chunk")])

    sdk.monkeypatch.setattr(service, "preview", preview)
    r = sdk.client.post("/api/v1/memories/forget",
                        json={"description": "my phone number", "user_id": "u1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "preview" and body["request_id"] == "req-1"
    # Each item says where it lives, so a person reviewing sees what goes and where.
    assert body["candidates"] == [
        {"memory_id": "m1", "content": "number is 512-555-0142",
         "memory_type": "fact", "store": "memory"},
        {"memory_id": "c1", "content": "I said it is 512-555-0142",
         "memory_type": "source text", "store": "chunk"}]
    assert body["deleted_ids"] == []
    assert seen["description"] == "my phone number"
    assert seen["requested_by"].startswith("sdk:")


def _record_modes(sdk):
    used = []

    async def preview(*a, **k):
        used.append("preview")
        return _outcome(request_id="req-1")

    async def direct(*a, **k):
        used.append("direct")
        return _outcome(status="deleted", deleted_ids=["m1"])

    sdk.monkeypatch.setattr(service, "preview", preview)
    sdk.monkeypatch.setattr(service, "direct", direct)
    return used


def test_core_deletes_when_the_request_names_no_mode(sdk):
    """Core has no dashboard to approve a preview, so an unnamed mode must act."""
    sdk.monkeypatch.setattr(edition, "forget_default_mode", "direct")  # Core
    used = _record_modes(sdk)
    r = sdk.client.post("/api/v1/memories/forget",
                        json={"description": "my phone number", "user_id": "u1"})
    assert r.status_code == 200 and r.json()["status"] == "deleted"
    assert used == ["direct"]


def test_core_still_honours_an_explicit_preview(sdk):
    sdk.monkeypatch.setattr(edition, "forget_default_mode", "direct")  # Core
    used = _record_modes(sdk)
    r = sdk.client.post("/api/v1/memories/forget",
                        json={"description": "my phone number", "user_id": "u1", "mode": "preview"})
    assert r.status_code == 200 and r.json()["status"] == "preview"
    assert used == ["preview"]


def test_cloud_previews_when_the_request_names_no_mode_and_honours_direct(sdk):
    sdk.monkeypatch.setattr(edition, "forget_default_mode", "preview")  # Cloud
    used = _record_modes(sdk)
    for body in ({}, {"mode": "direct"}):
        r = sdk.client.post("/api/v1/memories/forget",
                            json={"description": "my phone number", "user_id": "u1", **body})
        assert r.status_code == 200
    assert used == ["preview", "direct"]


def test_edition_default_modes():
    from synap.cloud.shared.edition import _CLOUD, _CORE
    assert (_CLOUD.forget_default_mode, _CORE.forget_default_mode) == ("preview", "direct")


def test_direct_mode_deletes_in_one_call(sdk):
    seen = {}

    async def direct(session, *, scope, description, deps, requested_by=None,
                     conversation_id=None, user_id=None, customer_id=None):
        seen["requested_by"] = requested_by
        seen["conversation_id"] = conversation_id
        seen["ids"] = (user_id, customer_id)
        return _outcome(status="deleted", deleted_ids=["m1"], request_id="r1")

    async def preview(*a, **k):
        raise AssertionError("direct mode must not file a preview")

    sdk.monkeypatch.setattr(service, "direct", direct)
    sdk.monkeypatch.setattr(service, "preview", preview)
    r = sdk.client.post("/api/v1/memories/forget",
                        json={"description": "my phone number", "user_id": "u1",
                              "mode": "direct"})
    assert r.status_code == 200 and r.json()["status"] == "deleted"
    assert r.json()["deleted_ids"] == ["m1"]
    assert r.json()["request_id"] == "r1"
    assert seen["requested_by"].startswith("sdk:")
    assert seen["conversation_id"] is None          # a caller that gives none is unchanged
    # the external ids the caller used are handed on, so the server can find its conversations
    assert seen["ids"] == ("u1", None)


def test_direct_mode_passes_the_conversation_id_in_its_normalised_form(sdk):
    from synap.cloud.shared.conversation_id_norm import coerce_conversation_id
    seen = {}

    async def direct(session, *, scope, description, deps, requested_by=None,
                     conversation_id=None, user_id=None, customer_id=None):
        seen["conversation_id"] = conversation_id
        return _outcome(status="nothing_matches")

    sdk.monkeypatch.setattr(service, "direct", direct)
    r = sdk.client.post("/api/v1/memories/forget",
                        json={"description": "my phone number", "user_id": "u1",
                              "mode": "direct", "conversation_id": "pgv2_abc123"})
    assert r.status_code == 200
    # the same key the message table uses, so the request position can be looked up
    assert seen["conversation_id"] == coerce_conversation_id("pgv2_abc123")


def test_nothing_matching_is_a_normal_answer(sdk):
    async def preview(*a, **k):
        return _outcome(status="nothing_matches")

    sdk.monkeypatch.setattr(service, "preview", preview)
    r = sdk.client.post("/api/v1/memories/forget",
                        json={"description": "my pet snake", "user_id": "u1"})
    assert r.status_code == 200 and r.json()["status"] == "nothing_matches"
    assert r.json()["candidates"] == []


def test_a_request_that_is_not_one_user_is_refused(sdk):
    sdk.state["scope"] = _Scope(level="customer", user=False)
    r = sdk.client.post("/api/v1/memories/forget", json={"description": "my number"})
    assert r.status_code == 400 and "one user" in r.json()["detail"]


def test_a_scope_error_is_a_400(sdk):
    sdk.state["resolve_error"] = "user_id needs a customer_id"
    r = sdk.client.post("/api/v1/memories/forget",
                        json={"description": "my number", "user_id": "u1"})
    assert r.status_code == 400 and "customer_id" in r.json()["detail"]


def test_an_unavailable_judge_is_a_503_that_says_nothing_was_deleted(sdk):
    async def preview(*a, **k):
        raise JudgeUnavailable("down")

    async def direct(*a, **k):
        raise JudgeUnavailable("down")

    sdk.monkeypatch.setattr(service, "preview", preview)
    sdk.monkeypatch.setattr(service, "direct", direct)
    for mode in ("preview", "direct"):
        r = sdk.client.post("/api/v1/memories/forget",
                            json={"description": "my number", "user_id": "u1", "mode": mode})
        assert r.status_code == 503 and "nothing was deleted" in r.json()["detail"]


def test_without_a_jev_key_the_424_says_forget_needs_one(sdk, caplog):
    async def no_key(*a, **k):
        raise JudgeNotConfigured("JEV_API_KEY is not set")

    sdk.monkeypatch.setattr(service, "preview", no_key)
    sdk.monkeypatch.setattr(service, "direct", no_key)
    for mode in ("preview", "direct"):
        r = sdk.client.post("/api/v1/memories/forget",
                            json={"description": "my number", "user_id": "u1", "mode": mode})
        detail = r.json()["detail"]
        # A 4xx: the Python SDK retries 5xx, and a missing key does not clear by retrying.
        assert r.status_code == 424
        assert "needs a Jev API key" in detail and "JEV_API_KEY" in detail and "by its id" in detail
    # and the server's own log says why, at a level that is shown by default
    assert any("JEV_API_KEY is not set on the server" in r.getMessage() and r.levelname == "WARNING"
               for r in caplog.records)


def test_a_partial_delete_names_the_store_that_still_holds_the_text(sdk):
    async def direct(*a, **k):
        return _outcome(status="partial", deleted_ids=["m1"], still_present_in=["cache"])

    sdk.monkeypatch.setattr(service, "direct", direct)
    r = sdk.client.post("/api/v1/memories/forget",
                        json={"description": "my number", "user_id": "u1", "mode": "direct"})
    assert r.json()["status"] == "partial" and r.json()["still_present_in"] == ["cache"]


def test_bad_input_is_a_422(sdk):
    assert sdk.client.post("/api/v1/memories/forget",
                           json={"description": "no", "user_id": "u1"}).status_code == 422
    assert sdk.client.post("/api/v1/memories/forget",
                           json={"description": "my number", "mode": "wipe"}).status_code == 422


def test_the_b2c_contract_check_runs(sdk):
    async def preview(*a, **k):
        return _outcome(status="nothing_matches")

    sdk.monkeypatch.setattr(service, "preview", preview)
    sdk.client.post("/api/v1/memories/forget", json={"description": "my number", "user_id": "u1"})
    assert sdk.state["b2c"] == ["POST /api/v1/memories/forget"]


# ------------------------------------------------------------ dashboard route

@dataclass
class _User:
    id: str = "u-1"
    email: str = "dana@acme.test"
    client_memberships: tuple = ("cli_ok",)


@pytest.fixture
def dash(monkeypatch):
    async def current_user():
        return _User()

    def get_client_context(user, client_id):
        if client_id not in user.client_memberships:
            raise HTTPException(status_code=403, detail="Not a member of this client")

    async def privacy_client_uuid(client_id):
        return CLIENT_UUID

    mod = _load(monkeypatch, "dashboard_deletion_requests", {
        f"{PKG}._helpers": _stub(f"{PKG}._helpers", get_client_context=get_client_context,
                                 privacy_client_uuid=privacy_client_uuid),
        f"{PKG}.auth": _stub(f"{PKG}.auth", get_current_dashboard_user=current_user),
    })
    app = FastAPI()
    app.include_router(mod.router)
    from synap.cloud.database.connection import get_async_session
    app.dependency_overrides[get_async_session] = _fake_session
    return types.SimpleNamespace(client=TestClient(app), mod=mod, monkeypatch=monkeypatch)


def _req(**kw):
    base = {"id": "r1", "client_id": CLIENT_UUID, "query": "my phone number",
            "status": "pending", "scope_level": "user", "customer_id": "c1", "user_id": "u1",
            "candidates": [{"memory_id": "m1", "content": "number is 512-555-0142",
                            "memory_type": "fact"}],
            "requested_by": "sdk:k", "decided_by": None, "decided_at": None,
            "result": None, "created_at": "2026-10-05T10:00:00Z"}
    return {**base, **kw}


def test_the_queue_lists_requests_for_a_member(dash):
    seen = {}

    async def list_requests(session, *, client_id, status, limit):
        seen.update(client_id=client_id, status=status)
        return [_req()]

    dash.monkeypatch.setattr(dash.mod.store, "list_requests", list_requests)
    r = dash.client.get("/dashboard/deletion-requests", params={"client_id": "cli_ok",
                                                                "status": "pending"})
    assert r.status_code == 200
    item = r.json()["requests"][0]
    assert item["candidates"][0]["content"] == "number is 512-555-0142"
    assert seen == {"client_id": CLIENT_UUID, "status": "pending"}     # the UUID, not cli_...


def test_a_non_member_gets_403_on_every_route(dash):
    for method, path in (("get", "/dashboard/deletion-requests"),
                         ("get", "/dashboard/deletion-requests/r1"),
                         ("post", "/dashboard/deletion-requests/r1/approve"),
                         ("post", "/dashboard/deletion-requests/r1/reject")):
        r = getattr(dash.client, method)(path, params={"client_id": "cli_other"})
        assert r.status_code == 403, path


def test_an_unknown_status_filter_is_a_400(dash):
    r = dash.client.get("/dashboard/deletion-requests",
                        params={"client_id": "cli_ok", "status": "bogus"})
    assert r.status_code == 400


def test_a_missing_request_is_a_404(dash):
    async def get_request(session, *, client_id, request_id):
        return None

    dash.monkeypatch.setattr(dash.mod.store, "get_request", get_request)
    r = dash.client.get("/dashboard/deletion-requests/nope", params={"client_id": "cli_ok"})
    assert r.status_code == 404


def test_approve_runs_the_delete_and_reports_what_happened(dash):
    seen = {}

    async def approve(session, *, client_id, request_id, decided_by, scope_of, deps_for):
        seen.update(client_id=client_id, request_id=request_id, decided_by=decided_by)
        return ForgetOutcome(status="deleted", deleted_ids=["m1"])

    dash.monkeypatch.setattr(service, "approve", approve)
    r = dash.client.post("/dashboard/deletion-requests/r1/approve",
                         params={"client_id": "cli_ok"})
    assert r.status_code == 200
    assert r.json() == {"status": "deleted", "deleted_ids": ["m1"], "failed_ids": [],
                        "still_present_in": []}
    assert seen == {"client_id": CLIENT_UUID, "request_id": "r1",
                    "decided_by": "dana@acme.test"}


def test_approving_a_decided_or_missing_request_is_a_409(dash):
    async def approve(*a, **k):
        return None

    dash.monkeypatch.setattr(service, "approve", approve)
    r = dash.client.post("/dashboard/deletion-requests/r1/approve",
                         params={"client_id": "cli_ok"})
    assert r.status_code == 409


def test_reject_and_a_second_reject(dash):
    results = iter([True, False])

    async def reject(session, *, client_id, request_id, decided_by):
        return next(results)

    dash.monkeypatch.setattr(service, "reject", reject)
    first = dash.client.post("/dashboard/deletion-requests/r1/reject",
                             params={"client_id": "cli_ok"})
    second = dash.client.post("/dashboard/deletion-requests/r1/reject",
                              params={"client_id": "cli_ok"})
    assert (first.status_code, first.json()) == (200, {"status": "rejected"})
    assert second.status_code == 409
