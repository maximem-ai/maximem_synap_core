"""The Core memory browser: read-only, no login of its own, nothing Cloud only.

TestClient and a fake session, so no Postgres is needed. The queries themselves are
checked against a real database in the Core end-to-end run.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from synap.cloud.application.sdk_api import core_browser, core_identity
from synap.cloud.database.connection import get_async_session
from synap.cloud.shared import edition

NOW = datetime(2026, 11, 2, 9, 30, tzinfo=timezone.utc)
BROWSER_DIR = Path(core_browser.DEFAULT_BROWSER_DIR)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def one(self):
        assert len(self._rows) == 1
        return self._rows[0]

    def scalar_one(self):
        return self._rows[0]


class FakeSession:
    """Answers each query with the rows registered for the first matching fragment."""

    def __init__(self, answers):
        self.answers = answers
        self.seen = []

    async def execute(self, stmt, params=None):
        sql = " ".join(str(stmt).split())
        self.seen.append((sql, dict(params or {})))
        for fragment, rows in self.answers:
            if fragment in sql:
                return _Result(rows)
        raise AssertionError(f"no fake answer for: {sql}")


def _memory(**over):
    row = dict(
        id=uuid.uuid4(), memory_type="facts", category="work", subcategory=None, content="Ana likes tea",
        confidence=0.9, created_at=NOW, updated_at=None, deprecated_at=None, superseded_by=None,
        superseded_at=None, temporal_category="perpetual", valid_from=None, valid_until=None,
        document_created_at=None, source_document_id=None, is_episode=False,
        customer_ext="cust-1", user_ext=None, replacement_content=None,
    )
    row.update(over)
    return SimpleNamespace(**row)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(core_identity.API_TOKEN_ENV, raising=False)
    monkeypatch.delenv(core_browser.BROWSER_DIR_ENV, raising=False)

    async def _mode(instance_id=core_identity.DEFAULT_INSTANCE_ID):
        return "equals_customer"

    monkeypatch.setattr(core_browser, "load_scoping_mode", _mode)
    monkeypatch.setattr(core_identity, "load_scoping_mode", _mode)


def _client(answers=()):
    app = FastAPI()
    app.include_router(core_browser.router)
    session = FakeSession(list(answers))

    async def _session():
        yield session

    app.dependency_overrides[get_async_session] = _session
    assert core_identity.install_edition_identity(app) is True
    return TestClient(app), session


@pytest.fixture(autouse=True)
def _core_identity(monkeypatch):
    monkeypatch.setattr(edition, "identity", "fixed")


# ------------------------------------------------------------------ read-only

def test_every_route_is_a_get():
    methods = {m for r in core_browser.router.routes for m in r.methods}
    assert methods <= {"GET", "HEAD"}, methods


def test_every_route_sits_under_browser():
    assert all(r.path.startswith("/browser") for r in core_browser.router.routes)


def test_the_router_is_in_the_core_router_only():
    sdk_api = Path(core_browser.__file__).parent
    assert "sdk_router.include_router(core_browser_router)" in (sdk_api / "router.py").read_text()
    admin_api = sdk_api.parent / "admin_api"
    assert "core_browser" not in "".join(p.read_text() for p in admin_api.rglob("*.py")), \
        "Cloud's admin_api must not include the Core browser"


# ------------------------------------------------------------------ the page

def test_the_page_files_exist_and_are_served():
    client, _ = _client()
    for name, media in core_browser.PAGE_FILES.items():
        assert (BROWSER_DIR / name).is_file(), name
        r = client.get(f"/browser/{name}")
        assert r.status_code == 200, name
        assert r.headers["content-type"].startswith(media.split(";")[0])
    index = client.get("/browser/")
    assert index.status_code == 200 and "Synap Core" in index.text


def test_browser_without_a_slash_redirects():
    client, _ = _client()
    r = client.get("/browser", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/browser/"


@pytest.mark.parametrize("name", ["app.js.map", "..%2Fapp.js", "README.md", "%2e%2e%2f%2e%2e%2fmain.py", "api", "index.htm"])
def test_only_the_named_files_are_served(name):
    client, _ = _client()
    assert client.get(f"/browser/{name}").status_code == 404


def test_the_page_carries_a_strict_content_security_policy():
    client, _ = _client()
    csp = client.get("/browser/").headers["content-security-policy"]
    assert "default-src 'none'" in csp and "script-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert client.get("/browser/").headers["x-content-type-options"] == "nosniff"


def test_a_missing_page_folder_is_a_404_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setenv(core_browser.BROWSER_DIR_ENV, str(tmp_path))
    client, _ = _client()
    assert client.get("/browser/").status_code == 404


def test_the_script_never_writes_html_and_calls_no_outside_host():
    js = (BROWSER_DIR / "app.js").read_text()
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert banned not in js, banned
    for name in ("app.js", "index.html", "style.css"):
        text = (BROWSER_DIR / name).read_text()
        assert not re.search(r"https?://(?!www\.w3\.org)", text), f"{name} names an outside host"


def test_the_page_has_no_cloud_only_screens():
    """The sidebar names exactly the Core views, and no Cloud-only word appears. "credits" is allowed:
    the MACA steps on the Config page say the plugin needs a Synap Cloud account with credits."""
    html = (BROWSER_DIR / "index.html").read_text()
    assert re.findall(r'data-nav="(\w+)"', html) == ["overview", "memories", "requests", "entities", "config"]
    text = ((BROWSER_DIR / "app.js").read_text() + html).lower()
    for word in ("billing", "api key", "invoice", "auth0", "login", "instances", "usage", "balance"):
        assert word not in text, word


def test_the_page_sets_no_inline_style():
    """The Content-Security-Policy refuses style attributes, so one would silently do nothing."""
    js = (BROWSER_DIR / "app.js").read_text()
    assert not re.search(r"\bstyle\s*:", js) and 'setAttribute("style"' not in js
    assert " style=" not in (BROWSER_DIR / "index.html").read_text()


# ------------------------------------------------------------------ the token

def test_no_token_set_means_open(monkeypatch):
    client, _ = _client([("count(*) FROM memories m WHERE", [SimpleNamespace(active=2, retired=1, customers=3, users=0)])])
    assert client.get("/browser/api/overview").status_code == 200


def test_token_set_gates_the_api_but_not_the_page(monkeypatch):
    monkeypatch.setenv(core_identity.API_TOKEN_ENV, "s3cret-token")
    client, _ = _client([("count(*) FROM memories m WHERE", [SimpleNamespace(active=2, retired=1, customers=3, users=0)])])
    assert client.get("/browser/").status_code == 200
    assert client.get("/browser/api/overview").status_code == 401
    assert client.get("/browser/api/overview", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/browser/api/overview", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200


# ------------------------------------------------------------------ the API

def test_overview_in_equals_customer_mode_has_one_kind_of_person():
    client, _ = _client([("count(*) FROM memories m WHERE", [SimpleNamespace(active=5, retired=2, customers=4, users=0)])])
    body = client.get("/browser/api/overview").json()
    assert body["edition"] == edition.name
    assert body["scoping_mode"] == "equals_customer"
    assert body["entity_kinds"] == ["customer"]
    assert (body["memories_active"], body["memories_retired"], body["customers"]) == (5, 2, 4)
    assert client.get("/browser/api/overview").headers["cache-control"] == "no-store"


def test_memories_list_shape_and_default_status_is_active():
    mem = _memory()
    client, session = _client([
        ("GROUP BY m.memory_type", [SimpleNamespace(mtype="facts", n=1)]),
        ("SELECT count(*)", [1]),
        ("ORDER BY m.created_at DESC", [mem]),
    ])
    body = client.get("/browser/api/memories").json()
    assert body["total"] == 1 and body["total_pages"] == 1 and body["by_type"] == [{"memory_type": "facts", "count": 1}]
    item = body["items"][0]
    assert item["content"] == "Ana likes tea" and item["scope"] == "customer" and item["status"] == "active"
    assert item["customer_id"] == "cust-1"
    list_sql = next(sql for sql, _ in session.seen if "ORDER BY m.created_at DESC" in sql)
    assert "deprecated_at IS NULL" in list_sql and "deleted_at IS NULL" in list_sql


def test_memories_search_is_a_bound_parameter_with_wildcards_escaped():
    client, session = _client([
        ("GROUP BY m.memory_type", []), ("SELECT count(*)", [0]), ("ORDER BY m.created_at DESC", []),
    ])
    evil = "x'; DROP TABLE memories; --%_"
    assert client.get("/browser/api/memories", params={"search": evil, "entity": "c1", "type": "facts"}).status_code == 200
    for sql, params in session.seen:
        assert "DROP TABLE" not in sql
    _, params = session.seen[-1]
    assert params["search"] == "%x'; DROP TABLE memories; --!%!_%"
    assert params["entity"] == "c1" and params["type"] == "facts"


def test_forgotten_memories_are_never_listed_even_with_status_all():
    client, session = _client([("GROUP BY m.memory_type", []), ("SELECT count(*)", [0]), ("ORDER BY m.created_at DESC", [])])
    client.get("/browser/api/memories", params={"status": "all"})
    assert all("deleted_at IS NULL" in sql for sql, _ in session.seen)


@pytest.mark.parametrize("params", [{"page": 0}, {"page_size": 101}, {"status": "everything"}])
def test_memories_rejects_bad_paging_and_status(params):
    client, _ = _client()
    assert client.get("/browser/api/memories", params=params).status_code == 422


def test_memory_detail_not_found_and_bad_id():
    client, _ = _client([("WHERE m.id = :id", [])])
    assert client.get(f"/browser/api/memories/{uuid.uuid4()}").status_code == 404
    assert client.get("/browser/api/memories/not-a-uuid").status_code == 404


def test_retired_memory_detail_shows_what_replaced_it_and_its_history():
    new_id = uuid.uuid4()
    mem = _memory(superseded_by=new_id, superseded_at=NOW, replacement_content="Ana likes coffee")
    event = SimpleNamespace(action="SUPERSEDE", content="Ana likes coffee", previous_content="Ana likes tea",
                            replaced_by_memory_id=new_id, reason="newer fact", created_at=NOW)
    client, _ = _client([("WHERE m.id = :id", [mem]), ("FROM memory_actions", [event])])
    body = client.get(f"/browser/api/memories/{mem.id}").json()
    assert body["status"] == "retired" and body["superseded_by"] == str(new_id)
    assert body["replacement_content"] == "Ana likes coffee"
    assert body["history"][0]["action"] == "SUPERSEDE" and body["history"][0]["previous_content"] == "Ana likes tea"
    assert "pii" not in " ".join(body).lower()


def test_entities_list_counts_memories_per_person():
    internal = uuid.uuid4()
    row = SimpleNamespace(internal_id=internal, external_id="cust-1", name="", description="", created_at=NOW, parent=None)
    client, _ = _client([
        ("SELECT count(*) FROM customer", [1]),
        ("FROM customer", [row]),
        ("GROUP BY m.customer_id", [SimpleNamespace(owner=internal, active=7, retired=1)]),
    ])
    body = client.get("/browser/api/entities", params={"kind": "customer"}).json()
    assert body["total"] == 1
    assert body["items"][0]["external_id"] == "cust-1" and body["items"][0]["memories_active"] == 7
    assert body["items"][0]["name"] is None


def test_entity_detail_not_found():
    client, _ = _client([("FROM customer WHERE customer_id = :ext", [])])
    assert client.get("/browser/api/entities/customer/nobody").status_code == 404
    assert client.get("/browser/api/entities/team/x").status_code == 422


def test_config_returns_the_file_and_the_mode(monkeypatch, tmp_path):
    path = tmp_path / "memory.yml"
    path.write_text("metadata:\n  client_id: default\nuser_context_isolation: equals_customer\n")
    monkeypatch.setenv("SYNAP_MEMORY_CONFIG", str(path))
    from synap.cloud.memory_config import file_loader

    monkeypatch.setattr(file_loader, "_shared", file_loader.FileConfigLoader(str(path)))
    client, _ = _client()
    body = client.get("/browser/api/config").json()
    assert body["exists"] is True and "equals_customer" in body["text"]
    assert body["path"] == str(path) and "scoping_mode" not in body


def test_config_with_no_file_says_so(monkeypatch, tmp_path):
    from synap.cloud.memory_config import file_loader

    monkeypatch.setattr(file_loader, "_shared", file_loader.FileConfigLoader(str(tmp_path / "missing.yml")))
    client, _ = _client()
    body = client.get("/browser/api/config").json()
    assert body["exists"] is False and body["text"] is None


def test_like_escaping():
    assert core_browser._like("50%_off!") == "%50!%!_off!!%"


def test_no_sql_alias_collides_with_a_row_attribute():
    """SQLAlchemy's Row has .t, .count, .index and others; an alias of the same name returns the method."""
    source = Path(core_browser.__file__).read_text()
    aliases = set(re.findall(r"\bAS (\w+)", source))
    assert not aliases & {"t", "count", "index", "tuple", "keys", "values", "items"}, aliases


def test_owners_are_joined_by_uuid5_hash_not_the_internal_id():
    source = Path(core_browser.__file__).read_text()
    assert "c.uuid5_hash = m.customer_id" in source and "u.uuid5_hash = m.user_id" in source
    assert "internal_customer_id" not in source and "internal_user_id" not in source


def test_the_script_only_replaces_children_through_show():
    """The DOM's replaceChildren prints "null" for a missing node and "[object ...]" for a list."""
    js = (BROWSER_DIR / "app.js").read_text()
    assert js.count(".replaceChildren(") == 1, "use show(el, ...nodes), which drops nulls and flattens lists"


def test_the_element_helper_flattens_nested_lists():
    """A list of [dt, dd] pairs is a list of lists; one level of flattening prints "[object HTMLElement]"."""
    js = (BROWSER_DIR / "app.js").read_text()
    assert "kids.flat(Infinity)" in js


def test_the_page_keeps_the_client_dashboards_theme():
    """The browser looks like the client dashboard. These are its light and dark tokens
    from ui/shared/src/styles/globals.css; if the dashboard's theme changes, change them here too."""
    css = (BROWSER_DIR / "style.css").read_text()
    for token in ("--primary: #e05d38", "--background: #f5f4ee", "--background: #0d0d0d", "--main-container: #000", "'Akatab', system-ui"):
        assert token in css, token
    shared = BROWSER_DIR.parent / "shared" / "src" / "styles" / "globals.css"
    if shared.is_file():
        source = shared.read_text()
        for token in ("--primary: #e05d38", "--background: #f5f4ee", "--background: #0d0d0d"):
            assert token in source, f"the dashboard's theme no longer has {token}"


# ------------------------------------------------------------------ analytics, requests, config

def _analytics_answers(**over):
    day = datetime.now(timezone.utc).date()
    base = [
        ("AS requests", [SimpleNamespace(active=5, retired=2, requests=3)]),
        ("AS ext", [SimpleNamespace(ext="cust-1", kind="customer", n=4)]),
        ("AS scope", [SimpleNamespace(scope="customer", n=5)]),
        ("AS mtype", [SimpleNamespace(mtype="facts", n=5)]),
        ("a.action AS kind", [SimpleNamespace(kind="CREATE", n=7), SimpleNamespace(kind="ENRICH", n=2)]),
        ("count(DISTINCT a.correlation_id) AS n", [SimpleNamespace(day=day, n=3)]),
        ("AS day, count(*) AS n FROM memories", [SimpleNamespace(day=day, n=7)]),
    ]
    return base


def test_analytics_fills_every_day_and_adds_up():
    client, session = _client(_analytics_answers())
    body = client.get("/browser/api/analytics", params={"days": 7}).json()
    assert len(body["per_day"]) == 7 and body["days"] == 7
    days = [p["day"] for p in body["per_day"]]
    assert days == sorted(days) and len(set(days)) == 7
    assert body["per_day"][-1] == {"day": days[-1], "memories": 7, "requests": 3}
    assert sum(p["memories"] for p in body["per_day"]) == body["memories_created"] == 7
    assert body["outcomes"][0] == {"action": "CREATE", "count": 7}
    assert body["top_entities"] == [{"external_id": "cust-1", "kind": "customer", "memories_active": 4}]
    assert body["memories_active"] == 5 and body["requests"] == 3
    assert all("deleted_at IS NULL" in sql for sql, _ in session.seen if "FROM memories m" in sql and "created_at" in sql)


@pytest.mark.parametrize("days", [0, 366, "x"])
def test_analytics_rejects_a_bad_range(days):
    client, _ = _client(_analytics_answers())
    assert client.get("/browser/api/analytics", params={"days": days}).status_code == 422


def test_requests_list_groups_writes_and_reads_outcomes():
    row = SimpleNamespace(rid="syn_1", started=NOW, actions=3, owner="cust-1", owners=1)
    client, session = _client([
        ("SELECT count(DISTINCT a.correlation_id) FROM", [1]),
        ("GROUP BY a.correlation_id ORDER BY", [row]),
        ("WHERE a.correlation_id IN", [SimpleNamespace(rid="syn_1", kind="CREATE", n=2), SimpleNamespace(rid="syn_1", kind="ENRICH", n=1)]),
    ])
    body = client.get("/browser/api/requests", params={"entity": "cust-1", "days": 7}).json()
    assert body["total"] == 1 and body["items"][0]["request_id"] == "syn_1"
    assert body["items"][0]["outcomes"] == {"CREATE": 2, "ENRICH": 1} and body["items"][0]["entity"] == "cust-1"
    assert any(params.get("entity") == "cust-1" and "since" in params for _, params in session.seen)
    assert all("DROP" not in sql for sql, _ in session.seen)


def test_request_detail_not_found_and_found():
    none = SimpleNamespace(started=None, ended=None, actions=0, owner=None, owners=0)
    client, _ = _client([("WHERE a.correlation_id = :rid", [none])])
    assert client.get("/browser/api/requests/nope").status_code == 404
    head = SimpleNamespace(started=NOW, ended=NOW, actions=1, owner="cust-1", owners=1)
    item = SimpleNamespace(kind="SUPERSEDE", memory_id=uuid.uuid4(), memory_type="facts", content="new", previous_content="old", reason="newer", created_at=NOW)
    client, _ = _client([
        ("WHERE a.correlation_id IN", [SimpleNamespace(rid="syn_1", kind="SUPERSEDE", n=1)]),
        ("min(a.created_at) AS started, max", [head]),
        ("ORDER BY a.created_at, a.id", [item]),
    ])
    body = client.get("/browser/api/requests/syn_1").json()
    assert body["items"][0]["previous_content"] == "old" and body["truncated"] is False and body["outcomes"] == {"SUPERSEDE": 1}


def _config_client(monkeypatch, tmp_path, text=None, extra=()):
    from synap.cloud.memory_config import file_loader

    path = tmp_path / "memory.yml"
    if text is not None:
        path.write_text(text)
    for name in extra:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        target.write_text("storage_architecture:\n  scoping:\n    user_context_isolation: strict\n")
    monkeypatch.setattr(file_loader, "_shared", file_loader.FileConfigLoader(str(path)))
    return _client()[0]


def test_config_valid_file_lists_its_sections(monkeypatch, tmp_path):
    client = _config_client(monkeypatch, tmp_path, "storage_architecture:\n  scoping:\n    user_context_isolation: equals_customer\n")
    body = client.get("/browser/api/config").json()
    assert body["valid"] is True and body["problems"] == []
    assert body["sections"] == [{"name": "storage_architecture", "settings": 1}]


def test_config_with_a_mistake_says_which_line(monkeypatch, tmp_path):
    client = _config_client(monkeypatch, tmp_path, "storage_architecture:\n  scoping:\n    user_context_isolation: maca_suggests\n")
    body = client.get("/browser/api/config").json()
    assert body["valid"] is False and body["problems"] and "memory.yml" in body["problems"][0]
    client = _config_client(monkeypatch, tmp_path, "a: [unclosed\n")
    body = client.get("/browser/api/config").json()
    assert body["valid"] is False and "line" in body["problems"][0]


def test_config_with_no_file_lists_the_samples(monkeypatch, tmp_path):
    client = _config_client(monkeypatch, tmp_path, None, extra=("samples/business-app.yml", "memory.generated.yml"))
    body = client.get("/browser/api/config").json()
    assert body["exists"] is False and body["valid"] is None and body["sections"] == []
    assert body["samples"] == ["business-app.yml"]
    assert "generated" not in body and "generated_invalid" not in body


def test_the_unchanged_sample_is_called_the_starter_config(monkeypatch, tmp_path):
    from synap.cloud.application.sdk_api.core_browser import STARTER_MARKER

    starter = STARTER_MARKER + "\n#\nstorage_architecture:\n  scoping:\n    user_context_isolation: equals_customer\n"
    body = _config_client(monkeypatch, tmp_path, starter).get("/browser/api/config").json()
    assert body["exists"] is True and body["starter"] is True and body["valid"] is True


def test_a_file_the_developer_wrote_is_not_the_starter(monkeypatch, tmp_path):
    mine = "storage_architecture:\n  scoping:\n    user_context_isolation: strict\n"
    assert _config_client(monkeypatch, tmp_path, mine).get("/browser/api/config").json()["starter"] is False
    assert _config_client(monkeypatch, tmp_path, None).get("/browser/api/config").json()["starter"] is False


def test_the_starter_config_nudges_to_the_maca_steps():
    js = (BROWSER_DIR / "app.js").read_text()
    config = js[js.index("async function configPage"):js.index("// ---------------------------------------------------------------- shell")]
    assert "You are on the starter config" in config and "Make it yours" in config and "setupSteps(c, true)" in config
    overview = js[js.index("async function overviewPage"):js.index("async function overviewPage") + 1800]
    assert "cfg.starter" in overview and "#/config" in overview


def test_the_config_and_entities_pages_drop_what_was_asked_to_go():
    """No name column (Core never stores a name, only the id), no scoping mode card and no
    generated-file notice on the Config page."""
    js = (BROWSER_DIR / "app.js").read_text()
    assert '"Name"' not in js[js.index("async function entitiesPage"):js.index("async function entityDetailPage")]
    config = js[js.index("async function configPage"):js.index("// ---------------------------------------------------------------- shell")]
    assert "Scoping mode" not in config and "waiting for review" not in config and "generatedNotice" not in js


def test_the_config_page_has_setup_steps_for_a_missing_file():
    js = (BROWSER_DIR / "app.js").read_text()
    for needle in ("No memory config yet", "equals_customer", "strict", "memory.generated.yml", "sdk.maca.generate", "SYNAP_CLOUD_API_KEY"):
        assert needle in js, needle


def test_the_logo_is_the_client_dashboards_own_file():
    """The page uses the dashboard's logo, under a name no older version used, so a browser that cached
    an earlier icon fetches this one."""
    logo = BROWSER_DIR / "logo.svg"
    assert not (BROWSER_DIR / "favicon.svg").exists() and "logo.svg" in (BROWSER_DIR / "index.html").read_text()
    shared = BROWSER_DIR.parent / "client" / "public" / "favicon.svg"
    if shared.is_file():
        assert logo.read_bytes() == shared.read_bytes()
