"""End-to-end checks that a fresh Synap Core install works, through the real SDK and the real stack.

Run order matters: later tests use what earlier ones stored. Each run uses new user ids, so the SDK's
own five-minute answer cache can never hide a result. Needs an LLM: the model in your .env is called
a few dozen times.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import pytest

from maximem_synap import MaximemSynapSDK

RUN = uuid.uuid4().hex[:8]
ALICE, BOB, NADIA = f"alice_{RUN}", f"bob_{RUN}", f"nadia_{RUN}"
S: dict = {}          # what earlier tests store for later ones


# -- helpers ------------------------------------------------------------------------------------
@contextmanager
def _sdk_env(url: str, token: str | None = None):
    saved = {k: v for k, v in os.environ.items() if k.startswith("SYNAP_")}
    for k in saved:
        del os.environ[k]
    os.environ["SYNAP_BASE_URL"] = url
    if token:
        os.environ["SYNAP_API_TOKEN"] = token
    try:
        yield
    finally:
        for k in [k for k in os.environ if k.startswith("SYNAP_")]:
            del os.environ[k]
        os.environ.update(saved)


def with_sdk(stack, fn, token: str | None = None):
    """Run `fn(sdk)` against the stack with a fresh keyless SDK."""
    async def go():
        sdk = MaximemSynapSDK()
        await sdk.initialize()
        try:
            return await fn(sdk)
        finally:
            await sdk.shutdown()
    with _sdk_env(stack.url, token):
        return asyncio.run(go())


def texts(ctx) -> list[str]:
    out = []
    for name in ("facts", "preferences", "episodes", "emotions", "temporal_events"):
        for item in getattr(ctx, name, None) or []:
            out.append(getattr(item, "content", str(item)))
    return out


async def store(sdk, user: str, document: str, document_type: str = "ai-chat-conversation"):
    created = await sdk.memories.create(document=document, document_type=document_type, user_id=user)
    return await sdk.memories.wait_for_completion(created.ingestion_id, timeout_seconds=400)


async def recall(sdk, user: str, *queries: str) -> list[str]:
    return texts(await sdk.user.context.fetch(user_id=user, search_query=list(queries)))


def browser_total(stack) -> int:
    status, body = stack.get("/browser/api/memories?limit=1&status=all")
    assert status == 200, body[:200]
    return json.loads(body)["total"]


# -- the install itself ---------------------------------------------------------------------------
def test_01_server_is_up_and_is_core(stack):
    status, body = stack.get("/health")
    assert status == 200 and json.loads(body) == {"status": "ok", "edition": "core"}


def test_02_the_setup_containers_finished_cleanly(stack):
    for service in ("secrets", "schema", "first-start"):
        assert stack.exit_code(service) == 0, f"{service} did not exit 0:\n{stack.logs(service)[-1500:]}"


def test_03_only_the_api_port_is_published_and_only_on_loopback(stack):
    rows = stack.docker("ps", "--filter", f"label=com.docker.compose.project={stack.project}",
                        "--format", "{{.Names}}|{{.Ports}}").strip().splitlines()
    assert rows, "no containers found"
    for row in rows:
        name, ports = row.split("|", 1)
        if name.endswith("-synap-1"):
            assert "127.0.0.1:8000->8000" in ports and "0.0.0.0" not in ports, ports
        else:
            assert "->" not in ports, f"{name} publishes a port: {ports}"


def test_04_passwords_are_generated_and_in_no_file(stack):
    secrets_env = stack.compose("exec", "-T", "synap", "cat", "/state/secrets.env").stdout
    passwords = dict(l.split("=", 1) for l in secrets_env.splitlines() if "=" in l)
    assert set(passwords) == {"POSTGRES_PASSWORD", "REDIS_PASSWORD", "NEO4J_PASSWORD"}
    assert all(len(v) >= 24 for v in passwords.values()) and len(set(passwords.values())) == 3
    for path in stack.root.rglob("*"):
        if path.is_file() and path.stat().st_size < 3_000_000 and ".git" not in path.parts:
            data = path.read_bytes()
            assert not any(v.encode() in data for v in passwords.values()), f"a password is in {path}"
    S["secrets_sha"] = hashlib.sha256(secrets_env.encode()).hexdigest()


def test_05_startup_warns_about_the_missing_token_and_logs_no_errors(stack):
    log = stack.logs("synap")
    assert "SYNAP_API_TOKEN is not set" in log
    errors = [l for l in log.splitlines() if re.search(r"\bERROR\b", l)]
    assert not errors, "\n".join(errors[:5])


def test_06_the_sdk_connects_with_no_key_and_core_reports_its_scoping_mode(stack):
    with_sdk(stack, lambda sdk: asyncio.sleep(0))
    status, body = stack.get("/api/v1/auth/whoami")
    assert status == 200 and json.loads(body)["user_context_isolation"] == "equals_customer"


# -- storing and recalling ------------------------------------------------------------------------
def test_07_a_memory_is_stored_and_recalled(stack):
    async def go(sdk):
        done = await store(sdk, ALICE, "User: I prefer dark mode in every app.\nAssistant: Noted!")
        S["alice_ids"] = done.memory_ids
        S["alice_status"] = done
        return await recall(sdk, ALICE, "preferences", "dark mode")
    got = with_sdk(stack, go)
    assert S["alice_status"].status.value == "completed" and S["alice_status"].memories_created >= 1
    assert any("dark" in t.lower() for t in got), got


def test_08_one_user_cannot_see_another_users_memories(stack):
    async def go(sdk):
        await store(sdk, BOB, "User: I am vegetarian and I never eat fish.\nAssistant: Understood.")
        bob = await recall(sdk, BOB, "diet", "vegetarian", "dark mode")
        alice = await recall(sdk, ALICE, "diet", "vegetarian", "dark mode")
        nobody = await recall(sdk, f"never_seen_{RUN}", "diet", "vegetarian", "dark mode")
        return bob, alice, nobody
    bob, alice, nobody = with_sdk(stack, go)
    assert any("vegetarian" in t.lower() for t in bob), bob
    assert not any("dark" in t.lower() for t in bob), bob
    assert not any("vegetarian" in t.lower() for t in alice), alice
    assert nobody == []


def test_09_a_request_that_does_not_match_the_scoping_mode_fails_loudly(stack):
    async def go(sdk):
        await sdk.memories.create(document="User: I like tea.\nAssistant: Nice.", document_type="ai-chat-conversation",
                                  user_id=ALICE, customer_id="some_customer")
    with pytest.raises(Exception) as err:
        with_sdk(stack, go)
    assert "customer" in str(err.value).lower(), str(err.value)


def test_10_text_that_is_too_short_is_skipped_and_says_why(stack):
    async def go(sdk):
        return await store(sdk, ALICE, "Hi there.", document_type="document")
    done = with_sdk(stack, go)
    assert done.memories_created == 0 and done.error_message, done


def test_11_an_uploaded_file_is_read(stack, tmp_path):
    note = tmp_path / "note.md"
    note.write_text("# Notes on Maria\n\nMaria Santos lives in Porto and is allergic to shellfish. "
                    "She works as a cartographer and plays the cello on weekends.\n")

    async def go(sdk):
        created = await sdk.memories.create_from_file(file_path=str(note), document_type="document", user_id=f"maria_{RUN}")
        return await sdk.memories.wait_for_completion(created.ingestion_id, timeout_seconds=400)
    done = with_sdk(stack, go)
    assert done.status.value == "completed" and done.memories_created >= 1, done


def test_12_a_conversation_is_recorded_and_compacted(stack):
    conversation = str(uuid.uuid4())
    turns = [("user", "Hi, I'm Nadia. I live in Lisbon and work as a night-shift nurse."),
             ("assistant", "Nice to meet you Nadia! Night shifts must be tiring."),
             ("user", "They are. My dog Mango keeps me company in the mornings."),
             ("assistant", "Mango sounds lovely. What breed?"),
             ("user", "A small beagle. I also train for a half marathon in March."),
             ("assistant", "Great goal. How is the training going?"),
             ("user", "Slowly, I run 10 km on weekends. I am vegetarian too."),
             ("assistant", "Good to know, I will keep that in mind.")]

    async def go(sdk):
        result = await sdk.conversation.record_messages_batch(
            [{"conversation_id": conversation, "role": r, "content": c, "user_id": NADIA} for r, c in turns])
        assert result["failed"] == 0 and result["succeeded"] == len(turns), result
        await sdk.conversation.context.compact(conversation, force=True)
        end = time.time() + 300
        while time.time() < end:
            st = await sdk.conversation.context.get_compaction_status(conversation)
            if str(getattr(st, "status", "")).lower().endswith("completed"):
                break
            assert "fail" not in str(getattr(st, "status", "")).lower(), st
            await asyncio.sleep(5)
        return await sdk.conversation.context.get_compacted(conversation)
    compacted = with_sdk(stack, go)
    text = str(getattr(compacted, "compacted_context", compacted)).lower()
    assert "lisbon" in text and "beagle" in text or "mango" in text, text[:600]


def test_13_one_memory_can_be_deleted_by_id(stack):
    assert S["alice_ids"], "test 07 stored no memory ids"
    target = UUID(S["alice_ids"][0])

    async def go(sdk):
        await sdk.memories.delete(target)
        with pytest.raises(Exception):
            await sdk.memories.get(target)
    with_sdk(stack, go)


def test_14_forget_answers_clearly(stack):
    async def go(sdk):
        unknown = await sdk.memories.forget(description="my old address", user_id=f"never_seen_{RUN}", mode="direct")
        with pytest.raises(Exception) as err:
            await sdk.memories.forget(description="the vegetarian diet", user_id=BOB, mode="direct")
        return unknown, str(err.value)
    unknown, message = with_sdk(stack, go)
    assert unknown.status == "nothing_matches"
    assert "Jev API key" in message and "Nothing was deleted" in message, message


def test_15_cloud_only_calls_say_they_are_not_available(stack):
    async def go(sdk):
        messages = []
        for call in (lambda: sdk.credits.get_balance(),
                     lambda: sdk.data.delete(scope="user", user_id=BOB, dry_run=True)):
            with pytest.raises(Exception) as err:
                await call()
            messages.append(str(err.value))
        return messages
    for message in with_sdk(stack, go):
        assert "not available in the open-source server" in message, message


def test_16_the_maca_plugin_without_a_cloud_key_says_so(stack, tmp_path):
    usecase = tmp_path / "use-case.md"
    usecase.write_text("# Agent\nA support assistant that remembers customers.\n")

    async def go(sdk):
        with pytest.raises(Exception) as err:
            await sdk.maca.generate(file_path=str(usecase), user_relationship="b2c")
        return str(err.value)
    assert "not configured" in with_sdk(stack, go)


def test_17_the_memory_browser_shows_memories_and_cannot_write(stack):
    status, html = stack.get("/browser/")
    assert status == 200 and "<html" in html.lower()
    assert browser_total(stack) >= 2
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        assert stack.send(method, "/browser/api/memories") == 405, method


# -- restart and access control ---------------------------------------------------------------------
def test_18_data_and_passwords_survive_a_restart(stack):
    before = browser_total(stack)
    stack.compose("down", timeout=300)
    stack.compose("up", "-d", timeout=900)
    stack.wait_healthy()
    assert browser_total(stack) == before
    secrets_env = stack.compose("exec", "-T", "synap", "cat", "/state/secrets.env").stdout
    assert hashlib.sha256(secrets_env.encode()).hexdigest() == S["secrets_sha"]
    got = with_sdk(stack, lambda sdk: recall(sdk, BOB, "diet", "vegetarian", f"after restart {secrets.token_hex(2)}"))
    assert any("vegetarian" in t.lower() for t in got), got


def test_19_a_token_closes_the_server_to_callers_without_it(stack):
    token = secrets.token_urlsafe(24)
    stack.set_env("SYNAP_API_TOKEN", token)
    stack.compose("up", "-d", timeout=600)
    stack.wait_healthy()
    assert stack.get("/api/v1/auth/whoami")[0] in (401, 403)
    assert stack.get("/browser/api/memories?limit=1")[0] in (401, 403)
    assert stack.get("/api/v1/auth/whoami", {"Authorization": f"Bearer {token}"})[0] == 200
    got = with_sdk(stack, lambda sdk: recall(sdk, BOB, "diet", "vegetarian"), token=token)
    assert any("vegetarian" in t.lower() for t in got), got
    with pytest.raises(Exception):
        with_sdk(stack, lambda sdk: recall(sdk, BOB, "diet"))
    assert "SYNAP_API_TOKEN is not set" not in stack.logs("synap").split("Started server process")[-1]
