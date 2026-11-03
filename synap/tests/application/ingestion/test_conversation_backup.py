"""One restorable object per conversation, treated by the PII gate, never stored raw."""
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from synap.cloud.application.ingestion.conversation_backup import (
    backup_conversation, conversation_key,
)
from synap.cloud.application.ingestion.manager import (
    IngestionManager, PIIPolicyEnforcementError, StorageTreatment,
)
from synap.cloud.application.ingestion.storage_provider import FileStoreStorageProvider

SCOPE = SimpleNamespace(client_id="c", instance_id="i", customer_id="u")
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def msg(n, content=None, role="user"):
    return SimpleNamespace(message_id=f"m{n}", role=role,
                           content=content if content is not None else f"text {n}",
                           timestamp=T0 + timedelta(minutes=n))


class FakeStore:
    def __init__(self):
        self.objects = {}
        self.meta = {}
        self.writes = 0

    async def read_object(self, path):
        return self.objects.get(path)

    async def write_object(self, path, content, metadata=None):
        self.objects[path] = content
        self.meta[path] = metadata
        self.writes += 1


def manager(treat=None):
    calls = []

    async def default(text, scope, request_id):
        calls.append(request_id)
        return StorageTreatment(text=text.replace("4111111111111111", "[CARD]"),
                                detected=True, floor_blocked=1)

    mgr = SimpleNamespace(_storage=FakeStore(),
                          treat_for_storage=AsyncMock(side_effect=treat or default))
    mgr.calls = calls
    return mgr


def stored(mgr, conv="conv-1"):
    return json.loads(mgr._storage.objects[conversation_key(SCOPE, conv)])


@pytest.mark.asyncio
async def test_one_object_per_conversation_with_ids_and_timestamps():
    mgr = manager()
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1), msg(2, role="assistant")])
    key = conversation_key(SCOPE, "conv-1")
    assert list(mgr._storage.objects) == [key] and "conv-1" in key
    doc = stored(mgr)
    assert doc["conversation_id"] == "conv-1"
    assert [m["message_id"] for m in doc["messages"]] == ["m1", "m2"]
    assert doc["messages"][1]["role"] == "assistant"
    assert doc["messages"][0]["timestamp"].startswith("2026-01-01T00:01")


@pytest.mark.asyncio
async def test_a_later_compaction_adds_only_new_messages_and_keeps_one_object():
    mgr = manager()
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1), msg(2)])
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1), msg(2), msg(3), msg(4)])
    assert len(mgr._storage.objects) == 1
    assert [m["message_id"] for m in stored(mgr)["messages"]] == ["m1", "m2", "m3", "m4"]
    # each message is treated once, not again on every compaction
    assert sorted(mgr.calls) == sorted(f"convbackup:conv-1:m{n}" for n in (1, 2, 3, 4))


@pytest.mark.asyncio
async def test_full_reprocessing_does_not_duplicate_messages():
    mgr = manager()
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1), msg(2)])
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1), msg(2)])
    assert len(stored(mgr)["messages"]) == 2
    assert mgr._storage.writes == 1  # nothing new, nothing rewritten


@pytest.mark.asyncio
async def test_the_stored_text_is_the_treated_text_not_the_original():
    mgr = manager()
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1, "card 4111111111111111 ok")])
    text = stored(mgr)["messages"][0]["content"]
    assert "4111111111111111" not in text and "[CARD]" in text


@pytest.mark.asyncio
async def test_a_message_whose_treatment_fails_is_stored_flagged_with_where_then_repaired(caplog):
    fail = {"on": True}

    async def treat(text, scope, request_id):
        if fail["on"] and request_id.endswith(":m2"):
            return StorageTreatment(text=text, failed_at="policy", reason="could not alias")
        return StorageTreatment(text=text.replace("secret", "[X]"), detected=True)

    mgr = manager(treat)
    with caplog.at_level("ERROR"):
        await backup_conversation(mgr, "conv-1", SCOPE, [msg(1), msg(2, "secret"), msg(3)])
    doc = stored(mgr)
    m2 = doc["messages"][1]
    assert m2["content"] == "secret"  # stored as received, per the decision
    assert m2["pii"] == {"status": "failed", "failed_at": "policy", "reason": "could not alias"}
    assert doc["messages"][0]["pii"]["status"] == "treated"
    assert doc["pii_failed_message_ids"] == ["m2"]
    assert "conversation_backup_holds_untreated_text" in caplog.text
    # the object itself says so, so a bucket scan finds it without opening files
    assert mgr._storage.meta[conversation_key(SCOPE, "conv-1")]["pii_failed_messages"] == "1"

    fail["on"] = False  # the gate recovers: the next compaction repairs it
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1), msg(2, "secret"), msg(3)])
    doc = stored(mgr)
    assert doc["messages"][1]["content"] == "[X]" and doc["messages"][1]["pii"]["status"] == "treated"
    assert doc["pii_failed_message_ids"] == []
    assert len(doc["messages"]) == 3


@pytest.mark.asyncio
async def test_a_message_that_keeps_failing_is_not_rewritten_every_time():
    async def treat(text, scope, request_id):
        return StorageTreatment(text=text, failed_at="detector", reason="x")
    mgr = manager(treat)
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1)])
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1)])
    assert mgr._storage.writes == 1


@pytest.mark.asyncio
async def test_treated_messages_record_counts_not_values():
    mgr = manager()
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1, "card 4111111111111111")])
    pii = stored(mgr)["messages"][0]["pii"]
    assert pii["status"] == "treated" and pii["floor_blocked"] == 1
    assert "4111" not in json.dumps(pii)


@pytest.mark.asyncio
async def test_messages_come_back_in_timestamp_order_even_when_added_late():
    mgr = manager()
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(2), msg(3)])
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1), msg(2), msg(3)])
    assert [m["message_id"] for m in stored(mgr)["messages"]] == ["m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_empty_messages_are_skipped_and_other_conversations_are_separate():
    mgr = manager()
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1, "  "), msg(2)])
    await backup_conversation(mgr, "conv-2", SCOPE, [msg(1)])
    assert [m["message_id"] for m in stored(mgr)["messages"]] == ["m2"]
    assert len(mgr._storage.objects) == 2


@pytest.mark.asyncio
async def test_a_storage_failure_never_raises_into_the_compaction(caplog):
    mgr = manager()
    mgr._storage.read_object = AsyncMock(side_effect=RuntimeError("s3 down"))
    with caplog.at_level("ERROR"):
        await backup_conversation(mgr, "conv-1", SCOPE, [msg(1)])
    assert "conversation_backup_failed" in caplog.text


# --- the manager method that runs the real gate -----------------------------------

def _real_manager(gate):
    mgr = IngestionManager.__new__(IngestionManager)
    mgr._run_ingress_gate = gate
    return mgr


@pytest.mark.asyncio
async def test_treat_for_storage_returns_what_the_gate_leaves():
    async def gate(req):
        req.content = req.content.replace("Asha", "[NAME_1]")
        req.metadata.update(pii_detected=True, pii_aliased=1, pii_policy_version=3)
    t = await _real_manager(gate).treat_for_storage("Asha called", SCOPE_REAL, request_id="r")
    assert (t.text, t.failed_at, t.aliased, t.policy_version) == ("[NAME_1] called", None, 1, 3)


@pytest.mark.asyncio
async def test_treat_for_storage_reports_a_policy_failure_and_returns_the_original(caplog):
    async def gate(req):
        raise PIIPolicyEnforcementError("could not apply policy v2")
    with caplog.at_level("ERROR"):
        t = await _real_manager(gate).treat_for_storage("x", SCOPE_REAL, request_id="r")
    assert (t.text, t.failed_at) == ("x", "policy") and "policy v2" in t.reason
    assert "backup_text_pii_failed" in caplog.text


@pytest.mark.asyncio
async def test_treat_for_storage_reports_a_detector_failure():
    async def gate(req):
        req.metadata["pii_check_failed"] = True
    t = await _real_manager(gate).treat_for_storage("x", SCOPE_REAL, request_id="r")
    assert (t.text, t.failed_at) == ("x", "detector")


@pytest.mark.asyncio
async def test_treat_for_storage_never_raises_on_a_gate_bug():
    async def gate(req):
        raise NameError("bug")
    t = await _real_manager(gate).treat_for_storage("x", SCOPE_REAL, request_id="r")
    assert t.failed_at == "gate" and "NameError" in t.reason


from synap.cloud.agents.ingestion.models.inputs import Scope  # noqa: E402
SCOPE_REAL = Scope(client_id=uuid.uuid4(), instance_id=uuid.uuid4(), customer_id=uuid.uuid4())


@pytest.mark.asyncio
async def test_provider_read_object_is_none_for_a_missing_object():
    store = SimpleNamespace(_bucket="b", download=lambda k: (_ for _ in ()).throw(FileNotFoundError(k)))
    with patch("synap.cloud.application.ingestion.storage_provider.get_file_store",
               return_value=store):
        p = FileStoreStorageProvider()
    assert await p.read_object("nope") is None


@pytest.mark.asyncio
async def test_no_pii_step_is_labelled_not_applied_and_not_retried():
    async def untreated(text, scope, request_id):
        return StorageTreatment(text=text, applied=False)
    mgr = manager(untreated)
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1)])
    assert stored(mgr)["messages"][0]["pii"] == {"status": "not_applied"}
    assert stored(mgr)["pii_failed_message_ids"] == []
    await backup_conversation(mgr, "conv-1", SCOPE, [msg(1)])
    assert mgr._storage.writes == 1 and mgr.treat_for_storage.await_count == 1


@pytest.mark.asyncio
async def test_a_build_with_no_pii_hook_reports_the_text_as_untreated():
    from unittest.mock import patch
    with patch("synap.cloud.application.ingestion.manager.edition.acall", AsyncMock(return_value=None)):
        out = await IngestionManager.treat_for_storage(SimpleNamespace(), "hello", SCOPE, request_id="r")
    assert out.text == "hello" and out.applied is False
