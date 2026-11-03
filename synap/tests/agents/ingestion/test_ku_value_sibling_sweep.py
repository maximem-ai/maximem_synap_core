"""Sibling value sweep (SYNAP_KU_VALUE_SIBLING_SWEEP).

A supersession retires exactly ONE row, but the retired value is usually
spread over paraphrases the classifier calls ENRICH ("the user prefers to be
contacted at old@x", "receipts go to old@x"). Those siblings stay CURRENT, so
retrieval serves the old email next to the new one — the "two emails on file"
shape seen in practice.

When on, a successful SUPERSEDE also retires every other current row carrying
the superseded value, the same way the supersession retired its own target
(superseded_by = the new row, SUPERSEDE changelog row, Chroma vector delete).
Rows that ALSO carry the new value are left alone. Flag off = unchanged
behavior (pinned below).
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.pipeline.knowledge_update import SupersessionAction
from synap.cloud.agents.ingestion.storage.coordinator import StorageCoordinator
from synap.tests.agents.ingestion._retire_shim import attach_retire_shim

OLD_EMAIL = "casey@old.example.com"
NEW_EMAIL = "casey@new.example.com"
OLD_ROW = f"The user's email is {OLD_EMAIL}"
NEW_ROW = f"The user's email is {NEW_EMAIL}"
SIBLING = f"The user prefers to be contacted at {OLD_EMAIL}"
SIBLING_2 = f"Receipts are sent to {OLD_EMAIL}"
MIGRATION_ROW = f"The user moved from {OLD_EMAIL} to {NEW_EMAIL}"


def _scope() -> Scope:
    return Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )


def _row(content, mtype="facts"):
    return {"id": str(uuid4()), "content": content, "memory_type": mtype}


def _coordinator(sibling_rows):
    rdbms = MagicMock()
    rdbms.soft_delete_memory = AsyncMock(return_value=True)
    rdbms.set_deprecated_at = AsyncMock(return_value=True)
    rdbms.insert_memory_link = AsyncMock()
    rdbms.insert_memory_action = AsyncMock()
    rdbms.current_memories_matching = AsyncMock(return_value=sibling_rows)
    rdbms.retired_memories_with_value = AsyncMock(return_value=[])
    vector = MagicMock()
    vector.delete_memory_vectors = AsyncMock()
    coord = StorageCoordinator.__new__(StorageCoordinator)
    coord._rdbms = rdbms
    attach_retire_shim(rdbms)
    coord._vector = vector
    return coord, rdbms, vector


def _action(old_content=OLD_ROW, new_content=NEW_ROW):
    return SupersessionAction(
        old_memory_id=str(uuid4()),
        old_memory_content=old_content,
        new_memory_id=str(uuid4()),
        new_memory_content=new_content,
        reason="the notification email changed",
        memory_type="facts",
    )


@pytest.mark.asyncio
class TestFlagOff:
    async def test_siblings_left_alone(self, monkeypatch):
        monkeypatch.delenv("SYNAP_KU_VALUE_SIBLING_SWEEP", raising=False)
        coord, rdbms, _ = _coordinator(sibling_rows=[_row(SIBLING)])
        action = _action()
        count = await coord.execute_supersessions(
            actions=[action], scope=_scope(),
        )
        assert count == 1
        rdbms.current_memories_matching.assert_not_awaited()
        # Only the pair's own target was retired.
        assert rdbms.soft_delete_memory.await_count == 1
        assert rdbms.soft_delete_memory.await_args.kwargs["memory_id"] == (
            action.old_memory_id
        )


@pytest.mark.asyncio
class TestSweepOn:
    async def test_paraphrase_siblings_retired(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_VALUE_SIBLING_SWEEP", "true")
        sib1, sib2 = _row(SIBLING), _row(SIBLING_2, "preferences")
        coord, rdbms, vector = _coordinator(sibling_rows=[sib1, sib2])
        action = _action()
        count = await coord.execute_supersessions(
            actions=[action], scope=_scope(),
            http_correlation_id="corr-1",
        )
        assert count == 1
        retired = {
            c.kwargs["memory_id"] for c in rdbms.soft_delete_memory.await_args_list
        }
        assert retired == {action.old_memory_id, sib1["id"], sib2["id"]}
        # Siblings point at the NEW memory as their survivor.
        for call in rdbms.soft_delete_memory.await_args_list:
            assert call.kwargs["superseded_by"] == action.new_memory_id
        # Vectors dropped for the target + both siblings.
        assert vector.delete_memory_vectors.await_count == 3
        sibling_actions = [
            c.kwargs for c in rdbms.insert_memory_action.await_args_list
            if c.kwargs["memory_id"] in {sib1["id"], sib2["id"]}
        ]
        assert len(sibling_actions) == 2
        assert all(a["action"] == "SUPERSEDE" for a in sibling_actions)
        assert all(
            a["replaced_by_memory_id"] == action.new_memory_id
            for a in sibling_actions
        )
        assert all("value-sibling-sweep" in a["reason"] for a in sibling_actions)
        assert {a["memory_type"] for a in sibling_actions} == {"facts", "preferences"}
        assert rdbms.current_memories_matching.call_args.kwargs[
            "content_tokens"
        ] == [OLD_EMAIL]

    async def test_row_carrying_the_new_value_is_skipped(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_VALUE_SIBLING_SWEEP", "true")
        sib, migration = _row(SIBLING), _row(MIGRATION_ROW)
        coord, rdbms, _ = _coordinator(sibling_rows=[sib, migration])
        action = _action()
        await coord.execute_supersessions(actions=[action], scope=_scope())
        retired = {
            c.kwargs["memory_id"] for c in rdbms.soft_delete_memory.await_args_list
        }
        assert sib["id"] in retired
        assert migration["id"] not in retired

    async def test_pair_ids_excluded_from_the_lookup(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_VALUE_SIBLING_SWEEP", "true")
        coord, rdbms, _ = _coordinator(sibling_rows=[])
        action = _action()
        await coord.execute_supersessions(actions=[action], scope=_scope())
        excluded = set(
            rdbms.current_memories_matching.call_args.kwargs["exclude_memory_ids"]
        )
        assert excluded == {action.old_memory_id, action.new_memory_id}

    async def test_no_high_entropy_value_skips_the_sweep(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_VALUE_SIBLING_SWEEP", "true")
        coord, rdbms, _ = _coordinator(sibling_rows=[_row(SIBLING)])
        count = await coord.execute_supersessions(
            actions=[_action(
                old_content="The user's plan is Basic",
                new_content="The user's plan is Pro",
            )],
            scope=_scope(),
        )
        assert count == 1
        rdbms.current_memories_matching.assert_not_awaited()

    async def test_unchanged_value_is_not_swept(self, monkeypatch):
        """Both sides carry the same email (the change was elsewhere) — the
        value isn't stale, so nothing sibling-sweeps on it."""
        monkeypatch.setenv("SYNAP_KU_VALUE_SIBLING_SWEEP", "true")
        coord, rdbms, _ = _coordinator(sibling_rows=[_row(SIBLING)])
        await coord.execute_supersessions(
            actions=[_action(
                old_content=f"The user's email {OLD_EMAIL} is unverified",
                new_content=f"The user's email {OLD_EMAIL} is verified",
            )],
            scope=_scope(),
        )
        rdbms.current_memories_matching.assert_not_awaited()

    async def test_lookup_failure_is_non_fatal(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_VALUE_SIBLING_SWEEP", "true")
        coord, rdbms, _ = _coordinator(sibling_rows=[])
        rdbms.current_memories_matching = AsyncMock(
            side_effect=RuntimeError("db down")
        )
        count = await coord.execute_supersessions(
            actions=[_action()], scope=_scope(),
        )
        assert count == 1  # the supersession itself already succeeded

    async def test_sibling_write_failure_is_non_fatal(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_VALUE_SIBLING_SWEEP", "true")
        sib = _row(SIBLING)
        coord, rdbms, _ = _coordinator(sibling_rows=[sib])
        action = _action()

        async def _soft_delete(memory_id, superseded_by):
            if memory_id == sib["id"]:
                raise RuntimeError("row locked")
            return True

        rdbms.soft_delete_memory = AsyncMock(side_effect=_soft_delete)
        count = await coord.execute_supersessions(
            actions=[action], scope=_scope(),
        )
        assert count == 1

    async def test_deprecate_with_link_path_writes_lineage(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_VALUE_SIBLING_SWEEP", "true")
        monkeypatch.setenv("ENABLE_KU_DEPRECATE_WITH_LINK", "true")
        sib = _row(SIBLING)
        coord, rdbms, _ = _coordinator(sibling_rows=[sib])
        action = _action()
        await coord.execute_supersessions(actions=[action], scope=_scope())
        links = {
            (c.kwargs["from_memory_id"], c.kwargs["to_memory_id"])
            for c in rdbms.insert_memory_link.await_args_list
        }
        assert (sib["id"], action.new_memory_id) in links
        deprecated = {
            c.kwargs["memory_id"] for c in rdbms.set_deprecated_at.await_args_list
        }
        assert sib["id"] in deprecated

    async def test_customer_scope_without_user_is_skipped(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_VALUE_SIBLING_SWEEP", "true")
        coord, rdbms, _ = _coordinator(sibling_rows=[_row(SIBLING)])
        scope = MagicMock()
        scope.user_id = None
        count = await coord.execute_supersessions(
            actions=[_action()], scope=scope,
        )
        assert count == 1
        rdbms.current_memories_matching.assert_not_awaited()

    async def test_no_sweep_when_supersession_itself_no_ops(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_VALUE_SIBLING_SWEEP", "true")
        coord, rdbms, _ = _coordinator(sibling_rows=[_row(SIBLING)])
        rdbms.soft_delete_memory = AsyncMock(return_value=False)
        count = await coord.execute_supersessions(
            actions=[_action()], scope=_scope(),
        )
        assert count == 0
        rdbms.current_memories_matching.assert_not_awaited()
