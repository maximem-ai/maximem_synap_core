"""Retired-value guard (SYNAP_KU_RETIRED_VALUE_GUARD).

Regression case: after the user
changed their notification email, a later extraction re-asserted the OLD
address and the classifier answered SUPERSEDE with reason "updates the domain
of the user's notification email address" — so the row carrying the NEW email
was retired and the email silently rolled back.

The guard blocks a SUPERSEDE whose NEW content carries a high-entropy value
(full email / URL / >=7-digit phone) that this user already RETIRED, unless
the text explicitly signals a flip ("switched back to old@x"), which is a
legitimate revert. Flag off = unchanged behavior (pinned below).
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.pipeline import knowledge_update as ku
from synap.cloud.agents.ingestion.pipeline.knowledge_update import SupersessionAction
from synap.cloud.agents.ingestion.storage.coordinator import StorageCoordinator
from synap.tests.agents.ingestion._retire_shim import attach_retire_shim

OLD_EMAIL = "casey@old.example.com"
NEW_EMAIL = "casey@new.example.com"
# The backwards pair: the CURRENT row holds the new email, and a stale
# extraction re-asserting the old one is trying to supersede it.
CURRENT_ROW = f"The user's notification email is {NEW_EMAIL}"
BACKWARDS_NEW = f"The user's notification email address is {OLD_EMAIL}"


def _scope() -> Scope:
    return Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )


def _coordinator(retired_rows):
    rdbms = MagicMock()
    rdbms.soft_delete_memory = AsyncMock(return_value=True)
    rdbms.set_deprecated_at = AsyncMock(return_value=True)
    rdbms.insert_memory_link = AsyncMock()
    rdbms.insert_memory_action = AsyncMock()
    rdbms.retired_memories_with_value = AsyncMock(return_value=retired_rows)
    rdbms.current_memories_matching = AsyncMock(return_value=[])
    vector = MagicMock()
    vector.delete_memory_vectors = AsyncMock()
    coord = StorageCoordinator.__new__(StorageCoordinator)
    coord._rdbms = rdbms
    attach_retire_shim(rdbms)
    coord._vector = vector
    return coord, rdbms, vector


def _action(new_content=BACKWARDS_NEW, old_content=CURRENT_ROW, source_span=None):
    return SupersessionAction(
        old_memory_id=str(uuid4()),
        old_memory_content=old_content,
        new_memory_id=str(uuid4()),
        new_memory_content=new_content,
        reason="updates the domain of the user's notification email address",
        memory_type="facts",
        new_memory_source_span=source_span,
    )


def _retired_row(content=f"The user's notification email is {OLD_EMAIL}"):
    return [{"id": str(uuid4()), "content": content, "memory_type": "facts"}]


# ---------------------------------------------------------------------------
# 1. Helpers: high-entropy value extraction + change markers
# ---------------------------------------------------------------------------

class TestHighEntropyValues:
    def test_email_extracted(self):
        assert ku._extract_high_entropy_values(BACKWARDS_NEW) == frozenset(
            {OLD_EMAIL}
        )

    def test_url_extracted(self):
        assert ku._extract_high_entropy_values(
            "The user's site is https://old.example.com/profile"
        ) == frozenset({"https://old.example.com/profile"})

    def test_phone_with_seven_plus_digits_extracted(self):
        assert ku._extract_high_entropy_values(
            "The user's mobile is 555-123-4567"
        ) == frozenset({"555-123-4567"})

    def test_short_digit_run_is_not_a_value(self):
        # "ends in 4242" repeats across unrelated facts — never a retire key
        # (this is where _extract_value_tokens is deliberately wider).
        assert ku._extract_high_entropy_values("The user's card ends in 4242") == frozenset()
        assert "4242" in ku._extract_value_tokens("The user's card ends in 4242")

    def test_class_keyword_alone_is_not_a_value(self):
        assert ku._extract_high_entropy_values(
            "The user prefers email over phone for notifications"
        ) == frozenset()

    def test_trailing_punctuation_trimmed_and_lowercased(self):
        assert ku._extract_high_entropy_values(
            "Mail goes to Casey.Old@Example.COM."
        ) == frozenset({"casey.old@example.com"})

    def test_empty_text(self):
        assert ku._extract_high_entropy_values("") == frozenset()
        assert ku._extract_high_entropy_values(None) == frozenset()


class TestChangeMarkers:
    @pytest.mark.parametrize("text", [
        f"The user changed their email back to {OLD_EMAIL}",
        f"The user switched to {OLD_EMAIL}",
        f"The user now uses {OLD_EMAIL}",
        f"The user wants mail at {OLD_EMAIL} instead",
        f"The user updated their email to {OLD_EMAIL}",
        f"The user reverted to {OLD_EMAIL}",
        f"Actually the user's email is {OLD_EMAIL}",
    ])
    def test_markers_detected(self, text):
        assert ku._has_value_change_marker(text)

    def test_backwards_case_has_no_marker(self):
        assert not ku._has_value_change_marker(BACKWARDS_NEW)

    def test_empty(self):
        assert not ku._has_value_change_marker("")
        assert not ku._has_value_change_marker(None)


# ---------------------------------------------------------------------------
# 2. Flag OFF = legacy behavior (regression pin)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFlagOff:
    async def test_backwards_supersede_still_executes(self, monkeypatch):
        monkeypatch.delenv("SYNAP_KU_RETIRED_VALUE_GUARD", raising=False)
        coord, rdbms, _ = _coordinator(retired_rows=_retired_row())
        action = _action()
        count = await coord.execute_supersessions(
            actions=[action], scope=_scope(),
        )
        assert count == 1
        rdbms.soft_delete_memory.assert_awaited_once_with(
            memory_id=action.old_memory_id,
            superseded_by=action.new_memory_id,
        )
        rdbms.retired_memories_with_value.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. Flag ON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGuardOn:
    async def test_backwards_supersede_blocked(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_RETIRED_VALUE_GUARD", "true")
        coord, rdbms, vector = _coordinator(retired_rows=_retired_row())
        count = await coord.execute_supersessions(
            actions=[_action()], scope=_scope(),
            http_correlation_id="corr-1",
        )
        assert count == 0
        rdbms.soft_delete_memory.assert_not_awaited()
        rdbms.insert_memory_action.assert_not_awaited()
        vector.delete_memory_vectors.assert_not_awaited()
        assert rdbms.retired_memories_with_value.call_args.kwargs[
            "value_token"
        ] == OLD_EMAIL

    async def test_legit_flip_back_allowed(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_RETIRED_VALUE_GUARD", "true")
        coord, rdbms, _ = _coordinator(retired_rows=_retired_row())
        count = await coord.execute_supersessions(
            actions=[_action(
                new_content=f"The user switched back to {OLD_EMAIL}",
            )],
            scope=_scope(),
        )
        assert count == 1
        rdbms.soft_delete_memory.assert_awaited_once()

    async def test_marker_in_source_span_allows_flip(self, monkeypatch):
        """The third-person extraction can drop the change wording; the raw
        utterance still carries it."""
        monkeypatch.setenv("SYNAP_KU_RETIRED_VALUE_GUARD", "true")
        coord, rdbms, _ = _coordinator(retired_rows=_retired_row())
        count = await coord.execute_supersessions(
            actions=[_action(source_span=f"actually use {OLD_EMAIL} again")],
            scope=_scope(),
        )
        assert count == 1
        rdbms.soft_delete_memory.assert_awaited_once()

    async def test_value_never_retired_is_allowed(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_RETIRED_VALUE_GUARD", "true")
        coord, rdbms, _ = _coordinator(retired_rows=[])
        count = await coord.execute_supersessions(
            actions=[_action()], scope=_scope(),
        )
        assert count == 1
        rdbms.retired_memories_with_value.assert_awaited()

    async def test_no_high_entropy_value_skips_the_lookup(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_RETIRED_VALUE_GUARD", "true")
        coord, rdbms, _ = _coordinator(retired_rows=_retired_row())
        count = await coord.execute_supersessions(
            actions=[_action(
                new_content="The user's manager is Person B",
                old_content="The user's manager is Person A",
            )],
            scope=_scope(),
        )
        assert count == 1
        rdbms.retired_memories_with_value.assert_not_awaited()

    async def test_lookup_failure_fails_open(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_RETIRED_VALUE_GUARD", "true")
        coord, rdbms, _ = _coordinator(retired_rows=[])
        rdbms.retired_memories_with_value = AsyncMock(
            side_effect=RuntimeError("db down")
        )
        count = await coord.execute_supersessions(
            actions=[_action()], scope=_scope(),
        )
        assert count == 1

    async def test_storage_impl_without_lookup_fails_open(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_RETIRED_VALUE_GUARD", "true")
        coord, rdbms, _ = _coordinator(retired_rows=[])
        del rdbms.retired_memories_with_value
        count = await coord.execute_supersessions(
            actions=[_action()], scope=_scope(),
        )
        assert count == 1

    async def test_customer_scope_without_user_is_skipped(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_RETIRED_VALUE_GUARD", "true")
        coord, rdbms, _ = _coordinator(retired_rows=_retired_row())
        scope = MagicMock()
        scope.user_id = None
        count = await coord.execute_supersessions(
            actions=[_action()], scope=scope,
        )
        assert count == 1
        rdbms.retired_memories_with_value.assert_not_awaited()

    async def test_other_actions_in_the_batch_still_execute(self, monkeypatch):
        monkeypatch.setenv("SYNAP_KU_RETIRED_VALUE_GUARD", "true")
        coord, rdbms, _ = _coordinator(retired_rows=_retired_row())

        async def _retired(scope, value_token, exclude_memory_ids=None, limit=1):
            return _retired_row() if value_token == OLD_EMAIL else []

        rdbms.retired_memories_with_value = AsyncMock(side_effect=_retired)
        blocked = _action()
        clean = _action(
            new_content="The user's manager is Person B",
            old_content="The user's manager is Person A",
        )
        count = await coord.execute_supersessions(
            actions=[blocked, clean], scope=_scope(),
        )
        assert count == 1
        assert rdbms.soft_delete_memory.await_args.kwargs["memory_id"] == (
            clean.old_memory_id
        )


# ---------------------------------------------------------------------------
# 4. Wiring + SQL pins
# ---------------------------------------------------------------------------

class TestWiring:
    def test_source_span_carried_onto_the_action(self):
        import inspect
        src = inspect.getsource(ku.KnowledgeUpdateDetector)
        assert "new_memory_source_span=getattr(" in src

    def test_retired_lookup_sql_is_retired_rows_only(self):
        import inspect
        from synap.cloud.agents.ingestion.storage import sqlalchemy_rdbms_wrapper
        src = inspect.getsource(
            sqlalchemy_rdbms_wrapper.SQLAlchemyRDBMSWrapper.retired_memories_with_value
        )
        assert "superseded_by IS NOT NULL OR deprecated_at IS NOT NULL" in src
        assert "ILIKE" in src
