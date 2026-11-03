"""KU echo guard (ENABLE_ECHO_GUARD, WS2 of the echo-poisoning plan).

Failure this protects against: conversation turns are ingested
with the assistant's reply included, so the agent restating a stale value
mints a fresh fact whose newer extraction time lets it SUPERSEDE the user's
genuinely newer statement (in an eval run, an agent echo superseded the
user's updated email shortly after the update).

Guard semantics (pair loop, code-level — the classifier addendum is only a
precision aid): an assistant-sourced NEW memory may not supersede a
user-sourced or unattributed EXISTING memory unless both sides carry the
same concrete value (a plain duplicate, where dedup is safe). Blocked pairs
become NONE drops: the echo never becomes servable and the decision is
audited in memory_actions.

Default OFF: with the flag unset, behavior is identical to today.
"""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    MemoryDestination,
    MemoryType,
    Scope,
)
from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    KnowledgeUpdateDetector,
    _extract_value_tokens,
)

USER_EMAIL_FACT = "The user's email address is casey.new@example.com"
ECHO_OLD_EMAIL = "The user receives updates at casey.old@example.com"
ECHO_SAME_EMAIL = "Updates for the user go to casey.new@example.com"


def _scope() -> Scope:
    return Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )


def _mem(content: str, assertion_source=None) -> ExtractedMemory:
    return ExtractedMemory(
        memory_id=uuid4(),
        memory_type=MemoryType.FACTS,
        content=content,
        confidence=0.9,
        destination=MemoryDestination.MEMORIES,
        source_document_id="doc-x",
        assertion_source=assertion_source,
    )


def _vector_with_hit(existing_content: str, existing_source=None, score: float = 0.9):
    vec = MagicMock()
    vec.batch_search = AsyncMock(return_value=[[
        {
            "id": str(uuid4()),
            "content": existing_content,
            "score": score,
            "metadata": {"memory_type": "facts", "assertion_source": existing_source},
        },
    ]])
    return vec


def _llm_classifier(returns_classification: str):
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=json.dumps({
        "classification": returns_classification,
        "reason": "test stub",
    }))
    return llm


async def _run(new_mem, vector, llm):
    detector = KnowledgeUpdateDetector(vector=vector, llm_client=llm)
    return await detector.detect_updates(
        memories=[new_mem],
        memory_embeddings=[[0.1] * 8],
        scope=_scope(),
    )


@pytest.mark.asyncio
class TestEchoGuard:
    async def test_flag_off_assistant_supersession_proceeds(self, monkeypatch):
        """Today's behavior is untouched with the flag unset."""
        monkeypatch.delenv("ENABLE_ECHO_GUARD", raising=False)
        result = await _run(
            _mem(ECHO_OLD_EMAIL, assertion_source="assistant"),
            _vector_with_hit(USER_EMAIL_FACT, existing_source="user"),
            _llm_classifier("SUPERSEDE"),
        )
        assert len(result.supersessions) == 1
        assert len(result.skips) == 0

    async def test_assistant_echo_blocked_from_superseding_user(self, monkeypatch):
        """The eval's killer edge: assistant echo of a stale email must not
        retire the user's newer statement. Becomes a NONE drop with an
        echo-guard reason."""
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        result = await _run(
            _mem(ECHO_OLD_EMAIL, assertion_source="assistant"),
            _vector_with_hit(USER_EMAIL_FACT, existing_source="user"),
            _llm_classifier("SUPERSEDE"),
        )
        assert len(result.supersessions) == 0
        assert len(result.skips) == 1
        assert "echo-guard" in result.skips[0].reason
        assert result.skips[0].existing_memory_content == USER_EMAIL_FACT

    async def test_assistant_echo_blocked_against_unattributed_legacy(self, monkeypatch):
        """Legacy rows (assertion_source NULL) get the same protection."""
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        result = await _run(
            _mem(ECHO_OLD_EMAIL, assertion_source="assistant"),
            _vector_with_hit(USER_EMAIL_FACT, existing_source=None),
            _llm_classifier("SUPERSEDE"),
        )
        assert len(result.supersessions) == 0
        assert len(result.skips) == 1

    async def test_same_value_dedup_still_allowed(self, monkeypatch):
        """Assistant restating the SAME value is a plain duplicate — dedup
        supersession stays legal."""
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        result = await _run(
            _mem(ECHO_SAME_EMAIL, assertion_source="assistant"),
            _vector_with_hit(USER_EMAIL_FACT, existing_source="user"),
            _llm_classifier("SUPERSEDE"),
        )
        assert len(result.supersessions) == 1
        assert len(result.skips) == 0

    async def test_user_sourced_new_memory_unaffected(self, monkeypatch):
        """User updates keep last-write-wins exactly as today."""
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        result = await _run(
            _mem(USER_EMAIL_FACT, assertion_source="user"),
            _vector_with_hit(ECHO_OLD_EMAIL, existing_source="assistant"),
            _llm_classifier("SUPERSEDE"),
        )
        assert len(result.supersessions) == 1

    async def test_assistant_to_assistant_supersession_allowed(self, monkeypatch):
        """Agents may update their own derived state."""
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        result = await _run(
            _mem(ECHO_OLD_EMAIL, assertion_source="assistant"),
            _vector_with_hit(ECHO_SAME_EMAIL, existing_source="assistant"),
            _llm_classifier("SUPERSEDE"),
        )
        assert len(result.supersessions) == 1

    async def test_unattributed_new_memory_unaffected(self, monkeypatch):
        """Attribution flag off upstream (assertion_source None on the new
        memory) means the guard never fires — flag interlock."""
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        result = await _run(
            _mem(ECHO_OLD_EMAIL, assertion_source=None),
            _vector_with_hit(USER_EMAIL_FACT, existing_source="user"),
            _llm_classifier("SUPERSEDE"),
        )
        assert len(result.supersessions) == 1


@pytest.mark.asyncio
class TestProvenanceAddendum:
    async def test_prompt_carries_provenance_when_on(self, monkeypatch):
        monkeypatch.setenv("ENABLE_ECHO_GUARD", "true")
        llm = _llm_classifier("UNRELATED")
        await _run(
            _mem(ECHO_OLD_EMAIL, assertion_source="assistant"),
            _vector_with_hit(USER_EMAIL_FACT, existing_source="user"),
            llm,
        )
        messages = llm.complete.call_args.kwargs["messages"]
        system = messages[0]["content"]
        user = messages[1]["content"]
        assert "Speaker provenance" in system
        assert "asserted by user" in user
        assert "asserted by assistant" in user

    async def test_prompt_unchanged_when_off(self, monkeypatch):
        monkeypatch.delenv("ENABLE_ECHO_GUARD", raising=False)
        llm = _llm_classifier("UNRELATED")
        await _run(
            _mem(ECHO_OLD_EMAIL, assertion_source="assistant"),
            _vector_with_hit(USER_EMAIL_FACT, existing_source="user"),
            llm,
        )
        messages = llm.complete.call_args.kwargs["messages"]
        assert "Speaker provenance" not in messages[0]["content"]
        assert "Provenance:" not in messages[1]["content"]


class TestValueTokens:
    def test_emails_extracted_and_normalized(self):
        assert _extract_value_tokens("email is Casey.New@Example.com") == {"casey.new@example.com"}

    def test_keyword_email_is_not_a_value(self):
        assert _extract_value_tokens("the user prefers email for updates") == frozenset()

    def test_phone_and_card_normalization(self):
        toks = _extract_value_tokens("call +1 (555) 010-2233, card ending 4242 1111")
        assert "+15550102233" in toks

    def test_different_emails_differ(self):
        a = _extract_value_tokens("updates go to a@x.com")
        b = _extract_value_tokens("updates go to b@x.com")
        assert a and b and a != b

    def test_same_value_different_phrasing_matches(self):
        a = _extract_value_tokens("The user's email address is casey.new@example.com")
        b = _extract_value_tokens("Updates for the user go to casey.new@example.com")
        assert a == b
