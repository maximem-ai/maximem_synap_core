"""Gap 5/6 recall-admission fixes — deterministic guards, no LLM/DB.

Companion to test_ku_delete_and_threshold.py. That file covers the flag/knob
plumbing PR #881 added; this one covers the three defects that kept the
features from ever firing end-to-end:

1. `_classify_pair` rejected a "DELETE" answer as an invalid classification
   (the whitelist was never extended), so the DELETE branch in the pair loop
   was unreachable even with ENABLE_KU_DELETE on.
2. A retraction memory references its target instead of restating it, so the
   pair sits at ~0.50-0.55 cosine (measured, arctic-l-v2.0) — below any usable
   global gate. Retraction-shaped new memories now admit candidates at a
   dedicated floor (default 0.45) when the delete flag is on.
3. High-entropy value updates (email/address) land at 0.71-0.78 — below the
   0.82 gate — so last-write-wins never ran for them. With
   ENABLE_KU_VALUE_AWARE_RECALL on, sub-gate pairs are admitted when both
   sides share an attribute class (email/phone/address/payment/url).

All admission changes are recall-only: every admitted pair still goes through
the LLM classifier, which remains the precision layer.

⚠ 2026-10-03: the main gate moved 0.82 -> 0.55 (see SIMILARITY_THRESHOLD). These
two features were built as narrow doors through a gate that was too high, and
that reading is now confirmed globally rather than per attribute class. Note what
it means for them:

  - value-aware recall is now INERT, not merely redundant. It was built for
    email/address pairs measured at 0.71-0.78, all of which now clear the main
    gate on their own. Worse, VALUE_RECALL_FLOOR is itself 0.55 and
    _value_recall_floor never returns a value above the gate, so its admission
    band is [0.55, 0.55) — empty. It admits nothing the gate does not already
    admit. The flag has always defaulted OFF, so nothing changes by default;
    this is dead wiring to remove or re-tune, not a regression. Left in place
    deliberately: deleting it is a separate decision from moving the gate.
  - retraction recall still matters: its floor is 0.45, below the new gate, and
    retraction pairs were measured at ~0.50-0.55, which straddles it.

The tests below assert that a SUB-GATE pair needs its feature's door. That intent
is unchanged; the scores moved under the new gate so they still test it. A test
that used 0.55 or 0.78 was testing the door while the pair now walks through the
gate.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory,
    MemoryDestination,
    MemoryType,
    Scope,
)
from synap.cloud.agents.ingestion.pipeline import knowledge_update as ku
from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    KnowledgeUpdateDetector,
)


# asyncio marks are applied per-class (not module-wide) because the floor,
# lexicon, and SQL-inspection tests are plain sync functions.


def _scope() -> Scope:
    return Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )


def _mem(content: str, mtype: MemoryType = MemoryType.FACTS) -> ExtractedMemory:
    return ExtractedMemory(
        memory_id=uuid4(),
        memory_type=mtype,
        content=content,
        confidence=0.9,
        destination=MemoryDestination.MEMORIES,
        source_document_id="doc-x",
        entities_mentioned=[],
    )


def _vector_with_hit(existing_content: str, score: float):
    vec = MagicMock()
    vec.batch_search = AsyncMock(return_value=[[
        {"id": str(uuid4()), "content": existing_content, "score": score},
    ]])
    return vec


def _llm_classifier(returns_classification: str):
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=json.dumps({
        "classification": returns_classification,
        "reason": "test stub",
    }))
    return llm


# ---------------------------------------------------------------------------
# 1. DELETE must survive _classify_pair's verdict whitelist (flag-gated)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDeleteVerdictWhitelist:
    async def test_delete_verdict_accepted_when_flag_on(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        detector = KnowledgeUpdateDetector(
            vector=MagicMock(), llm_client=_llm_classifier("DELETE"),
        )
        result = await detector._classify_pair(
            existing_content="the user's home address is 12 Oak St",
            new_content="the user asked to delete their saved home address",
            scope=_scope(),
        )
        assert result is not None
        assert result["classification"] == "DELETE"

    async def test_delete_verdict_rejected_when_flag_off(self, monkeypatch):
        monkeypatch.delenv("ENABLE_KU_DELETE", raising=False)
        detector = KnowledgeUpdateDetector(
            vector=MagicMock(), llm_client=_llm_classifier("DELETE"),
        )
        result = await detector._classify_pair(
            existing_content="the user's home address is 12 Oak St",
            new_content="the user asked to delete their saved home address",
            scope=_scope(),
        )
        assert result is None


# ---------------------------------------------------------------------------
# 2. Retraction recall floor (rides ENABLE_KU_DELETE, no extra flag)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRetractionRecall:
    RETRACTION = "The user asked not to keep their payment card on file"
    TARGET = "The user's saved payment card ends in 4242"

    async def test_retraction_pair_admitted_below_gate(self, monkeypatch):
        """Measured shape: retraction vs target = ~0.55. With the delete flag
        on, the pair must reach the classifier and produce a DeletionAction."""
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(self.TARGET, score=0.55),
            llm_client=_llm_classifier("DELETE"),
        )
        result = await detector.detect_updates(
            memories=[_mem(self.RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert len(result.deletions) == 1
        assert result.deletions[0].old_memory_content == self.TARGET

    async def test_non_retraction_memory_keeps_main_gate(self, monkeypatch):
        """A plain fact BELOW the main gate must NOT be admitted just because
        the delete flag is on — the lower floor applies only to
        retraction-shaped text. 0.50 is under the 0.55 gate; it was 0.55 when
        the gate was 0.82."""
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        llm = _llm_classifier("UNRELATED")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(self.TARGET, score=0.50), llm_client=llm,
        )
        result = await detector.detect_updates(
            memories=[_mem("The user enjoys weekend hiking in Marin")],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.deletions == []
        llm.complete.assert_not_called()

    async def test_retraction_below_floor_not_admitted(self, monkeypatch):
        """Noise territory (measured retraction-vs-unrelated ~0.27-0.30) stays
        excluded even for retraction-shaped memories."""
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        llm = _llm_classifier("DELETE")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit("The user goes hiking every weekend", score=0.30),
            llm_client=llm,
        )
        result = await detector.detect_updates(
            memories=[_mem(self.RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.deletions == []
        llm.complete.assert_not_called()

    async def test_flag_off_means_no_lower_gate(self, monkeypatch):
        """0.50 sits between the retraction floor (0.45) and the main gate
        (0.55), so only the flag can admit it."""
        monkeypatch.delenv("ENABLE_KU_DELETE", raising=False)
        llm = _llm_classifier("DELETE")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(self.TARGET, score=0.50), llm_client=llm,
        )
        result = await detector.detect_updates(
            memories=[_mem(self.RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.deletions == []
        llm.complete.assert_not_called()

    def test_floor_default_and_clamp(self, monkeypatch):
        monkeypatch.delenv("SYNAP_KU_RETRACTION_RECALL_FLOOR", raising=False)
        assert ku._retraction_recall_floor(0.82) == ku.RETRACTION_RECALL_FLOOR == 0.45
        monkeypatch.setenv("SYNAP_KU_RETRACTION_RECALL_FLOOR", "0.6")
        assert ku._retraction_recall_floor(0.82) == 0.6
        monkeypatch.setenv("SYNAP_KU_RETRACTION_RECALL_FLOOR", "0.05")
        assert ku._retraction_recall_floor(0.82) == 0.30  # floor of the floor
        monkeypatch.setenv("SYNAP_KU_RETRACTION_RECALL_FLOOR", "0.95")
        assert ku._retraction_recall_floor(0.82) == 0.82  # never above the gate
        monkeypatch.setenv("SYNAP_KU_RETRACTION_RECALL_FLOOR", "garbage")
        assert ku._retraction_recall_floor(0.82) == 0.45

    def test_retraction_intent_lexicon(self):
        positives = [
            "The user asked not to keep their payment card on file",
            "The user asked to delete their shipping address",
            "The user asked to forget their email address",
            "The user wants their phone number removed",
            "The user does not want their card details stored",
            "The user no longer wants their address kept on file",
            "The user asked to stop storing their location",
            "The user retracted their earlier statement about the dosage",
            # v2 — shapes the extractor actually produced in end-to-end runs
            # (it normalizes retraction requests into preference statements):
            "The user prefers not to have their payment card kept on file",
            "The user doesn't want their email address stored",
            "The user requested the removal of their shipping address",
            "The user would like their phone number gone from the system",
        ]
        negatives = [
            "The user's saved payment card ends in 4242",
            "The user's email address is john@example.com",
            "The user removed the old couch from their living room",
            "The user keeps a journal on their desk",
            "The user wants a new phone for their birthday",
            "The user saved money for a vacation to Portugal",
        ]
        for text in positives:
            assert ku._is_retraction_request(text), text
        for text in negatives:
            assert not ku._is_retraction_request(text), text


# ---------------------------------------------------------------------------
# 3. Value-aware recall (ENABLE_KU_VALUE_AWARE_RECALL, default OFF)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestValueAwareRecall:
    OLD_EMAIL = "The user's email address is john.smith1985@example.com"
    NEW_EMAIL = "The user's email address is j.smith.work@example.org"

    async def test_email_update_admitted_when_flag_on(self, monkeypatch):
        """Measured shape: distinct-email update = ~0.78, below the 0.82 gate.
        Shared 'email' class must admit the pair so SUPERSEDE can run."""
        monkeypatch.setenv("ENABLE_KU_VALUE_AWARE_RECALL", "true")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(self.OLD_EMAIL, score=0.78),
            llm_client=_llm_classifier("SUPERSEDE"),
        )
        result = await detector.detect_updates(
            memories=[_mem(self.NEW_EMAIL)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert len(result.supersessions) == 1
        assert result.supersessions[0].old_memory_content == self.OLD_EMAIL

    async def test_default_off_preserves_the_main_gate(self, monkeypatch):
        """A sub-gate email pair still needs the flag.

        Was 0.78, which is where real email pairs measure. With the gate at 0.55
        that score is admitted by the gate itself, which is the point of the
        change and the reason this feature is now redundant for its own
        motivating case. 0.50 is what still depends on the flag.
        """
        monkeypatch.delenv("ENABLE_KU_VALUE_AWARE_RECALL", raising=False)
        llm = _llm_classifier("SUPERSEDE")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(self.OLD_EMAIL, score=0.50), llm_client=llm,
        )
        result = await detector.detect_updates(
            memories=[_mem(self.NEW_EMAIL)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.supersessions == []
        llm.complete.assert_not_called()

    async def test_no_shared_class_not_admitted(self, monkeypatch):
        """Sub-gate similarity without a shared attribute class stays out."""
        monkeypatch.setenv("ENABLE_KU_VALUE_AWARE_RECALL", "true")
        llm = _llm_classifier("SUPERSEDE")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(
                "The user goes hiking every weekend in the Marin Headlands",
                score=0.50,
            ),
            llm_client=llm,
        )
        result = await detector.detect_updates(
            memories=[_mem(self.NEW_EMAIL)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.supersessions == []
        llm.complete.assert_not_called()

    async def test_shared_class_below_floor_not_admitted(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_VALUE_AWARE_RECALL", "true")
        llm = _llm_classifier("SUPERSEDE")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(self.OLD_EMAIL, score=0.50),  # < 0.55 floor
            llm_client=llm,
        )
        result = await detector.detect_updates(
            memories=[_mem(self.NEW_EMAIL)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.supersessions == []
        llm.complete.assert_not_called()

    async def test_above_gate_pairs_unaffected_by_flag(self, monkeypatch):
        """Sanity: the main gate is untouched — a 0.9 pair classifies with the
        flag off exactly as before."""
        monkeypatch.delenv("ENABLE_KU_VALUE_AWARE_RECALL", raising=False)
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(self.OLD_EMAIL, score=0.90),
            llm_client=_llm_classifier("SUPERSEDE"),
        )
        result = await detector.detect_updates(
            memories=[_mem(self.NEW_EMAIL)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert len(result.supersessions) == 1

    def test_floor_default_and_clamp(self, monkeypatch):
        monkeypatch.delenv("SYNAP_KU_VALUE_RECALL_FLOOR", raising=False)
        assert ku._value_recall_floor(0.82) == ku.VALUE_RECALL_FLOOR == 0.55
        monkeypatch.setenv("SYNAP_KU_VALUE_RECALL_FLOOR", "0.7")
        assert ku._value_recall_floor(0.82) == 0.7
        monkeypatch.setenv("SYNAP_KU_VALUE_RECALL_FLOOR", "0.2")
        assert ku._value_recall_floor(0.82) == 0.40
        monkeypatch.setenv("SYNAP_KU_VALUE_RECALL_FLOOR", "0.95")
        assert ku._value_recall_floor(0.82) == 0.82

    def test_attribute_classes(self):
        cases = {
            "The user's email address is a@b.com": {"email"},
            "Reach the user at their e-mail for updates": {"email"},
            "The user's phone number is 415-555-0134": {"phone"},
            "The user's shipping address is 221B Baker Street": {"address"},
            "The user's saved payment card ends in 4242": {"payment"},
            "The user's dashboard is https://grafana.example.com/d/x": {"url"},
            "The user goes hiking every weekend": set(),
        }
        for text, expected in cases.items():
            assert ku._attribute_classes(text) == frozenset(expected), text


# ---------------------------------------------------------------------------
# 4. Read path: an executed DELETE must stop being served by the PG sub-path
# ---------------------------------------------------------------------------

class TestDeprecatedAtReadFilter:
    def test_postgres_memory_queries_filter_deprecated_at(self):
        """execute_deletions sets only deprecated_at (no successor exists to
        point superseded_by at), so every memories query that filters
        superseded_by must also filter deprecated_at."""
        from synap.cloud.application.client_api import retrieval_manager
        source = Path(retrieval_manager.__file__).read_text()
        superseded = source.count("AND superseded_by IS NULL")
        deprecated = source.count("AND deprecated_at IS NULL")
        assert superseded > 0
        assert deprecated == superseded
