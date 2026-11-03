"""DELETE subject guard + retraction sweep (ENABLE_KU_DELETE_SUBJECT_GUARD).

Mis-target case: the user asked to
delete their home address; the retraction recall floor (0.45) paired the
retraction with "The user lives in Texas" (pure vector similarity, no subject
check), the classifier answered DELETE, and execute_deletions deprecated the
Texas fact while "The user's home address is 123 Main St, Springfield" stayed
current. Two fixes, one flag:

  (a) guard — a DELETE verdict only executes when the paired memory matches
      the retraction's subject (value token / subject-token / attribute-class
      overlap);
  (b) sweep — the confirmed retraction retires ALL current memories matching
      its subject via ILIKE, so duplicate copies die too, not just the one
      vector-paired row.

Flag off = byte-identical legacy behavior (pinned below).
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
from synap.cloud.agents.ingestion.pipeline import knowledge_update as ku
from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    KnowledgeUpdateDetector,
    RetractionSweep,
)
from synap.cloud.agents.ingestion.storage.coordinator import StorageCoordinator
from synap.tests.agents.ingestion._retire_shim import attach_retire_shim

RETRACTION = "The user prefers to have their home address deleted from the system"
MISMATCH_TARGET = "The user lives in Texas"
CORRECT_TARGET = "The user's home address is 123 Main St, Springfield"


def _scope() -> Scope:
    return Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )


def _mem(content: str, mtype: MemoryType = MemoryType.PREFERENCES) -> ExtractedMemory:
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
# 1. Helpers: subject tokens + subject match
# ---------------------------------------------------------------------------

class TestSubjectHelpers:
    def test_subject_tokens_strip_scaffolding(self):
        assert ku._subject_tokens(RETRACTION) == frozenset({"home", "address"})

    def test_subject_tokens_of_unrelated_fact(self):
        assert ku._subject_tokens(MISMATCH_TARGET) == frozenset({"lives", "texas"})

    def test_prod_mismatch_pair_does_not_match(self):
        assert not ku._delete_subject_match(RETRACTION, MISMATCH_TARGET)

    def test_correct_pair_matches_on_subject_tokens(self):
        assert ku._delete_subject_match(RETRACTION, CORRECT_TARGET)

    def test_value_token_match(self):
        assert ku._delete_subject_match(
            "The user asked to delete casey.old@example.com from their account",
            "Receipts are sent to casey.old@example.com",
        )

    def test_plural_prefix_overlap(self):
        assert ku._delete_subject_match(
            "The user wants their saved addresses removed",
            "The user's saved address is 12 Oak St",
        )

    def test_attribute_class_match_without_shared_tokens(self):
        # "card" class on both sides even though the subject nouns differ.
        assert ku._delete_subject_match(
            "The user requested the deletion of their card",
            "Payment method on file: Visa card ending 4242",
        )

    def test_no_signal_retraction_passes_through(self):
        # Nothing extractable on the retraction side -> guard cannot judge,
        # classifier verdict stands (precision-only guard).
        assert ku._delete_subject_match(
            "The user asked to delete this from the records",
            MISMATCH_TARGET,
        )


# ---------------------------------------------------------------------------
# 2. Flag OFF = legacy behavior, byte-identical (regression pin)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestFlagOffLegacy:
    async def test_mismatched_delete_still_executes_when_guard_off(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        monkeypatch.delenv("ENABLE_KU_DELETE_SUBJECT_GUARD", raising=False)
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(MISMATCH_TARGET, score=0.55),
            llm_client=_llm_classifier("DELETE"),
        )
        result = await detector.detect_updates(
            memories=[_mem(RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert len(result.deletions) == 1
        assert result.deletions[0].old_memory_content == MISMATCH_TARGET
        assert result.retraction_sweeps == []


# ---------------------------------------------------------------------------
# 3. Flag ON: guard blocks mis-targets, sweep entries are emitted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGuardOn:
    async def test_prod_mismatch_blocked_and_sweep_recorded(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(MISMATCH_TARGET, score=0.55),
            llm_client=_llm_classifier("DELETE"),
        )
        result = await detector.detect_updates(
            memories=[_mem(RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.deletions == []  # Texas fact survives
        assert len(result.retraction_sweeps) == 1
        sweep = result.retraction_sweeps[0]
        assert set(sweep.subject_tokens) == {"home", "address"}
        assert sweep.value_tokens == ()

    async def test_correct_pair_still_deleted_and_sweep_recorded(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(CORRECT_TARGET, score=0.55),
            llm_client=_llm_classifier("DELETE"),
        )
        result = await detector.detect_updates(
            memories=[_mem(RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert len(result.deletions) == 1
        assert result.deletions[0].old_memory_content == CORRECT_TARGET
        assert len(result.retraction_sweeps) == 1

    async def test_weak_subject_no_sweep_entry(self, monkeypatch):
        """One generic subject token and no value: too imprecise to ILIKE-sweep
        (would mass-match). Guard still applies to the pair."""
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(MISMATCH_TARGET, score=0.55),
            llm_client=_llm_classifier("DELETE"),
        )
        result = await detector.detect_updates(
            memories=[_mem("The user asked to delete their email")],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.deletions == []
        assert result.retraction_sweeps == []

    async def test_non_delete_verdicts_unaffected(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit("The user's email is a@b.example", score=0.9),
            llm_client=_llm_classifier("SUPERSEDE"),
        )
        result = await detector.detect_updates(
            memories=[_mem("The user's email is c@d.example", MemoryType.FACTS)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert len(result.supersessions) == 1
        assert result.retraction_sweeps == []


# ---------------------------------------------------------------------------
# 4. Coordinator sweep execution
# ---------------------------------------------------------------------------

def _coordinator(rows):
    rdbms = MagicMock()
    rdbms.current_memories_matching = AsyncMock(return_value=rows)
    rdbms.set_deprecated_at = AsyncMock(return_value=True)
    rdbms.insert_memory_action = AsyncMock()
    vector = MagicMock()
    vector.delete_memory_vectors = AsyncMock()
    coord = StorageCoordinator.__new__(StorageCoordinator)
    coord._rdbms = rdbms
    attach_retire_shim(rdbms)
    coord._vector = vector
    return coord, rdbms, vector


def _sweep(subject=("address", "home"), values=()):
    return RetractionSweep(
        new_memory_id=str(uuid4()),
        new_memory_content=RETRACTION,
        subject_tokens=tuple(subject),
        value_tokens=tuple(values),
        reason="test stub",
    )


@pytest.mark.asyncio
class TestExecuteRetractionSweeps:
    async def test_flag_off_is_noop(self, monkeypatch):
        monkeypatch.delenv("ENABLE_KU_DELETE_SUBJECT_GUARD", raising=False)
        coord, rdbms, _ = _coordinator(rows=[{"id": "x", "content": CORRECT_TARGET,
                                              "memory_type": "facts"}])
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep()], scope=_scope(),
        )
        assert swept == set()
        rdbms.current_memories_matching.assert_not_awaited()

    async def test_sweeps_matching_current_rows(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        rid = str(uuid4())
        coord, rdbms, vector = _coordinator(rows=[
            {"id": rid, "content": CORRECT_TARGET, "memory_type": "facts"},
        ])
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep()], scope=_scope(),
            http_correlation_id="corr-1",
        )
        assert swept == {rid}
        rdbms.set_deprecated_at.assert_awaited_once_with(memory_id=rid)
        vector.delete_memory_vectors.assert_awaited_once()
        action = rdbms.insert_memory_action.call_args.kwargs
        assert action["action"] == "DELETE"
        assert action["memory_id"] == rid
        assert action["previous_content"] == CORRECT_TARGET
        assert "delete-sweep" in action["reason"]

    async def test_retraction_intent_rows_never_swept(self, monkeypatch):
        """The stored retraction request itself contains the subject tokens —
        it must survive as the audit record of the user's intent."""
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        coord, rdbms, _ = _coordinator(rows=[
            {"id": str(uuid4()), "content": RETRACTION, "memory_type": "preferences"},
        ])
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep()], scope=_scope(),
        )
        assert swept == set()
        rdbms.set_deprecated_at.assert_not_awaited()

    async def test_value_tokens_preferred_over_subject(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        coord, rdbms, _ = _coordinator(rows=[])
        await coord.execute_retraction_sweeps(
            sweeps=[_sweep(values=("casey.old@example.com",))], scope=_scope(),
        )
        kwargs = rdbms.current_memories_matching.call_args.kwargs
        assert kwargs["content_tokens"] == ["casey.old@example.com"]

    async def test_batch_ids_excluded(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        coord, rdbms, _ = _coordinator(rows=[])
        await coord.execute_retraction_sweeps(
            sweeps=[_sweep()], scope=_scope(),
            exclude_memory_ids={"batch-1", "batch-2"},
        )
        excluded = set(rdbms.current_memories_matching.call_args.kwargs["exclude_memory_ids"])
        assert {"batch-1", "batch-2"} <= excluded

    async def test_lookup_failure_leaves_rows_live(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        coord, rdbms, _ = _coordinator(rows=[])
        rdbms.current_memories_matching = AsyncMock(side_effect=RuntimeError("db down"))
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep()], scope=_scope(),
        )
        assert swept == set()
        rdbms.set_deprecated_at.assert_not_awaited()

    async def test_scope_without_user_is_noop(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        coord, rdbms, _ = _coordinator(rows=[])
        scope = MagicMock()
        scope.user_id = None
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep()], scope=scope,
        )
        assert swept == set()
        rdbms.current_memories_matching.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. Wiring + SQL pins (inspect.getsource, style of test_ku_echo_resurrection)
# ---------------------------------------------------------------------------

class TestWiring:
    def test_stage4_sweep_runs_after_deletions_before_echo_sweep(self):
        import inspect
        from synap.cloud.agents.ingestion.pipeline import stage4_organization
        src = inspect.getsource(stage4_organization)
        assert "execute_retraction_sweeps" in src
        assert src.index("execute_deletions") < src.index("execute_retraction_sweeps")
        assert src.index("execute_retraction_sweeps") < src.index(
            "suppress_retired_value_echoes"
        )

    def test_sqlalchemy_lookup_is_current_rows_only(self):
        import inspect
        from synap.cloud.agents.ingestion.storage import sqlalchemy_rdbms_wrapper
        src = inspect.getsource(
            sqlalchemy_rdbms_wrapper.SQLAlchemyRDBMSWrapper.current_memories_matching
        )
        assert "superseded_by IS NULL" in src
        assert "deprecated_at IS NULL" in src
        assert "ILIKE" in src

    def test_delete_branch_guard_present(self):
        import inspect
        src = inspect.getsource(ku.KnowledgeUpdateDetector)
        assert "_delete_subject_match" in src
        assert "sweeps_by_new_id" in src
