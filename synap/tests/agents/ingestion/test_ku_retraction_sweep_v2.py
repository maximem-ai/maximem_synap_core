"""Retraction sweep v2 (SYNAP_KU_RETRACTION_SWEEP_V2).

Miss case: the user said "don't keep my card on file after this
transaction". The extractor normalized that to "The user prefers not to have
their payment card kept on file after the current transaction", so the v1
sweep derived subject tokens {payment, card, current, after, transaction} and
asked Postgres for rows containing ALL of them. The stored row reads "The
user's card ends in 4242" — zero matches, the card was never retired.

v2, one flag, two parts:
  (a) the conditional tail ("after the current transaction") is cut before
      tokenizing, so the subject is {payment, card};
  (b) rows sharing the retraction's ATTRIBUTE CLASS also qualify, confirmed in
      Python with the real regexes, with email/address precedence so an
      address retraction can never take an "email address is x@y" row.

Flag off = byte-identical v1 behavior (pinned below).
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

CARD_RETRACTION = (
    "The user prefers not to have their payment card kept on file "
    "after the current transaction"
)
CARD_ROW = "The user's card ends in 4242"
ADDRESS_RETRACTION = "The user prefers to have their home address deleted from the system"
ADDRESS_ROW = "The user's home address is 123 Main St, Springfield"
EMAIL_ROW = "The user's email address is casey@example.com"


def _scope() -> Scope:
    return Scope(
        client_id=uuid4(), instance_id=uuid4(),
        customer_id=uuid4(), user_id=uuid4(),
    )


def _mem(content: str, source_span: str = None) -> ExtractedMemory:
    return ExtractedMemory(
        memory_id=uuid4(),
        memory_type=MemoryType.PREFERENCES,
        content=content,
        confidence=0.9,
        destination=MemoryDestination.MEMORIES,
        source_document_id="doc-x",
        entities_mentioned=[],
        source_span=source_span,
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


def _coordinator(token_rows, class_rows=None):
    """Coordinator whose ALL-token lookup returns ``token_rows`` and whose v2
    class prefilter returns ``class_rows``."""
    rdbms = MagicMock()
    rdbms.current_memories_matching = AsyncMock(return_value=token_rows)
    rdbms.current_memories_matching_any = AsyncMock(
        return_value=class_rows if class_rows is not None else []
    )
    rdbms.set_deprecated_at = AsyncMock(return_value=True)
    rdbms.insert_memory_action = AsyncMock()
    vector = MagicMock()
    vector.delete_memory_vectors = AsyncMock()
    coord = StorageCoordinator.__new__(StorageCoordinator)
    coord._rdbms = rdbms
    attach_retire_shim(rdbms)
    coord._vector = vector
    return coord, rdbms, vector


def _row(content, mtype="facts"):
    return {"id": str(uuid4()), "content": content, "memory_type": mtype}


def _sweep(content, subject, values=()):
    return RetractionSweep(
        new_memory_id=str(uuid4()),
        new_memory_content=content,
        subject_tokens=tuple(subject),
        value_tokens=tuple(values),
        reason="test stub",
    )


# ---------------------------------------------------------------------------
# 1. Helpers: conditional-tail truncation + class matching
# ---------------------------------------------------------------------------

class TestConditionalTail:
    def test_card_tail_truncated(self):
        assert ku._truncate_conditional_tail(CARD_RETRACTION) == (
            "The user prefers not to have their payment card kept on file"
        )

    @pytest.mark.parametrize("connective", [
        "after", "once", "until", "when", "as soon as", "following",
    ])
    def test_every_connective_cuts(self, connective):
        text = f"The user wants their card removed {connective} the order ships"
        assert ku._truncate_conditional_tail(text) == (
            "The user wants their card removed"
        )

    def test_no_connective_is_identity(self):
        assert ku._truncate_conditional_tail(ADDRESS_RETRACTION) == ADDRESS_RETRACTION

    def test_empty_is_safe(self):
        assert ku._truncate_conditional_tail("") == ""
        assert ku._truncate_conditional_tail(None) == ""

    def test_tokens_before_and_after_truncation(self):
        assert ku._subject_tokens(CARD_RETRACTION) == frozenset(
            {"payment", "card", "current", "after", "transaction"}
        )
        assert ku._subject_tokens(
            ku._truncate_conditional_tail(CARD_RETRACTION)
        ) == frozenset({"payment", "card"})


class TestClassMatching:
    def test_card_row_shares_payment_class(self):
        classes = ku._attribute_classes(CARD_RETRACTION)
        assert classes == frozenset({"payment"})
        assert ku._retraction_class_match(classes, CARD_ROW)

    def test_address_row_shares_address_class(self):
        classes = ku._attribute_classes(ADDRESS_RETRACTION)
        assert classes == frozenset({"address"})
        assert ku._retraction_class_match(classes, ADDRESS_ROW)

    def test_address_retraction_does_not_match_email_row(self):
        classes = ku._attribute_classes(ADDRESS_RETRACTION)
        assert not ku._retraction_class_match(classes, EMAIL_ROW)

    def test_email_precedence_blocks_dual_class_row(self):
        # A row that IS postal-classed but also carries an email must not be
        # swept by an address-only retraction.
        row = "Mail for the user goes to the address casey@example.com"
        classes = ku._attribute_classes(ADDRESS_RETRACTION)
        assert "address" in ku._attribute_classes(row)
        assert "email" in ku._attribute_classes(row)
        assert not ku._retraction_class_match(classes, row)

    def test_email_retraction_still_reaches_email_row(self):
        classes = ku._attribute_classes(
            "The user asked to have their email address forgotten"
        )
        assert ku._retraction_class_match(classes, EMAIL_ROW)

    def test_unrelated_row_never_matches(self):
        classes = ku._attribute_classes(CARD_RETRACTION)
        assert not ku._retraction_class_match(classes, "The user lives in Texas")

    def test_no_class_retraction_matches_nothing(self):
        assert not ku._retraction_class_match(frozenset(), CARD_ROW)

    def test_sql_keywords_are_deduped_and_ordered(self):
        assert ku._class_sql_keywords({"payment"}) == ["card", "payment"]
        assert ku._class_sql_keywords({"address", "payment"}) == [
            "address", "street", "card", "payment",
        ]
        assert ku._class_sql_keywords(frozenset()) == []


# ---------------------------------------------------------------------------
# 2. Detector: sweep subject tokens (flag off = legacy tail-polluted tokens)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDetectorSubjectTokens:
    async def _sweep_for(self, monkeypatch, v2: bool):
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        if v2:
            monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        else:
            monkeypatch.delenv("SYNAP_KU_RETRACTION_SWEEP_V2", raising=False)
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(CARD_ROW, score=0.55),
            llm_client=_llm_classifier("DELETE"),
        )
        result = await detector.detect_updates(
            memories=[_mem(CARD_RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert len(result.retraction_sweeps) == 1
        return result.retraction_sweeps[0]

    async def test_flag_off_keeps_conditional_tail_tokens(self, monkeypatch):
        sweep = await self._sweep_for(monkeypatch, v2=False)
        assert set(sweep.subject_tokens) == {
            "payment", "card", "current", "after", "transaction",
        }

    async def test_flag_on_truncates_tail(self, monkeypatch):
        sweep = await self._sweep_for(monkeypatch, v2=True)
        assert set(sweep.subject_tokens) == {"payment", "card"}

    async def test_source_span_survives_a_tail_in_the_extraction(self, monkeypatch):
        """Truncation is per part, so a qualifier in the extracted text cannot
        swallow the raw utterance appended behind it."""
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(CARD_ROW, score=0.55),
            llm_client=_llm_classifier("DELETE"),
        )
        result = await detector.detect_updates(
            memories=[_mem(
                "The user wants their data removed after the trip",
                source_span="please forget my home address",
            )],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        tokens = set(result.retraction_sweeps[0].subject_tokens)
        assert {"home", "address"} <= tokens
        assert "trip" not in tokens


# ---------------------------------------------------------------------------
# 3. Coordinator: class-widened candidate fetch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCoordinatorSweepV2:
    async def test_card_case_flag_off_sweeps_nothing(self, monkeypatch):
        """Regression pin: v1 finds no ALL-token match and never looks further."""
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.delenv("SYNAP_KU_RETRACTION_SWEEP_V2", raising=False)
        coord, rdbms, _ = _coordinator(token_rows=[], class_rows=[_row(CARD_ROW)])
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep(CARD_RETRACTION, ("card", "payment"))],
            scope=_scope(),
        )
        assert swept == set()
        rdbms.current_memories_matching_any.assert_not_awaited()
        rdbms.set_deprecated_at.assert_not_awaited()

    async def test_card_case_flag_on_sweeps_the_card_row(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        card_row = _row(CARD_ROW)
        coord, rdbms, vector = _coordinator(token_rows=[], class_rows=[card_row])
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep(CARD_RETRACTION, ("card", "payment"))],
            scope=_scope(),
            http_correlation_id="corr-1",
        )
        assert swept == {card_row["id"]}
        rdbms.set_deprecated_at.assert_awaited_once_with(memory_id=card_row["id"])
        vector.delete_memory_vectors.assert_awaited_once()
        assert rdbms.current_memories_matching_any.call_args.kwargs[
            "content_keywords"
        ] == ["card", "payment"]
        action = rdbms.insert_memory_action.call_args.kwargs
        assert action["action"] == "DELETE"
        assert action["previous_content"] == CARD_ROW

    async def test_address_case_still_works_via_token_path(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        addr_row = _row(ADDRESS_ROW)
        coord, rdbms, _ = _coordinator(token_rows=[addr_row], class_rows=[])
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep(ADDRESS_RETRACTION, ("address", "home"))],
            scope=_scope(),
        )
        assert swept == {addr_row["id"]}
        # Value/subject-token path keeps its priority and runs first.
        assert rdbms.current_memories_matching.call_args.kwargs[
            "content_tokens"
        ] == ["address", "home"]

    async def test_email_row_not_swept_by_address_retraction(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        addr_row, email_row = _row(ADDRESS_ROW), _row(EMAIL_ROW)
        coord, rdbms, _ = _coordinator(
            token_rows=[], class_rows=[addr_row, email_row],
        )
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep(ADDRESS_RETRACTION, ("address", "home"))],
            scope=_scope(),
        )
        assert swept == {addr_row["id"]}
        assert email_row["id"] not in swept

    async def test_class_rows_confirmed_with_real_regexes(self, monkeypatch):
        """The ILIKE keyword net is loose on purpose; a row that merely
        contains a keyword-ish string but no class match is dropped."""
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        noise = _row("The user sent a postcard from Lisbon")
        coord, _, _ = _coordinator(token_rows=[], class_rows=[noise])
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep(CARD_RETRACTION, ("card", "payment"))],
            scope=_scope(),
        )
        assert swept == set()

    async def test_duplicate_rows_are_not_double_swept(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        card_row = _row(CARD_ROW)
        coord, rdbms, _ = _coordinator(
            token_rows=[card_row], class_rows=[card_row],
        )
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep(CARD_RETRACTION, ("card", "payment"))],
            scope=_scope(),
        )
        assert swept == {card_row["id"]}
        assert rdbms.set_deprecated_at.await_count == 1

    async def test_retraction_intent_rows_still_never_swept(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        coord, rdbms, _ = _coordinator(
            token_rows=[], class_rows=[_row(CARD_RETRACTION, "preferences")],
        )
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep(CARD_RETRACTION, ("card", "payment"))],
            scope=_scope(),
        )
        assert swept == set()
        rdbms.set_deprecated_at.assert_not_awaited()

    async def test_class_lookup_failure_is_non_fatal(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        token_row = _row(ADDRESS_ROW)
        coord, rdbms, _ = _coordinator(token_rows=[token_row])
        rdbms.current_memories_matching_any = AsyncMock(
            side_effect=RuntimeError("db down")
        )
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep(ADDRESS_RETRACTION, ("address", "home"))],
            scope=_scope(),
        )
        assert swept == {token_row["id"]}

    async def test_storage_impl_without_class_lookup_degrades(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        coord, rdbms, _ = _coordinator(token_rows=[])
        del rdbms.current_memories_matching_any
        swept = await coord.execute_retraction_sweeps(
            sweeps=[_sweep(CARD_RETRACTION, ("card", "payment"))],
            scope=_scope(),
        )
        assert swept == set()

    async def test_batch_ids_excluded_from_class_lookup(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        coord, rdbms, _ = _coordinator(token_rows=[], class_rows=[])
        await coord.execute_retraction_sweeps(
            sweeps=[_sweep(CARD_RETRACTION, ("card", "payment"))],
            scope=_scope(),
            exclude_memory_ids={"batch-1"},
        )
        excluded = set(
            rdbms.current_memories_matching_any.call_args.kwargs["exclude_memory_ids"]
        )
        assert "batch-1" in excluded


# ---------------------------------------------------------------------------
# 4. The motivating case, detector -> coordinator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCardCaseEndToEnd:
    async def _run(self, monkeypatch, v2: bool):
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        if v2:
            monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        else:
            monkeypatch.delenv("SYNAP_KU_RETRACTION_SWEEP_V2", raising=False)
        scope = _scope()
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit("The user lives in Texas", score=0.50),
            llm_client=_llm_classifier("DELETE"),
        )
        result = await detector.detect_updates(
            memories=[_mem(
                CARD_RETRACTION,
                source_span="don't keep my card on file after this",
            )],
            memory_embeddings=[[0.1] * 8],
            scope=scope,
        )
        card_row = _row(CARD_ROW)
        # Postgres side: the ALL-token ILIKE finds nothing (that is the bug);
        # the class prefilter returns the card row.
        coord, rdbms, _ = _coordinator(token_rows=[], class_rows=[card_row])
        swept = await coord.execute_retraction_sweeps(
            sweeps=result.retraction_sweeps,
            scope=scope,
            http_correlation_id="corr-1",
        )
        return swept, card_row, rdbms

    async def test_flag_off_card_survives(self, monkeypatch):
        swept, card_row, rdbms = await self._run(monkeypatch, v2=False)
        assert swept == set()
        rdbms.set_deprecated_at.assert_not_awaited()

    async def test_flag_on_card_is_retired(self, monkeypatch):
        swept, card_row, rdbms = await self._run(monkeypatch, v2=True)
        assert swept == {card_row["id"]}
        rdbms.set_deprecated_at.assert_awaited_once_with(memory_id=card_row["id"])


# ---------------------------------------------------------------------------
# 5. SQL pin
# ---------------------------------------------------------------------------

class TestSql:
    def test_any_lookup_is_current_rows_only_and_or_joined(self):
        import inspect
        from synap.cloud.agents.ingestion.storage import sqlalchemy_rdbms_wrapper
        src = inspect.getsource(
            sqlalchemy_rdbms_wrapper.SQLAlchemyRDBMSWrapper.current_memories_matching_any
        )
        assert "superseded_by IS NULL" in src
        assert "deprecated_at IS NULL" in src
        assert '" OR ".join(clauses)' in src
        assert "LIMIT :limit" in src


# ---------------------------------------------------------------------------
# 6. Standalone emission: the sweep must not depend on vector recall.
#    Miss case: "The user prefers not to keep their card on file
#    after transactions." embedded below the 0.45 retraction floor vs the
#    stored card row -> "no candidates found" -> no pair, no classifier, no
#    sweep, card survived. v2 emits the sweep from the lexicon signal alone.
# ---------------------------------------------------------------------------

LIVE_RETRACTION = (
    "The user prefers not to keep their card on file after transactions."
)


def _vector_no_hits():
    vec = MagicMock()
    vec.batch_search = AsyncMock(return_value=[[]])
    return vec


class TestStandaloneSweepEmission:
    def _env(self, monkeypatch, v2: bool):
        monkeypatch.setenv("ENABLE_KU_DELETE", "true")
        monkeypatch.setenv("ENABLE_KU_DELETE_SUBJECT_GUARD", "true")
        if v2:
            monkeypatch.setenv("SYNAP_KU_RETRACTION_SWEEP_V2", "true")
        else:
            monkeypatch.delenv("SYNAP_KU_RETRACTION_SWEEP_V2", raising=False)

    async def test_no_candidates_still_emits_sweep(self, monkeypatch):
        self._env(monkeypatch, v2=True)
        detector = KnowledgeUpdateDetector(
            vector=_vector_no_hits(), llm_client=MagicMock(),
        )
        result = await detector.detect_updates(
            memories=[_mem(LIVE_RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert len(result.retraction_sweeps) == 1
        sweep = result.retraction_sweeps[0]
        assert sweep.subject_tokens == ("card",)
        assert "standalone" in sweep.reason

    async def test_no_candidates_flag_off_keeps_v1_empty(self, monkeypatch):
        self._env(monkeypatch, v2=False)
        detector = KnowledgeUpdateDetector(
            vector=_vector_no_hits(), llm_client=MagicMock(),
        )
        result = await detector.detect_updates(
            memories=[_mem(LIVE_RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.retraction_sweeps == []
        assert result.supersessions == []

    async def test_pair_verdict_none_still_sweeps(self, monkeypatch):
        """Candidates recalled but the classifier declined DELETE — the
        lexicon signal still drives the sweep."""
        self._env(monkeypatch, v2=True)
        detector = KnowledgeUpdateDetector(
            vector=_vector_with_hit(CARD_ROW, score=0.55),
            llm_client=_llm_classifier("NONE"),
        )
        result = await detector.detect_updates(
            memories=[_mem(LIVE_RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert len(result.retraction_sweeps) == 1
        assert "standalone" in result.retraction_sweeps[0].reason

    async def test_one_token_without_class_does_not_emit(self, monkeypatch):
        """Single subject token with no attribute class stays below the
        strong-signal gate — a vague retraction can't mass-retire."""
        self._env(monkeypatch, v2=True)
        detector = KnowledgeUpdateDetector(
            vector=_vector_no_hits(), llm_client=MagicMock(),
        )
        result = await detector.detect_updates(
            memories=[_mem("The user asked to forget their nickname")],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.retraction_sweeps == []

    async def test_non_retraction_memory_unaffected(self, monkeypatch):
        self._env(monkeypatch, v2=True)
        detector = KnowledgeUpdateDetector(
            vector=_vector_no_hits(), llm_client=MagicMock(),
        )
        result = await detector.detect_updates(
            memories=[_mem("The user's favorite color is blue")],
            memory_embeddings=[[0.1] * 8],
            scope=_scope(),
        )
        assert result.retraction_sweeps == []
        assert result.supersessions == []

    async def test_standalone_sweep_end_to_end_retires_card(self, monkeypatch):
        """Full path of the miss case: no pair forms, standalone sweep is
        emitted, coordinator's class prefilter retires the card row."""
        self._env(monkeypatch, v2=True)
        scope = _scope()
        detector = KnowledgeUpdateDetector(
            vector=_vector_no_hits(), llm_client=MagicMock(),
        )
        result = await detector.detect_updates(
            memories=[_mem(LIVE_RETRACTION)],
            memory_embeddings=[[0.1] * 8],
            scope=scope,
        )
        card_row = _row(CARD_ROW)
        coord, rdbms, _ = _coordinator(token_rows=[], class_rows=[card_row])
        swept = await coord.execute_retraction_sweeps(
            sweeps=result.retraction_sweeps,
            scope=scope,
            http_correlation_id="corr-standalone",
        )
        assert swept == {card_row["id"]}
        rdbms.set_deprecated_at.assert_awaited_once_with(memory_id=card_row["id"])
