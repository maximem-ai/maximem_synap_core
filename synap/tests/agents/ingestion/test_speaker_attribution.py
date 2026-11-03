"""Speaker attribution (PR train 1+2 of the echo-poisoning fix plan).

Covers the ENABLE_SPEAKER_ATTRIBUTION flag surface:
- prompt addendum is injected in code (not in the 18 prompt-variant strings)
  only when the flag is on AND the content carries speaker-labelled turns;
- LLM-emitted assertion_source values are normalized/validated, junk → None;
- the field survives the stage-3 JSON parse whitelist;
- both RDBMS wrappers and both Chroma metadata sites persist the column
  (source-level checks, mirroring test_ku_recall_admission's approach).

Default-off invariant: with the flag unset, prompts and stored rows are
byte-identical to pre-change behavior (assertion_source stays None/NULL).
"""

import re
from unittest import mock

import pytest

from synap.cloud.agents.ingestion.models.outputs import ExtractedMemory
from synap.cloud.agents.ingestion.models import MemoryDestination, MemoryType
from synap.cloud.agents.ingestion.pipeline import stage3_extraction
from synap.cloud.agents.ingestion.pipeline.stage3_extraction import (
    _SPEAKER_ATTRIBUTION_ADDENDUM,
    _clean_assertion_source,
    _maybe_add_speaker_attribution,
    _speaker_attribution_enabled,
    Stage3Extractor,
)

CONVERSATION = "User: my email is a@b.com\n\nAssistant: noted, a@b.com it is."
PLAIN_DOC = "Quarterly report. Revenue grew 4%. Contact a@b.com for details."
PROMPT = "You are an extractor."


class TestFlagGating:
    def test_flag_default_off(self, monkeypatch):
        monkeypatch.delenv("ENABLE_SPEAKER_ATTRIBUTION", raising=False)
        assert _speaker_attribution_enabled() is False

    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("1", True), ("yes", True), ("TRUE", True),
        ("false", False), ("0", False), ("off", False), ("", False),
    ])
    def test_flag_parsing(self, monkeypatch, value, expected):
        monkeypatch.setenv("ENABLE_SPEAKER_ATTRIBUTION", value)
        assert _speaker_attribution_enabled() is expected

    def test_prompt_unchanged_when_flag_off(self, monkeypatch):
        monkeypatch.delenv("ENABLE_SPEAKER_ATTRIBUTION", raising=False)
        assert _maybe_add_speaker_attribution(PROMPT, CONVERSATION) == PROMPT

    def test_addendum_added_for_conversation_when_on(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SPEAKER_ATTRIBUTION", "true")
        out = _maybe_add_speaker_attribution(PROMPT, CONVERSATION)
        assert out.startswith(PROMPT)
        assert _SPEAKER_ATTRIBUTION_ADDENDUM in out
        assert "assertion_source" in out

    def test_no_addendum_for_plain_document(self, monkeypatch):
        monkeypatch.setenv("ENABLE_SPEAKER_ATTRIBUTION", "true")
        assert _maybe_add_speaker_attribution(PROMPT, PLAIN_DOC) == PROMPT

    @pytest.mark.parametrize("label", ["user", "assistant", "human", "ai", "system", "bot", "USER"])
    def test_all_stage0_turn_labels_trigger(self, monkeypatch, label):
        monkeypatch.setenv("ENABLE_SPEAKER_ATTRIBUTION", "true")
        content = f"{label}: hello there"
        assert _SPEAKER_ATTRIBUTION_ADDENDUM in _maybe_add_speaker_attribution(PROMPT, content)


class TestCleanAssertionSource:
    @pytest.mark.parametrize("raw,expected", [
        ("user", "user"), ("User", "user"), (" USER ", "user"), ("human", "user"),
        ("assistant", "assistant"), ("ai", "assistant"), ("bot", "assistant"),
        ("agent", "assistant"), ("ASSISTANT", "assistant"),
        ("system", "system"), ("tool", "system"),
    ])
    def test_valid_values_normalized(self, raw, expected):
        assert _clean_assertion_source(raw) == expected

    @pytest.mark.parametrize("junk", ["customer", "speaker1", "", "both", None, 5, {"a": 1}, ["user"]])
    def test_junk_becomes_none(self, junk):
        assert _clean_assertion_source(junk) is None


class TestParseWhitelist:
    """_try_parse_json_extraction must let assertion_source through (it drops
    non-whitelisted keys), and must not invent it when absent."""

    def _parse(self, payload):
        extractor = mock.Mock(spec=Stage3Extractor)
        return Stage3Extractor._try_parse_json_extraction(extractor, payload)

    def test_field_passes_whitelist(self):
        payload = '{"memories": [{"content": "email is a@b.com", "confidence": 0.9, "assertion_source": "assistant"}]}'
        parsed = self._parse(payload)
        assert len(parsed) == 1
        assert parsed[0]["assertion_source"] == "assistant"

    def test_absent_field_stays_absent(self):
        payload = '{"memories": [{"content": "email is a@b.com", "confidence": 0.9}]}'
        parsed = self._parse(payload)
        assert len(parsed) == 1
        assert "assertion_source" not in parsed[0]


class TestModelAndThreading:
    def _make_memory(self, **overrides):
        from uuid import uuid4
        kwargs = dict(
            memory_id=uuid4(),
            memory_type=MemoryType.FACTS,
            content="x",
            confidence=0.9,
            destination=MemoryDestination.MEMORIES,
            source_document_id="doc1",
        )
        kwargs.update(overrides)
        return ExtractedMemory(**kwargs)

    def test_model_default_is_none(self):
        assert self._make_memory().assertion_source is None

    def test_model_accepts_value(self):
        assert self._make_memory(assertion_source="assistant").assertion_source == "assistant"

    def test_process_memory_sanitizes_raw_value(self):
        """The ExtractedMemory constructor call in _process_memory routes the
        raw LLM value through _clean_assertion_source."""
        import inspect
        src = inspect.getsource(Stage3Extractor._process_memory)
        assert '_clean_assertion_source(raw.get("assertion_source"))' in src


class TestStoragePersistence:
    """Source-level checks that every storage write path carries the column
    (same style as TestDeprecatedAtReadFilter in test_ku_recall_admission)."""

    def _source(self, module):
        import inspect
        return inspect.getsource(module)

    def test_rdbms_wrapper_inserts_column(self):
        from synap.cloud.agents.ingestion.storage import rdbms_wrapper
        src = self._source(rdbms_wrapper)
        # memories + candidates INSERTs, and both ON CONFLICT SET clauses
        assert src.count("assertion_source") >= 4
        assert src.count("assertion_source = EXCLUDED.assertion_source") == 2

    def test_rdbms_wrapper_placeholder_counts_match(self):
        """Each INSERT's %s placeholder count must equal its column count —
        guards against a column/param mismatch slipping through."""
        from synap.cloud.agents.ingestion.storage import rdbms_wrapper
        src = self._source(rdbms_wrapper)
        for stmt in re.findall(r"INSERT INTO (?:memories|candidates) \((.*?)\) VALUES \((.*?)\)", src, re.DOTALL):
            cols = [c.strip() for c in stmt[0].split(",") if c.strip()]
            placeholders = stmt[1].count("%s")
            assert len(cols) == placeholders

    def test_sqlalchemy_wrapper_inserts_column(self):
        from synap.cloud.agents.ingestion.storage import sqlalchemy_rdbms_wrapper
        src = self._source(sqlalchemy_rdbms_wrapper)
        assert src.count(":assertion_source") == 2  # memories + candidates
        assert src.count("assertion_source = EXCLUDED.assertion_source") == 2
        assert src.count('"assertion_source": getattr(memory, \'assertion_source\', None)') == 2

    def test_vector_metadata_both_sites(self):
        from synap.cloud.agents.ingestion.storage import vector_wrapper
        src = self._source(vector_wrapper)
        # single upsert_memory + upsert_memories_batch, None-guarded
        assert src.count('metadata["assertion_source"] = memory.assertion_source') == 2

    def test_migration_covers_memories_and_candidates(self):
        from pathlib import Path
        repo_root = Path(__file__).resolve()
        while repo_root.name != "synap":
            repo_root = repo_root.parent
        mig = repo_root.parent / "synap" / "db" / "migrations" / "071_memories_assertion_source.sql"
        text = mig.read_text()
        assert "ALTER TABLE memories" in text
        assert "ALTER TABLE candidates" in text
        assert "assertion_source" in text
