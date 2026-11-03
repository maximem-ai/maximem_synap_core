"""Analysis-as-hints injection (spec §4.5).

Client-supplied call-analysis JSON (transcript ingest) is appended to the
extraction USER prompt as hints — in code, so all provider prompt variants
inherit it — gated by ENABLE_ANALYSIS_HINTS (default ON) and capped at
ANALYSIS_HINTS_MAX_CHARS of compact JSON.
"""

import inspect
import json

import pytest

from synap.cloud.agents.ingestion.pipeline.stage3_extraction import (
    ANALYSIS_HINTS_MAX_CHARS,
    _ANALYSIS_HINTS_HEADER,
    _maybe_add_analysis_hints,
    Stage3Extractor,
)
import synap.cloud.agents.ingestion.workers.stage23_worker as worker_mod
from synap.tests.agents.ingestion._retire_shim import attach_retire_shim

PROMPT = "Content to analyze:\nuser: hi"
HINTS = {"disposition": "qualified", "sentiment": "positive"}


class TestMaybeAddAnalysisHints:

    def test_appended_by_default(self, monkeypatch):
        monkeypatch.delenv("ENABLE_ANALYSIS_HINTS", raising=False)
        out = _maybe_add_analysis_hints(PROMPT, HINTS)
        assert out.startswith(PROMPT)
        assert "[CLIENT-PROVIDED CALL ANALYSIS" in out
        # The addendum must state the hints-not-source rule explicitly.
        assert "never as extraction source" in out
        assert "memories must cite transcript content" in out
        # Compact, key-sorted JSON payload.
        assert out.endswith(json.dumps(
            HINTS, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ))

    def test_kill_switch_off_is_noop(self, monkeypatch):
        monkeypatch.setenv("ENABLE_ANALYSIS_HINTS", "false")
        assert _maybe_add_analysis_hints(PROMPT, HINTS) == PROMPT

    @pytest.mark.parametrize("flag", ["true", "1", "yes"])
    def test_kill_switch_truthy_values(self, monkeypatch, flag):
        monkeypatch.setenv("ENABLE_ANALYSIS_HINTS", flag)
        assert _maybe_add_analysis_hints(PROMPT, HINTS) != PROMPT

    @pytest.mark.parametrize("hints", [None, {}, "a string", 5, ["list"]])
    def test_absent_or_nondict_hints_are_noop(self, monkeypatch, hints):
        monkeypatch.delenv("ENABLE_ANALYSIS_HINTS", raising=False)
        assert _maybe_add_analysis_hints(PROMPT, hints) == PROMPT

    def test_json_capped_at_max_chars(self, monkeypatch):
        monkeypatch.delenv("ENABLE_ANALYSIS_HINTS", raising=False)
        big = {"blob": "x" * (ANALYSIS_HINTS_MAX_CHARS * 3)}
        out = _maybe_add_analysis_hints(PROMPT, big)
        appended = out[len(PROMPT):]
        assert appended.startswith(_ANALYSIS_HINTS_HEADER)
        payload = appended[len(_ANALYSIS_HINTS_HEADER):]
        assert len(payload) == ANALYSIS_HINTS_MAX_CHARS

    def test_unserializable_hints_are_noop(self, monkeypatch):
        monkeypatch.delenv("ENABLE_ANALYSIS_HINTS", raising=False)
        circular = {}
        circular["self"] = circular
        assert _maybe_add_analysis_hints(PROMPT, circular) == PROMPT


class TestThreading:

    def test_extract_signature_accepts_analysis_hints(self):
        for method in (
            Stage3Extractor.extract,
            Stage3Extractor._extract_type,
            Stage3Extractor._extract_simple_group,
        ):
            assert "analysis_hints" in inspect.signature(method).parameters, method

    def test_extract_threads_hints_to_both_extraction_paths(self):
        """Baseline per-type fan-out AND the Option-D merged-simple call
        both receive the hints.

        Counted across ``extract`` and the ``_extract_impl`` it delegates to.
        It used to count two occurrences inside ``extract`` alone, which went
        red the moment the body moved into ``_extract_impl`` and stayed red on
        main: the hints still reached both paths, and the assertion was reading
        for a shape rather than for the property. ``_extract_impl`` is named
        explicitly so a delegation to a third method fails here rather than
        passing quietly.
        """
        src = inspect.getsource(Stage3Extractor.extract)
        src += inspect.getsource(Stage3Extractor._extract_impl)
        assert src.count("analysis_hints=analysis_hints") >= 2

    def test_addendum_applied_to_user_prompt_in_both_builders(self):
        for method in (
            Stage3Extractor._extract_type,
            Stage3Extractor._extract_simple_group,
        ):
            src = inspect.getsource(method)
            assert (
                "user_prompt = _maybe_add_analysis_hints(user_prompt, analysis_hints)"
                in src
            ), method

    def test_worker_passes_request_metadata_analysis(self):
        src = inspect.getsource(worker_mod)
        assert 'analysis_hints=(request.metadata or {}).get("analysis")' in src
