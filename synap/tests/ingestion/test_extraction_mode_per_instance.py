"""Stage-3 extraction mode is resolved PER INSTANCE, not once for the fleet.

Why this exists: baseline runs five memory calls per chunk where hybrid_graph
runs three, plus a separate entity and relationship call instead of one merged
graph call. Measured on 15 real LongMemEval sessions, the merged arm returned
70 memories against baseline's 125. We want one instance on baseline to test
that, without every other client paying for the extra calls.
"""
import ast
import inspect
import pathlib

import pytest

from synap.cloud.agents.ingestion.pipeline import stage3_extraction as s3

INST_A = "44eff2d7-2a3d-5075-8c39-addf2a0188d0"
INST_B = "4fb56f47-3fc9-57c7-aeed-5ac445e554ff"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("STAGE3_EXTRACTION_MODE", raising=False)
    monkeypatch.delenv("STAGE3_EXTRACTION_MODE_BY_INSTANCE", raising=False)


def test_default_is_baseline_when_nothing_is_set():
    assert s3.resolve_extraction_mode(INST_A) == "baseline"
    assert s3.resolve_extraction_mode(None) == "baseline"


def test_fleet_variable_still_applies_to_everyone(monkeypatch):
    """The existing deployment sets only the fleet variable. It must keep
    working untouched, or this change silently re-extracts for every client."""
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE", "hybrid_graph")
    assert s3.resolve_extraction_mode(INST_A) == "hybrid_graph"
    assert s3.resolve_extraction_mode(INST_B) == "hybrid_graph"
    assert s3.resolve_extraction_mode(None) == "hybrid_graph"


def test_one_instance_can_differ_from_the_fleet(monkeypatch):
    """The whole point: our benchmark instance on baseline, everyone else on
    hybrid_graph, in the same process."""
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE", "hybrid_graph")
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE_BY_INSTANCE", f"{INST_A}=baseline")
    assert s3.resolve_extraction_mode(INST_A) == "baseline"
    assert s3.resolve_extraction_mode(INST_B) == "hybrid_graph"


def test_override_list_tolerates_spacing_and_case(monkeypatch):
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE", "hybrid")
    monkeypatch.setenv(
        "STAGE3_EXTRACTION_MODE_BY_INSTANCE",
        f"  {INST_A.upper()} = BASELINE , {INST_B}=hybrid_graph ",
    )
    assert s3.resolve_extraction_mode(INST_A) == "baseline"
    assert s3.resolve_extraction_mode(INST_B) == "hybrid_graph"


def test_an_unknown_mode_never_silently_changes_extraction(monkeypatch):
    """A typo in an env var must not quietly change how memories are extracted
    for every client. Unknown values fall back, loudly, never through."""
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE", "hybrid_graph")
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE_BY_INSTANCE", f"{INST_A}=hybird")
    assert s3.resolve_extraction_mode(INST_A) == "hybrid_graph"

    monkeypatch.setenv("STAGE3_EXTRACTION_MODE", "nonsense")
    monkeypatch.delenv("STAGE3_EXTRACTION_MODE_BY_INSTANCE", raising=False)
    assert s3.resolve_extraction_mode(INST_A) == "baseline"


def test_malformed_entries_are_skipped_not_fatal(monkeypatch):
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE", "hybrid_graph")
    monkeypatch.setenv(
        "STAGE3_EXTRACTION_MODE_BY_INSTANCE",
        f",,no-equals-sign,{INST_A}=baseline,=orphan,{INST_B}=",
    )
    assert s3.resolve_extraction_mode(INST_A) == "baseline"
    assert s3.resolve_extraction_mode(INST_B) == "hybrid_graph"


def test_the_extractor_actually_consults_the_resolver():
    """Wiring lock. The resolver can be perfect and the extractor can still
    read os.environ directly, in which case every test above passes and
    the override is ignored entirely."""
    src = pathlib.Path(inspect.getfile(s3)).read_text()
    tree = ast.parse(src)
    # extract() is a thin wrapper since 2026-09-18 (it sets the per-call MACA
    # config and delegates), so look in whichever of the two does the work
    # rather than pinning the name of the method that happens to hold it.
    bodies = {
        n.name: ast.dump(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name in ("extract", "_extract_impl")
    }
    assert bodies, "Stage3Extractor has neither extract() nor _extract_impl()"
    joined = "\n".join(bodies.values())
    assert "resolve_extraction_mode" in joined, (
        "the extraction entry point no longer calls resolve_extraction_mode — the "
        "per-instance override is inert")
    assert "STAGE3_EXTRACTION_MODE" not in joined, (
        "the extraction entry point reads the env var directly again, which "
        "bypasses the per-instance override")
