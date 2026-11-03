"""These tests are worthless if CI does not run them.

Found by audit 2026-08-23: `.github/workflows/test.yml` enumerates its test
directories by hand and `scoping` was not among them, so all 461 tests here,
including the X-01 security property test, ran only when somebody remembered.
And with the DSN unset, 96 of the 461 skip: the entire L-13 concurrency set, the
whole grants suite and the five-rung fixture, which is to say most of the tests
that touch a real database and therefore most of the ones that would catch the
identifier and locking defects this feature keeps producing.

A skip and a pass are indistinguishable in a CI summary line. So the wiring
itself is pinned here, in the suite it protects, because a workflow file is
exactly the kind of thing that gets edited by someone solving a different
problem.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW = _ROOT / ".github" / "workflows" / "test.yml"
_SCOPING = Path(__file__).resolve().parent


def _workflow_text() -> str:
    if not _WORKFLOW.exists():
        pytest.skip(f"{_WORKFLOW} not present in this checkout")
    return _WORKFLOW.read_text(encoding="utf-8")


def test_ci_runs_this_directory():
    text = _workflow_text()
    dirs = re.search(r"for d in (.+?); do", text, re.S)
    assert dirs, "could not find the test-directory loop in the workflow"
    listed = dirs.group(1).replace("\\", " ").split()
    assert "scoping" in listed, (
        f"CI runs {sorted(listed)} and not `scoping`, so every test in this "
        "directory is a manual ritual"
    )


def test_ci_gives_this_directory_a_database():
    text = _workflow_text()
    assert "SYNAP_TEST_DATABASE_URL" in text, (
        "CI does not set SYNAP_TEST_DATABASE_URL, so the Postgres-backed files "
        "here would skip and report green"
    )
    assert re.search(r"image:\s*postgres:", text), (
        "CI sets a DSN but starts no Postgres service for it to point at"
    )


def test_ci_makes_a_missing_database_fatal_rather_than_a_skip():
    text = _workflow_text()
    assert 'SYNAP_REQUIRE_PG_TESTS: "1"' in text or "SYNAP_REQUIRE_PG_TESTS: 1" in text, (
        "without this, a database that fails to start turns 96 tests into "
        "skips and the run still reports success"
    )


def test_the_require_flag_actually_does_something():
    """The flag is only worth setting in CI if the code reads it."""
    conftest = (_SCOPING / "conftest.py").read_text(encoding="utf-8")
    assert "SYNAP_REQUIRE_PG_TESTS" in conftest
    assert "raise RuntimeError" in conftest, (
        "conftest reads the flag but does nothing fatal with it"
    )


def _files_mentioning_the_dsn() -> list[Path]:
    out = []
    for path in sorted(_SCOPING.glob("test_*.py")):
        if "SYNAP_TEST_DATABASE_URL" in path.read_text(encoding="utf-8"):
            out.append(path)
    return out


def test_every_database_backed_file_uses_the_one_shared_gate():
    """One definition, so a seventh file cannot invent a seventh spelling that
    the CI flag does not know to make fatal."""
    offenders = []
    for path in _files_mentioning_the_dsn():
        if path.name == "test_ci_wiring.py":
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        uses_shared = "requires_pg" in src
        own_skipif = any(
            isinstance(n, ast.Attribute) and n.attr == "skipif"
            for n in ast.walk(tree)
        )
        if own_skipif and not uses_shared:
            offenders.append(path.name)
    assert not offenders, (
        "these gate on the database with their own skipif instead of the "
        f"shared `requires_pg`, so CI cannot make them fatal: {offenders}"
    )


def test_the_guard_is_looking_at_real_files():
    """A guard that finds nothing passes for the wrong reason."""
    found = _files_mentioning_the_dsn()
    assert len(found) >= 6, (
        f"expected at least six database-backed files, found {[p.name for p in found]}"
    )
