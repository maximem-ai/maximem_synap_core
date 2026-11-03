"""The LLM steps on the retrieval path must not sample.

Two cold-cache recordings of the SAME build,
with no ingestion in the window, returned HTTP 200 for every request both times.
Many non-empty requests returned a **different set** of memories, with a mean
overlap near one half. Ordering was never the difference and totals matched exactly,
so it was membership.

Two `temperature=0.1` call sites on that path explain it:

  * `retrieval.llm_filter` decides which candidates survive. It judges
    relevance, not score, which is why dropped items spanned the whole score
    range rather than clustering at the cut.
  * `retrieval.subquery_generation` decides what to search for. A different
    subquery is a different vector search, which is the likeliest explanation
    for a top-scoring item being absent from one run entirely: the one-run
    maximum score (0.4569) was HIGHER than the both-runs maximum (0.4417).

Sampling on a step that decides which memories a user gets back buys nothing and
costs reproducibility, including the ability to measure any future change.

⚠ This does not by itself guarantee determinism. Providers can still vary at
temperature 0 through batching and floating-point non-associativity. It removes
the deliberate part, which is the part we control.
"""
from __future__ import annotations

import ast
import pathlib

RM = (
    pathlib.Path(__file__).resolve().parents[3]
    / "synap" / "cloud" / "application" / "client_api" / "retrieval_manager.py"
)


def _temperatures():
    """Every `temperature=` keyword passed to an llm complete() call."""
    tree = ast.parse(RM.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "complete"):
            continue
        for kw in node.keywords:
            if kw.arg == "temperature" and isinstance(kw.value, ast.Constant):
                found.append((node.lineno, kw.value.value))
    return found


def test_the_guard_finds_the_call_sites_at_all():
    """A guard that inspects nothing passes for the wrong reason."""
    temps = _temperatures()
    assert len(temps) >= 4, (
        f"expected at least 4 llm complete() calls with an explicit temperature "
        f"in retrieval_manager.py, found {len(temps)}. If they were refactored, "
        f"update this guard rather than deleting it."
    )


def test_no_retrieval_llm_call_samples():
    hot = [(ln, t) for ln, t in _temperatures() if t not in (0, 0.0)]
    assert not hot, (
        "retrieval LLM calls are sampling again at "
        + ", ".join(f"line {ln}: temperature={t}" for ln, t in hot)
        + ". On production this produced a different SET of memories for 9 of "
          "20 non-empty requests between two identical runs, and it makes every "
          "quality measurement on the retrieval path unmeasurable."
    )
