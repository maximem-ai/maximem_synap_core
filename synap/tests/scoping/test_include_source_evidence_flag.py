"""`include_source_evidence` is a payload knob, not a latency one.

Measured on a sample of real responses, and the two numbers point opposite ways,
which is the whole reason this flag is documented the way it is:

  * source_evidence is **41.5% of the response body** across a sample of real
  responses, roughly 400 bytes per item, on nearly every item. A caller
    that pipes our response into an LLM pays for all of it in every prompt.
  * the enrichment that produces it costs **p50 under 20ms** across a sample of
  real requests, under 1% of a fetch, because the chunk-evidence batching fix
    already took it down to a few milliseconds.

So it saves bytes and tokens, never meaningful time. These lock the wiring and
the default, so the flag cannot quietly stop working or start defaulting off.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
ROUTES = ROOT / "synap" / "cloud" / "application" / "sdk_api" / "sdk_context.py"
RM = ROOT / "synap" / "cloud" / "application" / "client_api" / "retrieval_manager.py"


def test_the_flag_defaults_to_todays_behaviour():
    """Default True or every existing caller silently loses evidence."""
    tree = ast.parse(ROUTES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ContextFetchRequest":
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and getattr(stmt.target, "id", None) == "include_source_evidence"
                ):
                    assert isinstance(stmt.value, ast.Constant) and stmt.value.value is True, (
                        "include_source_evidence must default to True; defaulting it off "
                        "would strip source text from every existing caller."
                    )
                    return
    pytest.fail("ContextFetchRequest has no include_source_evidence field")


def test_every_fetch_handler_threads_the_flag():
    """Four handlers build a decision: conversation, user, customer, client.
    A flag wired into three of them is a flag that works only sometimes."""
    src = ROUTES.read_text(encoding="utf-8")
    threaded = src.count('"include_source_evidence": request.include_source_evidence')
    handlers = src.count("request: ContextFetchRequest,")
    assert threaded >= 4, (
        f"include_source_evidence is threaded into {threaded} decision sites but "
        f"there are {handlers} handlers taking ContextFetchRequest. A flag wired "
        f"into some of them is worse than none, because it works on some routes."
    )


def test_the_retrieval_side_actually_reads_it():
    """A field nothing reads is the 'healthy, versioned, doing nothing' shape."""
    src = RM.read_text(encoding="utf-8")
    assert 'get("include_source_evidence"' in src, (
        "retrieval_manager never reads include_source_evidence, so the flag is inert"
    )
    assert "chunk_evidence_skipped_by_caller" in src, (
        "skipping should say so in the log, or nobody can tell a caller opted out "
        "from the enrichment silently failing"
    )


def test_the_env_var_and_the_request_flag_are_both_still_honoured():
    """They answer different questions: one is deployment-wide, one is per
    caller. Losing either is a regression."""
    src = RM.read_text(encoding="utf-8")
    assert "ENABLE_CHUNK_EVIDENCE" in src, "the deployment-wide switch was dropped"
    assert "_want_evidence and os.environ.get(" in src, (
        "the per-request flag must gate alongside the env var, not replace it"
    )
