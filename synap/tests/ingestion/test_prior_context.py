"""The leak guard for prior-memory context, kept as a test.

Showing extraction the person's earlier memories is the one change here that could
invent data: the extractor could lift a fact out of the context and store it as if
the person had just said it. Measured across 21 questions, 3 memory
types and 63 planted opportunities, with context up to 119 memories: zero leaks.

This file is what keeps that true. ``test_a_planted_fact_is_never_extracted`` is an
offline reproduction of that probe: it needs no model, because what it checks is
that the instruction forbidding it is actually in the prompt that goes out and that
the context is never confusable with the document.

The full probe with a live model lives in synap/evals/temporal/ and is run before
turning the flag on anywhere that matters, because the thing this test cannot check
is whether a model obeys.
"""

from datetime import datetime

import pytest

from synap.cloud.agents.ingestion.services import prior_context

CANARIES = [
    "The user's cousin Delphine Varrow lives in Portland and visits every spring.",
    "The user has a standing Tuesday appointment with a physiotherapist named Oswin Trask.",
    "The user owns a 2017 Volvo V60 with the licence plate KRX-8842.",
]


def mem(content, start=None, end=None, said="2023-05-27T06:10:00"):
    return {"content": content, "event_start": start, "event_end": end,
            "document_created_at": said}


# ── the format ────────────────────────────────────────────────────────────────

def test_an_unknown_event_date_says_so_rather_than_showing_the_saying_date():
    """The defect this whole change exists to remove is a date that reads as the
    event date when all we know is when it was mentioned. A context block that
    showed a bare date would teach the extractor exactly that."""
    block = prior_context.render([mem("The user stayed in Haight-Ashbury.")])
    assert "happened: UNKNOWN" in block
    assert "said on:  2023-05-27" in block
    # the two are never collapsed into one undifferentiated date
    assert block.count("2023-05-27") == 1


def test_a_known_window_is_shown_as_a_range():
    block = prior_context.render([
        mem("The user visited San Francisco for the wedding.",
            start=datetime(2023, 3, 27), end=datetime(2023, 3, 31))])
    assert "happened: 2023-03-27 to 2023-03-31" in block


def test_a_single_day_is_not_shown_as_a_range():
    block = prior_context.render([
        mem("The user went to the museum.",
            start=datetime(2023, 3, 27), end=datetime(2023, 3, 27))])
    assert "happened: 2023-03-27\n" in block
    assert " to " not in block.split("happened:")[1].split("\n")[0]


def test_an_empty_memory_is_skipped_not_rendered_blank():
    assert prior_context.render([mem("   ")]) == ""


def test_dates_may_arrive_as_strings_or_datetimes():
    a = prior_context.render([mem("x", start=datetime(2023, 3, 27))])
    b = prior_context.render([mem("x", start="2023-03-27T00:00:00+00:00")])
    assert "happened: 2023-03-27" in a and "happened: 2023-03-27" in b


# ── the leak guard ────────────────────────────────────────────────────────────

def test_the_prompt_forbids_extracting_from_the_context():
    """Three sentences carry the whole guarantee. If an edit softens them, the
    live probe is the only thing left, and it runs far less often than this."""
    out = prior_context.wrap("Content to analyze:\n---\nUser: hi\n---\n",
                             prior_context.render([mem("Something earlier.")]))
    assert "MUST NOT extract any memory from this section" in out
    assert "Every memory you output must be\nstated in the content below" in out
    assert "If something appears only in this section, do not output it" in out


def test_a_planted_fact_sits_only_in_the_context_and_is_fenced_off():
    """The offline half of the canary probe: a planted fact must appear inside the
    delimited context section and nowhere in the part of the prompt the extractor
    is told to extract from."""
    document = "Content to analyze:\n---\nUser: I'm planning a trip.\n---\n"
    out = prior_context.wrap(document,
                             prior_context.render([mem(c) for c in CANARIES]))
    context_part, _, document_part = out.partition("---\n\n")
    for canary in CANARIES:
        assert canary in context_part
        assert canary not in document_part
    for token in ("Delphine", "Varrow", "Oswin", "Trask", "KRX-8842"):
        assert token not in document_part


def test_the_context_is_delimited_so_it_cannot_read_as_document_content():
    out = prior_context.wrap("DOC", prior_context.render([mem("earlier fact")]))
    assert out.endswith("DOC")
    assert out.count("---") >= 2


# ── the no-op guarantee ───────────────────────────────────────────────────────

def test_no_context_leaves_the_prompt_byte_identical():
    prompt = "Content to analyze:\n---\nUser: hi\n---\n"
    assert prior_context.wrap(prompt, "") == prompt
    assert prior_context.render([]) == ""


def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SYNAP_STAGE3_PRIOR_CONTEXT", raising=False)
    monkeypatch.delenv("SYNAP_STAGE3_PRIOR_CONTEXT_BY_INSTANCE", raising=False)
    assert prior_context.enabled_for(None) is False
    assert prior_context.enabled_for("inst_abc") is False


def test_per_instance_override_beats_the_fleet_flag(monkeypatch):
    monkeypatch.setenv("SYNAP_STAGE3_PRIOR_CONTEXT", "on")
    monkeypatch.setenv("SYNAP_STAGE3_PRIOR_CONTEXT_BY_INSTANCE", "inst_abc=off")
    assert prior_context.enabled_for("inst_other") is True
    assert prior_context.enabled_for("inst_abc") is False


def test_the_cap_is_bounded_and_survives_a_bad_value(monkeypatch):
    monkeypatch.setenv("SYNAP_STAGE3_PRIOR_CONTEXT_MAX", "not a number")
    assert prior_context.max_memories() == prior_context.DEFAULT_CAP
    monkeypatch.setenv("SYNAP_STAGE3_PRIOR_CONTEXT_MAX", "0")
    assert prior_context.max_memories() == 1
    monkeypatch.setenv("SYNAP_STAGE3_PRIOR_CONTEXT_MAX", "12")
    assert prior_context.max_memories() == 12


# ── the fetcher ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_failed_read_yields_no_context_rather_than_raising():
    """Context is an enhancement. A database hiccup must not fail an ingest."""
    class Boom:
        def __call__(self):
            raise RuntimeError("pool exhausted")

    class Scope:
        user_id = "11111111-1111-1111-1111-111111111111"

    fetch = prior_context.build_fetcher(Boom())
    assert await fetch(Scope(), "doc-1") == ""


@pytest.mark.asyncio
async def test_no_user_or_no_factory_yields_no_context():
    class Scope:
        user_id = None

    assert await prior_context.build_fetcher(None)(Scope(), "doc") == ""
    fetch = prior_context.build_fetcher(object())
    assert await fetch(Scope(), "doc") == ""


@pytest.mark.asyncio
async def test_the_document_being_ingested_is_excluded():
    """A retry or a re-ingest would otherwise hand the document its own previous
    output back as "what we already know", which is a loop, not context."""
    captured = {}

    class Result:
        @staticmethod
        def fetchall():
            return []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, stmt, params):
            captured["sql"] = str(stmt)
            captured["params"] = params
            return Result()

    class Scope:
        user_id = "11111111-1111-1111-1111-111111111111"

    fetch = prior_context.build_fetcher(lambda: Session())
    await fetch(Scope(), "longterm:doc-1")
    assert captured["params"]["doc"] == "longterm:doc-1"
    assert "source_document_id IS DISTINCT FROM" in captured["sql"]
    assert "superseded_by IS NULL" in captured["sql"]
    assert captured["params"]["lim"] == prior_context.max_memories()


# ---------------------------------------------------------------------------
# The clause that stopped this feature running at all.
#
# `(:doc IS NULL OR source_document_id IS DISTINCT FROM :doc)` looks harmless
# and is valid SQL, but a bind parameter alone on the left of `IS NULL` gives
# Postgres nothing to infer a type from, so asyncpg's prepare step raised
# `AmbiguousParameterError: could not determine data type of parameter $2`.
# It failed on all 44 documents of the 3 Oct benchmark run, and because the
# read is deliberately non-fatal it degraded silently: a warning per call and
# an empty context block, so extraction behaved exactly as if the flag were off.
#
# test_the_document_being_ingested_is_excluded above passed throughout, because
# it only checked that the clause was present and the parameter bound.
# ---------------------------------------------------------------------------

class _Capture:
    """Captures the statement without needing a database."""

    def __init__(self):
        self.sql = None
        self.params = None

    def factory(self):
        capture = self

        class Result:
            @staticmethod
            def fetchall():
                return []

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def execute(self, stmt, params):
                capture.sql = str(stmt)
                capture.params = params
                return Result()

        return lambda: Session()


class _Scope:
    user_id = "11111111-1111-1111-1111-111111111111"


@pytest.mark.asyncio
@pytest.mark.parametrize("document_id", [None, "", "longterm:doc-1"])
async def test_no_bind_parameter_is_ever_compared_to_null_untyped(document_id):
    """The regression, stated as the rule that was broken. Postgres must be able
    to determine the type of every parameter, and a bare ``:name IS NULL`` is
    the one shape where it cannot."""
    import re
    cap = _Capture()
    fetch = prior_context.build_fetcher(cap.factory())
    await fetch(_Scope(), document_id)
    offending = re.findall(r":\w+\s+IS\s+(?:NOT\s+)?NULL", cap.sql, re.IGNORECASE)
    assert offending == [], (
        f"a bind parameter compared to NULL with no cast: {offending}. "
        "asyncpg cannot prepare this and the whole read fails.")


@pytest.mark.asyncio
async def test_with_no_document_the_parameter_is_not_bound_at_all():
    """The clause is built conditionally, so the untypeable comparison does not
    exist rather than being cast around."""
    cap = _Capture()
    fetch = prior_context.build_fetcher(cap.factory())
    await fetch(_Scope(), None)
    assert "doc" not in cap.params
    assert ":doc" not in cap.sql
    # and the read still does its real job
    assert "superseded_by IS NULL" in cap.sql
    assert cap.params["lim"] == prior_context.max_memories()


@pytest.mark.asyncio
async def test_the_document_exclusion_is_cast_so_its_type_is_known():
    cap = _Capture()
    fetch = prior_context.build_fetcher(cap.factory())
    await fetch(_Scope(), "longterm:doc-1")
    assert "source_document_id IS DISTINCT FROM CAST(:doc AS text)" in cap.sql
    assert cap.params["doc"] == "longterm:doc-1"


@pytest.mark.asyncio
async def test_the_user_filter_is_cast_too():
    """`user_id = :uid` against a uuid column has the same failure shape."""
    cap = _Capture()
    fetch = prior_context.build_fetcher(cap.factory())
    await fetch(_Scope(), None)
    assert "CAST(:uid AS uuid)" in cap.sql
