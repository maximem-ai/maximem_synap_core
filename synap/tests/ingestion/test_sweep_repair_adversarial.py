"""Adversarial cases for the sweep+attach repair pass.

Each case attacks one of three properties, whatever the model returns:
  1. it never invents a fact
  2. it never loses a fact (no row vanishes or shrinks)
  3. it never fails an ingest

Cases marked xfail are REAL HOLES found by reading the code: the test states
the property that should hold and fails today. Do not "fix" the test.
"""
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory, MemoryDestination, MemoryType,
)
from synap.cloud.agents.ingestion.pipeline.stage3_extraction import Stage3Extractor
from synap.cloud.agents.ingestion.services import sweep_repair as sr

CHUNK = "User: I paid $60 for the taxi. I left home at 7 AM. My cat Miso is 3."


def mem(content, **kw):
    return ExtractedMemory(
        memory_id=uuid4(), memory_type=MemoryType.FACTS, content=content,
        confidence=0.9, destination=MemoryDestination.MEMORIES,
        source_document_id="d", **kw)


class FakeLLM:
    def __init__(self, *replies, raise_on=None):
        self.replies = list(replies)
        self.calls = 0
        self.raise_on = raise_on

    async def complete(self, **kw):
        self.calls += 1
        if self.raise_on == self.calls:
            raise RuntimeError("boom")
        r = self.replies.pop(0)
        return r if isinstance(r, (str, dict, list)) or r is None else r


def S(quote="I paid $60 for the taxi", content="The user paid $60 for a taxi",
      said_by="person"):
    return {"quote": quote, "content": content, "said_by": said_by}


def sweep(*stmts):
    return json.dumps({"statements": list(stmts)})


def decide(*ds):
    return json.dumps({"decisions": list(ds)})


async def run(memories, llm, chunk=CHUNK, scope=None):
    scope = scope or SimpleNamespace(scope_level="user", instance_id="i")
    return await Stage3Extractor._sweep_repair(
        None, content=chunk, memories=memories, llm=llm,
        request_scope=scope, correlation_id="c", document_id="d")


def texts(ms):
    return [m.content for m in ms]


# ---------------------------------------------------------------- property 3
@pytest.mark.parametrize("reply", [
    "[]", '"just a string"', "42", "null", "", None, "{not json",
    '{"wrong_key": []}', '{"statements": "oops"}', '{"statements": {"a": 1}}',
    '{"statements": [1, "x", null, [], {"quote": []}]}',
    "```json\n[1,2]\n```", [1, 2], 7,
])
async def test_malformed_sweep_reply_returns_memories_untouched(reply):
    """Sweep reply of any wrong shape must leave the rows exactly as they came."""
    ms = [mem("The user owns a cat named Miso")]
    out = await run(ms, FakeLLM(reply))
    assert texts(out) == texts(ms)


@pytest.mark.parametrize("reply", [
    "[]", '"s"', "null", None, '{"decisions": "x"}', '{"decisions": {"a": 1}}',
    '{"decisions": [1, null, "x", [], {}]}', '{"nope": 1}', "{{", [1],
])
async def test_malformed_attach_reply_returns_memories_untouched(reply):
    """Attach reply of any wrong shape must change nothing."""
    ms = [mem("The user owns a cat named Miso")]
    out = await run(ms, FakeLLM(sweep(S()), reply))
    assert texts(out) == texts(ms)


@pytest.mark.parametrize("which", [1, 2])
async def test_llm_exception_on_either_call_returns_same_list(which):
    """An exception in either call must not propagate and must keep every row."""
    ms = [mem("The user owns a cat named Miso")]
    out = await run(ms, FakeLLM(sweep(S()), decide(), raise_on=which))
    assert texts(out) == texts(ms)


async def test_exception_mid_apply_does_not_leave_half_applied_rows(monkeypatch):
    """If applying extend #2 blows up, extend #1 must not leak into the result."""
    import synap.cloud.agents.ingestion.pipeline.knowledge_update as ku
    calls = {"n": 0}
    real = ku.find_dropped_entities

    def flaky(a, b, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("x")
        return real(a, b, **k)
    monkeypatch.setattr(ku, "find_dropped_entities", flaky)
    ms = [mem("The user has a cat"), mem("The user has a dog")]
    llm = FakeLLM(sweep(S(), S("I left home at 7 AM", "left at 7 AM")), decide(
        {"statement_index": 0, "action": "extend", "row_index": 0,
         "text": "The user has a cat, paid $60 for a taxi"},
        {"statement_index": 1, "action": "extend", "row_index": 1,
         "text": "The user has a dog, left at 7 AM"}))
    out = await run(ms, llm)
    assert texts(out) == texts(ms)


async def test_empty_and_whitespace_chunk_makes_no_llm_call():
    """An empty chunk has nothing to sweep, so no call is spent."""
    for chunk in ("", ):
        llm = FakeLLM()
        out = await run([mem("a")], llm, chunk=chunk)
        assert llm.calls == 0 and texts(out) == ["a"]


async def test_whitespace_only_chunk_cannot_supply_a_quote():
    """A whitespace chunk must never validate a quote, even a whitespace one."""
    ms = [mem("a fact")]
    llm = FakeLLM(sweep(S(quote="   ", content="x paid $5")),
                  decide({"statement_index": 0, "action": "new", "text": "paid $5"}))
    out = await run(ms, llm, chunk="   \n\t ")
    assert texts(out) == ["a fact"]


async def test_no_memories_makes_no_llm_call():
    """Zero rows in means zero calls and an empty list back."""
    llm = FakeLLM()
    assert await run([], llm) == []
    assert llm.calls == 0


# ---------------------------------------------------------------- property 1
def test_quote_check_is_case_and_whitespace_insensitive_only():
    """A reflowed or recased quote is kept; a different word is not."""
    assert sr.surviving_statements([S(quote="i  PAID\n$60")], CHUNK)
    assert not sr.surviving_statements([S(quote="I paid $600")], CHUNK)


@pytest.mark.parametrize("quote", [".*", "[a-z]+", "(", "\\", "$60)", "I paid $6|"])
def test_regex_special_quotes_are_literal_substring_checks(quote):
    """Regex metacharacters in a quote must be matched literally, never raise."""
    assert sr.surviving_statements([S(quote=quote)], CHUNK) == []


def test_unicode_quote_survives_only_when_present():
    """Accents and non-Latin text compare literally after lowercasing.

    The content has to be consistent with its own quote: an earlier version of
    this test paired a coffee quote with the default "$60 taxi" content, which
    the content-versus-quote number check now refuses, and rightly.
    """
    chunk = "Usuario: Compré un café por €4,50 en İstanbul 東京"
    assert sr.surviving_statements(
        [S(quote="café por €4,50", content="The user bought a café for €4,50")], chunk)
    assert sr.surviving_statements(
        [S(quote="東京", content="The user mentioned 東京")], chunk)
    assert not sr.surviving_statements(
        [S(quote="cafe por €4,50", content="The user bought a cafe for €4,50")], chunk)


def test_statement_with_non_string_fields_does_not_raise():
    """Lists, ints and None in quote/content/said_by are tolerated."""
    raw = [{"quote": 5, "content": 6}, {"quote": None}, {"quote": ["I paid"],
           "content": {"a": 1}}, {"quote": "I paid", "content": "x", "said_by": 3}]
    sr.surviving_statements(raw, CHUNK)


# WAS A HOLE, now closed
async def test_content_that_contradicts_its_quote_cannot_become_a_row():
    """Quote 'I paid $60 for the taxi' is real; content claims $600 and a limo."""
    ms = [mem("The user owns a cat")]
    llm = FakeLLM(
        sweep(S(content="The user paid $600 for a limousine")),
        decide({"statement_index": 0, "action": "new",
                "text": "The user paid $600 for a limousine"}))
    out = await run(ms, llm)
    assert "limousine" not in " ".join(texts(out))


# WAS A HOLE, now closed
async def test_tiny_quote_does_not_authorise_arbitrary_content():
    """Quote 'a' exists in every chunk; content says the user bought a yacht."""
    ms = [mem("The user owns a cat")]
    llm = FakeLLM(
        sweep(S(quote="a", content="The user bought a yacht for $2000")),
        decide({"statement_index": 0, "action": "new",
                "text": "The user bought a yacht for $2000"}))
    out = await run(ms, llm)
    assert "yacht" not in " ".join(texts(out))


# WAS A HOLE, now closed
async def test_new_text_unrelated_to_any_statement_is_refused():
    """Attach returns a 'new' whose text is not the statement's content."""
    ms = [mem("The user owns a cat")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "new",
        "text": "The user bought a house in 2019"}))
    out = await run(ms, llm)
    assert "house" not in " ".join(texts(out))


# WAS A HOLE, now closed
@pytest.mark.parametrize("si", [99, -1, None, "0", 0.0, 1.5, True, [0]])
async def test_new_with_invalid_statement_index_is_refused(si):
    """A decision citing no real statement has no grounding and must not add."""
    ms = [mem("The user owns a cat")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": si, "action": "new",
        "text": "The user paid $60 for a taxi"}))
    out = await run(ms, llm)
    assert len(out) == 1


# WAS A HOLE, now closed
async def test_extend_with_unrelated_text_and_bad_statement_index_is_refused():
    """Extend cites statement 99, which does not exist, and adds invention."""
    ms = [mem("The user has a cat")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 99, "action": "extend", "row_index": 0,
        "text": "The user has a cat and is a licensed pilot"}))
    out = await run(ms, llm)
    assert texts(out) == ["The user has a cat"]


# ---------------------------------------------------------------- property 2
# WAS A HOLE, now closed
async def test_extend_that_negates_the_row_is_refused():
    """Row says the user owns 3 cats; extend keeps '3 cats' but says they do not."""
    ms = [mem("The user owns 3 cats")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0,
        "text": "The user does not own 3 cats"}))
    out = await run(ms, llm)
    assert texts(out) == ["The user owns 3 cats"]


# WAS A HOLE, now closed
async def test_extend_cannot_shrink_a_row_with_no_entities():
    """Row has plenty of plain words; extend returns four of them."""
    row = "the user really enjoys long slow weekend hikes with friends"
    ms = [mem(row)]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0,
        "text": "user likes walks"}))
    out = await run(ms, llm)
    assert texts(out) == [row]


# WAS A HOLE, now closed
async def test_extend_cannot_replace_row_with_a_different_fact():
    """Row 'user likes tea' rewritten to 'user paid $60 for a taxi'."""
    ms = [mem("the user likes tea in the morning")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0,
        "text": "The user paid $60 for the taxi"}))
    out = await run(ms, llm)
    assert texts(out) == ["the user likes tea in the morning"]


async def test_extend_dropping_a_quantity_is_refused():
    """Row has $60; extend keeps the story but loses the amount."""
    ms = [mem("The user paid $60 for the taxi")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0,
        "text": "The user paid for the taxi and left at 7 AM"}))
    assert texts(await run(ms, llm)) == texts(ms)


# WAS A HOLE, now closed")
async def test_extend_changing_a_quantity_is_refused():
    """Row has $60; extend writes $600."""
    ms = [mem("The user paid $60 for the taxi")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0,
        "text": "The user paid $600 for the taxi"}))
    assert texts(await run(ms, llm)) == texts(ms)


async def test_extend_dropping_a_name_is_refused():
    """Row names Miso; extend says 'the cat'."""
    ms = [mem("The user's cat is named Miso")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0,
        "text": "The user's cat is 3 years old"}))
    assert texts(await run(ms, llm)) == texts(ms)


async def test_extend_may_add_a_number_the_row_lacked():
    """Legitimate case pinned: adding the missing price is the point."""
    ms = [mem("The user took a taxi")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0,
        "text": "The user took a taxi that cost $60"}))
    assert texts(await run(ms, llm)) == ["The user took a taxi that cost $60"]


@pytest.mark.parametrize("ri", ["0", 0.0, 1.0, -1, None, 99, [0], {"a": 0}, "", 10**30])
async def test_extend_with_non_integer_or_out_of_range_row_index_is_ignored(ri):
    """String, float, negative, null, huge row_index must not touch any row."""
    ms = [mem("The user took a taxi"), mem("The user has a cat")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": ri,
        "text": "The user took a taxi that cost $60"}))
    assert texts(await run(ms, llm)) == texts(ms)


# WAS A HOLE, now closed
async def test_extend_with_boolean_row_index_is_ignored():
    """JSON true as row_index becomes row 1 through isinstance(int)."""
    ms = [mem("The user has a cat"), mem("The user took a taxi")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": True,
        "text": "The user took a taxi that cost $60"}))
    assert texts(await run(ms, llm)) == texts(ms)


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None, 
    ".",
    "...",
    "-",
    "?!"])
async def test_extend_with_empty_or_punctuation_text(text):
    """Empty/whitespace text never rewrites; punctuation-only must not either."""
    ms = [mem("The user took a taxi")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0, "text": text}))
    assert texts(await run(ms, llm)) == texts(ms)


async def test_extend_without_text_key_is_ignored():
    """Missing text on an extend must not blank the row."""
    ms = [mem("The user took a taxi")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0}))
    assert texts(await run(ms, llm)) == texts(ms)


async def test_drop_carrying_row_index_and_text_changes_nothing():
    """drop must be inert even if the model attaches a row_index and text."""
    ms = [mem("The user took a taxi")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "drop", "row_index": 0,
        "text": "something else $5"}))
    assert texts(await run(ms, llm)) == texts(ms)


@pytest.mark.parametrize("action", ["EXTEND", " Extend ", "extend\n"])
async def test_action_casing_and_whitespace_is_forgiven(action):
    """Odd casing/whitespace around a real action still applies it."""
    ms = [mem("The user took a taxi")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": action, "row_index": 0,
        "text": "The user took a taxi that cost $60"}))
    assert texts(await run(ms, llm)) == ["The user took a taxi that cost $60"]


@pytest.mark.parametrize("action", ["delete", "replace", "merge", "", None, 1,
                                    ["extend"], "extend,new", "ext end"])
async def test_unknown_action_changes_nothing(action):
    """An action outside extend/new/drop must be inert."""
    ms = [mem("The user took a taxi")]
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": action, "row_index": 0,
        "text": "The user paid $60"}))
    assert texts(await run(ms, llm)) == texts(ms)


async def test_two_extends_of_one_row_keep_the_first():
    """Second rewrite of the same row must not overwrite the first."""
    ms = [mem("The user took a taxi")]
    llm = FakeLLM(sweep(S(), S("I left home at 7 AM", "left at 7 AM")), decide(
        {"statement_index": 0, "action": "extend", "row_index": 0,
         "text": "The user took a taxi that cost $60"},
        {"statement_index": 1, "action": "extend", "row_index": 0,
         "text": "The user took a taxi at 7 AM"}))
    assert texts(await run(ms, llm)) == ["The user took a taxi that cost $60"]


# WAS A HOLE, now closed
async def test_refused_extend_does_not_block_a_valid_later_extend():
    """First extend drops a number (refused); second is fine and should land."""
    ms = [mem("The user paid $60 for the taxi")]
    llm = FakeLLM(sweep(S(), S("I left home at 7 AM", "left at 7 AM")), decide(
        {"statement_index": 0, "action": "extend", "row_index": 0,
         "text": "The user paid for the taxi"},
        {"statement_index": 1, "action": "extend", "row_index": 0,
         "text": "The user paid $60 for the taxi and left at 7 AM"}))
    assert "7 AM" in texts(await run(ms, llm))[0]


async def test_extend_keeps_every_other_field_of_the_row():
    """Only content changes: id, confidence, dates, type, source stay."""
    m = mem("The user took a taxi", assertion_source="user")
    llm = FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0,
        "text": "The user took a taxi that cost $60"}))
    out = (await run([m], llm))[0]
    assert (out.memory_id, out.confidence, out.destination, out.assertion_source,
            out.memory_type) == (m.memory_id, m.confidence, m.destination,
                                 "user", m.memory_type)


async def test_input_list_is_not_mutated():
    """The caller's list and rows are untouched when the pass edits and adds."""
    ms = [mem("The user took a taxi")]
    snap = list(ms)
    llm = FakeLLM(sweep(S(), S("I left home at 7 AM", "left at 7 AM")), decide(
        {"statement_index": 0, "action": "extend", "row_index": 0,
         "text": "The user took a taxi that cost $60"},
        {"statement_index": 1, "action": "new", "text": "The user left at 7 AM"}))
    out = await run(ms, llm)
    assert ms == snap and ms[0].content == "The user took a taxi" and len(out) == 2


async def test_no_row_ever_disappears_under_a_flood():
    """Thousands of mixed decisions never reduce the original row count."""
    ms = [mem(f"fact {i} costs ${i + 1}") for i in range(5)]
    ds = [{"statement_index": 0, "action": a, "row_index": i % 7,
           "text": f"x {i}"} for i in range(3000)
          for a in ("extend", "drop", "new")][:3000]
    out = await run(ms, FakeLLM(sweep(S()), decide(*ds)))
    assert len(out) >= len(ms)
    assert [m.memory_id for m in out[:5]] == [m.memory_id for m in ms]


# WAS A HOLE, now closed
async def test_adds_are_capped():
    """3000 'new' decisions with a price must not create 3000 memories."""
    ms = [mem("The user has a cat")]
    ds = [{"statement_index": 0, "action": "new",
           "text": f"The user paid ${i + 1} for a taxi"} for i in range(3000)]
    out = await run(ms, FakeLLM(sweep(S()), decide(*ds)))
    assert len(out) < 50


# WAS A HOLE, now closed
async def test_duplicate_new_decisions_add_one_row():
    """Two 'new' decisions with the same text, or text equal to an existing row."""
    ms = [mem("The user left home at 7 AM")]
    d = {"statement_index": 0, "action": "new",
         "text": "The user left home at 7 AM"}
    out = await run(ms, FakeLLM(sweep(S()), decide(d, d)))
    assert len(out) == 1


async def test_huge_reply_text_does_not_crash():
    """A megabyte of text in one decision must not raise."""
    ms = [mem("The user took a taxi")]
    big = "The user took a taxi that cost $60 " + "word " * 200000
    out = await run(ms, FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0, "text": big})))
    assert len(out) == 1


# WAS A HOLE, now closed
async def test_extend_cannot_balloon_a_row():
    """Extend text 1,000x the row length is a rewrite, not an addition."""
    ms = [mem("The user took a taxi")]
    big = "The user took a taxi that cost $60 " + "word " * 5000
    out = await run(ms, FakeLLM(sweep(S()), decide({
        "statement_index": 0, "action": "extend", "row_index": 0, "text": big})))
    assert len(texts(out)[0]) < 2000


# ------------------------------------------------- said_by / assertion_source
@pytest.mark.parametrize("said_by,want", [
    ("person", "user"), ("PERSON", "user"), (" assistant ", "assistant"),
    ("document", None), ("system", None), ("user", None), ("", None),
    (None, None), (7, None), ("__proto__", None)])
async def test_said_by_maps_only_to_known_speakers(said_by, want):
    """Only person/assistant yield a source; everything else stays unknown."""
    ms = [mem("The user has a cat")]
    llm = FakeLLM(sweep(S(said_by=said_by)), decide({
        "statement_index": 0, "action": "new",
        "text": "The user paid $60 for a taxi"}))
    out = await run(ms, llm)
    assert out[-1].assertion_source == want


@pytest.mark.parametrize("si", [-1, 99, None, "0", 0.0])
async def test_bad_statement_index_leaves_assertion_source_unknown(si):
    """An add that cites no real statement must not get a guessed speaker."""
    ms = [mem("The user has a cat")]
    llm = FakeLLM(sweep(S(said_by="person")), decide({
        "statement_index": si, "action": "new",
        "text": "The user paid $60 for a taxi"}))
    out = await run(ms, llm)
    for m in out[1:]:
        assert m.assertion_source is None


# ----------------------------------------------------------- Guard B / dest
@pytest.mark.parametrize("text", ["The user feels happy", "thinking of a studio",
                                  "wants to explore more", "The user likes tea"])
async def test_new_without_a_value_or_acquisition_is_refused(text):
    """Intentions and feelings do not become rows."""
    ms = [mem("The user has a cat")]
    llm = FakeLLM(sweep(S()), decide({"statement_index": 0, "action": "new",
                                      "text": text}))
    assert len(await run(ms, llm)) == 1


@pytest.mark.parametrize("level,dest", [
    ("user", MemoryDestination.MEMORIES),
    ("customer", MemoryDestination.CUSTOMER_KNOWLEDGE),
    ("client", MemoryDestination.CLIENT_KNOWLEDGE),
    (None, MemoryDestination.MEMORIES), ("weird", MemoryDestination.MEMORIES)])
async def test_added_row_destination_comes_from_scope_not_siblings(level, dest):
    """A low-confidence sibling in `candidates` must not pull the add there."""
    sib = mem("The user has a cat")
    sib = sib.model_copy(update={"destination": MemoryDestination.CANDIDATES,
                                 "confidence": 0.1})
    llm = FakeLLM(sweep(S()), decide({"statement_index": 0, "action": "new",
                                      "text": "The user paid $60 for a taxi"}))
    out = await run([sib], llm, scope=SimpleNamespace(scope_level=level,
                                                      instance_id=None))
    assert out[-1].destination == dest


# -------------------------------------------------------------- flag parsing
@pytest.mark.parametrize("env,inst,want", [
    ("", "a", False), ("true", "a", True), (" TRUE ", None, True),
    ("false", "a", False), ("0", "a", False), ("maybe", "a", False),
])
def test_flag_fleet_value(monkeypatch, env, inst, want):
    """Fleet flag accepts only the documented truthy words."""
    monkeypatch.delenv("SYNAP_INGEST_SWEEP_REPAIR_BY_INSTANCE", raising=False)
    monkeypatch.setenv("SYNAP_INGEST_SWEEP_REPAIR", env)
    assert sr.enabled_for(inst) is want


def test_flag_instance_override_beats_fleet_and_tolerates_garbage(monkeypatch):
    """A per-instance off wins over fleet on; malformed pairs are skipped."""
    monkeypatch.setenv("SYNAP_INGEST_SWEEP_REPAIR", "true")
    monkeypatch.setenv("SYNAP_INGEST_SWEEP_REPAIR_BY_INSTANCE",
                       ",,junk,=1, A = off ,b=on")
    assert sr.enabled_for("a") is False
    assert sr.enabled_for("B") is True
    assert sr.enabled_for("zzz") is True
