"""The JEV pick: strict, and closed when JEV cannot answer."""

import pytest

from synap.cloud.application.memory_forget import judge
from synap.cloud.application.memory_forget.models import Candidate


def _c(i, text):
    return Candidate(memory_id=f"m{i}", content=text)


def _answer(level, p=0.9):
    """A JEV answer putting probability p on `level`, the rest on level 0."""
    probs = {str(i): 0.0 for i in range(len(judge.RUBRIC))}
    probs[str(level)] = p
    probs["0"] += 1.0 - p
    return {"probabilities": probs}


def _ask_by_text(table):
    """A fake ask(): reads each '[i] text' line out of the state and answers by text."""
    async def ask(state, questions):
        lines = [l for l in state.splitlines() if l.startswith("[")]
        out = {}
        for i, line in enumerate(lines):
            text = line.split("] ", 1)[1]
            out[f"s{i}"] = _answer(table(text))
        return out
    return ask


async def test_the_value_goes_and_the_related_preference_stays():
    cands = [_c(1, "The user's contact number is 512-555-0142."),
             _c(2, "The user prefers email over phone calls."),
             _c(3, "The user lives in Austin.")]
    table = lambda t: 3 if "512-555" in t else (1 if "phone" in t else 0)
    got = await judge.pick("my phone number", cands, ask_fn=_ask_by_text(table))
    assert [c.memory_id for c in got] == ["m1"]


async def test_a_removal_note_that_still_holds_the_value_is_picked():
    cands = [_c(1, "The user's contact number 512-555-0142 was removed from their account.")]
    got = await judge.pick("my phone number", cands, ask_fn=_ask_by_text(lambda t: 3))
    assert [c.memory_id for c in got] == ["m1"]


async def test_nothing_matching_returns_nothing():
    cands = [_c(1, "The user likes jazz."), _c(2, "The user has a dog.")]
    assert await judge.pick("my phone number", cands, ask_fn=_ask_by_text(lambda t: 0)) == []


async def test_the_bar_is_strict_a_borderline_score_is_kept():
    # P(level>=2) = 0.55 is below the 0.6 bar, so the memory survives.
    async def ask(state, questions):
        probs = {"0": 0.45, "1": 0.0, "2": 0.55, "3": 0.0}
        return {"s0": {"probabilities": probs}}
    assert await judge.pick("x", [_c(1, "maybe")], ask_fn=ask) == []


async def test_a_missing_or_malformed_answer_is_not_admitted():
    async def ask(state, questions):
        return {"s0": "yes", "s1": {"probabilities": None}, "s2": None}
    cands = [_c(1, "a"), _c(2, "b"), _c(3, "c")]
    assert await judge.pick("x", cands, ask_fn=ask) == []


async def test_a_failing_judge_raises_so_nothing_is_deleted():
    async def ask(state, questions):
        raise TimeoutError("slow")
    with pytest.raises(judge.JudgeUnavailable):
        await judge.pick("x", [_c(1, "a")], ask_fn=ask)


async def test_no_api_key_raises_and_does_not_guess(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(judge.JudgeUnavailable):
        await judge.pick("x", [_c(1, "a")])


async def test_the_live_call_waits_long_and_retries_at_most_once(monkeypatch):
    # A pooled call holds ~100 candidates and takes longer than the client's 3s default.
    # A timed-out call JEV already processed would be sent again and billed twice.
    seen = {}

    async def fake_ask(state, questions, **kw):
        seen.update(kw)
        return {"s0": {"probabilities": {"3": 1.0}}}

    from synap.cloud.shared import jev_client
    monkeypatch.setattr(jev_client, "ask", fake_ask)
    monkeypatch.setattr(jev_client, "jev_key_set", lambda: True)
    out = await judge.pick("my phone number", [Candidate("m1", "number is 512-555-0142")])
    assert [c.memory_id for c in out] == ["m1"]
    assert seen == {"timeout": 30.0, "retries": 1, "for_forget": True}


async def test_without_a_key_forget_says_it_needs_one(monkeypatch):
    from synap.cloud.shared import jev_client
    monkeypatch.setattr(jev_client, "jev_key_set", lambda: False)
    with pytest.raises(judge.JudgeNotConfigured, match="JEV_API_KEY"):
        await judge.pick("my phone number", [Candidate("m1", "number is 512-555-0142")])


async def test_empty_candidates_need_no_call():
    async def ask(state, questions):
        raise AssertionError("should not be called")
    assert await judge.pick("x", [], ask_fn=ask) == []


async def test_a_long_list_is_judged_in_batches_and_one_bad_batch_fails_all():
    calls = []

    async def ask(state, questions):
        calls.append(len(questions))
        if len(calls) == 2:
            raise RuntimeError("second batch fails")
        return {f"s{i}": _answer(3) for i in range(len(questions))}

    cands = [_c(i, f"t{i}") for i in range(judge.BATCH + 5)]
    with pytest.raises(judge.JudgeUnavailable):
        await judge.pick("x", cands, ask_fn=ask)
    assert calls == [judge.BATCH, 5]


async def test_batches_keep_the_original_order():
    async def ask(state, questions):
        return {f"s{i}": _answer(3) for i in range(len(questions))}
    cands = [_c(i, f"t{i}") for i in range(judge.BATCH + 3)]
    got = await judge.pick("x", cands, ask_fn=ask)
    assert [c.memory_id for c in got] == [c.memory_id for c in cands]


def test_the_state_names_what_to_forget_and_numbers_the_memories():
    state = judge.build_state("my phone number", [_c(1, "alpha"), _c(2, "beta")])
    assert "my phone number" in state
    assert "[0] alpha" in state and "[1] beta" in state


def test_the_rubric_is_an_ordered_list_not_a_dict():
    q = judge.build_questions(2)
    assert isinstance(q["s0"]["criteria"], list) and len(q) == 2
