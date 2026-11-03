"""The Jev gate that decides whether extraction gets the temporal rules.

The failure that matters is a false NO: the document's memories lose their
dates. So every way Jev can fail must come out as "include the rules".
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from synap.cloud.agents.ingestion.services import temporal_gate as tg
from synap.cloud.shared import jev_client

DCT = datetime(2023, 5, 30, 18, 3)
CHAT = ("User: I'm looking for throw pillows for my living room.\n"
        "Assistant: Sure! Here are some ideas:\n1. Velvet\n2. Linen\n"
        "User: I got a new area rug a month ago.\n"
        "Assistant: Nice choice.")


@pytest.fixture(autouse=True)
def _jev_on(monkeypatch):
    from synap.cloud.shared import edition
    monkeypatch.setattr(edition, "engine_uses_jev", True)  # these test the Jev path, which Cloud runs
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    monkeypatch.delenv("SYNAP_TEMPORAL_GATE", raising=False)


def _fake(scores_by_substring, calls):
    async def ask(state, questions, **kw):
        calls.append(state)
        msg = state["message"]
        v = next((s for sub, s in scores_by_substring.items() if sub in msg), 0.05)
        return {k: {"type": "noul", "noul": v if k == "relative" else 0.01} for k in questions}
    return ask


def test_split_keeps_speakers_and_multiline_bodies():
    msgs = tg.split_messages(CHAT)
    assert [s for s, _ in msgs] == ["user", "assistant", "user", "assistant"]
    assert "2. Linen" in msgs[1][1]


def test_text_without_speakers_is_cut_into_slices():
    text = "\n\n".join(["para " * 100] * 10)
    msgs = tg.split_messages(text)
    assert len(msgs) > 1 and all(s == "document" for s, _ in msgs)


def test_state_is_one_message_with_its_date_and_speaker(monkeypatch):
    calls = []
    monkeypatch.setattr(jev_client, "ask", _fake({"a month ago": 0.97}, calls))
    d = asyncio.run(tg.decide(CHAT, DCT))
    assert d.needs_rules and d.source == "jev" and d.max_score == pytest.approx(0.97)
    assert calls[0] == {"conversation_date": "2023-05-30 (Tuesday)", "speaker": "user",
                        "message": "I'm looking for throw pillows for my living room."}


def test_no_time_reference_means_no_rules(monkeypatch):
    calls = []
    monkeypatch.setattr(jev_client, "ask", _fake({}, calls))
    d = asyncio.run(tg.decide("User: I like jazz.\nAssistant: Great.", DCT))
    assert not d.needs_rules and d.messages_checked == 2


def test_it_stops_at_the_first_batch_with_a_yes(monkeypatch):
    calls = []
    monkeypatch.setattr(jev_client, "ask", _fake({"yesterday": 0.9}, calls))
    monkeypatch.setattr(tg, "_BATCH", 2)
    chat = "User: I went hiking yesterday.\nAssistant: ok\n" + "User: more\nAssistant: ok\n" * 10
    d = asyncio.run(tg.decide(chat, DCT))
    assert d.needs_rules and len(calls) == 2


@pytest.mark.parametrize("error", [jev_client.JevUnavailable("HTTP 429"), RuntimeError("boom")])
def test_any_jev_failure_includes_the_rules(monkeypatch, error):
    async def ask(*a, **k):
        raise error
    monkeypatch.setattr(jev_client, "ask", ask)
    d = asyncio.run(tg.decide("User: I like jazz.\nAssistant: Great.", DCT))
    assert d.needs_rules and d.source == "jev_error"


def test_no_key_falls_back_to_the_regex(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY")
    assert asyncio.run(tg.decide("User: I went there yesterday.", DCT)).source == "regex"


def test_the_kill_switch_falls_back_to_the_regex(monkeypatch):
    monkeypatch.setenv("SYNAP_TEMPORAL_GATE", "regex")
    d = asyncio.run(tg.decide("User: I went there last month.", DCT))
    assert d.source == "regex" and d.needs_rules


def test_a_document_longer_than_the_cap_checks_the_rest_with_the_regex(monkeypatch):
    calls = []
    monkeypatch.setattr(jev_client, "ask", _fake({}, calls))
    monkeypatch.setattr(tg, "_MAX_MESSAGES", 4)
    chat = "User: hi\nAssistant: ok\n" * 3 + "User: I moved here last year.\nAssistant: ok\n"
    d = asyncio.run(tg.decide(chat, DCT))
    assert len(calls) == 4 and d.source == "regex_tail" and d.needs_rules


def test_empty_text_needs_nothing():
    assert asyncio.run(tg.decide("   ", DCT)).source == "empty"


def test_the_questions_are_the_measured_three():
    assert set(tg.QUESTIONS) == {"relative", "calendar", "duration"}
    assert all(q["type"] == "noul" and "`message`" in q["instructions"]["question"] for q in tg.QUESTIONS.values())
