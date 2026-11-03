"""Compaction v3: record merge, code checks, rendering, chunking, the writer
engine and the manager around it (COMPACTION_V3_PLAN.md §6-§8).

The LLM is scripted: each test queues the JSON the writer "returns", so what
is asserted is the code around the model, which is where the guarantees live
(unchanged sections copied byte for byte, quotes checked, one repair, then
drop, literals never lost, a failed update stores nothing).
"""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap.cloud.application.compaction.budgets import agent_budget
from synap.cloud.application.compaction.interfaces import CompactionTriggerType
from synap.cloud.application.compaction.manager import CompactionError, CompactionManager
from synap.cloud.application.compaction.models import CompactionConfig, CompactedContext, ConversationMessage
from synap.cloud.application.compaction.passes.base import COMPACTION_PROFILE_ENV, GROQ_PROFILE
from synap.cloud.application.compaction.v3 import (
    FRAMING,
    CompactionRecord,
    CompactionWriter,
    EngineSettings,
    Section,
    chunk_turns,
    is_banned_header,
    label_turns,
    render_context_block,
    render_document,
    strip_citations,
)
from synap.cloud.application.compaction.v3.checks import check_quotes, check_size, size_bounds
from synap.cloud.application.compaction.v3.record import LedgerChange, LedgerEntry, merge_output
from synap.cloud.infrastructure.llm import LLMProvider

T0 = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)


def msgs(*texts, start=0, ids=None):
    return [
        ConversationMessage(
            message_id=(ids[i] if ids else f"id{start + i}"),
            role="user" if (start + i) % 2 == 0 else "assistant",
            content=text,
            timestamp=T0 + timedelta(minutes=start + i),
        )
        for i, text in enumerate(texts)
    ]


class ScriptedLLM:
    """Returns the queued outputs in order and records every prompt."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            raise AssertionError("unexpected LLM call")
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(content=json.dumps(out), input_tokens=1000, output_tokens=200,
                               total_tokens=1200, provider=LLMProvider.GROQ, model=GROQ_PROFILE.model,
                               finish_reason="stop")

    def get_provider(self, provider):
        return MagicMock()

    @property
    def user_prompts(self):
        return [c["messages"][1]["content"] for c in self.calls]


SETTINGS = EngineSettings(target_ratio=0.30, record_ceiling=7_500, chunk_tokens=15_000)

FIRST_TURNS = (
    "Hi, I'm Priya. Order ORD-88213-KX arrived broken.",
    "Sorry, Priya. I'll send you a return label today.",
)

# Sections that do not state the order number, so code must keep it.
VAGUE_SECTIONS = [{"header": "Customer and order", "text": "Priya reported that her order arrived broken. [m1]"}]

FIRST_OUTPUT = {
    "plan": {"use_case": "customer_support", "headers": ["Customer and order", "Conversation timeline"]},
    "changed_sections": [
        {"header": "Customer and order", "text": "Priya reported that order ORD-88213-KX arrived broken. [m1]"},
        {"header": "Conversation timeline", "text": "The assistant apologised. [m2]"},
    ],
    "literals": [{"label": "Order", "value": "ORD-88213-KX", "m": "m1"}],
    "agreed_actions": [{"who": "the assistant", "what": "send a return label today",
                        "quotes": [{"text": "I'll send you a return label today", "m": "m2"}], "status": "open"}],
    "segment_density": [{"start": "m1", "end": "m2", "level": "high"}],
}


def writer(llm):
    return CompactionWriter(llm, GROQ_PROFILE)


# =============================================================================
# record merge
# =============================================================================

class TestMerge:
    def test_unchanged_sections_are_copied_byte_for_byte(self):
        record = CompactionRecord(sections=[Section("s1", "Profile", "Alex is a Senior Data Engineer.  [m1]"),
                                            Section("s2", "Timeline", "Old timeline. [m2]")])
        merge_output(record, {"changed_sections": [{"id": "s2", "text": "New timeline. [m5]"}]}, {})
        assert record.sections[0].text == "Alex is a Senior Data Engineer.  [m1]"
        assert record.sections[1].text == "New timeline. [m5]"

    def test_the_plan_is_frozen_but_can_grow(self):
        record = CompactionRecord(sections=[Section("s1", "Profile", "x")])
        merge_output(record, {"plan": {"headers": ["Something else"]},
                              "changed_sections": [{"header": "Badge access", "text": "y"}]}, {})
        assert [s.header for s in record.sections] == ["Profile", "Badge access"]
        assert record.sections[1].id == "s2"

    @pytest.mark.parametrize("header", ["Next Steps", "## What's Next", "Action Items", "Follow-ups",
                                        "Next Meeting Items", "To-do", "Recommendations", "next step:",
                                        "Open questions and next steps", "Status and follow-ups"])
    def test_next_step_headers_are_banned(self, header):
        assert is_banned_header(header)
        record = CompactionRecord()
        notes = merge_output(record, {"changed_sections": [{"header": header, "text": "Do the thing."}]}, {})
        assert record.sections == [] and notes.rejected_headers

    @pytest.mark.parametrize("header", ["Customer and order", "Open questions", "Root cause", "Steps tried"])
    def test_descriptive_headers_are_allowed(self, header):
        assert not is_banned_header(header)

    def test_the_ledger_is_additive_and_keeps_the_old_value(self):
        record = CompactionRecord(ledger=[LedgerEntry("Base salary", "$185,000", "m3")])
        merge_output(record, {"literals": [{"label": "base salary", "value": "$192,500", "m": "m21",
                                            "replaces": "$185,000"}]}, {"m21": T0.isoformat()})
        (entry,) = record.ledger
        assert entry.value == "$192,500" and [h.value for h in entry.history] == ["$185,000"]
        assert "Base salary: $192,500 (was $185,000; changed at turn 21, 17 Oct 2026)" in render_document(record)

    def test_superseded_facts_join_the_ledger(self):
        record = CompactionRecord(ledger=[LedgerEntry("Manager", "Tom Becker", "m2")])
        merge_output(record, {"superseded": [{"label": "Manager", "old": "Tom Becker", "new": "Lena Fischer",
                                              "m": "m5"}]}, {})
        assert record.ledger[0].value == "Lena Fischer"
        assert record.ledger[0].history[0].value == "Tom Becker"

    def test_an_action_is_updated_by_id_never_duplicated(self):
        record = CompactionRecord()
        merge_output(record, FIRST_OUTPUT, {})
        merge_output(record, {"agreed_actions": [{"id": "a1", "status": "done"}]}, {})
        assert len(record.actions) == 1 and record.actions[0].status == "done"

    def test_round_trip(self):
        record = CompactionRecord()
        merge_output(record, FIRST_OUTPUT, {})
        assert CompactionRecord.from_dict(record.to_dict()).to_dict() == record.to_dict()
        assert CompactionRecord.from_dict({"version": 2}) is None


# =============================================================================
# checks
# =============================================================================

class TestChecks:
    def test_a_quote_must_appear_in_a_turn(self):
        output = {"agreed_actions": [{"who": "the assistant", "what": "email a summary",
                                      "quotes": [{"text": "I will email you a summary", "m": "m2"}]}]}
        problems = check_quotes(output, {"m1": "Hi", "m2": "Sorry about that."})
        assert problems and "no turn contains the quote" in problems[0].message

    def test_a_miscited_quote_is_corrected_not_failed(self):
        output = {"agreed_actions": [{"who": "u", "what": "x", "quotes": [{"text": "I’ll  SEND it", "m": "m1"}]}]}
        assert check_quotes(output, {"m1": "Hi", "m2": "Sure, I'll send it now."}) == []
        assert output["agreed_actions"][0]["quotes"][0]["m"] == "m2"

    def test_size_band(self):
        assert size_bounds(10_000, 0.30) == (2_500, 3_500)
        assert size_bounds(500, 0.30) == (125, 400)  # the ~400-token floor
        assert check_size(3_000, 10_000, 0.30) == []
        assert check_size(4_000, 10_000, 0.30)[0].kind == "size_over"
        assert check_size(1_000, 10_000, 0.30)[0].kind == "size_under"
        assert check_size(0, 600, 0.30) == []  # a short span may add nothing


# =============================================================================
# rendering
# =============================================================================

class TestRender:
    def test_the_block_is_framed_and_citations_are_stripped(self):
        record = CompactionRecord()
        merge_output(record, FIRST_OUTPUT, {})
        record.covers = {"start_m": "m1", "end_m": "m2", "message_count": 2,
                         "start_timestamp": T0.isoformat(), "end_timestamp": (T0 + timedelta(minutes=1)).isoformat()}
        block = render_context_block(record)
        assert block.startswith("## Conversation so far (compacted: turns 1–2, 2026-10-17 09:00–09:01 UTC)")
        assert FRAMING in block
        assert "[m1]" not in block and "### Customer and order" in block
        assert "### Agreed actions earlier\n- the assistant: send a return label today (open)" in block
        # The section states the value, so the ledger does not print it again.
        assert block.count("ORD-88213-KX") == 1

    def test_strip_citations(self):
        assert strip_citations("A [m1]. B [m2, m4]. C [m3–m5].") == "A. B. C."


# =============================================================================
# chunker
# =============================================================================

class TestChunker:
    def test_chunks_split_at_turn_boundaries(self):
        turns = label_turns(msgs(*["word " * 300] * 10))
        chunks = chunk_turns(turns, limit=1_000)
        assert len(chunks) > 1
        assert [t.label for c in chunks for t in c] == [f"m{i}" for i in range(1, 11)]

    def test_a_turn_bigger_than_the_chunk_is_cut_and_labelled(self):
        (turn,) = label_turns(msgs("\n\n".join(["paragraph " * 200] * 6)))
        chunks = chunk_turns([turn], limit=500)
        parts = [t.part for c in chunks for t in c]
        assert len(parts) > 1 and parts[0] == f"part 1/{len(parts)} of m1"

    def test_turns_carry_timestamps(self):
        (turn,) = label_turns(msgs("hello"))
        assert turn.render().startswith("[m1 · 2026-10-17 09:00 UTC (Saturday) · user]")


# =============================================================================
# engine
# =============================================================================

class TestEngine:
    @pytest.mark.asyncio
    async def test_a_clean_update_is_one_call(self):
        llm = ScriptedLLM(FIRST_OUTPUT)
        result = await writer(llm).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)
        assert result.calls == 1 and result.repairs == 0
        assert result.record.covers["message_count"] == 2
        assert result.record.ledger[0].value == "ORD-88213-KX"
        assert "[m1 · 2026-10-17 09:00 UTC" in llm.user_prompts[0]

    @pytest.mark.asyncio
    async def test_an_invented_commitment_gets_one_repair_then_is_dropped(self):
        invented = {**FIRST_OUTPUT, "agreed_actions": FIRST_OUTPUT["agreed_actions"] + [
            {"who": "the assistant", "what": "email a summary of the call",
             "quotes": [{"text": "I'll email you a summary", "m": "m2"}], "status": "open"}]}
        llm = ScriptedLLM(invented, invented)
        result = await writer(llm).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)
        assert result.calls == 2 and result.repairs == 1
        assert 'no turn contains the quote "I\'ll email you a summary"' in llm.user_prompts[1]
        assert [a.what for a in result.record.actions] == ["send a return label today"]
        assert result.dropped == ["action: the assistant: email a summary of the call"]

    @pytest.mark.asyncio
    async def test_a_provider_json_error_is_retried(self):
        """Groq answers a malformed JSON-mode generation with an error, not text."""
        llm = ScriptedLLM(RuntimeError("Failed to generate JSON. Please adjust your prompt."), FIRST_OUTPUT)
        result = await writer(llm).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)
        assert len(llm.calls) == 2 and result.record.ledger[0].value == "ORD-88213-KX"

    @pytest.mark.asyncio
    async def test_other_provider_errors_are_not_retried(self):
        llm = ScriptedLLM(RuntimeError("rate limited"), FIRST_OUTPUT)
        with pytest.raises(RuntimeError):
            await writer(llm).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)

    @pytest.mark.asyncio
    async def test_a_repair_that_fixes_it_is_kept(self):
        broken = {**FIRST_OUTPUT, "agreed_actions": [{**FIRST_OUTPUT["agreed_actions"][0],
                                                      "quotes": [{"text": "I will mail a label", "m": "m2"}]}]}
        llm = ScriptedLLM(broken, FIRST_OUTPUT)
        result = await writer(llm).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)
        assert 'no turn contains the quote "I will mail a label"' in llm.user_prompts[1]
        assert result.repairs == 1 and not result.problems_after_repair

    @pytest.mark.asyncio
    async def test_a_missing_value_alone_is_added_by_code_without_a_repair(self):
        llm = ScriptedLLM({**FIRST_OUTPUT, "literals": [], "changed_sections": VAGUE_SECTIONS})
        result = await writer(llm).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)
        assert len(llm.calls) == 1 and result.auto_literals == ["ORD-88213-KX"]

    @pytest.mark.asyncio
    async def test_a_missing_value_the_prose_states_is_not_added_again(self):
        llm = ScriptedLLM({**FIRST_OUTPUT, "literals": []})
        result = await writer(llm).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)
        assert result.auto_literals == [] and render_document(result.record).count("ORD-88213-KX") == 1

    @pytest.mark.asyncio
    async def test_a_literal_left_out_twice_is_added_by_code(self):
        broken = {**FIRST_OUTPUT, "literals": [], "changed_sections": VAGUE_SECTIONS}
        result = await writer(ScriptedLLM(broken, broken)).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)
        assert result.auto_literals == ["ORD-88213-KX"]
        # Kept with the sentence it was said in, so a reader knows what it is.
        assert '- ORD-88213-KX: "Order ORD-88213-KX arrived broken." (turn 1)' in render_document(result.record)

    @pytest.mark.asyncio
    async def test_an_update_sends_only_new_turns_and_keeps_old_sections(self):
        first = await writer(ScriptedLLM(FIRST_OUTPUT)).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)
        before = first.record.sections[0].text
        llm = ScriptedLLM({"changed_sections": [{"id": "s2", "text": "The assistant apologised and the user thanked them. [m2, m3]"}]})
        new = label_turns(msgs("Thanks, that works.", start=2), first_position=3)
        result = await writer(llm).update(first.record, new, SETTINGS)
        assert result.record.sections[0].text == before
        assert "[m3 · " in llm.user_prompts[0] and "[m1 · " not in llm.user_prompts[0]
        assert result.record.covers["start_m"] == "m1" and result.record.covers["end_m"] == "m3"

    @pytest.mark.asyncio
    async def test_long_spans_fold_chunk_by_chunk_with_recaps(self):
        settings = EngineSettings(target_ratio=0.30, record_ceiling=50_000, chunk_tokens=700)
        out = lambda n: {"changed_sections": [{"header": "Timeline", "text": f"Chunk {n} happened."}],
                         "chunk_recap": f"recap of chunk {n}"}
        llm = ScriptedLLM(*[out(i) for i in range(1, 10)])
        result = await writer(llm).update(None, label_turns(msgs(*["talk " * 150] * 8)), settings)
        assert result.chunks > 1 and result.calls >= result.chunks
        assert "recap of chunk 1" not in llm.user_prompts[0]
        assert "EARLIER CHUNKS OF THIS UPDATE" in llm.user_prompts[-1] and "recap of chunk 1" in llm.user_prompts[-1]

    @pytest.mark.asyncio
    async def test_the_ceiling_rolls_up_sections_never_the_ledger(self):
        record = CompactionRecord(
            sections=[Section("s1", "History", "Long settled history. " * 400)],
            ledger=[LedgerEntry("Order", "ORD-1", "m1", history=[LedgerChange("ORD-0", "m1")])],
            covers={"start_m": "m1", "end_m": "m1", "message_count": 1},
        )
        condensed = "Settled history. " * 300
        llm = ScriptedLLM({"changed_sections": []}, {"changed_sections": [{"id": "s1", "text": condensed}]})
        settings = EngineSettings(target_ratio=0.30, record_ceiling=500, chunk_tokens=15_000)
        result = await writer(llm).update(record, label_turns(msgs("ok", start=1), first_position=2), settings)
        assert result.rollup and result.record.sections[0].text == condensed.strip()
        assert result.record.ledger[0].history[0].value == "ORD-0"

    @pytest.mark.asyncio
    async def test_a_roll_up_far_below_its_target_is_discarded(self):
        """Seen: asked to go from 5,100 to 4,000 tokens, the roll-up wrote 1,300 and lost half the answers."""
        record = CompactionRecord(sections=[Section("s1", "History", "Long settled history. " * 400)],
                                  covers={"start_m": "m1", "end_m": "m1", "message_count": 1})
        llm = ScriptedLLM({"changed_sections": []}, {"changed_sections": [{"id": "s1", "text": "Short history."}]})
        settings = EngineSettings(target_ratio=0.30, record_ceiling=500, chunk_tokens=15_000)
        result = await writer(llm).update(record, label_turns(msgs("ok", start=1), first_position=2), settings)
        assert not result.rollup and result.record.sections[0].text.startswith("Long settled history.")

    @pytest.mark.asyncio
    async def test_a_writer_failure_raises(self):
        with pytest.raises(RuntimeError):
            await writer(ScriptedLLM(RuntimeError("groq down"))).update(None, label_turns(msgs("hi")), SETTINGS)


# =============================================================================
# manager
# =============================================================================

def manager_with(llm, monkeypatch, existing=None, min_span=0):
    monkeypatch.setenv(COMPACTION_PROFILE_ENV, "groq")
    # The turns here are a few words each: the minimum span is tested apart.
    import synap.cloud.application.compaction.manager as manager_module
    monkeypatch.setattr(manager_module, "MIN_SPAN_TOKENS", min_span)
    storage = MagicMock()
    storage.get_latest = AsyncMock(return_value=existing)
    storage.save = AsyncMock()
    manager = CompactionManager(llm_manager=llm, storage=storage)
    manager._load_config = AsyncMock(return_value=CompactionConfig())
    manager._resolve_client_id = AsyncMock(return_value=None)
    return manager, storage


# Core compacts on the configured LLM and refuses to start without one; this gives it one (a no-op in Cloud).
@pytest.mark.usefixtures("core_llm_env")
class TestManager:
    def test_triggers(self, monkeypatch):
        manager, _ = manager_with(ScriptedLLM(), monkeypatch)
        config = CompactionConfig()
        assert manager.should_compact(msgs(*["hi"] * 29), config) == (False, CompactionTriggerType.MANUAL_API)
        assert manager.should_compact(msgs(*["hi"] * 30), config)[1] == CompactionTriggerType.MESSAGE_COUNT
        long = msgs(*["token " * 2_000] * 8)  # ~16K tokens, 8 turns
        assert manager.should_compact(long, config)[1] == CompactionTriggerType.TOKEN_THRESHOLD
        # A bigger agent window raises T: the same turns are not due.
        big = agent_budget(400_000, manager.limits.record_ceiling)
        assert manager.should_compact(long, config, budget=big) == (False, CompactionTriggerType.MANUAL_API)

    def test_a_span_under_the_minimum_is_not_compacted(self, monkeypatch):
        manager, _ = manager_with(ScriptedLLM(), monkeypatch, min_span=2_000)
        config = CompactionConfig()
        short = msgs(*["hi"] * 30)  # 30 turns: the message trigger, but a few tokens
        assert manager.should_compact(short, config) == (False, CompactionTriggerType.MANUAL_API)
        enough = msgs(*["word " * 80] * 30)  # ~2.4K tokens
        assert manager.should_compact(enough, config)[1] == CompactionTriggerType.MESSAGE_COUNT

    @pytest.mark.asyncio
    async def test_the_staleness_check_reads_the_instances_recompaction_idle_minutes(self, monkeypatch):
        first, _ = manager_with(ScriptedLLM(FIRST_OUTPUT), monkeypatch)
        existing = await first.compact("c1", "inst", msgs(*FIRST_TURNS), force=True)
        existing.metadata.created_at = datetime.now(timezone.utc) - timedelta(minutes=3)
        turns = msgs(*FIRST_TURNS, "Thanks!")

        async def run(minutes):
            manager, storage = manager_with(ScriptedLLM(), monkeypatch, existing=existing)
            manager._load_config = AsyncMock(
                return_value=CompactionConfig(recompaction_idle_minutes=minutes))
            storage.get_from_cache = AsyncMock(return_value=existing)
            storage.set_compaction_in_progress = AsyncMock(return_value=True)
            storage.clear_compaction_in_progress = AsyncMock()
            manager.compact = AsyncMock(return_value="recompacted")
            result = await manager.get_or_create("c1", "inst", turns)
            return result, manager.compact.await_count

        # Three minutes of silence: a one-minute setting is stale, a ten-minute one is not,
        # and neither is the five-minute code default that used to decide both.
        assert await run(1) == ("recompacted", 1)
        assert await run(10) == (existing, 0)

    @pytest.mark.asyncio
    async def test_a_forced_update_under_the_minimum_keeps_the_record(self, monkeypatch):
        first, _ = manager_with(ScriptedLLM(FIRST_OUTPUT), monkeypatch)
        existing = await first.compact("c1", "inst", msgs(*FIRST_TURNS), force=True)
        manager, storage = manager_with(ScriptedLLM(), monkeypatch, existing=existing, min_span=2_000)
        result = await manager.compact("c1", "inst", msgs(*FIRST_TURNS, "Thanks!"), force=True)
        assert result is existing and not storage.save.await_count

    def test_the_agent_model_comes_from_the_latest_turn(self, monkeypatch):
        manager, _ = manager_with(ScriptedLLM(), monkeypatch)
        turns = msgs("a", "b")
        turns[0].metadata = {"agent_model": "gpt-4o"}
        turns[1].metadata = {"agent_model": "gpt-5"}
        assert manager.budget_for(turns).window == 400_000
        assert manager.budget_for(msgs("a")).window == 128_000

    @pytest.mark.asyncio
    async def test_first_compaction_stores_one_framed_block(self, monkeypatch):
        manager, storage = manager_with(ScriptedLLM(FIRST_OUTPUT), monkeypatch)
        result = await manager.compact("c1", "inst", msgs(*FIRST_TURNS), force=True)
        storage.save.assert_awaited_once()
        assert result.context_block.startswith("## Conversation so far")
        assert result.narrative_summary.overview == result.context_block
        assert result.narrative_summary.factual_paragraph == "" == result.narrative_summary.conversational_paragraph
        assert result.narrative_summary.current_state == {} and result.verbatim_recent_messages == []
        assert result.get_injection_format() == result.context_block
        smr = result.metadata.source_message_range
        assert (smr["message_count"], smr["end_id"], smr["incremental"]) == (2, "id1", False)
        assert result.agreed_actions[0]["who"] == "the assistant"
        assert CompactedContext.from_dict(result.to_dict()).context_block == result.context_block

    @pytest.mark.asyncio
    async def test_an_update_reads_only_the_turns_after_the_boundary(self, monkeypatch):
        first, _ = manager_with(ScriptedLLM(FIRST_OUTPUT), monkeypatch)
        existing = await first.compact("c1", "inst", msgs(*FIRST_TURNS), force=True)
        llm = ScriptedLLM({"changed_sections": [{"id": "s2", "text": "The user said thanks. [m3]"}]})
        manager, _ = manager_with(llm, monkeypatch, existing=existing)
        result = await manager.compact("c1", "inst", msgs(*FIRST_TURNS, "Thanks!"), force=True)
        assert len(llm.calls) == 1
        assert "[m3 · " in llm.user_prompts[0] and "Hi, I'm Priya" not in llm.user_prompts[0]
        assert result.metadata.source_message_range["incremental"] is True

    @pytest.mark.asyncio
    async def test_a_changed_history_is_rebuilt(self, monkeypatch):
        first, _ = manager_with(ScriptedLLM(FIRST_OUTPUT), monkeypatch)
        existing = await first.compact("c1", "inst", msgs(*FIRST_TURNS), force=True)
        llm = ScriptedLLM(FIRST_OUTPUT)
        manager, _ = manager_with(llm, monkeypatch, existing=existing)
        edited = msgs("Hi, I'm Priya. Order ORD-88213-KX arrived broken.", "Sorry, Priya. I'll send you a return label today.", "ok",
                      ids=["idX", "id1", "id2"])
        await manager.compact("c1", "inst", edited, force=True)
        assert "(empty: this is the first compaction" in llm.user_prompts[0]

    @pytest.mark.asyncio
    async def test_no_new_turns_means_no_call(self, monkeypatch):
        first, _ = manager_with(ScriptedLLM(FIRST_OUTPUT), monkeypatch)
        existing = await first.compact("c1", "inst", msgs(*FIRST_TURNS), force=True)
        manager, storage = manager_with(ScriptedLLM(), monkeypatch, existing=existing)
        assert await manager.compact("c1", "inst", msgs(*FIRST_TURNS), force=True) is existing
        storage.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_failed_update_stores_nothing(self, monkeypatch):
        manager, storage = manager_with(ScriptedLLM(RuntimeError("groq down")), monkeypatch)
        with pytest.raises(CompactionError):
            await manager.compact("c1", "inst", msgs(*FIRST_TURNS), force=True)
        storage.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_metering_uses_the_real_token_counts(self, monkeypatch):
        manager, _ = manager_with(ScriptedLLM(FIRST_OUTPUT), monkeypatch)
        manager._track_compaction_usage = MagicMock()
        manager._track_compaction_event = MagicMock()
        await manager.compact("c1", "inst", msgs(*FIRST_TURNS), force=True, client_id="cli")
        kwargs = manager._track_compaction_usage.call_args.kwargs
        assert (kwargs["input_tokens"], kwargs["output_tokens"]) == (1000, 200)


class TestResolvedDates:
    def test_a_relative_date_keeps_its_words_and_shows_its_date(self):
        record = CompactionRecord()
        merge_output(record, {"literals": [{"label": "School form deadline", "value": "the 30th",
                                            "resolved": "2026-10-30", "m": "m7"}]}, {})
        assert check_quotes({"literals": [{"label": "x", "value": "the 30th", "resolved": "2026-10-30", "m": "m7"}]},
                            {"m7": "The school needs it by the 30th."}) == []
        assert "- School form deadline: the 30th (2026-10-30)" in render_document(record)


def test_a_replacement_inside_one_span_keeps_the_old_value():
    """First compaction: Lisbon -> Porto both happen in the new turns."""
    record = CompactionRecord()
    merge_output(record, {"literals": [{"label": "Location", "value": "Porto", "m": "m7", "replaces": "Lisbon"}]},
                 {"m7": T0.isoformat()})
    assert "- Location: Porto (was Lisbon; changed at turn 7, 17 Oct 2026)" in render_document(record)


class TestInventoryCoverage:
    RECORD = CompactionRecord(sections=[Section("s1", "Decisions and reasons",
                                                "Dmitri was rejected: he has no liability insurance. [m5]")])

    def test_an_item_the_record_carries_passes(self):
        from synap.cloud.application.compaction.v3.checks import check_coverage

        inventory = [{"type": "rejected", "item": "User rejected Dmitri because he lacks liability insurance",
                      "m": "m5"}]
        assert check_coverage(self.RECORD, inventory) == []

    def test_an_item_the_record_lost_is_a_problem_and_values_are_skipped(self):
        from synap.cloud.application.compaction.v3.checks import check_coverage

        inventory = [{"type": "feeling", "item": "User is angry: loved the car, did nothing wrong", "m": "m49"},
                     {"type": "value", "item": "Claim CLM-88310-K", "m": "m4"}]
        (problem,) = check_coverage(self.RECORD, inventory)
        assert problem.kind == "coverage" and problem.ref["m"] == "m49"

    @pytest.mark.asyncio
    async def test_inventory_items_are_written_under_their_type(self):
        output = {**FIRST_OUTPUT, "inventory": [
            {"type": "feeling", "item": "Priya is upset the vase was a wedding gift", "m": "m1"},
            {"type": "advice", "item": "The assistant suggested a refund; Priya said maybe", "m": "m2"},
            {"type": "open", "item": "User asked whether the label is prepaid?", "m": "m1"},
            {"type": "value", "item": "Order ORD-88213-KX", "m": "m1"}]}
        result = await writer(ScriptedLLM(output)).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)
        record = result.record
        assert record.section_by_header("How the user feels").text == "- Priya is upset the vase was a wedding gift. [m1]"
        assert record.section_by_header("Proposals and responses").text.startswith("- The assistant suggested")
        assert record.section_by_header("Open items") is None and record.section_by_header("Facts") is None

    @pytest.mark.asyncio
    async def test_items_grouped_by_turn_are_written_with_their_why(self):
        output = {**FIRST_OUTPUT, "turns": [
            {"m": "m1", "items": [{"type": "rejected", "item": "Priya turned down a replacement",
                                   "why": "she needs the money back before rent"}]},
            {"m": "m2", "items": [{"type": "advice", "item": "The assistant suggested photos of the box",
                                   "why": "Priya said she threw it away"}]}]}
        result = await writer(ScriptedLLM(output)).update(None, label_turns(msgs(*FIRST_TURNS)), SETTINGS)
        record = result.record
        assert record.section_by_header("Decisions and reasons").text == (
            "- Priya turned down a replacement — she needs the money back before rent. [m1]")
        assert record.section_by_header("Proposals and responses").text == (
            "- The assistant suggested photos of the box, response: Priya said she threw it away. [m2]")

    def test_a_why_the_item_already_says_or_that_starts_with_a_connective_is_not_doubled(self):
        from synap.cloud.application.compaction.v3.record import item_text

        assert item_text({"type": "decision", "item": "Chose rail because it is cheaper",
                          "why": "it is cheaper"}) == "Chose rail because it is cheaper"
        assert item_text({"type": "decision", "item": "Chose rail", "why": "since it is cheaper"}) == (
            "Chose rail, since it is cheaper")
        assert item_text({"type": "fact", "item": "Lease ends in May", "why": ""}) == "Lease ends in May"

    def test_a_value_the_prose_mentions_without_its_meaning_stays_in_key_values(self):
        record = CompactionRecord(sections=[Section("s1", "Facts", "- Loan-to-value at $438,000 is about 71%. [m21]\n"
                                                                    "- The appraiser revised the value. [m21]")])
        merge_output(record, {"literals": [{"label": "Revised appraisal value", "value": "$438,000", "m": "m21"},
                                           {"label": "Loan-to-value", "value": "71%", "m": "m21"}]}, {})
        document = render_document(record)
        assert "- Revised appraisal value: $438,000" in document
        assert "- Loan-to-value: 71%" not in document

    @pytest.mark.asyncio
    async def test_a_value_the_prose_has_only_in_another_sense_is_still_kept(self):
        """ "15:12" in "abandoned between 14:05 and 15:12" does not say a release went out at 15:12."""
        turns = ("Checkouts were abandoned between 14:05 and 15:12.", "Okay.", "v4.18.3 is deployed at 15:12 UTC.")
        output = {"changed_sections": [{"header": "Facts", "text": "- Checkouts were abandoned between 14:05 and 15:12. [m1]"}],
                  "literals": [{"label": "Abandoned from", "value": "14:05", "m": "m1"},
                               {"label": "Abandoned until", "value": "15:12", "m": "m1"}]}
        result = await writer(ScriptedLLM(output, output)).update(None, label_turns(msgs(*turns)), SETTINGS)
        assert '"v4.18.3 is deployed at 15:12 UTC." (turn 3)' in render_document(result.record)

    @pytest.mark.asyncio
    async def test_a_suggestion_the_writer_left_out_is_kept_verbatim_with_the_reply(self):
        turns = ("Order ORD-88213-KX arrived broken.",
                 "Sorry. You might also want to photograph the box for the claim.",
                 "Hm, I already threw it away.")
        output = {**FIRST_OUTPUT, "agreed_actions": []}
        result = await writer(ScriptedLLM(output, output)).update(None, label_turns(msgs(*turns)), SETTINGS)
        assert result.record.section_by_header("Proposals and responses").text == (
            '- the assistant suggested: "You might also want to photograph the box for the claim."; '
            'the user replied: "Hm, I already threw it away." [m2]')

    @pytest.mark.asyncio
    async def test_a_response_that_is_not_the_reply_is_replaced_by_the_reply(self):
        turns = ("Order ORD-88213-KX arrived broken.",
                 "Sorry. You might also want to photograph the box for the claim.",
                 "Hm, I already threw it away.")
        output = {**FIRST_OUTPUT, "agreed_actions": [], "turns": [{"m": "m2", "items": [
            {"type": "advice", "item": "the assistant suggested photographing the box", "why": "no reply"}]}]}
        result = await writer(ScriptedLLM(output, output)).update(None, label_turns(msgs(*turns)), SETTINGS)
        assert result.record.section_by_header("Proposals and responses").text == (
            '- the assistant suggested photographing the box, response: the user replied: '
            '"Hm, I already threw it away.". [m2]')

    @pytest.mark.asyncio
    async def test_the_suggesters_own_reason_written_as_a_response_stays_a_reason(self):
        turns = ("Order ORD-88213-KX arrived broken.",
                 "Sorry. You might also want to photograph the box for the claim.",
                 "What's the refund amount?")
        output = {**FIRST_OUTPUT, "agreed_actions": [], "turns": [{"m": "m2", "items": [
            {"type": "advice", "item": "the assistant suggested photographing the box", "why": "the claim needs photos"}]}]}
        result = await writer(ScriptedLLM(output, output)).update(None, label_turns(msgs(*turns)), SETTINGS)
        assert result.record.section_by_header("Proposals and responses").text == (
            "- the assistant suggested photographing the box — the claim needs photos. [m2]")

    @pytest.mark.asyncio
    async def test_a_promise_the_writer_left_out_is_kept_verbatim(self):
        turns = ("Hi, I'm Priya. Order ORD-88213-KX arrived broken.",
                 "Sorry, Priya. I'll send you a return label today.",
                 "Thanks. I'll post the box back on Friday.")
        result = await writer(ScriptedLLM(FIRST_OUTPUT)).update(None, label_turns(msgs(*turns)), SETTINGS)
        section = result.record.section_by_header("Promises, as said")
        assert section.text == 'the user said: "I\'ll post the box back on Friday." [m3]'

    def test_names_of_two_capitalised_words_are_literals_but_sentence_openers_are_not(self):
        from synap.cloud.application.compaction.v3.checks import literal_candidates

        found = literal_candidates(["The adjuster is Marisol Ekwueme. Okay That works. Ask Mrs. Halvorsen."])
        assert "Marisol Ekwueme" in found and "Mrs. Halvorsen" in found and "Okay That" not in found

    def test_a_change_a_later_turn_states_updates_the_stale_value(self):
        from synap.cloud.application.compaction.v3.record import apply_stated_changes

        record = CompactionRecord(sections=[Section("s1", "Facts", "- Visa appointment on 18 November at 10:15. [m5]")])
        merge_output(record, {"literals": [
            {"label": "Visa appointment date and time", "value": "18 November at 10:15", "m": "m5"},
            {"label": "start date", "value": "1 February 2027", "m": "m6"}]}, {})
        apply_stated_changes(record, {
            "m40": "The visa appointment moved from 18 November at 10:15 to 25 November at 09:40.",
            "m41": "So the start date moves from 1 February to 15 February 2027 because Sasha is away."}, {})
        assert [(e.label, e.value, [h.value for h in e.history]) for e in record.ledger] == [
            ("Visa appointment date and time", "25 November at 09:40", ["18 November at 10:15"]),
            ("start date", "15 February 2027", ["1 February 2027"])]
        assert "(later changed to 25 November at 09:40, turn 40)" in record.sections[0].text

    def test_a_new_quantity_or_a_summary_is_not_a_change(self):
        from synap.cloud.application.compaction.v3.record import apply_stated_changes

        record = CompactionRecord()
        merge_output(record, {"literals": [{"label": "Current loan balance", "value": "$312,000", "m": "m2"},
                                           {"label": "Hotel cost per night", "value": "EUR 210", "m": "m3"},
                                           {"label": "Last day at work", "value": "15 January", "m": "m4"},
                                           {"label": "original budget", "value": "$40,000", "m": "m5"}]}, {})
        apply_stated_changes(record, {
            "m37": "Rolling $6,850 into the loan brings the new balance to $318,850.",
            "m40": "That's everything updated: Hotel Ribeira Oeste per night, budget EUR 2,300.",
            "m41": "Your last day at work stays 15 January even though the start moved to 15 February.",
            "m42": "Budget revised from $40,000 to $47,000."}, {})
        assert [e.value for e in record.ledger] == ["$312,000", "EUR 210", "15 January", "$40,000"]

    def test_sentences_are_not_split_after_titles(self):
        from synap.cloud.application.compaction.v3.record import split_sentences

        assert split_sentences("Ask Dr. Sandhu. Mrs. Halvorsen is annoyed! Fine.") == [
            "Ask Dr. Sandhu.", "Mrs. Halvorsen is annoyed!", "Fine."]

    def test_written_out_dates_are_literals(self):
        from synap.cloud.application.compaction.v3.checks import literal_candidates

        found = literal_candidates(["Register by 24 October, and the lock runs to December 1. May I ask?"])
        assert "24 October" in found and "December 1" in found
        assert not any(v.startswith("May") for v in found)


class TestProbeGapFixes:
    def test_two_values_under_one_label_in_one_output_are_two_items(self):
        """Three commits named in one turn came out as "9c0d5e8 (was b7e41d0; 3f9a2c1)"."""
        record = CompactionRecord()
        merge_output(record, {"literals": [{"label": "Commit", "value": "3f9a2c1", "m": "m13"},
                                           {"label": "Commit", "value": "b7e41d0", "m": "m13"},
                                           {"label": "Commit", "value": "9c0d5e8", "m": "m13"}]}, {})
        # No numbered labels: "Rental coverage per day (2): $61" said $61 was
        # coverage, and it was the rental's price. The others stay unlabelled
        # until the engine gives them their sentence.
        assert [(e.label, e.value, e.history, e.auto) for e in record.ledger] == [
            ("Commit", "3f9a2c1", [], False), ("value from m13", "b7e41d0", [], True),
            ("value from m13", "9c0d5e8", [], True)]

    async def test_an_unlabelled_value_is_shown_with_its_sentence(self):
        output = {**FIRST_OUTPUT, "literals": [
            {"label": "Rental coverage per day", "value": "$45", "m": "m1"},
            {"label": "Rental coverage per day", "value": "$61", "m": "m1"}]}
        turns = label_turns(msgs("Coverage is $45 a day. The AWD rental is $61 a day, so I pay the gap.",
                                 "Understood."))
        result = await writer(ScriptedLLM(output)).update(None, turns, SETTINGS)
        entry = next(e for e in result.record.ledger if e.value == "$61")
        assert entry.auto and entry.label == "The AWD rental is $61 a day, so I pay the gap."
        assert "(2)" not in render_document(result.record)


class TestInventedClaimGates:
    """P2 of COMPACTION_HYBRID_PLAN.md: lines the claims judge found invented on 19 Oct."""

    def test_a_reply_must_be_about_the_suggestion(self):
        from synap.cloud.application.compaction.v3.checks import answers

        about = "You might want to keep the tire receipt"
        assert answers(about, "Good point, I kept the receipt.")  # shares a word
        assert answers(about, "Maybe later.")  # a direct answer
        assert answers(about, "Hm, I already threw it away.")  # points back
        assert not answers(about, "What's the offer?")
        assert not answers(about, "The adjuster called at 3pm with the valuation figure for the car.")

    def test_code_never_writes_because(self):
        from synap.cloud.application.compaction.v3.record import item_text

        assert item_text({"type": "feeling", "item": "the user is stressed about the neighbor",
                          "why": "more than the money"}) == "the user is stressed about the neighbor — more than the money"

    def test_a_rejected_item_says_it_was_turned_down(self):
        from synap.cloud.application.compaction.v3.record import item_text

        assert item_text({"type": "rejected", "item": "rolling back the whole release",
                          "why": "it would remove the tax fix"}) == (
            "Turned down: rolling back the whole release — it would remove the tax fix")
        assert item_text({"type": "rejected", "item": "Tomas ruled out the lake house", "why": ""}) == (
            "Tomas ruled out the lake house")

    def test_a_resolved_date_the_value_contradicts_is_dropped(self):
        from synap.cloud.application.compaction.v3.record import resolved_date_ok

        at = "2026-09-20T09:05:00+00:00"  # a Sunday
        assert resolved_date_ok("Monday", "2026-09-14", at)
        assert resolved_date_ok("Monday", "2026-09-21", at)
        assert not resolved_date_ok("Monday", "2026-09-22", at)  # a Tuesday
        assert not resolved_date_ok("Monday", "2026-11-02", at)  # a Monday, but six weeks out
        assert resolved_date_ok("December 4", "2026-12-04", at)
        assert not resolved_date_ok("December 4", "2026-12-14", at)
        assert not resolved_date_ok("the 30th", "2026-10-29", at)
        assert resolved_date_ok("next week", "2026-09-28", at)  # nothing to contradict
        record = CompactionRecord()
        merge_output(record, {"literals": [{"label": "Inspection", "value": "Wednesday", "resolved": "2026-09-24",
                                            "m": "m3"}]}, {"m3": at})
        assert record.ledger[0].resolved == ""

    def test_one_value_written_twice_under_one_label_is_one_item(self):
        record = CompactionRecord()
        merge_output(record, {"literals": [{"label": "Price advantage", "value": "$3,050", "m": "m10"},
                                           {"label": "Price advantage", "value": "3,050", "m": "m10"}]}, {})
        assert [(e.label, e.value) for e in record.ledger] == [("Price advantage", "$3,050")]

    def test_a_marked_replacement_in_one_output_is_still_a_replacement(self):
        record = CompactionRecord()
        merge_output(record, {"literals": [{"label": "Plan", "value": "Aetna PPO", "m": "m1"},
                                           {"label": "Plan", "value": "Aetna HSA", "m": "m8", "replaces": "Aetna PPO"}]}, {})
        assert [(e.value, [h.value for h in e.history]) for e in record.ledger] == [("Aetna HSA", ["Aetna PPO"])]

    def test_a_new_value_for_a_stored_label_is_a_replacement(self):
        record = CompactionRecord(ledger=[LedgerEntry("Start date", "October 6", "m1")])
        merge_output(record, {"literals": [{"label": "Start date", "value": "October 13", "m": "m9"}]}, {})
        assert record.ledger[0].value == "October 13" and record.ledger[0].history[0].value == "October 6"

    def test_the_literal_check_sees_paths_numbers_hashes_percentages_and_times(self):
        from synap.cloud.application.compaction.v3.checks import literal_candidates

        found = literal_candidates(["jobs/orders_enriched/conf.yaml; 7be21d0; 1,204,551 rows; 22%; 15:30; "
                                    "https://x.io/a/b.html"])
        assert {"jobs/orders_enriched/conf.yaml", "7be21d0", "1,204,551", "22%", "15:30"} <= set(found)
        assert "io/a/b.html" not in found


class TestStatuses:
    """Plans and promises say whether a later turn reports them done (19 Oct)."""

    def _record(self, writer_status):
        from synap.cloud.application.compaction.v3.record import add_inventory, MergeNotes

        record = CompactionRecord(covers={"end_m": "m9"})
        add_inventory(record, [{"type": "plan", "item": "the user will sign the Brightline contract on Thursday",
                                "m": "m3", "status": writer_status}], MergeNotes())
        return record

    def test_a_plan_the_writer_and_code_both_see_as_not_done_says_so(self):
        from synap.cloud.application.compaction.v3.record import track_statuses

        record = self._record("not_done")
        track_statuses(record, {"m4": "Great, talk Thursday.", "m9": "Thanks!"})
        assert "- the user will sign the Brightline contract on Thursday (not reported done by turn 9)." in (
            render_document(record))

    def test_a_later_turn_that_says_it_happened_marks_it_done_with_its_words(self):
        from synap.cloud.application.compaction.v3.record import track_statuses

        record = self._record("not_done")
        track_statuses(record, {"m7": "Is it signed? Yes: we signed the Brightline contract this morning."})
        doc = render_document(record)
        assert '(done, turn 7: "Yes: we signed the Brightline contract this morning.")' in doc
        assert "not reported done" not in doc

    def test_without_the_writers_not_done_nothing_is_claimed(self):
        from synap.cloud.application.compaction.v3.record import track_statuses

        for status in ("unclear", "done", ""):
            record = self._record(status)
            track_statuses(record, {"m4": "Let's not worry about that contract now."})
            doc = render_document(record)
            assert "not reported done" not in doc and "(done," not in doc, status

    def test_a_question_is_not_evidence_and_open_items_get_no_status(self):
        from synap.cloud.application.compaction.v3.record import track_statuses

        record = self._record("not_done")
        record.sections.append(Section("s9", "Open items", "- could it be the payment provider? [m2]"))
        track_statuses(record, {"m5": "Have you signed the Brightline contract yet?"})
        doc = render_document(record)
        assert "(not reported done by turn 9)" in doc
        assert "- could it be the payment provider?\n" in doc + "\n"

    def test_statuses_survive_a_round_trip(self):
        record = self._record("not_done")
        assert CompactionRecord.from_dict(record.to_dict()).statuses == record.statuses


# =============================================================================
# statuses the writer carries from chunk to chunk (20 Oct experiment)
# =============================================================================
class TestCarriedStatuses:
    TURNS = ("The initial term is five years and the go-live is 1 February.",
             "Noted: five-year term, go-live 1 February.",
             "Update from Renata: the term is now three years, and go-live moved to 1 March.",
             "Got it.")

    FIRST = {
        "turns": [{"m": "m1", "items": [
            {"type": "fact", "item": "The initial term is five years", "why": ""},
            {"type": "plan", "item": "Go-live is planned for 1 February", "why": "", "status": "not_done"}]}],
        "literals": [{"label": "Go-live date", "value": "1 February", "m": "m1"}],
    }

    def _second(self, statuses):
        return {"turns": [{"m": "m3", "items": [{"type": "fact", "item": "Renata said the term is now three years",
                                                  "why": ""}]}],
                "literals": [], "statuses": statuses}

    async def _run(self, mode, statuses):
        llm = ScriptedLLM(self.FIRST, self._second(statuses))
        settings = EngineSettings(target_ratio=0.30, record_ceiling=7_500, chunk_tokens=15_000, status_mode=mode)
        first = await writer(llm).update(None, label_turns(msgs(*self.TURNS[:2])), settings)
        second = await writer(llm).update(first.record, label_turns(msgs(*self.TURNS[2:], start=2), first_position=3),
                                          settings)
        return llm, second

    @pytest.mark.asyncio
    async def test_the_writer_sees_ids_and_the_rules(self):
        llm, _ = await self._run("restate", [])
        system = llm.calls[1]["messages"][0]["content"]
        assert "give one entry for EVERY id" in system
        view = json.loads(llm.user_prompts[1].split("CURRENT RECORD\n", 1)[1].split("\n\nNEW TURNS", 1)[0])
        assert view["sections"][0]["lines"][0] == {"id": "L1", "text": "- The initial term is five years. [m1]"}
        assert view["ledger"][0]["id"] == "V1"

    @pytest.mark.asyncio
    async def test_a_replaced_line_leaves_its_section_with_what_it_is_now(self):
        _, result = await self._run("delta", [
            {"id": "L1", "status": "replaced", "now": "three years", "m": "m3",
             "quote": "the term is now three years"},
            {"id": "V1", "status": "replaced", "now": "1 March", "m": "m3", "quote": "go-live moved to 1 March"}])
        doc = render_document(result.record)
        facts = doc.split("### Facts\n")[1].split("###")[0]
        assert "five years" not in facts
        assert "- The initial term is five years → now: three years (turn 3)." in doc.split("### No longer current")[1]
        entry = result.record.ledger_by_label("Go-live date")
        assert entry.value == "1 March" and [h.value for h in entry.history] == ["1 February"]
        assert result.status_counts == {"applied": 2, "ignored": 0, "listed": 2}

    @pytest.mark.asyncio
    async def test_a_status_without_a_quote_from_the_new_turns_is_ignored(self):
        _, result = await self._run("restate", [
            {"id": "L1", "status": "replaced", "now": "three years", "m": "m3", "quote": "the term was renegotiated"},
            {"id": "L2", "status": "done", "m": "m1", "quote": "the go-live is 1 February"},  # not a new turn
            {"id": "L1", "status": "current"}])
        assert "No longer current" not in render_document(result.record)
        assert result.status_counts["ignored"] == 2

    @pytest.mark.asyncio
    async def test_a_done_plan_says_so_in_place_and_code_adds_no_statuses(self):
        llm, result = await self._run("delta", [
            {"id": "L2", "status": "done", "m": "m3", "quote": "go-live moved to 1 March"}])
        doc = render_document(result.record)
        assert '- Go-live is planned for 1 February (done, turn 3: "go-live moved to 1 March").' in doc
        assert "not reported done" not in doc and "later changed to" not in doc

    def test_line_states_survive_a_round_trip(self):
        record = CompactionRecord(status_mode="delta", line_states={"x": {"state": "done", "m": "m2"}},
                                  ledger=[LedgerEntry("Cap", "6 months", "m1", stale={"state": "declined", "m": "m4"})])
        again = CompactionRecord.from_dict(record.to_dict())
        assert again.status_mode == "delta" and again.line_states == record.line_states
        assert again.ledger[0].stale == {"state": "declined", "m": "m4"}


class TestValueNet:
    LOG = ("The deploy failed again, see below.\n```\n2026-10-19 07:02:41 ERROR pod ledgerhook-api-5b7c9d8f6 "
           "port 9187 refused\n```\nTomasz says ticket PAY-2291 covers it.")

    @pytest.mark.asyncio
    async def test_trim_skips_values_found_only_in_pasted_logs(self):
        out = {"turns": [{"m": "m1", "items": [{"type": "fact", "item": "The deploy failed again", "why": ""}]}],
               "literals": []}
        for net, kept in (("full", True), ("trim", False)):
            llm = ScriptedLLM(out, out)
            settings = EngineSettings(target_ratio=0.30, record_ceiling=7_500, chunk_tokens=15_000, value_net=net)
            result = await writer(llm).update(None, label_turns(msgs(self.LOG, "Ok.")), settings)
            values = [e.value for e in result.record.ledger]
            assert ("5b7c9d8f6" in values) is kept, net
            assert "PAY-2291" in values, net  # said in someone's own words
            if net == "trim":
                assert "5b7c9d8f6" in result.skipped_values

    def test_value_kinds(self):
        from synap.cloud.application.compaction.v3.checks import value_kind

        assert [value_kind(v) for v in ("$3.55", "99.7%", "e7c2d18", "src/app/main.py", "18,442.70",
                                        "2.14.1-rc1", "PAY-2291", "Nadia Ferris", "21 November")] == [
            "money", "percent", "commit_hash", "file_path", "number", "version", "code_or_id", "name", "date"]


class TestStatusBlock:
    """One "where things stand" block, returned whole by each writer call."""

    @pytest.mark.asyncio
    async def test_the_block_is_carried_and_replaced_whole(self):
        first = {**TestCarriedStatuses.FIRST, "current_status": ["Initial term: five years", "Go-live: 1 February"]}
        second = {"turns": [], "literals": [], "current_status": ["Initial term: three years (was five years)",
                                                                   "Go-live: 1 March (was 1 February)"]}
        llm = ScriptedLLM(first, second)
        settings = EngineSettings(target_ratio=0.30, record_ceiling=7_500, chunk_tokens=15_000, status_mode="block")
        one = await writer(llm).update(None, label_turns(msgs(*TestCarriedStatuses.TURNS[:2])), settings)
        two = await writer(llm).update(one.record, label_turns(msgs(*TestCarriedStatuses.TURNS[2:], start=2),
                                                               first_position=3), settings)
        assert "CURRENT STATUS\n- Initial term: five years\n- Go-live: 1 February" in llm.user_prompts[1]
        assert '"current_status": [' in llm.calls[1]["messages"][0]["content"]
        doc = render_document(two.record)
        assert doc.startswith("### Where things stand now")
        assert "- Initial term: three years (was five years)" in doc and "- Go-live: 1 February" not in doc

    def test_an_empty_answer_keeps_the_block(self):
        from synap.cloud.application.compaction.v3.statuses import apply_block

        record = CompactionRecord(status_block=["Contract: unsigned"])
        assert apply_block(record, {"current_status": []}) == 0 and record.status_block == ["Contract: unsigned"]


class TestValueLinks:
    TABLE = "| Corvane | Storage | 2,100 pallets x 10 months | $18.50 | $388,500 |\n| TOTAL FY27 | | | | $2,181,418 |"
    FIX = "| Line | Was | Now |\n| Storage | $388,500 | $357,000 |\n| Total FY27 | $2,181,418 | $2,148,343 |"

    @pytest.mark.asyncio
    async def test_a_corrected_table_value_replaces_the_one_code_kept(self):
        empty = {"turns": [], "literals": []}
        for link, chained in ((False, False), (True, True)):
            llm = ScriptedLLM(empty, empty, empty, empty)
            settings = EngineSettings(target_ratio=0.30, record_ceiling=7_500, chunk_tokens=15_000, link_values=link)
            one = await writer(llm).update(None, label_turns(msgs(self.TABLE, "Thanks.")), settings)
            two = await writer(llm).update(one.record, label_turns(msgs(self.FIX, "Ok.", start=2), first_position=3),
                                           settings)
            doc = render_document(two.record)
            assert ("$388,500 → $357,000 (now $357,000)" in doc) is chained, link
            assert ("$2,181,418 → $2,148,343 (now $2,148,343)" in doc) is chained, link

    def test_keys_and_ambiguity(self):
        from synap.cloud.application.compaction.v3.checks import linked_entry, value_key

        assert value_key("$388,500", self.TABLE) == "corvan storag"
        assert value_key("$2,181,418", self.TABLE) == "total"
        assert value_key("$40", "The storage fee is $40 a pallet.") == "storag fee"
        two = [LedgerEntry("a", "$10", "m1", auto=True, key="storag fee corvan"),
               LedgerEntry("b", "$12", "m2", auto=True, key="storag fee ostran")]
        assert linked_entry(two, "$11", "storag fee", "m5") is None  # two candidates: no link
        assert linked_entry(two, "$11", "storag fee", "m5", "| Storage fee | $12 | $11 |") is two[1]
        assert linked_entry(two, "$11", "storag fee ostran", "m5") is two[1]  # the same key
        assert linked_entry(two, "11%", "storag fee ostran", "m5") is None  # another kind of value


class TestDocumentFixes:
    """20 Oct, documents case: contact values always kept; table values say which document they came from."""

    SIGNATURE = ("Thanks, talk soon.\n\nRenata Okafor\nSenior Account Executive\n"
                 "Direct: (775) 555-0148 | Mobile: (775) 555-0193\nrenata.okafor@corvane.example")

    @pytest.mark.asyncio
    async def test_contact_values_survive_a_kinds_list_without_them(self):
        out = {"turns": [{"m": "m1", "items": [{"type": "fact", "item": "Renata signed off", "why": ""}]}],
               "literals": []}
        llm = ScriptedLLM(out, out)
        settings = EngineSettings(target_ratio=0.30, record_ceiling=7_500, chunk_tokens=15_000,
                                  value_net="kinds", value_kinds=("money", "date"))
        result = await writer(llm).update(None, label_turns(msgs(self.SIGNATURE, "Ok.")), settings)
        values = [e.value for e in result.record.ledger]
        assert "(775) 555-0148" in values and "renata.okafor@corvane.example" in values

    def test_a_table_value_is_labelled_with_the_table_caption(self):
        from synap.cloud.application.compaction.v3.engine import _context

        text = "Corvane sent the revised schedule.\n\nSchedule B - Fees (Rev. 3)\n| Item | Value |\n|---|---|\n| Monthly minimum | 16,000 |"
        assert _context("16,000", text).startswith("[Schedule B - Fees (Rev. 3)] ")
        assert _context("16,000", "The minimum is 16,000 now.") == "The minimum is 16,000 now."


class TestValueHygiene:
    """20 Oct, documents case: no self-labelled values; amounts keep their unit."""

    def test_a_value_labelled_with_itself_is_dropped(self):
        record = CompactionRecord()
        merge_output(record, {"literals": [{"label": "This Agreement", "value": "This Agreement", "m": "m5"},
                                           {"label": "Initial term", "value": "five years", "m": "m5"}]}, {})
        assert [e.value for e in record.ledger] == ["five years"]

    def test_an_amount_keeps_its_unit(self):
        from synap.cloud.application.compaction.v3.checks import literal_candidates
        from synap.cloud.application.compaction.v3.engine import _complete_suffixes

        assert literal_candidates(["Fulfillment goes from $2.4M to $2.25M; we take $150k out."]) == [
            "$2.4M", "$2.25M", "$150k"]
        record = CompactionRecord(ledger=[LedgerEntry("First-year estimate", "$2.01", "m10")])
        _complete_suffixes(record, {"m10": "The first-year estimate is $2.01M all in."})
        assert record.ledger[0].value == "$2.01M"
