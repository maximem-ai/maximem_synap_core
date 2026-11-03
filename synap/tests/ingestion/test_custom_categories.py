"""memory_categories.custom_categories in extraction: the write side.

A memory of a custom category is stored as ``facts`` with the category's canonical name in
``category``. These tests drive the real Stage3Extractor and the real Core prompt text with
a model client that records what it is sent and replies with fixed items.
"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedMemory, MemoryDestination, Stage2Output,
)
from synap.cloud.agents.ingestion.models.enums import MemoryType
from synap.cloud.agents.ingestion.pipeline import stage3_extraction as s3
from synap.cloud.agents.ingestion.services import coalesce, custom_categories as cc
from synap.cloud.memory_config.config_schema import ConfigMetadata, MemoryArchitectureConfig
from synap.cloud.shared.prompt_switch import fill, pack_text

FLAG = "SYNAP_CUSTOM_CATEGORIES"
INST = "44eff2d7-2a3d-5075-8c39-addf2a0188d0"
OTHER = "4fb56f47-3fc9-57c7-aeed-5ac445e554ff"
CONTENT = "user: I checked in at the Hilton on Friday and I adopted a dog named Rex."
# sha256 of core_prompts/merged_extraction/system.txt as it was before A12e
ORIGINAL_SYSTEM_SHA = "b954e1f3b3ddc3ed0f66922253fcdf7eaa83ad8ba9edd95d378f9281704e34c1"

CHECKIN = {"name": "Check In", "description": "A stay at a hotel or a flight boarded.",
           "examples": ["checked in at the Ritz", "boarded flight 12"]}
CATEGORIES = [CHECKIN, {"name": "Warranty", "description": "Coverage on a purchase.", "examples": []}]

REPLY = json.dumps({"reasoning": "", "extracted": [
    {"type": "check_in", "content": "The user checked in at the Hilton on Friday", "confidence": 0.95},
    {"type": "facts", "content": "The user adopted a dog named Rex", "confidence": 0.95},
    {"type": "warranty", "content": "The user has a two year warranty on the laptop", "confidence": 0.2},
]})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (FLAG, f"{FLAG}_INSTANCES", "SYNAP_MACA_EXTRACTION_INSTANCES",
                "STAGE3_EXTRACTION_MODE", "STAGE3_EXTRACTION_MODE_BY_INSTANCE",
                "SYNAP_ASSISTANT_RECALL_INSTANCES", "ENABLE_SPEAKER_ATTRIBUTION", "EVAL_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE", "hybrid")


@pytest.fixture
def core_prompt(monkeypatch):
    """Core's merged prompt behind the interface, as Core binds it."""
    from synap.cloud.agents.ingestion.prompts.extraction import merged_simple as interface
    from synap.cloud.agents.ingestion.prompts.extraction.core import merged_simple as core
    monkeypatch.setattr(interface, "MergedSimpleExtractionPrompt", core.MergedSimpleExtractionPrompt)


def _config(custom=None, simple_enabled=True) -> MemoryArchitectureConfig:
    raw = {"metadata": ConfigMetadata(
        client_id="c", instance_id="i", config_version="1.0.0",
        generated_at=datetime.now(timezone.utc), generated_by="test")}
    if custom is not None:
        raw["memory_categories"] = {"custom_categories": custom}
    cfg = MemoryArchitectureConfig(**raw)
    if not simple_enabled:
        for name in ("facts", "preferences", "emotions"):
            getattr(cfg.memory_categories, name).enabled = False
    return cfg


class Capture:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def complete(self, messages=None, **kwargs):
        self.calls.append(messages if messages is not None else kwargs.get("prompt"))
        return self.reply


async def _extract(config, instance_id=INST, reply=REPLY, startup=None, priorities=None):
    main, light = Capture(reply), Capture(reply)
    ex = s3.Stage3Extractor(config=startup or _config(), llm_client=main, llm_client_light=light)
    scope = SimpleNamespace(instance_id=instance_id, client_id=None, customer_id=None,
                            user_id=None, scope_level="user")
    out = await ex.extract(
        content=CONTENT,
        stage2_output=Stage2Output(
            input_id="t", categories=[],
            extraction_priorities=priorities or [MemoryType.FACTS]),
        document_id="d", dct=None, has_temporal_content=False, request_scope=scope,
        correlation_id="c", config=config)
    return out, main.calls + light.calls


def _system(calls):
    return calls[0][0]["content"]


# --- the helper ---------------------------------------------------------------
def test_the_helper_is_empty_while_the_switch_is_off():
    assert cc.custom_categories_for(_config(CATEGORIES), INST) == ()


def test_the_helper_never_reads_the_config_with_the_switch_off():
    class Wrapper:
        @property
        def memory_categories(self):
            raise AssertionError("config read with the switch off")
    assert cc.custom_categories_for(Wrapper(), INST) == ()


def test_names_are_canonical_and_the_rest_is_kept(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    got = cc.custom_categories_for(_config(CATEGORIES), INST)
    assert [c.name for c in got] == ["check_in", "warranty"]
    assert got[0].description == "A stay at a hotel or a flight boarded."
    assert got[0].examples == ("checked in at the Ritz", "boarded flight 12")


def test_a_disabled_one_a_built_in_name_a_repeat_and_an_empty_name_are_left_out(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    cats = [
        {"name": "Hidden", "description": "d", "enabled": False},
        {"name": " Facts ", "description": "collides with a built-in"},
        {"name": "temporal events", "description": "collides once spaces become underscores"},
        {"name": "   ", "description": "empty"},
        {"name": "Trip Plan", "description": "one"},
        {"name": "trip   plan", "description": "the same name again"},
    ]
    assert [c.name for c in cc.custom_categories_for(_config(cats), INST)] == ["trip_plan"]


def test_a_long_name_is_cut_to_the_column_width_and_examples_to_five(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    cats = [{"name": "x" * 150, "description": "d", "examples": [f"e{i}" for i in range(9)] + [" "]}]
    (only,) = cc.custom_categories_for(_config(cats), INST)
    assert len(only.name) == 100
    assert only.examples == ("e0", "e1", "e2", "e3", "e4")


def test_the_switch_can_be_named_per_instance(monkeypatch):
    monkeypatch.setenv(f"{FLAG}_INSTANCES", INST)
    assert cc.custom_categories_for(_config(CATEGORIES), INST)
    assert cc.custom_categories_for(_config(CATEGORIES), OTHER) == ()


def test_the_prompt_lines_are_data_only(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    lines = cc.format_for_prompt(cc.custom_categories_for(_config(CATEGORIES), INST)).split("\n")
    assert lines == [
        '- check_in: A stay at a hotel or a flight boarded. '
        '(examples: "checked in at the Ritz"; "boarded flight 12")',
        "- warranty: Coverage on a purchase.",
    ]


def test_the_compaction_loader_and_the_helper_agree_on_who_is_enabled(monkeypatch):
    from synap.cloud.application.compaction.templates.loader import TemplateLoader
    monkeypatch.setenv(FLAG, "true")
    raw = [{"name": "checkin", "description": "d"}, {"name": "off", "description": "d", "enabled": False},
           {"name": "warranty", "description": "d"}]
    loader_names = TemplateLoader._get_maca_custom_categories(None, {"memory_categories": {"custom_categories": raw}})
    assert loader_names == [c.name for c in cc.custom_categories_for(_config(raw), INST)]


# --- the prompt ---------------------------------------------------------------
def test_the_pack_text_with_no_custom_types_is_the_text_it_was():
    out = fill(pack_text("merged_extraction", "system.txt"), builtin_only=True, custom_types="")
    assert hashlib.sha256(out.encode()).hexdigest() == ORIGINAL_SYSTEM_SHA


def test_the_pack_text_with_custom_types_drops_the_three_types_sentence():
    out = fill(pack_text("merged_extraction", "system.txt"), builtin_only=False, custom_types="- a: b")
    assert "one of three types" not in out
    assert 'exactly one of "facts", "preferences", "emotions".' not in out
    assert "- a: b" in out and "the name of one custom type" in out
    assert "[[" not in out


@pytest.mark.asyncio
async def test_switch_off_the_prompt_is_the_same_with_or_without_custom_categories(core_prompt):
    _, plain = await _extract(_config())
    _, listed = await _extract(_config(CATEGORIES))
    assert json.dumps(plain) == json.dumps(listed)
    system = _system(listed)
    assert "check_in" not in system and "Coverage on a purchase" not in system
    original = fill(pack_text("merged_extraction", "system.txt"), builtin_only=True, custom_types="")
    assert hashlib.sha256(original.encode()).hexdigest() == ORIGINAL_SYSTEM_SHA
    assert system.startswith(original)


@pytest.mark.asyncio
async def test_switch_on_the_prompt_names_each_type_with_its_description_and_examples(core_prompt, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    _, calls = await _extract(_config(CATEGORIES))
    system = _system(calls)
    assert "- check_in: A stay at a hotel or a flight boarded." in system
    assert '"checked in at the Ritz"' in system and '"boarded flight 12"' in system
    assert "- warranty: Coverage on a purchase." in system
    assert "one of three types" not in system and "[[" not in system


@pytest.mark.asyncio
async def test_a_disabled_category_is_not_offered(core_prompt, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    cats = [CHECKIN, {"name": "Warranty", "description": "Coverage on a purchase.", "enabled": False}]
    _, calls = await _extract(_config(cats))
    assert "check_in" in _system(calls) and "Coverage on a purchase" not in _system(calls)


@pytest.mark.asyncio
async def test_a_built_in_name_and_a_repeat_are_offered_once_or_not_at_all(core_prompt, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    cats = [{"name": "Facts", "description": "BUILTIN-CLASH"}, CHECKIN, dict(CHECKIN, description="REPEAT")]
    _, calls = await _extract(_config(cats))
    system = _system(calls)
    assert "BUILTIN-CLASH" not in system and "REPEAT" not in system
    assert system.count("- check_in:") == 1


# --- the accept rule ----------------------------------------------------------
@pytest.mark.asyncio
async def test_switch_off_a_custom_type_from_the_model_is_dropped(core_prompt):
    out, _ = await _extract(_config(CATEGORIES))
    assert [m.content for m in out.memories] == ["The user adopted a dog named Rex"]
    assert all(m.category is None for m in out.memories)


@pytest.mark.asyncio
async def test_switch_on_a_custom_item_becomes_a_fact_with_the_name_as_its_category(core_prompt, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    out, _ = await _extract(_config(CATEGORIES))
    by_text = {m.content: m for m in out.memories}
    hotel = by_text["The user checked in at the Hilton on Friday"]
    assert hotel.memory_type is MemoryType.FACTS and hotel.category == "check_in"
    dog = by_text["The user adopted a dog named Rex"]
    assert dog.memory_type is MemoryType.FACTS and dog.category is None


@pytest.mark.asyncio
async def test_the_model_may_spell_the_name_loosely(core_prompt, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    reply = json.dumps({"extracted": [{"type": " Check In ", "content": "c", "confidence": 0.9}]})
    out, _ = await _extract(_config(CATEGORIES), reply=reply)
    assert [(m.memory_type, m.category) for m in out.memories] == [(MemoryType.FACTS, "check_in")]


@pytest.mark.asyncio
async def test_a_custom_item_gets_the_facts_confidence_threshold(core_prompt, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    out, _ = await _extract(_config(CATEGORIES))
    floor = _config().memory_extraction.filters.min_confidence_threshold
    low = next(m for m in out.memories if m.category == "warranty")
    assert 0.2 < floor and low.destination is MemoryDestination.CANDIDATES
    high = next(m for m in out.memories if m.category == "check_in")
    assert high.destination is MemoryDestination.MEMORIES


@pytest.mark.asyncio
async def test_a_disabled_category_is_not_accepted_when_the_model_sends_it_anyway(core_prompt, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    cats = [CHECKIN, {"name": "Warranty", "description": "d", "enabled": False}]
    out, _ = await _extract(_config(cats))
    assert sorted(m.category or "-" for m in out.memories) == ["-", "check_in"]


@pytest.mark.asyncio
async def test_a_built_in_name_in_the_config_does_not_change_what_a_built_in_type_does(core_prompt, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    out, _ = await _extract(_config([{"name": "facts", "description": "x"}, CHECKIN]))
    dog = next(m for m in out.memories if m.content.endswith("Rex"))
    assert dog.memory_type is MemoryType.FACTS and dog.category is None


@pytest.mark.asyncio
async def test_custom_types_still_run_when_all_three_simple_types_are_off(core_prompt, monkeypatch):
    # This counts the model calls that read memories. The graph call (on in Core, which needs no entity
    # extractor) is a different call and is switched off here.
    monkeypatch.setenv("SYNAP_GRAPH_EXTRACTION_V2", "false")
    monkeypatch.setenv(FLAG, "true")
    startup = _config(simple_enabled=False)
    prio = [MemoryType.FACTS, MemoryType.PREFERENCES, MemoryType.EMOTIONS]
    out, calls = await _extract(_config(CATEGORIES), startup=startup, priorities=prio)
    assert calls, "no model call was made"
    assert sorted(m.category for m in out.memories) == ["check_in", "warranty"]

    monkeypatch.delenv(FLAG)
    out, calls = await _extract(_config(CATEGORIES), startup=startup, priorities=prio)
    assert calls == [] and out.memories == []


@pytest.mark.asyncio
async def test_baseline_mode_changes_nothing_and_warns_once(core_prompt, monkeypatch, caplog):
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE", "baseline")
    _, off = await _extract(_config(CATEGORIES))
    monkeypatch.setenv(FLAG, "true")
    with caplog.at_level(logging.WARNING):
        out, on = await _extract(_config(CATEGORIES))
    assert json.dumps(off) == json.dumps(on)
    assert all(m.category is None for m in out.memories)
    assert caplog.text.count("stage3_custom_categories_skipped_in_baseline") == 1


@pytest.mark.asyncio
async def test_the_list_does_not_leak_into_the_next_call(core_prompt, monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    await _extract(_config(CATEGORIES))
    assert cc.ACTIVE.get() == ()


# --- what is written ----------------------------------------------------------
def _memory(category=None, mtype=MemoryType.FACTS, content="c"):
    return ExtractedMemory(
        memory_id=uuid4(), memory_type=mtype, content=content, confidence=0.9,
        destination=MemoryDestination.MEMORIES, source_document_id="doc", category=category)


@pytest.mark.asyncio
async def test_the_postgres_insert_carries_the_category():
    from synap.cloud.agents.ingestion.storage.sqlalchemy_rdbms_wrapper import SQLAlchemyRDBMSWrapper
    seen = {}

    async def fake_execute(sql, params):
        seen.update(params)
    wrapper = SQLAlchemyRDBMSWrapper.__new__(SQLAlchemyRDBMSWrapper)
    wrapper._execute = fake_execute
    scope = SimpleNamespace(client_id=uuid4(), instance_id=uuid4(), customer_id=uuid4(), user_id=uuid4())
    await wrapper.insert_memory(_memory("check_in"), scope, "doc")
    assert seen["memory_type"] == "facts" and seen["category"] == "check_in"


class _FakeChroma:
    def __init__(self):
        self.docs = []

    def upsert(self, namespace, docs):
        self.docs.extend(docs)


@pytest.mark.asyncio
async def test_the_vector_metadata_carries_the_category_on_both_write_paths():
    from synap.cloud.agents.ingestion.storage.vector_wrapper import AsyncVectorWrapper
    provider = _FakeChroma()
    wrapper = AsyncVectorWrapper(provider)
    scope = SimpleNamespace(client_id=uuid4(), instance_id=uuid4(), customer_id=uuid4(),
                            user_id=uuid4(), scope_level="user")
    await wrapper.upsert_memory(_memory("check_in"), [0.1, 0.2], scope)
    await wrapper.upsert_memories_batch([_memory("check_in")], [[0.1, 0.2]], scope)
    assert len(provider.docs) == 2
    assert [d.metadata["category"] for d in provider.docs] == ["check_in", "check_in"]
    assert [d.metadata["memory_type"] for d in provider.docs] == ["facts", "facts"]


@pytest.mark.asyncio
async def test_coalesce_keeps_a_custom_category_when_it_merges_with_a_plain_fact():
    plain, custom = _memory(None, content="plain"), _memory("check_in", content="custom")

    async def group(_contents, _types):
        return {"groups": [{"members": [1, 2], "reason": "same fact"}]}

    async def merge(contents, _types):
        return {"merged_content": " ".join(contents), "rationale": "joined"}

    for members in ([plain, custom], [custom, plain]):
        result = await coalesce.coalesce_memories(members, group_fn=group, merge_fn=merge)
        assert [m.category for m in result.memories] == ["check_in"]


@pytest.mark.asyncio
async def test_coalesce_without_a_category_stays_without_one():
    a, b = _memory(None, content="a"), _memory(None, content="b")

    async def group(_contents, _types):
        return {"groups": [{"members": [1, 2], "reason": "same fact"}]}

    async def merge(contents, _types):
        return {"merged_content": " ".join(contents), "rationale": "joined"}

    result = await coalesce.coalesce_memories([a, b], group_fn=group, merge_fn=merge)
    assert [m.category for m in result.memories] == [None]


# --- the config file ----------------------------------------------------------
def test_a_core_config_accepts_the_five_fields_and_still_rejects_the_cut_two():
    from synap.cloud.memory_config import core_rules
    ok = {"memory_categories": {"custom_categories": [
        {"name": "x", "enabled": True, "priority": "high", "description": "d", "examples": ["e"]}]}}
    assert core_rules.find_problems(ok) == []
    for cut in ("append_only", "retention_days"):
        bad = {"memory_categories": {"custom_categories": [{"name": "x", "description": "d", cut: 1}]}}
        assert core_rules.find_problems(bad), cut
