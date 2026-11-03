"""memory_extraction.filters.require_entity_grounding, applied at extraction.

These tests drive the real Stage3Extractor with a model client that replies with
fixed memories and an entity extractor that returns fixed entities (or fails).
"""
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from synap.cloud.agents.ingestion.models import ExtractedEntity, Stage2Output
from synap.cloud.agents.ingestion.models.enums import MemoryType
from synap.cloud.agents.ingestion.pipeline import stage3_extraction as s3
from synap.cloud.agents.ingestion.prompts import maca_context_builder as builder
from synap.cloud.agents.ingestion.services import guardrails, writer_entity
from synap.cloud.memory_config.config_schema import ConfigMetadata, MemoryArchitectureConfig

FLAG = "SYNAP_GUARDRAIL_ENTITY_GROUNDING"
INST = "44eff2d7-2a3d-5075-8c39-addf2a0188d0"
OTHER = "4fb56f47-3fc9-57c7-aeed-5ac445e554ff"
CONTENT = "user: I adopted a dog named Rex. Rex is a beagle. Keep answers short."

GROUNDED = "Rex is a beagle"
UNGROUNDED = "Prefers short answers from the assistant"
ABOUT_USER = "The user adopted a dog"
REPLY = json.dumps({"reasoning": "", "extracted": [
    {"type": "facts", "content": GROUNDED, "confidence": 0.95},
    {"type": "facts", "content": UNGROUNDED, "confidence": 0.95},
    {"type": "facts", "content": ABOUT_USER, "confidence": 0.95},
]})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (FLAG, f"{FLAG}_INSTANCES", "SYNAP_MACA_EXTRACTION_INSTANCES",
                "STAGE3_EXTRACTION_MODE_BY_INSTANCE", "SYNAP_ASSISTANT_RECALL_INSTANCES",
                "ENABLE_SPEAKER_ATTRIBUTION", "EVAL_PROVIDER", "SYNAP_GRAPH_EXTRACTION_V2",
                "SYNAP_WRITER_ENTITY", "SYNAP_CUSTOMER_ENTITY",
                "GRAPH_EXTRACTION_INSTANCE_ALLOWLIST", "ENABLE_GRAPH_EXTRACTION"):
        monkeypatch.delenv(var, raising=False)
    # hybrid: one merged call, the path Core runs; each item carries its "type" (the merged prompt contract).
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE", "hybrid")


def _config(grounding="absent") -> MemoryArchitectureConfig:
    """"absent": the developer wrote nothing about extraction; True/False: wrote it;
    "default": wrote the filters block but not this field."""
    raw = {"metadata": ConfigMetadata(
        client_id="c", instance_id="i", config_version="1.0.0",
        generated_at=datetime.now(timezone.utc), generated_by="test")}
    if grounding == "default":
        raw["memory_extraction"] = {"filters": {"min_confidence_threshold": 0.7}}
    elif grounding != "absent":
        raw["memory_extraction"] = {"filters": {"require_entity_grounding": grounding}}
    return MemoryArchitectureConfig(**raw)


class Capture:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def complete(self, messages=None, **kwargs):
        self.calls.append(messages if messages is not None else kwargs.get("prompt"))
        return self.reply


class Entities:
    """An entity extractor returning fixed names, or failing."""
    def __init__(self, names=("Rex",), fail=False):
        self.names, self.fail = names, fail

    async def extract(self, content, domain_hints, **_):
        if self.fail:
            raise RuntimeError("entity model down")
        return [ExtractedEntity(entity_id=f"e{i}", entity_type="animal", name=n)
                for i, n in enumerate(self.names)]


async def _extract(config, instance_id=INST, entities=None, user_id=None):
    main, light = Capture(REPLY), Capture(REPLY)
    ex = s3.Stage3Extractor(config=_config(), llm_client=main, llm_client_light=light,
                            entity_extractor=entities if entities is not None else Entities())
    scope = SimpleNamespace(instance_id=instance_id, client_id=None, customer_id=None,
                            user_id=user_id, scope_level="user")
    out = await ex.extract(
        content=CONTENT,
        stage2_output=Stage2Output(input_id="t", categories=[],
                                   extraction_priorities=[MemoryType.FACTS]),
        document_id="d", dct=None, has_temporal_content=False, request_scope=scope,
        correlation_id="c", config=config)
    return out, json.dumps([main.calls, light.calls])


def _texts(out):
    return [m.content for m in out.memories]


# --- enforcement -------------------------------------------------------------
@pytest.mark.asyncio
async def test_written_true_drops_the_ungrounded_memory_and_keeps_the_grounded_one(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    out, _ = await _extract(_config(True))
    assert GROUNDED in _texts(out)
    assert UNGROUNDED not in _texts(out)


@pytest.mark.asyncio
async def test_the_writer_label_counts_as_grounding_when_the_note_has_a_writer(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    with_writer, _ = await _extract(_config(True), user_id="u-1")
    assert ABOUT_USER in _texts(with_writer)
    no_writer, _ = await _extract(_config(True), user_id=None)
    assert ABOUT_USER not in _texts(no_writer)


def test_a_writer_entity_in_the_graph_counts_too():
    writer = writer_entity.entity(SimpleNamespace(user_id="u-1"))
    mems = [SimpleNamespace(content=ABOUT_USER), SimpleNamespace(content=UNGROUNDED)]
    kept = s3.Stage3Extractor._apply_entity_grounding(
        mems, [writer], SimpleNamespace(instance_id=INST, user_id=None), "c")
    assert [m.content for m in kept] == [ABOUT_USER]


def test_the_customer_entity_counts_too():
    customer = ExtractedEntity(entity_id="c1", entity_type="organization",
                               name=writer_entity.CUSTOMER_LABEL,
                               properties={writer_entity.CUSTOMER_PROPERTY: True})
    mems = [SimpleNamespace(content="The user's company was charged twice"),
            SimpleNamespace(content=UNGROUNDED)]
    kept = s3.Stage3Extractor._apply_entity_grounding(
        mems, [customer], SimpleNamespace(instance_id=INST, user_id=None), "c")
    assert [m.content for m in kept] == ["The user's company was charged twice"]


@pytest.mark.asyncio
async def test_matching_is_whole_word_and_case_insensitive(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    out, _ = await _extract(_config(True), entities=Entities(("rEX",)))
    assert GROUNDED in _texts(out)
    out, _ = await _extract(_config(True), entities=Entities(("Re",)))   # a prefix is not a word
    assert GROUNDED not in _texts(out)


# --- only an explicit true enforces -------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("written", ["absent", "default", False])
async def test_absent_default_or_false_changes_nothing(monkeypatch, written):
    base, base_prompts = await _extract(_config())
    monkeypatch.setenv(FLAG, "true")
    out, prompts = await _extract(_config(written))
    assert _texts(out) == _texts(base) and UNGROUNDED in _texts(out)
    assert prompts == base_prompts and "<grounding>" not in prompts


def test_the_schema_default_is_true_which_is_why_the_written_record_is_read():
    cfg = _config("default")
    assert cfg.memory_extraction.filters.require_entity_grounding is True
    assert guardrails.entity_grounding_required(cfg, INST) is False


def test_each_level_must_be_written(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    assert guardrails.entity_grounding_required(_config(True), INST) is True
    assert guardrails.entity_grounding_required(_config("absent"), INST) is False
    empty_block = MemoryArchitectureConfig(
        metadata=_config().metadata, memory_extraction={})
    assert guardrails.entity_grounding_required(empty_block, INST) is False
    assert guardrails.entity_grounding_required(None, INST) is False


# --- the switch ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_switch_off_is_unchanged_and_the_field_is_never_read():
    plain, base = await _extract(_config())
    written, off = await _extract(_config(True))
    assert base == off and _texts(plain) == _texts(written)

    class Wrapper:
        def __init__(self, inner): self._inner = inner
        @property
        def memory_extraction(self): raise AssertionError("field read with the switch off")
        def __getattr__(self, name): return getattr(self._inner, name)
    await _extract(Wrapper(_config(True)))


@pytest.mark.asyncio
async def test_switch_can_be_named_per_instance(monkeypatch):
    monkeypatch.setenv(f"{FLAG}_INSTANCES", INST)
    on, on_prompts = await _extract(_config(True), INST)
    other, other_prompts = await _extract(_config(True), OTHER)
    assert UNGROUNDED not in _texts(on) and "<grounding>" in on_prompts
    assert UNGROUNDED in _texts(other) and "<grounding>" not in other_prompts


# --- an extraction failure must not erase a conversation ------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("entities", [Entities(fail=True), Entities(names=())])
async def test_no_entities_keeps_everything_and_warns(monkeypatch, caplog, entities):
    monkeypatch.setenv(FLAG, "true")
    with caplog.at_level(logging.INFO):
        out, _ = await _extract(_config(True), entities=entities)
    assert _texts(out) == [GROUNDED, UNGROUNDED, ABOUT_USER]
    warned = [r for r in caplog.records
              if r.levelno == logging.WARNING and "stage3_entity_grounding_skipped" in r.getMessage()]
    assert len(warned) == 1 and "memories=3" in warned[0].getMessage()


# --- the log ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_one_log_line_with_counts_and_no_memory_text(monkeypatch, caplog):
    monkeypatch.setenv(FLAG, "true")
    with caplog.at_level(logging.INFO):
        await _extract(_config(True), entities=Entities(("Rex",)))
    lines = [r.getMessage() for r in caplog.records if "stage3_entity_grounding" in r.getMessage()]
    assert len(lines) == 1
    assert "dropped=2 kept=1" in lines[0]
    for text in (GROUNDED, UNGROUNDED, ABOUT_USER, "beagle", "short answers", "Rex"):
        assert text not in lines[0]


# --- the prompt line ----------------------------------------------------------
@pytest.mark.asyncio
async def test_prompt_line_only_when_enforcing(monkeypatch):
    _, off = await _extract(_config(True))
    assert "<grounding>" not in off
    monkeypatch.setenv(FLAG, "true")
    _, on = await _extract(_config(True))
    assert "<grounding>" in on


def test_prompt_line_survives_a_config_that_overflows_the_budget():
    cfg = _config(True)
    cfg.agent_hints.ingestion_agent.focus_areas = ["padding " * 10] * 100
    token = builder.REQUIRE_ENTITY_GROUNDING.set(True)
    try:
        block = builder.MACAContextBuilder.build_extraction_context(cfg, MemoryType.FACTS)
    finally:
        builder.REQUIRE_ENTITY_GROUNDING.reset(token)
    assert "<grounding>" in block and block.endswith("</context>")


@pytest.mark.asyncio
async def test_the_setting_does_not_leak_into_the_next_call(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    await _extract(_config(True))
    assert builder.REQUIRE_ENTITY_GROUNDING.get() is False


def test_helper_keeps_order_and_counts():
    mems = [SimpleNamespace(content="Rex barks"), SimpleNamespace(content="Nothing here"),
            SimpleNamespace(content="REX again")]
    kept, dropped = guardrails.drop_ungrounded(mems, ["rex"], s3._names_entity)
    assert [m.content for m in kept] == ["Rex barks", "REX again"] and dropped == 1
