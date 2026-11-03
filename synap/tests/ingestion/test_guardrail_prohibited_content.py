"""guardrails.compliance.prohibited_memory_content, applied at extraction.

The prompt line is guidance to the model; the backstop only catches a literal
listed phrase. These tests drive the real Stage3Extractor with a model client
that records what it is sent and replies with fixed memories.
"""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from synap.cloud.agents.ingestion.models import Stage2Output
from synap.cloud.agents.ingestion.models.enums import MemoryType
from synap.cloud.agents.ingestion.pipeline import stage3_extraction as s3
from synap.cloud.agents.ingestion.prompts import maca_context_builder as builder
from synap.cloud.agents.ingestion.services import guardrails
from synap.cloud.memory_config.config_schema import ConfigMetadata, MemoryArchitectureConfig

FLAG = "SYNAP_GUARDRAIL_PROHIBITED_CONTENT"
INST = "44eff2d7-2a3d-5075-8c39-addf2a0188d0"
OTHER = "4fb56f47-3fc9-57c7-aeed-5ac445e554ff"
CONTENT = "user: My Social Security Number is 123-45-6789 and I adopted a dog named Rex."
ITEMS = ["Social Security number", "diagnosis"]

REPLY = json.dumps({"reasoning": "", "extracted": [
    {"type": "facts", "content": "The user's SOCIAL SECURITY NUMBER is 123-45-6789", "confidence": 0.95},
    {"type": "facts", "content": "The user adopted a dog named Rex", "confidence": 0.95},
    {"type": "facts", "content": "The user has a new cat named Mia", "confidence": 0.95},
]})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (FLAG, f"{FLAG}_INSTANCES", "SYNAP_MACA_EXTRACTION_INSTANCES",
                "STAGE3_EXTRACTION_MODE", "STAGE3_EXTRACTION_MODE_BY_INSTANCE",
                "SYNAP_ASSISTANT_RECALL_INSTANCES", "ENABLE_SPEAKER_ATTRIBUTION", "EVAL_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    # hybrid: one merged call, the path Core runs; each item carries its "type" (the merged prompt contract).
    monkeypatch.setenv("STAGE3_EXTRACTION_MODE", "hybrid")


def _config(prohibited=None) -> MemoryArchitectureConfig:
    raw = {"metadata": ConfigMetadata(
        client_id="c", instance_id="i", config_version="1.0.0",
        generated_at=datetime.now(timezone.utc), generated_by="test")}
    if prohibited is not None:
        raw["guardrails"] = {"compliance": {"prohibited_memory_content": prohibited}}
    return MemoryArchitectureConfig(**raw)


class Capture:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def complete(self, messages=None, **kwargs):
        self.calls.append(messages if messages is not None else kwargs.get("prompt"))
        return self.reply


async def _extract(config, instance_id=INST):
    main, light = Capture(REPLY), Capture(REPLY)
    ex = s3.Stage3Extractor(config=_config(), llm_client=main, llm_client_light=light)
    scope = SimpleNamespace(instance_id=instance_id, client_id=None, customer_id=None,
                            user_id=None, scope_level="user")
    out = await ex.extract(
        content=CONTENT,
        stage2_output=Stage2Output(input_id="t", categories=[],
                                   extraction_priorities=[MemoryType.FACTS]),
        document_id="d", dct=None, has_temporal_content=False, request_scope=scope,
        correlation_id="c", config=config)
    return out, json.dumps([main.calls, light.calls])


# --- the prompt line ---------------------------------------------------------
@pytest.mark.asyncio
async def test_prompt_names_the_items_only_when_the_switch_is_on(monkeypatch):
    _, off = await _extract(_config(ITEMS))
    assert "Social Security number" not in off and "<prohibited>" not in off

    monkeypatch.setenv(FLAG, "true")
    _, on = await _extract(_config(ITEMS))
    assert "<prohibited>" in on
    assert "- Social Security number" in on and "- diagnosis" in on


@pytest.mark.asyncio
async def test_switch_off_is_byte_identical_and_never_reads_the_guardrails():
    plain, base = await _extract(_config())
    listed, off = await _extract(_config(ITEMS))
    assert base == off
    assert [m.content for m in plain.memories] == [m.content for m in listed.memories]
    class Wrapper:
        def __init__(self, inner): self._inner = inner
        @property
        def guardrails(self): raise AssertionError("guardrails read with the switch off")
        def __getattr__(self, name): return getattr(self._inner, name)
    await _extract(Wrapper(_config(ITEMS)))


@pytest.mark.asyncio
async def test_switch_can_be_named_per_instance(monkeypatch):
    monkeypatch.setenv(f"{FLAG}_INSTANCES", INST)
    _, on = await _extract(_config(ITEMS), INST)
    _, other = await _extract(_config(ITEMS), OTHER)
    assert "<prohibited>" in on and "<prohibited>" not in other


@pytest.mark.asyncio
async def test_empty_list_changes_nothing(monkeypatch):
    _, base = await _extract(_config())
    monkeypatch.setenv(FLAG, "true")
    out, empty = await _extract(_config([]))
    assert base == empty
    assert len(out.memories) == 3


@pytest.mark.asyncio
async def test_the_item_survives_a_config_that_overflows_the_budget(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    cfg = _config(ITEMS)
    cfg.agent_hints.ingestion_agent.focus_areas = ["padding " * 10] * 100
    token = builder.PROHIBITED_CONTENT.set(tuple(ITEMS))
    try:
        block = builder.MACAContextBuilder.build_extraction_context(cfg, MemoryType.FACTS)
    finally:
        builder.PROHIBITED_CONTENT.reset(token)
    assert "- Social Security number" in block and block.endswith("</context>")


@pytest.mark.asyncio
async def test_the_setting_does_not_leak_into_the_next_call(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    await _extract(_config(ITEMS))
    assert builder.PROHIBITED_CONTENT.get() == ()


# --- the backstop ------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_literal_match_is_dropped_case_insensitively_and_the_rest_kept(monkeypatch, caplog):
    monkeypatch.setenv(FLAG, "true")
    with caplog.at_level("INFO"):
        out, _ = await _extract(_config(ITEMS))
    texts = [m.content for m in out.memories]
    assert not any("social security" in t.lower() for t in texts)
    assert "The user adopted a dog named Rex" in texts
    assert "The user has a new cat named Mia" in texts
    assert "stage3_prohibited_content_dropped dropped=1 kept=2" in caplog.text


@pytest.mark.asyncio
async def test_switch_off_keeps_the_matching_memory():
    out, _ = await _extract(_config(ITEMS))
    assert any("SOCIAL SECURITY" in m.content for m in out.memories)


def test_backstop_helpers():
    mems = [SimpleNamespace(content="Has a Diagnosis of flu"), SimpleNamespace(content="Likes tea")]
    kept, dropped = guardrails.drop_prohibited(mems, ["diagnosis"])
    assert [m.content for m in kept] == ["Likes tea"] and dropped == 1
    kept, dropped = guardrails.drop_prohibited(mems, [])
    assert len(kept) == 2 and dropped == 0
    kept, dropped = guardrails.drop_prohibited(mems, ["  ", ""])
    assert len(kept) == 2 and dropped == 0
    assert guardrails.drop_prohibited(mems, ["flu sickness"])[1] == 0   # a paraphrase passes: documented limit
