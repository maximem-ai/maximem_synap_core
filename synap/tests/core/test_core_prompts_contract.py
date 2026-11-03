"""One contract test per Core prompt.

Each case gives a Core prompt a fixed input, takes a recorded model reply, and checks that the
engine's own parser accepts it. No model is called: the reply is written down here. These tests read
only Core files (the Core prompt modules and the texts in ``synap/cloud/core_prompts``), never a Cloud
prompt file and never ``meta.json``, so they ship with Core and run there.

The Core prompt modules are imported by their own names, so the tests do not depend on which edition
is running. Where the engine imports a prompt by its neutral name (the knowledge-update classifier and
the compaction writer), the Core module is put in place of it for the length of the test.

What this does NOT prove: that a real model, given these texts, writes a reply like the recorded one.
That is measured by the eval harness, not here.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import sys
import textwrap
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from synap.cloud.shared import prompt_switch as ps

ING = "synap.cloud.agents.ingestion.prompts"
EXT = ING + ".extraction"
V3 = "synap.cloud.application.compaction.v3.prompts"


def core(interface: str):
    """The Core module behind a neutral interface name."""
    return importlib.import_module(ps.core_name(interface))


def put_core_in_place(monkeypatch, interfaces, engines):
    """Make the engine modules read the Core text: the interface names point at the Core modules."""
    for interface in interfaces:
        importlib.import_module(interface)  # load the real one first so it is restored afterwards
    originals = {name: importlib.import_module(name) for name in engines}
    for interface in interfaces:
        module = core(interface)
        monkeypatch.setitem(sys.modules, interface, module)
        parent_name, _, leaf = interface.rpartition(".")
        monkeypatch.setattr(importlib.import_module(parent_name), leaf, module, raising=False)
    fresh = {}
    for name, original in originals.items():
        parent_name, _, leaf = name.rpartition(".")
        monkeypatch.setattr(importlib.import_module(parent_name), leaf, original, raising=False)
        monkeypatch.delitem(sys.modules, name, raising=False)
        fresh[name] = importlib.import_module(name)
    return fresh


def keys_read(*functions) -> set:
    """The JSON keys the real parser code reads: every string used as ``x["key"]`` or ``x.get("key")``."""
    found = set()
    for function in functions:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, str):
                found.add(node.slice.value)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                found.add(node.args[0].value)
    return found


def keys_asked(*texts: str) -> set:
    """Every quoted word in the prompt text: the keys it names."""
    return {w for t in texts for w in re.findall(r'"([A-Za-z_]+)"', t)}


def keys_in(value) -> set:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys_in(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys_in(v)}
    return set()


def assert_keys_agree(texts, reply, parser_keys, *, required, ignore=()):
    """The text names every key the parser reads (bar ``ignore``); the reply uses only keys the text names
    and carries every ``required`` key."""
    asked, used = keys_asked(*texts), keys_in(reply)
    assert not (parser_keys - asked - set(ignore)), f"the parser reads keys the prompt never names: {parser_keys - asked - set(ignore)}"
    assert not (used - asked), f"the recorded reply uses keys the prompt never names: {used - asked}"
    assert set(required) <= used, f"the recorded reply lacks {set(required) - used}"
    assert set(required) <= asked, f"the prompt never names {set(required) - asked}"


# --- the pack as a whole --------------------------------------------------------------------

def test_every_prompt_core_owes_has_a_core_text():
    assert ps.check_core_pack() == []


def test_a_core_prompt_module_holds_no_leftover_field():
    """No [[NAME]] field is left in a text the module hands out as finished."""
    finished = [
        core(ING + ".knowledge_update").CLASSIFICATION_SYSTEM_PROMPT,
        core(ING + ".coalesce").COALESCE_SYSTEM_PROMPT,
        core(ING + ".merge").MERGE_SYSTEM_PROMPT,
        core(EXT + ".merged_simple").MergedSimpleExtractionPrompt.SYSTEM_PROMPT,
        core(EXT + ".episodes").EpisodesExtractionPrompt.SYSTEM_PROMPT,
        core(EXT + ".temporal").TemporalExtractionPrompt.SYSTEM_PROMPT,
        core(V3 + ".writer").WRITER_SYSTEM,
        core(V3 + ".writer").ROLLUP_SYSTEM,
        core(V3 + ".framing").FRAMING,
        core("synap.cloud.application.ingestion.prompts.image_description").VISION_PROMPT,
        core("synap.cloud.application.memory_forget.prompts.judge").PURPOSE,
    ]
    for text in finished:
        assert text.strip() and "[[" not in text and "]]" not in text


# --- P1: facts, preferences and emotions in one call ---------------------------------------

def test_merged_extraction_builds_and_its_reply_is_parsed():
    from synap.cloud.agents.ingestion.pipeline.stage3_extraction import Stage3Extractor

    prompt = core(EXT + ".merged_simple").MergedSimpleExtractionPrompt(MagicMock())
    system, user = prompt.build("user: I moved to Berlin and I hate cold mornings.", [], instance_id=uuid4())
    assert "facts" in system and "preferences" in system and "emotions" in system
    assert "I moved to Berlin and I hate cold mornings." in user and "[[" not in user
    reply = {"reasoning": "one move, one dislike", "extracted": [
        {"type": "facts", "content": "The user moved to Berlin.", "confidence": 0.95},
        {"type": "preferences", "content": "The user dislikes cold mornings.", "confidence": 0.8},
    ]}
    # The parser reads "extracted" and "content"; the caller reads "type" and "confidence" off each item.
    assert_keys_agree([system], reply, keys_read(Stage3Extractor._parse_merged_extraction) | {"type", "confidence"},
                      required={"extracted", "content", "type", "confidence"})
    extractor = Stage3Extractor.__new__(Stage3Extractor)  # a real object; the parser uses no state
    items = extractor._parse_merged_extraction(json.dumps(reply))
    assert [i["type"] for i in items] == ["facts", "preferences"]


def test_merged_extraction_adds_the_date_guidance_when_the_content_has_dates():
    prompt = core(EXT + ".merged_simple").MergedSimpleExtractionPrompt(MagicMock())
    when = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc)
    _system, user = prompt.build("user: see you next Tuesday", [], dct=when, has_temporal_content=True,
                                 instance_id=None)
    assert "2026-03-16" in user and "Monday" in user and "event_start" in user


# --- the date guidance -----------------------------------------------------------------------

def test_date_guidance_names_the_date_and_weekday_or_says_there_is_none():
    render = core(EXT + ".temporal_addendum").render_temporal_addendum
    dated = render(datetime(2026, 3, 16, tzinfo=timezone.utc), instance_id=None)
    assert "2026-03-16" in dated and "Monday" in dated and "not known" not in dated
    undated = render(None, instance_id=None)
    assert "not known" in undated and "[[" not in undated and "event_start" in undated


# --- P2 and P3: graph extraction, the writer and the writer's company -------------------------

GRAPH_REPLY = json.dumps({
    "entities": [
        {"name": "the user", "entity_type": "person", "aliases": [], "about": "the writer", "source_span": "I"},
        {"name": "the user's company", "entity_type": "organization", "aliases": [], "about": "employer",
         "source_span": "we"},
        {"name": "Zendesk", "entity_type": "product", "aliases": [], "about": "ticket tool", "source_span": "Zendesk"},
    ],
    "relationships": [{
        "source_entity": "the user's company", "source_type": "organization", "target_entity": "Zendesk",
        "target_type": "product", "relationship_type": "USES", "inverse_type": "USED_BY", "basis": "stated",
        "confidence": 0.95, "evidence": "we use Zendesk", "properties": {}}],
    "writer_name": None, "writer_organization": None,
})


def test_graph_extraction_with_a_company_builds_and_its_reply_is_parsed():
    from synap.cloud.agents.ingestion.services import graph_v2

    prompt = core(EXT + ".graph_v2").GraphExtractionPromptV2(MagicMock())
    system, user = prompt.build("I opened a ticket. We use Zendesk.", writer="the user",
                                customer="the user's company")
    assert "the user's company" in system and "writer_organization" in system
    assert "We use Zendesk." in user and "[[" not in system + user
    parsed = graph_v2.parse(json.loads(GRAPH_REPLY), "I opened a ticket. We use Zendesk.",
                            writer="the user", customer="the user's company")
    assert {e.name for e in parsed.entities} >= {"the user's company", "Zendesk"}
    assert [r.relationship_type for r in parsed.relationships] == ["USES"]
    # Every key graph_v2.parse reads for an entity or a relationship, and for the writer, is named in the prompt.
    assert_keys_agree([system], json.loads(GRAPH_REPLY), keys_read(graph_v2.parse),
                      required={"entities", "relationships", "name", "entity_type", "source_entity",
                                "target_entity", "relationship_type", "writer_name", "writer_organization"},
                      ignore={"statement", "statements"})  # the statements layer is off in Core


def test_graph_extraction_without_a_company_leaves_the_company_block_out():
    prompt = core(EXT + ".graph_v2").GraphExtractionPromptV2(MagicMock())
    system, _user = prompt.build("I opened a ticket.", writer="the user", customer=None)
    assert "THE WRITER'S ORGANIZATION" not in system and "writer_organization" not in system


def test_the_company_rule_is_stated_once_and_has_its_three_rules():
    prompt = core(EXT + ".graph_v2").GraphExtractionPromptV2(MagicMock())
    system, _ = prompt.build("We use Zendesk.", writer="the user", customer="the user's company")
    assert system.count("THE WRITER'S ORGANIZATION") == 1
    assert system.count('mean the writer\'s own organization') == 1  # the rule is not repeated
    # 1. one entity, with the exact name
    assert "Create one entity for that organization" in system and "the name exactly the user's company" in system
    # 2. that exact string in every relationship about "we"
    assert "use that exact string as source_entity or target_entity in every relationship" in system
    # 3. no "we", no entity
    assert 'If the note has no "we", "us" or "our" meaning the writer\'s organization, create no entity for it.' in system
    assert "[[" not in system


def test_without_a_company_the_company_rule_is_not_in_the_prompt():
    prompt = core(EXT + ".graph_v2").GraphExtractionPromptV2(MagicMock())
    system, _ = prompt.build("I opened a ticket.", writer="the user", customer=None)
    assert "ORGANIZATION" not in system and "create no entity for it" not in system
    assert "writer_organization" not in system


def test_the_no_we_no_entity_sentence_is_in_the_text_the_prompt_is_built_from():
    """Fails if customer_graph.txt loses the sentence, or if the assembly stops using it."""
    sentence = "create no entity for it"
    assert sentence in ps.pack_text("person_and_company", "customer_graph.txt")
    system, _ = core(EXT + ".graph_v2").GraphExtractionPromptV2(MagicMock()).build(
        "x", writer="the user", customer="the user's company")
    assert sentence in system


def test_the_graph_prompt_stops_if_its_company_block_cannot_be_found(monkeypatch):
    real = ps.pack_text
    monkeypatch.setattr(core(EXT + ".graph_v2"), "pack_text",
                        lambda folder, name: "no company block here" if name == "system.txt" else real(folder, name))
    with pytest.raises(ps.CorePromptMissing):
        core(EXT + ".graph_v2").GraphExtractionPromptV2(MagicMock()).build("x", writer="w", customer="c")


def test_graph_extraction_lists_earlier_references_only_when_given():
    prompt = core(EXT + ".graph_v2").GraphExtractionPromptV2(MagicMock())
    plain, _ = prompt.build("She called.", writer="the user")
    assert "EARLIER REFERENCES" not in plain
    with_refs, _ = prompt.build("She called.", writer="the user", references=[("she", "Ines Carvalho", "person")])
    assert "EARLIER REFERENCES" in with_refs and "Ines Carvalho (person)" in with_refs


# --- P3: the company wording for memories, and keeping the person ---------------------------

def test_the_company_wording_for_memories_uses_the_exact_name():
    section = core(ING + ".stage3_extraction").CUSTOMER_WE_SECTION.format(
        customer="the user's company", customer_cap="The user's company")
    assert "the user's company" in section and "The user's company" in section
    assert "[[" not in section and section.startswith("\n\n")


def test_dated_event_prompt_keeps_the_person_when_the_switch_is_on(monkeypatch):
    monkeypatch.setenv("SYNAP_TEMPORAL_KEEP_PERSON", "true")
    prompt = core(EXT + ".temporal").TemporalExtractionPrompt(MagicMock())
    system, user = prompt.build("Leopold wants the lane report by Thursday.", [], instance_id=None)
    assert core(EXT + ".temporal").PERSON_SECTION in system
    assert "Leopold wants the lane report by Thursday." in user


def test_dated_event_prompt_leaves_the_person_block_out_when_the_switch_is_off(monkeypatch):
    monkeypatch.delenv("SYNAP_TEMPORAL_KEEP_PERSON", raising=False)
    monkeypatch.delenv("SYNAP_TEMPORAL_KEEP_PERSON_INSTANCES", raising=False)
    prompt = core(EXT + ".temporal").TemporalExtractionPrompt(MagicMock())
    system, _ = prompt.build("Leopold wants the lane report by Thursday.", [], instance_id=None)
    assert core(EXT + ".temporal").PERSON_SECTION not in system


@pytest.mark.parametrize("module,cls,kind,reply_text", [
    ("episodes", "EpisodesExtractionPrompt", "EPISODES", "The user visited Lisbon last spring."),
    ("temporal", "TemporalExtractionPrompt", "TEMPORAL_EVENTS", "The user's tax return is due on 31 March."),
])
def test_episode_and_temporal_prompts_build_and_their_replies_are_parsed(module, cls, kind, reply_text):
    from synap.cloud.agents.ingestion.models.enums import MemoryType
    from synap.cloud.agents.ingestion.pipeline.stage3_extraction import Stage3Extractor

    prompt = getattr(core(f"{EXT}.{module}"), cls)(MagicMock())
    system, user = prompt.build("user: some content", [], instance_id=None)
    assert "extracted" in system and "some content" in user
    reply = json.dumps({"extracted": [{"content": reply_text, "confidence": 0.9, "source_span": "x"}]})
    extractor = Stage3Extractor.__new__(Stage3Extractor)
    items = extractor._parse_extraction(reply, getattr(MemoryType, kind))
    assert [i["content"] for i in items] == [reply_text]
    # The parser reads "extracted" and "content"; the caller reads "confidence" and "source_span" off each item.
    # The date keys the parser also reads are asked for by the date guidance, which is added to the request.
    guidance = core(EXT + ".temporal_addendum").render_temporal_addendum(
        datetime(2026, 3, 16, tzinfo=timezone.utc), instance_id=None)
    assert_keys_agree([system, guidance], json.loads(reply), keys_read(Stage3Extractor._try_parse_json_extraction)
                      | {"confidence", "source_span"}, required={"extracted", "content", "confidence"},
                      ignore={"reasoning", "output", "assertion_source"})  # speaker attribution is off in Core


# --- P4: the knowledge-update classifier ----------------------------------------------------

@pytest.mark.asyncio
async def test_the_classifier_text_is_what_the_model_is_sent_and_its_reply_is_accepted(monkeypatch):
    from synap.cloud.agents.ingestion.models import Scope

    ku = ING + ".knowledge_update"
    engines = put_core_in_place(monkeypatch, [ku], ["synap.cloud.agents.ingestion.pipeline.knowledge_update"])
    engine = engines["synap.cloud.agents.ingestion.pipeline.knowledge_update"]
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=json.dumps(
        {"classification": "SUPERSEDE", "reason": "moved cities", "confidence": 0.9}))
    detector = engine.KnowledgeUpdateDetector(vector=MagicMock(), llm_client=llm)
    scope = Scope(client_id=uuid4(), instance_id=uuid4(), customer_id=uuid4(), user_id=uuid4())
    parsed = await detector._classify_pair(existing_content="lives in Paris", new_content="moved to Berlin",
                                           scope=scope)
    messages = llm.complete.await_args.kwargs["messages"]
    assert messages[0]["content"] == core(ku).CLASSIFICATION_SYSTEM_PROMPT
    assert "lives in Paris" in messages[1]["content"] and "moved to Berlin" in messages[1]["content"]
    assert parsed is not None and parsed["classification"] == "SUPERSEDE"
    recorded = {"classification": "SUPERSEDE", "reason": "moved cities", "confidence": 0.9}
    assert_keys_agree([messages[0]["content"]], recorded, keys_read(engine.KnowledgeUpdateDetector._classify_pair)
                      | {"classification", "reason", "confidence"}, required=set(recorded),
                      ignore={"content", "role"})


def test_the_classifier_blocks_for_switches_that_are_off_in_core_stop_when_used():
    ku = core(ING + ".knowledge_update")
    for gap in (ku.DELETE_ADDENDUM, ku.PROVENANCE_ADDENDUM, ku.BATCH_CLASSIFICATION_ADDENDUM):
        with pytest.raises(ps.CorePromptMissing):
            "x" + gap
    with pytest.raises(ps.CorePromptMissing):
        ku.build_batch_user_prompt("a", [])


# --- P5: the user-profile merge -------------------------------------------------------------

def _profile_config(enabled=True):
    attribute = SimpleNamespace(name="plan", type="enum", description="Pricing plan", required=False,
                                allowed_values=["free", "pro"], examples=None)
    return SimpleNamespace(critical_attributes=[attribute], overview_enabled=enabled, max_profile_tokens=120)


def test_the_profile_merge_prompt_builds_and_its_reply_is_accepted():
    from synap.cloud.application.profile import updater

    text = core(ING + ".user_profile")
    cfg = _profile_config()
    rule = text.render_overview_rule(cfg)
    prompt = text.build_merge_prompt_text(
        attrs_spec=[{"name": "plan", "type": "enum", "allowed_values": ["free", "pro"]}],
        current_view={"attributes": {}, "overview": "", "extras": {}},
        memory_lines=["The user upgraded to the pro plan."], analysis=None,
        conversation_id="conv-1", overview_rule=rule)
    assert "The user upgraded to the pro plan." in prompt and '"conv-1"' in prompt
    assert "Overview enabled: yes" in prompt and "120" in prompt and "[[" not in prompt
    reply = json.dumps({"attributes": {"plan": {"value": "pro", "confidence": 0.9,
                                                "source_conversation_id": "conv-1"}},
                        "overview": "A pro customer.", "extras": {}})
    doc = updater._parse_llm_json(reply)
    normal = updater._normalize_llm_doc(doc, user_profile_cfg=cfg, current_profile=None,
                                        conversation_id="conv-1", now_iso="2026-10-30T00:00:00Z")
    assert normal["attributes"]["plan"]["value"] == "pro"
    assert_keys_agree([prompt], json.loads(reply), keys_read(updater._normalize_llm_doc, updater._parse_llm_json),
                      required={"attributes", "value", "confidence", "source_conversation_id", "overview", "extras"},
                      ignore={"updated_at", "_meta"})


def test_the_profile_merge_prompt_says_the_overview_is_off_when_it_is_off():
    text = core(ING + ".user_profile")
    rule = text.render_overview_rule(_profile_config(enabled=False))
    prompt = text.build_merge_prompt_text(attrs_spec=[], current_view={}, memory_lines=[], analysis=None,
                                          conversation_id=None, overview_rule=rule)
    assert "Overview enabled: no" in prompt


# --- coalesce and merge ----------------------------------------------------------------------

def test_coalesce_builds_a_numbered_list_and_its_reply_is_accepted():
    from synap.cloud.agents.ingestion.services.coalesce import parse_groups

    text = core(ING + ".coalesce")
    contents = ["The user started a yoga class.", "The user likes yoga.", "The user owns a bike."]
    user = text.build_coalesce_user_prompt(contents, ["facts", "preferences", "facts"])
    assert "1. [facts] The user started a yoga class." in user and "3. [facts] The user owns a bike." in user
    assert '"groups"' in text.COALESCE_SYSTEM_PROMPT
    reply = {"groups": [{"members": [1, 2], "reason": "same yoga fact"}]}
    groups = parse_groups(reply, len(contents))
    assert groups == [([0, 1], "same yoga fact")]
    assert_keys_agree([text.COALESCE_SYSTEM_PROMPT], reply, keys_read(parse_groups),
                      required={"groups", "members", "reason"})


@pytest.mark.asyncio
async def test_merge_builds_a_numbered_list_and_the_real_coalescer_accepts_its_reply():
    from synap.cloud.agents.ingestion.models import ExtractedMemory, MemoryDestination, MemoryType
    from synap.cloud.agents.ingestion.pipeline.stage3_extraction import Stage3Extractor
    from synap.cloud.agents.ingestion.services import coalesce

    text = core(ING + ".merge")
    contents = ["Ada works at Acme.", "Ada is an engineer at Acme."]
    user = text.build_merge_user_prompt(contents, ["facts", "facts"])
    assert "1. [facts] Ada works at Acme." in user and "2. [facts] Ada is an engineer at Acme." in user
    memories = [ExtractedMemory(memory_id=uuid4(), memory_type=MemoryType.FACTS, content=c, confidence=0.9,
                                destination=MemoryDestination.MEMORIES, source_document_id="doc-1")
                for c in contents]
    recorded = {"merged_content": "Ada is an engineer at Acme.", "rationale": "same job"}

    async def group_fn(_contents, _types):
        return {"groups": [{"members": [1, 2], "reason": "same job"}]}

    async def merge_fn(_contents, _types):  # the reply goes through the extractor's own JSON reader
        return Stage3Extractor._loads_merged(None, json.dumps(recorded))

    result = await coalesce.coalesce_memories(memories, group_fn=group_fn, merge_fn=merge_fn)
    assert result.counts["groups_merged"] == 1 and [m.content for m in result.memories] == [recorded["merged_content"]]
    # coalesce_memories also reads its own counters; the one key it takes from the model's reply is this.
    assert "merged_content" in keys_read(coalesce.coalesce_memories)
    assert_keys_agree([text.MERGE_SYSTEM_PROMPT], recorded, {"merged_content"},
                      required={"merged_content", "rationale"})


# --- P7: image description -------------------------------------------------------------------

def test_the_image_prompt_is_the_text_the_vision_call_sends():
    prompt = core("synap.cloud.application.ingestion.prompts.image_description").VISION_PROMPT
    assert "image" in prompt.lower() and "[[" not in prompt


async def test_the_vision_call_sends_the_core_prompt_and_keeps_the_recorded_reply(monkeypatch):
    import io
    from PIL import Image

    iface = "synap.cloud.application.ingestion.prompts.image_description"
    engines = put_core_in_place(monkeypatch, [iface], ["synap.cloud.application.ingestion.multimodal"])
    mm = engines["synap.cloud.application.ingestion.multimodal"]
    core_prompt = core(iface).VISION_PROMPT
    assert mm.VISION_PROMPT == core_prompt  # the engine reads the Core text, not the Cloud file

    png = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(png, format="PNG")
    recorded = "A whiteboard with the words 'Ship Friday' written on it."  # a recorded reply: plain text
    manager = MagicMock()
    manager.complete = AsyncMock(return_value=MagicMock(content=recorded))
    import synap.cloud.infrastructure.llm as llm
    monkeypatch.setattr(llm, "get_llm_manager", lambda: manager)
    mm._description_cache.clear()

    reads = await mm.read_images([mm.ImageInput(png.getvalue(), "image/png", "board.png")])
    sent = manager.complete.await_args.kwargs["messages"][0]["content"]
    assert {"type": "text", "text": core_prompt} in sent  # the call carries the Core text
    assert reads[0].ok and reads[0].description == recorded
    assert mm.render_parts(reads) == f"[image: board.png]\n{recorded}\n[/image]"

    # an empty reply is kept as a marker, not a description
    manager.complete = AsyncMock(return_value=MagicMock(content="  "))
    mm._description_cache.clear()
    empty = await mm.read_images([mm.ImageInput(png.getvalue(), "image/png", "board.png")])
    assert not empty[0].ok
    assert mm.render_parts(empty) == "[image: board.png (could not be read)]"


# --- P8: the forget judge --------------------------------------------------------------------

def test_the_forget_judge_has_four_levels_and_its_answer_is_read():
    from synap.cloud.application.memory_forget import judge

    text = core("synap.cloud.application.memory_forget.prompts.judge")
    assert len(text.RUBRIC) == len(judge.RUBRIC) == 4
    assert all(text.RUBRIC) and text.PURPOSE
    assert "position 2:" in text.rate_instruction(2)
    assert "position 0:" in text.rate_instruction(0)
    # A recorded Jev answer: memory 0 is clearly the thing to forget, memory 1 is not.
    answers = {"s0": {"probabilities": {"0": 0.0, "1": 0.05, "2": 0.15, "3": 0.8}},
               "s1": {"probabilities": {"0": 0.9, "1": 0.1, "2": 0.0, "3": 0.0}}}
    assert judge.admitted_from_answers(answers, 2) == {0}


# --- P6: compaction v3 -----------------------------------------------------------------------

COMPACTION_REPLY = {
    "turns": [{"m": "m1", "items": [{"type": "fact", "item": "Priya's order ORD-88213-KX arrived broken.",
                                     "why": "", "status": ""}]},
              {"m": "m2", "items": [{"type": "commitment", "item": "The assistant will send a return label.",
                                     "why": "", "status": ""}]}],
    "use_case": "customer_support",
    "changed_sections": [],
    "literals": [{"label": "Order number", "value": "ORD-88213-KX", "resolved": "", "m": "m1", "replaces": ""}],
    "superseded": [],
    "agreed_actions": [{"id": "", "who": "the assistant", "what": "send a return label today",
                        "quotes": [{"text": "I'll send you a return label today", "m": "m2"}], "status": "open"}],
    "segment_density": [{"start": "m1", "end": "m2", "level": "high"}],
    "chunk_recap": "Priya reported a broken order; a return label was promised.",
}


class _ScriptedLLM:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outputs.pop(0)
        return SimpleNamespace(content=json.dumps(out), input_tokens=1000, output_tokens=200, total_tokens=1200,
                               finish_reason="stop")


@pytest.mark.asyncio
async def test_the_compaction_writer_text_is_sent_and_its_reply_is_accepted(monkeypatch):
    from datetime import timedelta

    from synap.cloud.application.compaction.passes.base import GROQ_PROFILE

    names = [V3 + ".writer", V3 + ".checks", V3 + ".framing", V3 + ".retry"]
    engines = put_core_in_place(monkeypatch, names, ["synap.cloud.application.compaction.v3.engine"])
    engine = engines["synap.cloud.application.compaction.v3.engine"]
    from synap.cloud.application.compaction.v3.chunker import label_turns
    from synap.cloud.application.compaction.v3.engine import EngineSettings
    from synap.cloud.application.compaction.v3 import CompactionWriter  # noqa: F401  (loads the package)

    start = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)
    texts = ["Hi, I'm Priya. Order ORD-88213-KX arrived broken.", "Sorry, Priya. I'll send you a return label today."]
    messages = [SimpleNamespace(message_id=f"id{i}", role="user" if i % 2 == 0 else "assistant", content=t,
                                timestamp=start + timedelta(minutes=i)) for i, t in enumerate(texts)]
    llm = _ScriptedLLM(COMPACTION_REPLY)
    result = await engine.CompactionWriter(llm, GROQ_PROFILE).update(
        None, label_turns(messages), EngineSettings(target_ratio=0.30, record_ceiling=7_500, chunk_tokens=15_000))
    sent = llm.calls[0]["messages"]
    assert sent[0]["content"].startswith(core(V3 + ".writer").WRITER_SYSTEM)
    assert "ORD-88213-KX" in sent[1]["content"] and "[[" not in sent[1]["content"]
    assert [entry.value for entry in result.record.ledger] == ["ORD-88213-KX"]
    assert [action.what for action in result.record.actions] == ["send a return label today"]
    from synap.cloud.application.compaction.v3 import record as v3_record

    assert_keys_agree([sent[0]["content"]], COMPACTION_REPLY, keys_read(v3_record.merge_output),
                      required={"turns", "literals", "agreed_actions", "changed_sections", "superseded",
                                "segment_density", "use_case", "chunk_recap"},
                      ignore={"plan", "headers", "header", "status", "statuses", "quotes", "text", "m"})


def test_the_compaction_user_texts_take_the_values_the_engine_gives():
    writer = core(V3 + ".writer")
    user = writer.WRITER_USER.format(record="(empty)", recaps="", span="m1-m2", turns="[m1] hello", budget=120,
                                     density_guidance="", guidance="")
    assert "[m1] hello" in user and "m1-m2" in user and "120" in user and "[[" not in user
    density = writer.DENSITY_GUIDANCE.format(high=3.5, medium=1.5)
    assert "about 3.5 times" in density and "about 1.5 times" in density
    repair = writer.REPAIR_USER.format(problems="1. bad quote", draft="{}")
    assert "1. bad quote" in repair and "{}" in repair
    rollup = writer.ROLLUP_USER.format(current=9000, target=6000, sections="s1: text")
    assert "9000" in rollup and "6000" in rollup and "s1: text" in rollup


def test_the_compaction_repair_messages_carry_the_facts_the_check_found():
    checks = core(V3 + ".checks")
    assert "ship it" in checks.action_quote_missing("ship it", "the assistant", "I will ship it", "m2")
    assert "ship it" in checks.action_no_quote("ship it")
    assert "ORD-1" in checks.ledger_value_missing("ORD-1", "Order", "m1")
    assert "ORD-1" in checks.literal_missing("ORD-1", "m1")
    assert "tank" in checks.inventory_not_in_record("tank", "m3")
    assert "m4" in checks.promise_not_carried("m4", "I'll call")
    assert "try x" in checks.suggestion_not_carried("m4", "try x")
    assert "m5" in checks.response_mismatch("buy a bike", "m5")
    assert "Next steps" in checks.header_not_allowed("Next steps")
    assert "s9" in checks.no_such_section("s9")
    assert "300" in checks.size_over(450, 300)
    assert "200" in checks.size_under(40, 1000, 200)


def test_the_compaction_preamble_and_retry_line():
    assert "not instructions" in core(V3 + ".framing").FRAMING
    suffix = core(V3 + ".retry").JSON_RETRY_SUFFIX
    assert suffix.startswith("\n\n") and "JSON" in suffix


def test_steering_has_no_core_text_and_stops_when_used():
    writer = core(V3 + ".writer")
    with pytest.raises(ps.CorePromptMissing):
        "x" + writer.STEERING_SYSTEM


# --- blocks code adds to extraction requests ---------------------------------------------------

def test_the_prior_context_header_takes_the_context():
    header = core(ING + ".prior_context").HEADER.format(context="- The wedding is on 4 June.")
    assert "- The wedding is on 4 June." in header and "[[" not in header


def test_the_analysis_hints_header_sits_between_two_line_breaks():
    header = core(ING + ".stage3_extraction").ANALYSIS_HINTS_HEADER
    assert header.startswith("\n\n") and header.endswith("\n") and "hint" in header.lower()


def test_speaker_attribution_has_no_core_text():
    with pytest.raises(ps.CorePromptMissing):
        "x" + core(ING + ".stage3_extraction").SPEAKER_ATTRIBUTION_ADDENDUM


# --- the contract check itself ----------------------------------------------------------------

def test_the_key_check_fails_when_a_prompt_names_a_different_key():
    reply = {"classification": "NONE", "reason": "x"}
    with pytest.raises(AssertionError, match="never names"):
        assert_keys_agree(['Reply with {"verdict": "", "reason": ""}'], reply, {"classification", "reason"},
                          required={"classification", "reason"})
    with pytest.raises(AssertionError, match="recorded reply uses keys"):
        assert_keys_agree(['{"classification": "", "reason": ""}'], {"classification": "NONE", "why": "x"},
                          {"classification"}, required={"classification"})


# --- fill() fails loudly --------------------------------------------------------------------

@pytest.mark.parametrize("text,fields,message", [
    ("a [[IF TYPO]]b[[END IF]]", {"real": "x"}, "gave no TYPO"),
    ("a [[IF X]]b", {"x": "1"}, "no \\[\\[END IF\\]\\]"),
    ("a b[[END IF]]", {"x": "1"}, "no \\[\\[IF"),
    ("[[IF X]]a[[IF Y]]b[[END IF]][[END IF]]", {"x": "1", "y": "1"}, "do not nest"),
    ("hello [[NAME]]", {}, "needs the field \\[\\[NAME\\]\\]"),
    ("hello [[NAME]]", {"name": None}, "needs the field \\[\\[NAME\\]\\]"),
])
def test_fill_refuses_a_text_it_cannot_fill_exactly(text, fields, message):
    with pytest.raises(ps.CorePromptMissing, match=message):
        ps.fill(text, **fields)


def test_fill_keeps_or_drops_a_block_and_accepts_an_empty_value():
    text = "a[[IF X]] b [[X]][[END IF]] c [[Y]]."
    assert ps.fill(text, x="1", y="") == "a b 1 c ."
    assert ps.fill(text, x=None, y="") == "a c ."
    assert ps.fill(text, x="", y="z") == "a c z."


# --- steering ------------------------------------------------------------------------------------

def test_core_refuses_the_steering_setting_with_a_clear_message():
    from synap.cloud.memory_config import core_rules

    for value in (True, False):
        problems = core_rules.find_problems({"compaction_config": {"adaptive_thresholds": {"steering_enabled": value}}})
        assert len(problems) == 1
        path, message = problems[0]
        assert path == ("compaction_config", "adaptive_thresholds", "steering_enabled")
        assert "removed from Core" in message and "steering" in message and "no steering texts" in message
