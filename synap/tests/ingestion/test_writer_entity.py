"""The writer of a document as an entity (services/writer_entity), and the
temporal prompt keeping who a deadline is for. Both are off by default."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synap.cloud.agents.ingestion.models import ExtractedEntity, Scope
from synap.cloud.agents.ingestion.pipeline.stage3_extraction import Stage3Extractor
from synap.cloud.agents.ingestion.prompts.extraction import temporal
from synap.cloud.agents.ingestion.resolution.node_key import NODE_KEY_PROPERTY
from synap.cloud.agents.ingestion.services import writer_entity as we
from synap.cloud.application.ingestion.startup import _get_default_maca_config

USER = uuid.UUID("11111111-2222-3333-4444-555555555555")
OTHER = uuid.UUID("11111111-2222-3333-4444-666666666666")
SCOPE = SimpleNamespace(user_id=USER, customer_id="c", instance_id=None, client_id=None)
NOTE = "I report to Halvard Finch on the Night Freight team."


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("SYNAP_WRITER_ENTITY", "true")


def _person(name):
    return ExtractedEntity(entity_id=str(uuid.uuid4()), entity_type="person", name=name)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SYNAP_WRITER_ENTITY", raising=False)
    assert we.label_for(NOTE, SCOPE) is None
    assert we.anchor_for(["Who do I report to?"], SCOPE, [we.node_name(USER)]) is None


def test_label_only_for_a_first_person_note_with_a_user(on):
    assert we.label_for(NOTE, SCOPE) == "the user"
    assert we.label_for("Halvard Finch leads the Night Freight team.", SCOPE) is None
    assert we.label_for(NOTE, SimpleNamespace(user_id=None)) is None


def test_one_node_per_user_keyed_on_the_user_id(on):
    a, b = we.entity(SCOPE), we.entity(SimpleNamespace(user_id=OTHER))
    assert a.name == b.name == "the user"
    assert a.canonical_name != b.canonical_name
    assert a.properties[NODE_KEY_PROPERTY] == f"user:{USER}"
    assert we.is_writer(a) and not we.is_writer(_person("Halvard Finch"))


def test_stamped_marks_the_models_own_entity_or_adds_one(on):
    returned = [_person("Halvard Finch"), _person("The User")]
    out = we.stamped(returned, SCOPE)
    assert len(out) == 2 and we.is_writer(out[1]) and out[1].name == "the user"
    out = we.stamped([_person("Halvard Finch")], SCOPE)
    assert len(out) == 2 and we.is_writer(out[1])


def test_writer_with_no_relationship_is_dropped(on):
    w, p = we.entity(SCOPE), _person("Halvard Finch")
    rel = SimpleNamespace(source_entity_id=w.entity_id, target_entity_id=p.entity_id)
    assert we.without_unused([p, w], [rel]) == [p, w]
    assert we.without_unused([p, w], []) == [p]


def test_anchor_only_for_a_first_person_question_and_an_existing_node(on):
    name = we.node_name(USER)
    assert we.anchor_for(["Who do I report to?"], SCOPE, [name, "Halvard Finch"]) == name
    assert we.anchor_for(["Who leads Night Freight?"], SCOPE, [name]) is None
    assert we.anchor_for(["Who do I report to?"], SCOPE, ["Halvard Finch"]) is None


def _extractor(reply):
    llm = SimpleNamespace(complete=AsyncMock(return_value=json.dumps(reply)))
    return Stage3Extractor(config=_get_default_maca_config(), llm_client=llm, entity_extractor=object()), llm


@pytest.mark.asyncio
async def test_separate_path_gives_the_relationship_its_true_source(on):
    reply = {"relationships": [{"source_entity": "the user", "target_entity": "Halvard Finch",
                                "relationship_type": "REPORTS_TO", "confidence": 0.9}]}
    extractor, llm = _extractor(reply)
    boss, writer = _person("Halvard Finch"), we.entity(SCOPE)
    rels = await extractor._extract_relationships(NOTE, [boss], request_scope=SCOPE, writer=writer)
    assert [(r.source_entity_id, r.target_entity_id) for r in rels] == [(writer.entity_id, boss.entity_id)]
    user_prompt = llm.complete.call_args.kwargs["messages"][1]["content"]
    assert "the user (person): the writer of this content" in user_prompt


@pytest.mark.asyncio
async def test_separate_path_without_a_writer_is_unchanged(on):
    extractor, llm = _extractor({"relationships": []})
    assert await extractor._extract_relationships(NOTE, [_person("Halvard Finch")], request_scope=SCOPE) == []
    llm.complete.assert_not_called()      # one entity, no writer: nothing to relate


@pytest.mark.asyncio
async def test_single_call_path_marks_the_writer(on):
    reply = {"entities": [{"name": "the user", "entity_type": "person"},
                          {"name": "Halvard Finch", "entity_type": "person"}],
             "relationships": [{"source_entity": "the user", "target_entity": "Halvard Finch",
                                "relationship_type": "REPORTS_TO", "confidence": 0.9}]}
    extractor, _ = _extractor(reply)
    entities, rels = await extractor._extract_graph(NOTE, request_scope=SCOPE, writer="the user")
    writer = next(e for e in entities if we.is_writer(e))
    assert writer.canonical_name == we.node_name(USER)
    assert rels and rels[0].source_entity_id == writer.entity_id


@pytest.mark.asyncio
async def test_resolution_leaves_the_writer_alone(on, monkeypatch):
    from synap.cloud.agents.ingestion.resolution import entity_resolver as er
    monkeypatch.setattr(er, "adjudicated_mode", lambda: False)
    seen = []

    async def resolve(mention, **_):
        seen.append(mention)
        return SimpleNamespace(resolved=False, confidence=0.0)

    service = er.EntityResolutionService.__new__(er.EntityResolutionService)
    service._enabled, service._similarity_threshold = True, 0.9
    service._resolver = SimpleNamespace(resolve=resolve)
    service._auto_register = AsyncMock()
    scope = Scope(client_id=uuid.uuid4(), instance_id=uuid.uuid4(), customer_id=uuid.uuid4(),
                  user_id=USER, scope_level="user")
    writer, boss = we.entity(scope), _person("Halvard Finch")
    out = await service.resolve_entities([writer, boss], scope)
    assert seen == ["Halvard Finch"]
    assert writer in out and boss in out


def test_temporal_prompt_unchanged_unless_asked(monkeypatch):
    prompt = temporal.TemporalExtractionPrompt(_get_default_maca_config())
    monkeypatch.delenv("SYNAP_TEMPORAL_KEEP_PERSON", raising=False)
    off, _ = prompt.build("Halvard Finch wants the safety audit by Tuesday.", [], instance_id=None)
    assert "who_is_involved" not in off
    monkeypatch.setenv("SYNAP_TEMPORAL_KEEP_PERSON", "true")
    on_, _ = prompt.build("Halvard Finch wants the safety audit by Tuesday.", [], instance_id=None)
    assert on_.startswith(off[:200]) and "who_is_involved" in on_
