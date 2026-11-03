"""Two faults on the hybrid_graph path: a nested property that failed a
document's whole graph write, and configured node types dropped after the
model returned them."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synap.cloud.agents.ingestion.models import ExtractedEntity, ExtractedRelationship, Scope
from synap.cloud.agents.ingestion.pipeline.stage3_extraction import Stage3Extractor
from synap.cloud.agents.ingestion.storage.graph_wrapper import AsyncGraphWrapper, storable_properties
from synap.cloud.application.ingestion.startup import _get_default_maca_config


def test_a_value_neo4j_cannot_hold_becomes_its_json_text():
    out = storable_properties({
        "operation": "sailings", "count": 3, "ok": True, "gone": None,
        "change": {"from": "Monday", "to": "Wednesday"},
        "days": ["Monday", "Wednesday"], "mixed": ["a", 1], "nested": [{"a": 1}], "empty": [],
    })
    assert out["operation"] == "sailings" and out["count"] == 3 and out["ok"] is True
    assert "gone" not in out
    assert json.loads(out["change"]) == {"from": "Monday", "to": "Wednesday"}
    assert out["days"] == ["Monday", "Wednesday"] and out["empty"] == []
    assert json.loads(out["mixed"]) == ["a", 1] and json.loads(out["nested"]) == [{"a": 1}]


@pytest.mark.asyncio
async def test_a_relationship_with_a_nested_property_is_written():
    provider = SimpleNamespace(add_edge=lambda ns, a, b, t, props: props)
    scope = Scope(client_id=uuid.uuid4(), instance_id=uuid.uuid4(), customer_id=uuid.uuid4(),
                  user_id=uuid.uuid4(), scope_level="user")
    rel = ExtractedRelationship(source_entity_id="a", target_entity_id="b", relationship_type="MOVED",
                                properties={"change": {"from": "Monday", "to": "Wednesday"}})
    written = await AsyncGraphWrapper(provider).create_relationship(rel, scope)
    assert isinstance(written["change"], str)
    assert written["user_id"] == str(scope.user_id)


def _extractor(config):
    return Stage3Extractor(config=config, llm_client=SimpleNamespace(complete=AsyncMock()),
                           entity_extractor=object())


REPLY = json.dumps({"entities": [{"name": "Halvard Finch", "entity_type": "person"},
                                 {"name": "Zentrolex", "entity_type": "drug"}]})


def test_single_call_path_keeps_a_configured_node_type():
    config = _get_default_maca_config()
    config.memory_extraction.domain_hints.key_entities = ["person", "Drug"]
    names = [e.name for e in _extractor(config)._parse_graph_entities(REPLY)]
    assert names == ["Halvard Finch", "Zentrolex"]


def test_single_call_path_unchanged_when_nothing_is_configured():
    names = [e.name for e in _extractor(_get_default_maca_config())._parse_graph_entities(REPLY)]
    assert names == ["Halvard Finch"]


class _ConstraintError(Exception):
    pass


def _race_provider(losses):
    """A graph provider whose first *losses* creates lose to a concurrent worker."""
    calls = {"add": 0, "winner": None}

    def find(ns, props, limit):
        w = calls["winner"]
        return [w] if w is not None and props.get("canonical_name") == w.properties["canonical_name"] else []

    def add(ns, etype, props):
        calls["add"] += 1
        if calls["add"] <= losses:
            calls["winner"] = SimpleNamespace(id="winner", properties={
                "canonical_name": props["canonical_name"], "er_node_key": "someone-else"})
            raise _ConstraintError("Neo.ClientError.Schema.ConstraintValidationFailed: Node(197) already exists")
        return "created"

    return SimpleNamespace(find_nodes_by_properties_any_type=find, add_node=add,
                           update_node=lambda *a, **k: None), calls


def _keyed_entity():
    return ExtractedEntity(entity_id="e1", name="user", entity_type="person", properties={"er_node_key": "key:abcdefgh"})


@pytest.mark.asyncio
@pytest.mark.parametrize("losses", [1, 2, 3])
async def test_a_keyed_entity_write_survives_losing_the_create_race_more_than_once(losses, monkeypatch):
    # Twelve workers write the user's node for every session; losing the retry once is not rare.
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    provider, calls = _race_provider(losses)
    scope = Scope(client_id=uuid.uuid4(), instance_id=uuid.uuid4(), customer_id=uuid.uuid4(),
                  user_id=uuid.uuid4(), scope_level="user")
    node_id = await AsyncGraphWrapper(provider).upsert_entity_node(_keyed_entity(), scope)
    assert node_id in ("created", "winner")
    assert calls["add"] <= losses + 1


@pytest.mark.asyncio
async def test_a_keyed_entity_write_still_fails_when_the_race_never_ends(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    provider, calls = _race_provider(losses=99)
    scope = Scope(client_id=uuid.uuid4(), instance_id=uuid.uuid4(), customer_id=uuid.uuid4(),
                  user_id=uuid.uuid4(), scope_level="user")
    with pytest.raises(_ConstraintError):
        await AsyncGraphWrapper(provider).upsert_entity_node(_keyed_entity(), scope)
    assert calls["add"] == 5
