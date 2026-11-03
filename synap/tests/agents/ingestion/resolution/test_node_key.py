"""Graph nodes keyed by entity, not by spelling."""

import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.agents.ingestion.models import ExtractedEntity, Scope
from synap.cloud.agents.ingestion.resolution.node_key import (
    NODE_KEY_PROPERTY,
    is_provisional,
    may_use_node,
    new_provisional_key,
    node_name_candidates,
)
from synap.cloud.agents.ingestion.storage.graph_wrapper import AsyncGraphWrapper


class TestNaming:
    def test_resolved_entity_tries_name_then_type_then_key(self):
        assert node_name_candidates("Paris", "person", "0f3a9c1e-aaaa-bbbb-cccc-1234abcd5678") == [
            "Paris", "Paris (person)", "Paris [abcd5678]"]

    def test_unresolved_mention_skips_the_type_form(self):
        key = new_provisional_key()
        assert is_provisional(key)
        assert node_name_candidates("R. Smith", "person", key) == [
            "R. Smith", f"R. Smith [unresolved {key[-8:]}]"]

    def test_same_key_is_the_same_node(self):
        assert may_use_node("k1", "k1")
        assert not may_use_node("k1", "k2")

    def test_resolved_entity_adopts_a_node_written_before_keys(self):
        assert may_use_node("k1", None)

    def test_unresolved_mention_does_not_adopt_an_established_node(self):
        assert not may_use_node(new_provisional_key(), None)


class FakeGraph:
    """Enough of the provider: nodes by id, one node per canonical_name."""

    def __init__(self):
        self.nodes = {}

    def find_nodes_by_properties_any_type(self, namespace, properties, limit=10):
        return [n for n in self.nodes.values()
                if all(n.properties.get(k) == v for k, v in properties.items())][:limit]

    def add_node(self, namespace, node_type, properties):
        if self.find_nodes_by_properties_any_type(namespace, {"canonical_name": properties["canonical_name"]}):
            raise RuntimeError("ConstraintValidationFailed")
        node = SimpleNamespace(id=str(uuid.uuid4()), properties=dict(properties))
        self.nodes[node.id] = node
        return node.id

    def update_node(self, namespace, node_id, properties):
        self.nodes[node_id].properties.update(properties)


@pytest.fixture
def scope():
    return Scope(client_id=uuid.uuid4(), instance_id=uuid.uuid4(),
                 customer_id=uuid.uuid4(), user_id=uuid.uuid4())


@pytest.fixture
def graph():
    wrapper = AsyncGraphWrapper.__new__(AsyncGraphWrapper)
    wrapper._provider = FakeGraph()
    wrapper._get_namespace = lambda scope: "ns"
    wrapper._chain_properties = lambda scope: {}
    return wrapper


def entity(name, entity_type="person", key=None, canonical=None):
    e = ExtractedEntity(entity_id=str(uuid.uuid4()), entity_type=entity_type, name=name,
                        canonical_name=canonical)
    if key:
        e.properties[NODE_KEY_PROPERTY] = key
    return e


class TestKeyedUpsert:
    async def test_same_key_merges_and_keeps_the_spelling_as_an_alias(self, graph, scope):
        first = await graph.upsert_entity_node(entity("Robert Smith", key="k1"), scope)
        second = await graph.upsert_entity_node(
            entity("Bob Smith", key="k1", canonical="Robert Smith"), scope)
        assert first == second
        assert graph._provider.nodes[first].properties["aliases"] == ["Bob Smith"]

    async def test_same_spelling_different_entities_get_two_nodes(self, graph, scope):
        place = await graph.upsert_entity_node(entity("Paris", "location", key="k-place"), scope)
        person = await graph.upsert_entity_node(entity("Paris", "person", key="k-person"), scope)
        assert place != person
        props = graph._provider.nodes[person].properties
        assert props["canonical_name"] == "Paris (person)"
        assert props["display_name"] == "Paris" and props["name"] == "Paris"
        # ...and the person comes back to their own node next time.
        assert await graph.upsert_entity_node(entity("Paris", "person", key="k-person"), scope) == person

    async def test_two_unresolved_mentions_never_share_a_node(self, graph, scope):
        a = await graph.upsert_entity_node(entity("R. Smith", key=new_provisional_key()), scope)
        b = await graph.upsert_entity_node(entity("R. Smith", key=new_provisional_key()), scope)
        assert a != b
        assert graph._provider.nodes[b].properties["canonical_name"].startswith("R. Smith [unresolved ")

    async def test_resolved_entity_adopts_a_node_from_before_keys(self, graph, scope):
        legacy = await graph.upsert_entity_node(entity("Acme Corporation", "organization"), scope)
        keyed = await graph.upsert_entity_node(
            entity("Acme Corporation", "organization", key="k1"), scope)
        assert keyed == legacy
        assert graph._provider.nodes[legacy].properties[NODE_KEY_PROPERTY] == "k1"

    async def test_unresolved_mention_leaves_an_established_node_alone(self, graph, scope):
        legacy = await graph.upsert_entity_node(entity("R. Smith"), scope)
        unresolved = await graph.upsert_entity_node(
            entity("R. Smith", key=new_provisional_key()), scope)
        assert unresolved != legacy
        assert NODE_KEY_PROPERTY not in graph._provider.nodes[legacy].properties

    async def test_without_a_key_nothing_changes(self, graph, scope):
        """Legacy path: created by name; a second create of the name updates it."""
        first = await graph.upsert_entity_node(entity("Orion Health", "organization"), scope)
        second = await graph.upsert_entity_node(entity("Orion Health", "organization"), scope)
        assert first == second
