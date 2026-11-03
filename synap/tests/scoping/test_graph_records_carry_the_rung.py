"""Graph records carry the rung, and carry nothing new when there is no rung.

A vector document has carried `node_id` plus one `anc_<level_key>` per ancestor
since S-03. Graph records carried neither, so the only address a graph record
had was the flat customer/user pair, and a rung between the two had nowhere to
live. That is why the read filter and the graph arm could answer differently
about the same record: one of them had a field for the question and the other
did not.

The keys are additive. A caller with no resolved chain writes exactly what it
wrote before, which is every request from clients running the
default three rungs while nested scoping is off, and that is asserted here
rather than assumed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from synap.cloud.agents.ingestion.models import (
    ExtractedEntity,
    ExtractedRelationship,
    Scope,
    ScopeRung,
)
from synap.cloud.agents.ingestion.storage.graph_wrapper import AsyncGraphWrapper

pytestmark = pytest.mark.asyncio

CLIENT = UUID("11111111-1111-5111-8111-111111111111")
INST = UUID("aaaaaaaa-1111-5111-8111-111111111111")
CUST = UUID("22222222-2222-5222-8222-222222222222")
PRACTICE = UUID("33333333-3333-5333-8333-333333333333")
TEAM = UUID("44444444-4444-5444-8444-444444444444")
USER = UUID("66666666-6666-5666-8666-666666666666")


def _deep_chain():
    return [
        ScopeRung(node_id=CLIENT, level_key="client", role="root", roles=["root"]),
        ScopeRung(node_id=CUST, level_key="customer", role="tenant_of_record",
                  roles=["tenant_of_record"]),
        ScopeRung(node_id=PRACTICE, level_key="practice"),
        ScopeRung(node_id=TEAM, level_key="care_team"),
        ScopeRung(node_id=USER, level_key="user", role="identity", roles=["identity"]),
    ]


def _scope(chain=None, scope_level="user") -> Scope:
    return Scope(client_id=CLIENT, instance_id=INST, customer_id=CUST,
                 user_id=USER, scope_level=scope_level, chain=chain)


def _provider():
    p = MagicMock()
    p.add_node = MagicMock(return_value="node-1")
    p.add_edge = MagicMock(return_value="edge-1")
    p.get_edge = MagicMock(return_value=None)
    p.update_edge = MagicMock(return_value=None)
    return p


def _entity() -> ExtractedEntity:
    return ExtractedEntity(entity_id="e1", name="Ravi", entity_type="person",
                           canonical_name="Ravi")


async def test_an_entity_node_carries_its_node_id_and_every_ancestor():
    p = _provider()
    await AsyncGraphWrapper(p).upsert_entity_node(_entity(), _scope(_deep_chain()))

    props = p.add_node.call_args[0][2]
    assert props["node_id"] == str(USER)
    assert props["anc_practice"] == str(PRACTICE)
    assert props["anc_care_team"] == str(TEAM)
    assert props["anc_client"] == str(CLIENT)
    assert props["anc_customer"] == str(CUST)


async def test_a_relationship_edge_carries_the_same_keys():
    p = _provider()
    rel = ExtractedRelationship(source_entity_id="e1", target_entity_id="e2",
                                relationship_type="KNOWS")
    await AsyncGraphWrapper(p).create_relationship(rel, _scope(_deep_chain()))

    props = p.add_edge.call_args[0][4]
    assert props["node_id"] == str(USER)
    assert props["anc_care_team"] == str(TEAM)


async def test_a_write_filed_at_an_intermediate_rung_stamps_that_rung():
    """`scope_level="care_team"` is the whole point of an arbitrary ladder.

    The record must be filed at the team, not at the person, and it must carry
    the ancestors above the team and none below it.
    """
    p = _provider()
    await AsyncGraphWrapper(p).upsert_entity_node(
        _entity(), _scope(_deep_chain(), scope_level="care_team")
    )

    props = p.add_node.call_args[0][2]
    assert props["node_id"] == str(TEAM)
    assert props["anc_practice"] == str(PRACTICE)
    assert "anc_user" not in props


async def test_no_chain_adds_no_keys_at_all():
    """Clients on the default three rungs, with the flag off."""
    p = _provider()
    await AsyncGraphWrapper(p).upsert_entity_node(_entity(), _scope(None))

    props = p.add_node.call_args[0][2]
    assert "node_id" not in props
    assert [k for k in props if k.startswith("anc_")] == []
    # And the keys that were always there still are.
    assert props["customer_id"] == str(CUST)
    assert props["user_id"] == str(USER)


async def test_a_memory_attached_to_an_edge_carries_the_rung():
    """`memory_contents` is what the Neo4j arm actually serves as content."""
    from datetime import datetime, timezone

    from synap.cloud.agents.ingestion.models import (
        ExtractedMemory,
        MemoryDestination,
        MemoryType,
    )

    p = _provider()
    mem = ExtractedMemory(
        memory_id=uuid4(), content="a fact", memory_type=MemoryType.FACTS,
        confidence=0.9, entities_mentioned=["e1", "e2"],
        destination=MemoryDestination.MEMORIES, source_document_id="doc-1",
        extracted_at=datetime.now(timezone.utc),
    )
    rel = ExtractedRelationship(source_entity_id="e1", target_entity_id="e2",
                                relationship_type="KNOWS")
    updated = await AsyncGraphWrapper(p).attach_memories_to_relationships(
        [mem], [rel], ["edge-1"], _scope(_deep_chain()),
    )

    assert updated == 1
    import json

    payload = json.loads(p.update_edge.call_args[0][2]["memory_contents"])
    assert payload[0]["node_id"] == str(USER)
    assert payload[0]["anc_care_team"] == str(TEAM)
