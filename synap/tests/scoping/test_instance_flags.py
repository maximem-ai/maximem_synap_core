"""A flag on for the fleet or for named instances (shared/instance_flags)."""

import uuid
from types import SimpleNamespace

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.resolution import entity_resolver as er
from synap.cloud.agents.ingestion.services import node_memories, writer_entity
from synap.cloud.shared import scope_key
from synap.cloud.shared.instance_flags import enabled_for
from synap.cloud.shared.scope import ScopeHierarchy, parse_or_deterministic_uuid

FLAG = "SYNAP_TEST_FLAG"


def test_off_fleet_and_listed(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(FLAG + "_INSTANCES", raising=False)
    assert not enabled_for(FLAG, "inst_a")
    monkeypatch.setenv(FLAG + "_INSTANCES", "inst_a, inst_b")
    assert enabled_for(FLAG, "inst_a") and not enabled_for(FLAG, "inst_c") and not enabled_for(FLAG)
    # either id form names the instance
    assert enabled_for(FLAG, parse_or_deterministic_uuid("inst_b"))
    monkeypatch.setenv(FLAG + "_INSTANCES", "*")
    assert enabled_for(FLAG, "anything") and enabled_for(FLAG)
    monkeypatch.delenv(FLAG + "_INSTANCES")
    monkeypatch.setenv(FLAG, "true")
    assert enabled_for(FLAG, "inst_c")


def test_one_instance_gets_its_own_graph_and_another_does_not(monkeypatch):
    on, off, customer, user, client = (uuid.uuid4() for _ in range(5))
    for flag in ("SYNAP_GRAPH_PER_SCOPE", "SYNAP_ER_SCOPED_REGISTRY", "SYNAP_WRITER_ENTITY",
                 "SYNAP_GRAPH_NODE_MEMORIES"):
        monkeypatch.delenv(flag, raising=False)
        monkeypatch.setenv(flag + "_INSTANCES", str(on))

    def scope(instance):
        return Scope(client_id=client, instance_id=instance, customer_id=customer, user_id=user,
                     scope_level="user")

    # The write and the read of one instance agree on where its records are.
    assert scope_key.write_scope_id(scope(on)) == str(user)
    assert scope_key.write_scope_id(scope(off)) == str(customer)
    reader = ScopeHierarchy(instance_id=str(on), client_id=str(client), customer_id=str(customer),
                            user_id=str(user))
    assert scope_key.readable_scope_ids(reader)[0] == str(user)
    reader.instance_id = str(off)
    assert scope_key.readable_scope_ids(reader) == [str(customer), str(client)]

    assert er.registry_scope(scope(on)) == ("USER", str(user))
    assert er.registry_scope(scope(off)) == ("CUSTOMER", str(customer))
    assert writer_entity.label_for("I report to Halvard Finch.", scope(on)) == "the user"
    assert writer_entity.label_for("I report to Halvard Finch.", scope(off)) is None
    assert node_memories.enabled(on) and not node_memories.enabled(off)
