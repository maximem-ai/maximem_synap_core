"""GraphLifecycle.release (W3): both storage formats, retire vs erase, the rule.

Before this service no path changed the graph when a memory was retired or
deleted (issue12_graph.md section 2): supersession never called the graph, and
the delete helper only understood copies and reported success on 0 edges.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

import pytest

from synap.cloud.agents.ingestion.models import Scope
from synap.cloud.agents.ingestion.services.graph_lifecycle_rule import SupporterRow
from synap.cloud.agents.ingestion.storage.graph_lifecycle import (
    GraphLifecycle, ReleaseResult, release_namespaces)
from synap.cloud.agents.ingestion.storage.graph_wrapper import AsyncGraphWrapper

from _glc_fakes import FakeGraph, FakeVector, supporters_loader

CLIENT = UUID("00000000-0000-0000-0000-00000000000c")
INST = UUID("00000000-0000-0000-0000-00000000000a")
CUST = UUID("00000000-0000-0000-0000-0000000000c1")
USER = UUID("00000000-0000-0000-0000-0000000000a1")


@pytest.fixture(autouse=True)
def _per_scope(monkeypatch):
    monkeypatch.setenv("SYNAP_GRAPH_PER_SCOPE", "true")


def _scope():
    return Scope(client_id=CLIENT, instance_id=INST, customer_id=CUST, user_id=USER, scope_level="user")


NS = str(USER)
D = lambda day: datetime(2026, 10, day, tzinfo=timezone.utc)  # noqa: E731


def _rows(**rows):
    return supporters_loader(rows)


def _acme_graph(memory_ids, node_ids=None, **edge_props):
    g = FakeGraph()
    g.node(NS, "u", "the user", is_writer=True)
    g.node(NS, "acme", "Acme Corp", "organization", memory_ids=list(node_ids or memory_ids))
    g.edge(NS, "e1", "u", "acme", "WORKS_AT", memory_ids=list(memory_ids), **edge_props)
    return g


def test_namespaces_cover_write_legacy_and_readable_ones():
    ns = release_namespaces(_scope())
    assert ns[0] == NS
    assert str(CUST) in ns and str(CLIENT) in ns


async def test_supersession_retires_edge_and_node_with_no_support_left():
    g = _acme_graph(["m1"])
    rows = {"m1": SupporterRow(id="m1", superseded_by="m2", superseded_at=D(26), document_created_at=D(1),
                               superseder_document_created_at=D(26))}
    out = await GraphLifecycle(g, load_supporters=_rows(**rows)).release(
        _scope(), ["m1"], mode="retire", reason="superseded", superseded_by="m2")
    e = g.e(NS, "e1")
    assert e["memory_ids"] == [] and e["released_memory_ids"] == ["m1"]
    assert e["retired_at"] and e["retired_by_memory_id"] == "m2" and e["retired_reason"] == "superseded"
    assert g.n(NS, "acme")["retired_at"]
    assert out.edges_retired == 1 and out.nodes_retired == 1 and out.edges_deleted == 0
    assert NS in out.namespaces


async def test_t2_closes_edge_that_an_older_live_memory_still_names():
    # The 4b case: WORKS_AT Acme [m_old(superseded by n), m_live(older, live)].
    g = _acme_graph(["m_old", "m_live"])
    rows = {
        "m_old": SupporterRow(id="m_old", superseded_by="n", document_created_at=D(1),
                              superseder_document_created_at=D(26)),
        "m_live": SupporterRow(id="m_live", document_created_at=D(1)),
    }
    out = await GraphLifecycle(g, load_supporters=_rows(**rows)).release(
        _scope(), ["m_old"], mode="retire", reason="superseded", superseded_by="n")
    e = g.e(NS, "e1")
    assert e["memory_ids"] == ["m_live"], "the live supporter stays referenced"
    assert e["retired_at"] and e["retired_by_memory_id"] == "n"
    assert e["valid_to"].startswith("2026-10-26")
    assert out.records_closed >= 1


async def test_mixed_support_said_after_the_supersession_stays_open():
    g = _acme_graph(["m_old", "m_new_same"])
    rows = {
        "m_old": SupporterRow(id="m_old", superseded_by="n", document_created_at=D(1),
                              superseder_document_created_at=D(20)),
        "m_new_same": SupporterRow(id="m_new_same", document_created_at=D(25)),
    }
    await GraphLifecycle(g, load_supporters=_rows(**rows)).release(
        _scope(), ["m_old"], mode="retire", reason="superseded", superseded_by="n")
    assert "retired_at" not in g.e(NS, "e1")


async def test_erase_deletes_unsupported_edge_orphan_node_and_their_vectors():
    g = _acme_graph(["m1"])
    vec = FakeVector()
    out = await GraphLifecycle(g, vector=vec, load_supporters=_rows()).release(
        _scope(), ["m1"], mode="erase", reason="deleted")
    assert g.e(NS, "e1") is None
    assert g.n(NS, "acme") is None, "Acme has no edge and no memory left"
    assert g.n(NS, "u") is not None, "the writer's node is never deleted"
    assert out.edges_deleted == 1 and out.nodes_deleted == 1
    collections = dict(vec.deleted)
    assert len(collections["triples"]) == 1 and len(collections["entities"]) == 1
    from synap.cloud.agents.ingestion.storage.vector_wrapper import deterministic_entity_id
    acme = deterministic_entity_id("Acme Corp", "organization", str(CLIENT), NS)
    assert collections["entities"] == [acme]
    assert collections["triples"][0].endswith(f"_WORKS_AT_{acme}")


async def test_erase_keeps_edge_another_live_memory_supports():
    g = _acme_graph(["m1", "m2"])
    out = await GraphLifecycle(g, load_supporters=_rows(m2=SupporterRow(id="m2"))).release(
        _scope(), ["m1"], mode="erase", reason="deleted")
    e = g.e(NS, "e1")
    assert e is not None and e["memory_ids"] == ["m2"] and "retired_at" not in e
    assert out.edges_deleted == 0 and out.edges_pruned == 1


async def test_writer_and_customer_nodes_survive_erase():
    g = FakeGraph()
    g.node(NS, "u", "the user", is_writer=True, memory_ids=["m1"])
    g.node(NS, "co", "the user's company", is_customer="True")
    g.edge(NS, "e1", "u", "co", "WORKS_AT", memory_ids=["m1"])
    await GraphLifecycle(g, load_supporters=_rows()).release(_scope(), ["m1"], mode="erase", reason="deleted")
    assert g.e(NS, "e1") is None
    assert g.n(NS, "u") is not None and g.n(NS, "co") is not None
    assert "retired_at" not in g.n(NS, "u")


async def test_unbacked_edge_is_untouched():
    g = _acme_graph(["m1"])
    g.node(NS, "doc", "doc 4", "document")
    g.edge(NS, "e2", "doc", "acme", "ABOUT")
    await GraphLifecycle(g, load_supporters=_rows()).release(_scope(), ["m1"], mode="erase", reason="deleted")
    assert g.e(NS, "e2") is not None and "retired_at" not in g.e(NS, "e2")
    assert g.n(NS, "acme") is not None, "still has the unbacked edge"


async def test_release_is_idempotent():
    g = _acme_graph(["m1", "m2"])
    lc = GraphLifecycle(g, load_supporters=_rows(m2=SupporterRow(id="m2")))
    await lc.release(_scope(), ["m1"], mode="erase", reason="deleted")
    snapshot = json.dumps(g.edges, sort_keys=True, default=str)
    again = await lc.release(_scope(), ["m1"], mode="erase", reason="deleted")
    assert json.dumps(g.edges, sort_keys=True, default=str) == snapshot
    assert not again.changed


async def test_reopen_on_append_of_a_live_memory():
    g = _acme_graph(["m1"])
    await GraphLifecycle(g, load_supporters=_rows()).release(_scope(), ["m1"], mode="retire", reason="deleted")
    assert g.e(NS, "e1")["retired_at"]
    g.append_to_edge_list(NS, "e1", "memory_ids", ["m9"], 500)
    assert "retired_at" not in g.e(NS, "e1")
    # Re-appending an id already there (a replay) does not reopen.
    g.set_record_states(NS, [{"kind": "edge", "id": "e1", "open": False, "reason": "superseded"}])
    g.append_to_edge_list(NS, "e1", "memory_ids", ["m9"], 500)
    assert g.e(NS, "e1")["retired_at"]


async def test_update_retires_old_support_with_reason_updated():
    g = _acme_graph(["m1"])
    # The memory is alive (its text was edited), so it is NOT support any more.
    out = await GraphLifecycle(g, load_supporters=_rows(m1=SupporterRow(id="m1"))).release(
        _scope(), ["m1"], mode="retire", reason="updated")
    e = g.e(NS, "e1")
    assert e["retired_at"] and e["retired_reason"] == "updated"
    assert out.edges_retired == 1


# --- copies mode (memory_contents) -------------------------------------------------

def _copies_graph(entries):
    g = FakeGraph()
    g.node(NS, "u", "the user", is_writer=True)
    g.node(NS, "acme", "Acme Corp", "organization")
    g.edge(NS, "e1", "u", "acme", "WORKS_AT", memory_contents=json.dumps(entries))
    return g


def _entry(mid, text="works at Acme"):
    return {"memory_id": mid, "content": text, "embedding": [0.1], "is_current": True, "is_deleted": False}


async def test_copies_mode_retire_marks_entry_and_retires_edge():
    g = _copies_graph([_entry("m1")])
    rows = {"m1": SupporterRow(id="m1", superseded_by="m2", superseder_document_created_at=D(26))}
    out = await GraphLifecycle(g, load_supporters=_rows(**rows)).release(
        _scope(), ["m1"], mode="retire", reason="superseded", superseded_by="m2")
    e = g.e(NS, "e1")
    entry = json.loads(e["memory_contents"])[0]
    assert entry["is_current"] is False and entry["superseded_by"] == "m2"
    assert e["retired_at"] and e["retired_by_memory_id"] == "m2"
    assert out.legacy_entries_marked == 1


async def test_copies_mode_forget_redacts_and_erases_unsupported_edge():
    g = _copies_graph([_entry("m1")])
    out = await GraphLifecycle(g, load_supporters=_rows()).release(
        _scope(), ["m1"], mode="erase", reason="deleted", redact=True)
    assert g.e(NS, "e1") is None and g.n(NS, "acme") is None
    assert out.legacy_entries_marked == 1 and out.edges_deleted == 1


async def test_copies_mode_erase_keeps_edge_with_live_entry_and_redacts():
    g = _copies_graph([_entry("m1", "secret"), _entry("m2")])
    await GraphLifecycle(g, load_supporters=_rows(m2=SupporterRow(id="m2"))).release(
        _scope(), ["m1"], mode="erase", reason="deleted", redact=True)
    entries = {x["memory_id"]: x for x in json.loads(g.e(NS, "e1")["memory_contents"])}
    assert entries["m1"]["is_deleted"] is True and entries["m1"]["content"] == ""
    assert "embedding" not in entries["m1"]
    assert entries["m2"]["is_current"] is True
    assert "retired_at" not in g.e(NS, "e1")


# --- legacy helpers are thin wrappers ---------------------------------------------

async def test_soft_delete_helper_now_works_in_references_mode(monkeypatch):
    import synap.cloud.agents.ingestion.storage.graph_lifecycle as glc
    monkeypatch.setattr(glc, "_default_loader", lambda: _rows())
    g = _acme_graph(["m1"])
    n = await AsyncGraphWrapper(g).soft_delete_memory_from_edges("m1", _scope())
    assert n >= 1, "it used to find 0 edges in references mode and report success"
    assert g.e(NS, "e1") is None


async def test_update_helper_now_works_in_references_mode(monkeypatch):
    import synap.cloud.agents.ingestion.storage.graph_lifecycle as glc
    monkeypatch.setattr(glc, "_default_loader", lambda: _rows(m1=SupporterRow(id="m1")))
    g = _acme_graph(["m1"])
    n = await AsyncGraphWrapper(g).update_memory_on_edges("m1", "new text", "fact", 0.9, _scope())
    assert n >= 1
    assert g.e(NS, "e1")["retired_reason"] == "updated"


async def test_graph_failure_raises_so_the_outbox_keeps_the_row():
    g = _acme_graph(["m1"])
    g.fail_release = True
    with pytest.raises(RuntimeError):
        await GraphLifecycle(g, load_supporters=_rows()).release(_scope(), ["m1"], mode="erase", reason="deleted")


async def test_erase_record_by_graph_id():
    g = _acme_graph(["m1"])
    erased = await GraphLifecycle(g, load_supporters=_rows()).erase_record(_scope(), "e1")
    assert erased.kind == "relationship" and erased.memory_ids == ["m1"]
    assert erased.edges_deleted == 1 and erased.nodes_deleted == 0, "Acme still carries m1"
    assert await GraphLifecycle(g).erase_record(_scope(), "nope") is None
    # The writer's node cannot be erased by id.
    assert await GraphLifecycle(g).erase_record(_scope(), "u") is None
    assert g.n(NS, "u") is not None


def test_release_result_adds_up():
    a = ReleaseResult(edges_pruned=1, namespaces=["x"])
    a.add(ReleaseResult(edges_pruned=2, namespaces=["x", "y"]))
    assert a.edges_pruned == 3 and a.namespaces == ["x", "y"]
