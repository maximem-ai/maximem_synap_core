"""Unit tests for cross-document memory->edge relink matching."""
from synap.cloud.agents.ingestion.storage.graph_wrapper import AsyncGraphWrapper


def test_match_attaches_memory_naming_both_endpoints():
    nodes = {"n1": "STATUS-247", "n2": "TEST-STATUS-247-V1", "n3": "DEC-051"}
    edges = [
        {"edge_id": "e1", "from_id": "n1", "to_id": "n2"},   # STATUS-247 -- TEST
        {"edge_id": "e2", "from_id": "n1", "to_id": "n3"},   # STATUS-247 -- DEC-051
    ]
    memories = [
        {"memory_id": "m1", "content": "The test report TEST-STATUS-247-V1 is a current blocker on STATUS-247."},
        {"memory_id": "m2", "content": "STATUS-247 implements decision DEC-051."},
        {"memory_id": "m3", "content": "An unrelated note about onboarding."},
    ]
    out = AsyncGraphWrapper._match_memories_to_canonical_edges(memories, nodes, edges)
    # blocker memory names STATUS-247 + TEST-STATUS-247-V1 -> attaches to e1
    assert [m["memory_id"] for m in out["e1"]] == ["m1"]
    # implements memory names STATUS-247 + DEC-051 -> attaches to e2
    assert [m["memory_id"] for m in out["e2"]] == ["m2"]


def test_match_requires_both_endpoints():
    nodes = {"n1": "STATUS-247", "n2": "TEST-STATUS-247-V1"}
    edges = [{"edge_id": "e1", "from_id": "n1", "to_id": "n2"}]
    # only names one endpoint -> no attach
    memories = [{"memory_id": "m1", "content": "STATUS-247 is in progress."}]
    out = AsyncGraphWrapper._match_memories_to_canonical_edges(memories, nodes, edges)
    assert out == {}


def test_match_word_boundary_no_substring_falsepos():
    # "STATUS-24" must not match inside "STATUS-247"
    nodes = {"n1": "STATUS-24", "n2": "DEC-051"}
    edges = [{"edge_id": "e1", "from_id": "n1", "to_id": "n2"}]
    memories = [{"memory_id": "m1", "content": "STATUS-247 relates to DEC-051."}]
    out = AsyncGraphWrapper._match_memories_to_canonical_edges(memories, nodes, edges)
    assert out == {}  # STATUS-24 not present as a whole word


def test_match_skips_self_loops_and_unknown_nodes():
    nodes = {"n1": "STATUS-247"}
    edges = [
        {"edge_id": "e_self", "from_id": "n1", "to_id": "n1"},      # self-loop
        {"edge_id": "e_unk", "from_id": "n1", "to_id": "n_missing"},  # unknown endpoint
    ]
    memories = [{"memory_id": "m1", "content": "STATUS-247 and STATUS-247 again."}]
    out = AsyncGraphWrapper._match_memories_to_canonical_edges(memories, nodes, edges)
    assert out == {}
