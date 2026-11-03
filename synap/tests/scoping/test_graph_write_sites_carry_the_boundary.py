"""Every place that writes a graph record must say who wrote it.

WHY THIS FILE EXISTS
--------------------
The instance boundary was lost in 2026-03 and nothing noticed for seven months,
because every test asked "does the filter compile" and none asked "does every
store carry what the filter reads". This walks the source of the graph writer
and fails when a write site exists that nobody has classified. It does not prove
the writes are right (the behavioural tests do that); it makes a NEW write site
impossible to add silently.

If this fails because you added a function that writes to the graph: decide which
list it belongs in, and say why in the comment beside it.
"""
from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2] / "cloud" / "agents" / "ingestion" / "storage"
_WRITER = _ROOT / "graph_wrapper.py"

_WRITE_CALLS = {"add_node", "add_edge", "update_node", "update_edge",
                "append_to_node_list", "append_to_edge_list"}

# Functions that CREATE a node or relationship, or write its scope keys. Each must
# stamp the record with `graph_boundary.node_edge_properties`.
CREATORS = {"upsert_entity_node", "_upsert_keyed_entity_node", "create_relationship"}

# Functions that only change a record that already exists, never its ownership:
#   attach_memories_to_relationships  builds the memory dicts (stamped via memory_properties)
#   add_memory_references             appends memory ids to a node
#   relink / soft-delete / restore  rewrite existing memory dicts in place
#
# update_memory_on_edges is NOT a pure modifier: it APPENDS a new versioned memory
# entry, so it is listed in MEMORY_BUILDERS and must stamp it.
MODIFIERS = {
    "attach_memories_to_relationships",
    "add_memory_references",
    "relink_memories_to_canonical_edges",
    "soft_delete_memory_from_edges",
    "restore_memory_on_edges",
}


# Functions that BUILD a memory entry. Each must stamp it with its instance.
MEMORY_BUILDERS = {
    "attach_memories_to_relationships": "graph_boundary.memory_properties",
    "update_memory_on_edges": "graph_boundary.INSTANCE_ID",
}


def _functions():
    tree = ast.parse(_WRITER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            yield node


def _calls(fn):
    """Every adapter method the function touches: called, or handed to
    `asyncio.to_thread(self._provider.add_node, ...)`, which is how the writer
    calls the (synchronous) adapter."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute):
            yield node.attr


def _mentions(fn, dotted):
    return dotted in ast.get_source_segment(_WRITER.read_text(), fn)


def _writers():
    return {fn.name: fn for fn in _functions() if _WRITE_CALLS & set(_calls(fn))}


def test_no_unclassified_function_writes_to_the_graph():
    unknown = set(_writers()) - CREATORS - MODIFIERS - set(MEMORY_BUILDERS)
    assert not unknown, (
        f"{sorted(unknown)} write graph records and are in neither CREATORS nor MODIFIERS. "
        "Decide whether they set ownership (CREATORS: must stamp instance_ids) or only change an "
        "existing record (MODIFIERS), and add them here with a reason.")


def test_every_creator_is_still_a_writer_so_the_lists_cannot_rot():
    missing = (CREATORS | MODIFIERS | set(MEMORY_BUILDERS)) - set(_writers())
    assert not missing, f"listed but no longer writing: {sorted(missing)}; remove them from the lists"


def test_every_creator_stamps_the_record():
    writers = _writers()
    for name in sorted(CREATORS):
        assert _mentions(writers[name], "graph_boundary.node_edge_properties"), (
            f"{name} creates graph records without graph_boundary.node_edge_properties(scope)")


def test_the_memory_dicts_are_stamped_where_they_are_built():
    writers = _writers()
    for name, needle in MEMORY_BUILDERS.items():
        assert _mentions(writers[name], needle), f"{name} builds a memory entry without stamping its instance"


def test_the_adapter_never_loses_the_list_to_a_plain_set():
    """`SET n += $props` over a list would drop every other instance."""
    adapter = pathlib.Path(__file__).resolve().parents[2] / "providers" / "graph" / "adapters" / "neo4j.py"
    text = adapter.read_text()
    tree = ast.parse(text)
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in {"update_node", "update_edge", "add_edge"}:
            body = ast.get_source_segment(text, fn)
            assert "split_instance_ids" in body, f"{fn.name} must route instance_ids through the union"


def test_nothing_outside_the_writer_creates_graph_records():
    """A second writer would be a second place to forget the stamp."""
    import re
    cloud = pathlib.Path(__file__).resolve().parents[2] / "cloud"
    pattern = re.compile(r"(?:_provider|graph_provider|graph_adapter)\.(?:add_node|add_edge)\b")
    offenders = []
    for path in cloud.rglob("*.py"):
        if path == _WRITER or "tests" in path.parts:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(cloud.parent)}:{number}")
    assert not offenders, f"graph records created outside graph_wrapper: {offenders}"
