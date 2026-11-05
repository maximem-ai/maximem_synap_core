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
#   relink / restore                  rewrite existing memory dicts in place
#
# soft_delete_memory_from_edges and update_memory_on_edges no longer write the
# graph themselves: they delegate to the graph lifecycle service (DELEGATORS
# below), whose writes are guarded by the LIFECYCLE tests further down.
MODIFIERS = {
    "attach_memories_to_relationships",
    "add_memory_references",
    "relink_memories_to_canonical_edges",
    "restore_memory_on_edges",
}


# Functions that BUILD a memory entry. Each must stamp it with its instance.
MEMORY_BUILDERS = {
    "attach_memories_to_relationships": "graph_boundary.memory_properties",
}

# Writer functions that change the graph only through GraphLifecycle. Each must
# still delegate (a direct write would bypass the lifecycle tests below).
DELEGATORS = {"soft_delete_memory_from_edges", "update_memory_on_edges"}


# --- Graph lifecycle (releasing memories, closing and erasing records) -------
#
# These writes never create a record or change who owns it; the boundary they
# must carry is the NAMESPACE: every one is scoped to one namespace, which the
# service derives from the scope (scope_key.write_scope_id / readable_scope_ids,
# graph_lifecycle.release_namespaces) or is handed by its caller, and every
# Cypher statement of the adapter primitive matches `_namespace: $ns`.
_LIFECYCLE = _ROOT / "graph_lifecycle.py"
_ADAPTER = pathlib.Path(__file__).resolve().parents[2] / "providers" / "graph" / "adapters" / "neo4j.py"
LIFECYCLE_WRITE_PRIMITIVES = {"release_memory_ids", "set_record_states", "erase_records",
                              "rewrite_memory_contents"}


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


def test_the_legacy_helpers_delegate_to_the_lifecycle_service():
    writers = _writers()
    for fn in _functions():
        if fn.name in DELEGATORS:
            assert fn.name not in writers, f"{fn.name} writes the graph directly again"
            assert _mentions(fn, "GraphLifecycle"), f"{fn.name} no longer goes through GraphLifecycle"


def _lifecycle_calls():
    """(function name, primitive, namespace argument) for every primitive the
    lifecycle service calls through `self._call("<primitive>", ns, ...)`."""
    tree = ast.parse(_LIFECYCLE.read_text())
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_call" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in LIFECYCLE_WRITE_PRIMITIVES):
                yield fn.name, node.args[0].value, (node.args[1] if len(node.args) > 1 else None)


def test_every_lifecycle_write_names_its_namespace():
    calls = list(_lifecycle_calls())
    assert {p for _, p, _ in calls} == LIFECYCLE_WRITE_PRIMITIVES, (
        "the lifecycle service no longer calls every listed write primitive, or calls a new one: "
        "classify it here")
    for name, primitive, ns in calls:
        assert isinstance(ns, ast.Name) and ns.id == "ns", (
            f"{name} calls {primitive} without a namespace taken from the scope (`ns`)")


def test_the_lifecycle_namespaces_come_from_the_scope():
    text = _LIFECYCLE.read_text()
    tree = ast.parse(text)
    fns = {f.name: ast.get_source_segment(text, f) for f in ast.walk(tree)
           if isinstance(f, (ast.AsyncFunctionDef, ast.FunctionDef))}
    assert "scope_key.write_scope_id" in fns["release_namespaces"]
    assert "scope_key" in fns["erase_record"]


def test_every_lifecycle_primitive_matches_on_the_namespace():
    """Each Cypher statement the primitives run is bounded by `_namespace: $ns`."""
    import importlib
    adapter = importlib.import_module("synap.providers.graph.adapters.neo4j")
    text = _ADAPTER.read_text()
    tree = ast.parse(text)
    # Module-level Cypher, as the module computes it (some are built with format()).
    consts = {k: v for k, v in vars(adapter).items() if isinstance(v, str)}
    seen = set()
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name in LIFECYCLE_WRITE_PRIMITIVES):
            continue
        seen.add(fn.name)
        statements = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id in consts and "MATCH" in consts[node.id]:
                statements.append((node.id, consts[node.id]))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "MATCH" in node.value:
                statements.append(("inline", node.value))
        assert statements, f"{fn.name}: no Cypher found to check"
        for label, cypher in statements:
            assert "_namespace: $ns" in cypher, f"{fn.name} runs {label} without `_namespace: $ns`"
    assert seen == LIFECYCLE_WRITE_PRIMITIVES, f"adapter primitives missing: {LIFECYCLE_WRITE_PRIMITIVES - seen}"


def test_nothing_outside_the_service_calls_the_lifecycle_primitives():
    """A second caller would be a second place to get the namespace wrong."""
    import re
    cloud = pathlib.Path(__file__).resolve().parents[2] / "cloud"
    pattern = re.compile(r"\.(?:" + "|".join(sorted(LIFECYCLE_WRITE_PRIMITIVES)) + r")\(")
    allowed = {_LIFECYCLE, _WRITER}
    offenders = []
    for path in cloud.rglob("*.py"):
        if path in allowed or "tests" in path.parts:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(cloud.parent)}:{number}")
    assert not offenders, f"lifecycle primitives called outside GraphLifecycle: {offenders}"
