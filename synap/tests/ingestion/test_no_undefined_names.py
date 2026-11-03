"""Every name a function reads must exist somewhere it can be found.

On 3 Oct 2026 every Stage 4 run on the LongMemEval instance failed with
``NameError: name 'correlation_id' is not defined``: the open-anchor block read a
variable organize_and_persist never had, and its own ``except`` read it again, so
the "never fails a persist" guard raised from inside its handler. 44 ingestions
went to the dead-letter queue before anyone looked. The unit tests passed
because they call the resolver directly and never reach that block.

This reads the source with ``symtable`` rather than importing it, so it needs no
database, no model and no extra dependency, and fails on the line that would
raise at run time.
"""
from __future__ import annotations

import ast
import builtins
import symtable
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

FILES = [
    "cloud/agents/ingestion/pipeline/stage4_organization.py",
    "cloud/agents/ingestion/pipeline/stage3_extraction.py",
    "cloud/agents/ingestion/services/anchor_resolver.py",
    "cloud/agents/ingestion/services/date_prepass.py",
    "cloud/agents/ingestion/services/prior_context.py",
    "cloud/application/conversation/tracker.py",
]


def _module_names(tree: ast.Module) -> set:
    """Names bound anywhere at module level, including inside if/try blocks."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


def _unresolved(table: symtable.SymbolTable, known: set, out: list) -> None:
    if table.get_type() == "function":
        for sym in table.get_symbols():
            if (sym.is_referenced() and sym.is_global() and not sym.is_declared_global()
                    and sym.get_name() not in known):
                out.append(f"{table.get_name()}: {sym.get_name()}")
    for child in table.get_children():
        _unresolved(child, known, out)


@pytest.mark.parametrize("rel", FILES)
def test_every_name_read_in_a_function_is_defined(rel):
    path = _ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not on this branch")
    src = path.read_text()
    known = _module_names(ast.parse(src)) | set(dir(builtins)) | {"__file__", "__name__"}
    missing: list = []
    _unresolved(symtable.symtable(src, str(path), "exec"), known, missing)
    assert not missing, f"{rel} reads names nothing defines: {missing}"
