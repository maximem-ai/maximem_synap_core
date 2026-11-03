"""A function-level import of a name already imported at module level makes
that name local for the WHOLE function, including lines above the import.

Found in a running deployment. `main.lifespan` imports
`get_session_factory` at module level (line 19) and again inside the function
about 180 lines further down. Python binds the name as a local for the entire
function body, so the reference near the top raised:

    UnboundLocalError: cannot access local variable 'get_session_factory'
    where it is not associated with a value

and entity resolution was disabled on every boot, in every environment. The
handler logged "synap-er initialization failed (tables may not exist yet)",
which is a plausible, specific and entirely wrong explanation, so anyone who
investigated went looking at the database.

This is the shape worth guarding, not the one instance: it is silent, it
survives every test that does not boot the app, and the error message points
away from the cause.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "synap" / "cloud"


def _shadowing_imports(tree: ast.Module) -> list[tuple[str, str, int]]:
    """Names imported at module level and re-imported inside a function."""
    module_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                module_level.add(a.asname or a.name.split(".")[0])

    problems: list[tuple[str, str, int]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for a in node.names:
                name = a.asname or a.name.split(".")[0]
                if name not in module_level:
                    continue
                # Only a problem when the name is also READ somewhere in the
                # function before the import line. A re-import below its only
                # use is harmless, and flagging it would make the guard noisy
                # enough to be switched off.
                used_earlier = any(
                    isinstance(u, ast.Name)
                    and u.id == name
                    and isinstance(u.ctx, ast.Load)
                    and u.lineno < node.lineno
                    for u in ast.walk(fn)
                )
                if used_earlier:
                    problems.append((fn.name, name, node.lineno))
    return problems


def _python_files():
    for p in sorted(SRC.rglob("*.py")):
        if "/tests/" in str(p):
            continue
        yield p


def test_no_function_reimports_a_module_level_name_it_already_used():
    found: list[str] = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for fn_name, name, lineno in _shadowing_imports(tree):
            found.append(
                f"{path.relative_to(ROOT)}:{lineno} function {fn_name}() re-imports "
                f"{name!r}, which it already uses further up. Python makes the name "
                f"local for the whole function, so the earlier use raises "
                f"UnboundLocalError at runtime."
            )
    assert not found, "shadowed module-level imports:\n  " + "\n  ".join(found)


def test_the_guard_catches_the_bug_it_was_written_for():
    """Mutation check. A guard that cannot fail is not a guard."""
    src = (
        "from db import get_session_factory\n"
        "def lifespan():\n"
        "    f = get_session_factory()\n"
        "    from db import get_session_factory\n"
        "    return f\n"
    )
    assert _shadowing_imports(ast.parse(src)) == [("lifespan", "get_session_factory", 4)]


def test_the_guard_does_not_flag_a_reimport_below_its_only_use():
    """A local import of a name never read earlier in the function is legal and
    common (deferred imports for startup cost), and must not be flagged."""
    src = (
        "from db import thing\n"
        "def f():\n"
        "    from db import thing\n"
        "    return thing()\n"
    )
    assert _shadowing_imports(ast.parse(src)) == []
