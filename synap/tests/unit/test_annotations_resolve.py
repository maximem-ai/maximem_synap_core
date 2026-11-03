"""Annotations have to name something that exists, on the Python we deploy.

This suite runs on 3.14, where annotations are evaluated lazily: writing
`Optional[str]` in a signature without importing `Optional` is fine, because
nothing ever looks the name up. The container runs 3.11, where annotations on
a `def` are evaluated as the module is imported, and the same code raises

    NameError: name 'Optional' is not defined

at import time. Not in a request, not under load: the application does not
start. That is exactly what happened, and every test here passed first.

So this reproduces the strict rule on the lenient interpreter. A module that
opts into `from __future__ import annotations` is lazy on 3.11 too and is
skipped, because for those the leniency is deliberate rather than accidental.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path
from typing import List, Set, Tuple

import pytest

_ROOT = Path(__file__).resolve().parents[2] / "cloud"


def _module_files() -> List[Path]:
    return [
        p
        for p in sorted(_ROOT.rglob("*.py"))
        if "__pycache__" not in p.parts and "ui" not in p.parts
    ]


def _is_lazy(tree: ast.Module) -> bool:
    """True if the module opted into deferred annotations."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(a.name == "annotations" for a in node.names):
                return True
    return False


def _bound_names(tree: ast.Module) -> Set[str]:
    """Everything the module namespace holds by the time it finishes importing.

    Deliberately generous: anything bound anywhere in the module counts, even
    inside a function, because the point is to catch the name that is bound
    *nowhere*, not to model scope precisely. A false alarm here would be a
    test nobody trusts.
    """
    names: Set[str] = set(dir(builtins))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)

    return names


def _annotation_names(node: ast.AST) -> Set[str]:
    """Every bare name an annotation expression would look up."""
    out: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            out.add(child.id)
        elif isinstance(child, ast.Attribute):
            # `t.Optional[str]` looks up `t` only.
            base = child
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                out.add(base.id)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            # A string annotation is not evaluated at def time on any version.
            continue
    return out


def _unresolved(path: Path) -> List[Tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if _is_lazy(tree):
        return []

    bound = _bound_names(tree)
    problems: List[Tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        annotations = []
        args = node.args
        for arg in args.args + args.kwonlyargs + args.posonlyargs:
            if arg.annotation is not None:
                annotations.append(arg.annotation)
        for extra in (args.vararg, args.kwarg):
            if extra is not None and extra.annotation is not None:
                annotations.append(extra.annotation)
        if node.returns is not None:
            annotations.append(node.returns)

        for annotation in annotations:
            for name in _annotation_names(annotation):
                if name not in bound:
                    problems.append((annotation.lineno, name))

    return problems


def test_every_annotation_names_something_that_exists():
    offenders = []
    for path in _module_files():
        for lineno, name in _unresolved(path):
            rel = path.relative_to(_ROOT.parent.parent)
            offenders.append(f"{rel}:{lineno} annotation uses undefined {name!r}")

    assert offenders == [], (
        "these import fine on 3.14 and raise NameError on the 3.11 in the "
        "container, which means the application does not start:\n"
        + "\n".join(sorted(offenders))
    )


def test_the_scan_reads_the_modules_it_should():
    """The control. An empty file list passes the assertion above silently."""
    files = _module_files()
    assert len(files) > 100, f"only found {len(files)} modules"
    assert any(p.name == "_helpers.py" for p in files)


def test_it_catches_the_bug_it_was_written_for(tmp_path):
    """The exact shape that once took a deployment down, and the fix, side by side."""
    broken = tmp_path / "broken.py"
    broken.write_text(
        "from typing import Set\n\n"
        "async def f(x: Optional[str]) -> None:\n    return None\n"
    )
    assert [n for _, n in _unresolved(broken)] == ["Optional"]

    fixed = tmp_path / "fixed.py"
    fixed.write_text(
        "from typing import Optional, Set\n\n"
        "async def f(x: Optional[str]) -> None:\n    return None\n"
    )
    assert _unresolved(fixed) == []


def test_a_future_import_is_respected(tmp_path):
    """Deferred annotations are lazy on 3.11 too, so the same code is fine."""
    lazy = tmp_path / "lazy.py"
    lazy.write_text(
        "from __future__ import annotations\n\n"
        "async def f(x: Optional[str]) -> None:\n    return None\n"
    )
    assert _unresolved(lazy) == []
