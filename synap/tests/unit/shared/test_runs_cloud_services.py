"""The edition value ``runs_cloud_services`` and the start-up blocks it gates in main.py."""
from __future__ import annotations

import ast
from pathlib import Path

from synap.cloud.shared import edition as ed

ROOT = Path(__file__).resolve().parents[3].parent
MAIN = ROOT / "synap/cloud/application/main.py"

# Packages Core does not ship; main.py must not import them unless a plain edition value allows it.
CLOUD_ONLY = (
    "synap.cloud.application.credits",
    "synap.cloud.application.notifications",
    "synap.cloud.application.lifecycle",
    "synap.cloud.application.admin_api",
    "synap.cloud.application.migration",
    "synap.cloud.application.playground",
    "synap.cloud.agents.connectors",
    "synap.cloud.application.grpc",
)
GUARDS = {"edition.runs_cloud_services", "edition.serves_grpc"}


def test_only_cloud_runs_cloud_services():
    assert ed.resolve({}).runs_cloud_services is True
    assert ed.resolve({"SYNAP_EDITION": "core"}).runs_cloud_services is False
    assert isinstance(ed.runs_cloud_services, bool)


def _unguarded_imports(tree: ast.AST) -> list[str]:
    found: list[str] = []

    def walk(node: ast.AST, guarded: bool) -> None:
        for child in ast.iter_child_nodes(node):
            g = guarded
            if isinstance(child, ast.If) and ast.unparse(child.test) in GUARDS:
                # Only the body is guarded; an else branch runs when the value is false.
                for stmt in child.body:
                    walk_stmt(stmt, True)
                for stmt in child.orelse:
                    walk_stmt(stmt, guarded)
                continue
            if isinstance(child, ast.ImportFrom) and not g and (child.module or "").startswith(CLOUD_ONLY):
                found.append(f"line {child.lineno}: {child.module}")
            walk(child, g)

    def walk_stmt(stmt: ast.AST, guarded: bool) -> None:
        wrapper = ast.Module(body=[stmt], type_ignores=[])
        walk(wrapper, guarded)

    walk(tree, False)
    return found


def test_main_does_not_import_cloud_only_packages_outside_a_guard():
    tree = ast.parse(MAIN.read_text())
    assert _unguarded_imports(tree) == []
