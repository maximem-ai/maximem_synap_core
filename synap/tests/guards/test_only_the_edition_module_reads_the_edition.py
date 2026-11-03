"""Only ``shared/edition.py`` may name ``SYNAP_EDITION``.

The edition is one decision made in one place. A second reader
would be an ``if edition == ...`` check by another route, and the sync would
then ship a Cloud-versus-Core difference nobody can find by reading the module.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ALLOWED = ROOT / "synap" / "cloud" / "shared" / "edition.py"
NAME = "SYNAP_EDITION"


def _is_test(path: Path) -> bool:
    return "tests" in path.parts or path.name.startswith("test_") or path.name == "conftest.py"


def test_only_the_edition_module_names_the_edition_variable():
    offenders = []
    for path in (ROOT / "synap").rglob("*.py"):
        if "__pycache__" in path.parts or _is_test(path) or path == ALLOWED:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and NAME in node.value:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, (
        f"{NAME} is named at {offenders}. Ask synap.cloud.shared.edition for the "
        "value you need instead of reading the variable."
    )


def test_the_edition_module_does_name_it():
    tree = ast.parse(ALLOWED.read_text(encoding="utf-8"))
    assert any(
        isinstance(n, ast.Constant) and n.value == NAME for n in ast.walk(tree)
    ), "the premise: the module reads the variable, or the guard above guards nothing"
