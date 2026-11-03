"""Two rules the SHIP-12 walk found were defended only by their own source.

S-12 and G-14 were both implemented, correctly as far as reading them goes, and
neither was named by anything that ran. That is the state the walk exists to
turn up: a rule whose only evidence is the code the rule is about.
"""
from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]

# What a copy must carry. The four ids are obvious. These two are not, and both
# omissions are silent: `scope_level` picks the namespace, so dropping it sends
# a client-level write to `synap_None_v2`, a namespace this repo has produced by
# accident before; `chain` picks the node, so dropping it files a customer's
# shared turn as one person's and makes the record invisible to nested reads.
MUST_COPY = ("scope_level", "chain")


def _scope_copies(tree: ast.AST):
    """Every `Scope(...)` whose arguments read from another Scope.

    An AST walk rather than a text search, because the thing being asserted is
    about which keywords are present in a call, and a regex over source cannot
    tell a keyword from the same word in a comment. It is also why this is a
    guard on call sites rather than a behavioural test: there is one copy site
    today and the risk is the second one, written later, by someone who copies
    the four ids because those are the ones they were thinking about.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "Scope":
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        reads_a_scope = any(
            isinstance(v, ast.Attribute)
            and isinstance(v.value, ast.Name)
            and "scope" in v.value.id.lower()
            for v in kw.values()
        )
        if reads_a_scope:
            yield node, set(kw)


def test_s12_every_scope_built_from_another_scope_carries_level_and_chain():
    """S-12. A copy that omits a field is not a copy, and both omissions are
    silent: nothing raises, the write simply lands somewhere else."""
    offenders = []
    for path in (REPO / "synap" / "cloud").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node, keys in _scope_copies(tree):
            missing = [f for f in MUST_COPY if f not in keys]
            if missing:
                offenders.append(
                    f"{path.relative_to(REPO)}:{node.lineno} omits {missing}")
    assert not offenders, (
        "a Scope copied from another Scope must carry everything that decides "
        "where data lands:\n  " + "\n  ".join(offenders))


def test_s12_the_guard_is_looking_at_something():
    """A guard that finds no call sites passes for free, which is how a
    call-site test quietly stops testing anything after a refactor renames the
    thing it was looking for."""
    found = 0
    for path in (REPO / "synap" / "cloud").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        found += sum(1 for _ in _scope_copies(tree))
    assert found >= 1, "no Scope-from-Scope construction found; the guard is blind"


def test_g14_grants_and_instance_visibility_are_separate_axes():
    """G-14. They answer different questions and a client can want one without
    the other, so neither migration may reference the other's tables.

    Checked against the migrations rather than the ORM, because the coupling
    that would matter is a foreign key, and a foreign key is a thing the
    database has whether or not any model mentions it.
    """
    migrations = REPO / "synap" / "db" / "migrations"
    grants = (migrations / "098_scope_grants.sql").read_text().lower()
    visibility = (migrations / "069_cross_instance_visibility.sql").read_text().lower()

    visibility_tables = ("instance_visibility_policy", "instance_visibility_grant",
                         "instance_shareable_flag")
    for t in visibility_tables:
        assert t not in grants, f"the grants migration references {t}"
    for t in ("scope_grant", "scope_node", "scope_level"):
        assert t not in visibility, f"the visibility migration references {t}"
