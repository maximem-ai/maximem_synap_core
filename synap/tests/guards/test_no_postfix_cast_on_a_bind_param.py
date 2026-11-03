""":param::type is not a bind parameter, and Postgres says so at runtime.

Found in a running deployment, repeatedly within hours.

SQLAlchemy's `text()` finds bind parameters with a colon-name pattern, and it
does NOT recognise one immediately followed by `::`. So this:

    VALUES (:instance_id, :relationship_type, :embedding::vector, :source)

compiled to this, with three of the four bound and one left as literal text:

    VALUES ($1, $2, :embedding::vector, $3)

asyncpg then hands Postgres a statement containing a bare colon and the whole
INSERT dies with `syntax error at or near ":"`.

WHY IT MATTERED AND WHY NOBODY SAW IT

The statement registers a relationship type together with the embedding used to
tell one type name from another. Every one of those writes failed, so the
registry filled with types carrying no embedding and the semantic dedup of type
names had nothing to compare. It fails inside ingestion, where a persistence
error is logged and the ingest still reports success, so there was nothing to
notice from outside.

THE FIX IS CAST(:param AS type), which SQLAlchemy binds correctly.

This guard covers the CLASS. The same file had two more of the identical shape
in its similarity query, which had not fired yet only because that path runs
less often.
"""
from __future__ import annotations

import os
import re

# `:name` immediately followed by `::`. The name must not itself begin after a
# colon, or the `::vector` inside an already-correct cast would match.
POSTFIX_CAST = re.compile(r"(?<![:\w]):[a-zA-Z_]\w*::[a-zA-Z_]")

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "cloud")


def _offenders():
    hits = []
    for base, _, files in os.walk(ROOT):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(base, f)
            try:
                text = open(p, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                # A line that is only a comment is documentation, including the
                # note this fix left behind explaining the broken form.
                if stripped.startswith("#") or stripped.startswith("--"):
                    continue
                if POSTFIX_CAST.search(line):
                    rel = os.path.relpath(p, os.path.join(ROOT, "..", ".."))
                    hits.append(f"{rel}:{i}: {stripped[:90]}")
    return hits


def test_no_bind_parameter_carries_a_postfix_cast():
    offenders = _offenders()
    assert not offenders, (
        "a bind parameter is written `:name::type`. SQLAlchemy will not bind "
        "it, the colon reaches Postgres, and the statement fails at runtime "
        "with `syntax error at or near \":\"`. Write CAST(:name AS type):\n  "
        + "\n  ".join(offenders)
    )


def test_the_pattern_actually_recognises_the_bug_it_was_written_for():
    """Positive control, in the same run.

    A guard that finds nothing and a guard that looks for nothing produce the
    same output. This asserts the pattern still matches the exact line that was
    failing at runtime, so a clean result above means something.
    """
    broken = "(:instance_id, :relationship_type, :embedding::vector, :source)"
    assert POSTFIX_CAST.search(broken), "the pattern no longer finds the original defect"

    fixed = "(:instance_id, :relationship_type, CAST(:embedding AS vector), :source)"
    assert not POSTFIX_CAST.search(fixed), "the pattern flags the correct form"

    # And it must not fire on a cast with no bind parameter in front of it,
    # which is ordinary SQL and appears throughout this codebase.
    assert not POSTFIX_CAST.search("WHERE id = ANY($1::uuid[])")
    assert not POSTFIX_CAST.search("SELECT scope_path::text FROM scope_node")
