"""A forgotten memory must be hidden by every read that hides retired ones.

Soft delete only works if each reader of ``memories`` honours ``deleted_at``. The
readers that decide "is this memory live" all spell it the same way, so this finds
every spelling of "not superseded and not deprecated" and checks the same predicate
list also says ``deleted_at IS NULL``. A new reader added later in the old shape
fails here instead of quietly serving a forgotten memory.
"""

import re
from pathlib import Path

CLOUD = Path(__file__).resolve().parents[3] / "cloud"

# "superseded_by IS NULL" then "deprecated_at IS NULL", with whatever separates two
# adjacent conditions: spaces, a newline, or the quote that ends one Python string
# literal and opens the next.
_LIVE = re.compile(r"superseded_by IS NULL[\s\"']*AND[\s\"']*(?:\w+\.)?deprecated_at IS NULL")
_DELETED = re.compile(r"deleted_at IS NULL")
_REACH = 160    # characters after the match in which deleted_at must appear

# Files that are allowed to say "live" without checking deleted_at, and why.
ALLOWED = {
    # The forget service's own SELECT already carries it; this is its definition.
    "application/memory_forget/service.py": "carries deleted_at itself",
}


def _sources():
    for path in CLOUD.rglob("*.py"):
        rel = path.relative_to(CLOUD).as_posix()
        if rel in ALLOWED or "/tests/" in rel:
            continue
        yield rel, path.read_text(encoding="utf-8", errors="ignore")


def test_every_live_row_predicate_also_excludes_forgotten_memories():
    missing = []
    for rel, src in _sources():
        for m in _LIVE.finditer(src):
            tail = src[m.start(): m.end() + _REACH]
            if not _DELETED.search(tail):
                line = src.count("\n", 0, m.start()) + 1
                missing.append(f"{rel}:{line}")
    assert not missing, (
        "These readers decide a memory is live without checking deleted_at, so a "
        "forgotten memory would still be served:\n  " + "\n  ".join(missing))


def test_the_guard_actually_finds_the_known_readers():
    """If the pattern stopped matching, the test above would pass on nothing."""
    hits = sum(len(_LIVE.findall(src)) for _, src in _sources())
    # 15 in the monorepo; 14 in Core, where one consolidation file does not ship. The minimum
    # that holds in both: the point is that the pattern still matches, not the exact count.
    assert hits >= 14, f"expected the known live-row predicates, found {hits}"


def test_delete_by_id_is_untouched():
    """Delete by ID stays a hard delete: no deleted_at anywhere in that handler."""
    src = (CLOUD / "application/sdk_api/sdk_memories.py").read_text()
    start = src.index("async def delete_memory(")
    handler = src[start:]
    assert "DELETE FROM memories" in handler
    # The response model has a `deleted_at` timestamp field; the SQL must not.
    assert "deleted_at IS" not in handler and "SET deleted_at" not in handler


def test_reversal_and_the_integrity_sweep_never_touch_deleted_at():
    for rel in ("application/consolidation/reversal.py", "application/consolidation/sleep.py"):
        src = (CLOUD / rel).read_text()
        assert not re.search(r"SET[^;\"]*\bdeleted_at\b", src), rel
