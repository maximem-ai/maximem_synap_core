"""The flag that stops the applier's own vector writes being shipped back."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from synap.providers.vector.replication_context import (
    applying_replication,
    is_replication_apply,
)


def test_off_by_default():
    """Ingestion must be logged. Only the applier opts out."""
    assert is_replication_apply() is False


def test_on_inside_the_block_and_off_after():
    with applying_replication():
        assert is_replication_apply() is True
    assert is_replication_apply() is False


def test_restored_even_if_the_write_raises():
    """A failed apply must not leave logging disabled for everything after it."""
    try:
        with applying_replication():
            raise RuntimeError("chroma said no")
    except RuntimeError:
        pass
    assert is_replication_apply() is False


def test_it_does_not_leak_into_a_concurrent_task():
    """The whole reason this is a ContextVar and not a global or an attribute.

    The adapter is shared. An ingestion write happening at the same moment as
    an apply must still be logged, or its vector never reaches the other
    region and nothing anywhere records that it was supposed to.
    """
    seen = {}

    async def ingestion():
        await asyncio.sleep(0)          # yield, so it interleaves mid-apply
        seen["ingestion"] = is_replication_apply()

    async def applier():
        with applying_replication():
            seen["inside"] = is_replication_apply()
            await asyncio.sleep(0.01)   # hold the flag across an await
            seen["still_inside"] = is_replication_apply()

    async def main():
        await asyncio.gather(applier(), ingestion())

    asyncio.run(main())

    assert seen["inside"] is True
    assert seen["still_inside"] is True, "the flag must survive an await"
    assert seen["ingestion"] is False, (
        "the applier's flag leaked into a concurrent ingestion write; that "
        "write would silently never be replicated"
    )


def test_the_adapter_wrapper_actually_consults_it():
    """Source-level, because importing the wrapper drags in the whole
    ingestion stack. Weak, but it catches the guard being deleted, which is
    the failure that turns into an infinite vector loop between regions."""
    src = Path("synap/cloud/agents/ingestion/storage/vector_wrapper.py").read_text()
    m = re.search(r"if write_args is not None[^\n]*:", src)
    assert m, "the write-log hook is gone from _run_chroma"
    assert "is_replication_apply()" in m.group(0), (
        "_run_chroma logs every write regardless of who made it; the applier's "
        "own writes will be shipped back to the region they came from"
    )


def test_the_leaf_module_stays_dependency_free():
    """Both the applier and the ingestion stack import this.

    If it grows an import, the replication process inherits whatever that
    pulls in, which is how a langgraph import error becomes a replication
    outage.
    """
    src = Path("synap/providers/vector/replication_context.py").read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", src, re.M)
    allowed = {"__future__", "contextlib", "contextvars"}
    assert set(imports) <= allowed, "replication_context grew a dependency: %s" % (
        set(imports) - allowed
    )
