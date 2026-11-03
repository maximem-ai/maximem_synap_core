"""A user's first fetch, before any memory exists, is an empty answer and not a warning with a traceback.

Found at the final Docker test: seven "Failed to fetch from ... collection ... does not exist" warnings,
each with a stack, for new users. The collection is created with the first memory.
"""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from synap.cloud.application.client_api.retrieval_manager import RetrievalManager


class NotFoundError(Exception):
    pass


def _manager(error):
    m = object.__new__(RetrievalManager)
    m.vector = MagicMock()
    m.vector.get_all_from_collection = AsyncMock(side_effect=error)
    return m


async def test_a_collection_that_does_not_exist_yet_is_empty_and_quiet(caplog):
    caplog.set_level(logging.INFO)
    m = _manager(NotFoundError("Collection [synap_x_v2] does not exist."))
    out = await m._fetch_all_from_collection(SimpleNamespace(client_id="c"), "memories")
    assert out == []
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("No memories collection yet" in r.getMessage() for r in caplog.records)


async def test_any_other_failure_is_still_a_warning_with_its_stack(caplog):
    caplog.set_level(logging.INFO)
    m = _manager(RuntimeError("connection reset"))
    assert await m._fetch_all_from_collection(SimpleNamespace(client_id="c"), "memories") == []
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and warned[0].exc_info is not None
