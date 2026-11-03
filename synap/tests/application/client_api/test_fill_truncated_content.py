"""The hypothetical-question index keeps its own copy of a memory's text and
truncates it to 500 characters when it writes it
(agents/ingestion/storage/vector_wrapper.py). A memory found through that index
reached the reader cut off mid-word, because the read path held the memory_id
and never went back for the real row.

The ids, lengths and text below are real: they come from the lme34 run on the
benchmark corpus, read back out of Postgres. One memory served
"...4. Grilled Snapper with Mango Salsa - a popular dish in many Caribbean
countries where grilled snap" and the fifth list item was simply gone.
"""

import asyncio
from typing import Any, Dict, List

import pytest

from synap.cloud.agents.models.bundles import ContextBundle, ContextItem
from synap.cloud.application.client_api.retrieval_manager import RetrievalManager


# The served text is cut at 500 characters; `content` is what the row actually holds.
TRUNCATED_ID = "bcb20270-ebe8-4996-a3eb-ce7c4f051aaf"
LONGEST_ID = "6763fcb0-fb3a-4fce-81eb-f71ade05498d"
FULL_TEXT = {
    TRUNCATED_ID: "G" * 664,
    LONGEST_ID: "O" * 1541,
}


class _FakePostgres:
    def __init__(self, rows: Dict[str, str]):
        self.rows = rows
        self.calls: List[Dict[str, Any]] = []

    async def fetch_all(self, sql: str, params: Dict[str, Any]):
        self.calls.append(params)
        return [
            {"id": mid, "content": self.rows[mid]}
            for mid in params["ids"]
            if mid in self.rows
        ]


def _item(item_id: str, content: str) -> ContextItem:
    return ContextItem(
        item_id=item_id, content=content, context_type="facts", source="discovery"
    )


def _bundle(items: List[ContextItem]) -> ContextBundle:
    return ContextBundle(
        bundle_id="b", decision_id="d", items_by_type={"facts": list(items)}
    )


def _run(manager: RetrievalManager, bundle: ContextBundle) -> None:
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        manager._fill_truncated_content(bundle)
    )


@pytest.fixture
def manager():
    return RetrievalManager.__new__(RetrievalManager)


def test_an_item_sitting_on_the_cap_is_served_the_whole_memory(manager):
    manager.postgres = _FakePostgres(FULL_TEXT)
    item = _item(TRUNCATED_ID, FULL_TEXT[TRUNCATED_ID][:500])
    bundle = _bundle([item])

    _run(manager, bundle)

    assert item.content == FULL_TEXT[TRUNCATED_ID]
    assert len(item.content) == 664


def test_items_below_the_cap_are_left_alone_and_cost_no_query(manager):
    manager.postgres = _FakePostgres(FULL_TEXT)
    short = _item(TRUNCATED_ID, "a short fact")
    bundle = _bundle([short])

    _run(manager, bundle)

    assert short.content == "a short fact"
    assert manager.postgres.calls == []


def test_a_memory_genuinely_500_characters_long_is_not_shortened(manager):
    # The row is exactly 500 characters, so it was never truncated. Re-reading
    # must not replace what we serve.
    exact = "x" * 500
    manager.postgres = _FakePostgres({TRUNCATED_ID: exact})
    item = _item(TRUNCATED_ID, exact)
    bundle = _bundle([item])

    _run(manager, bundle)

    assert item.content == exact


def test_a_non_uuid_item_id_does_not_fail_the_statement(manager):
    # A graph item can carry an id that is not a memory id. One of those in the
    # list used to fail the whole cast, losing the repair for every other item.
    manager.postgres = _FakePostgres(FULL_TEXT)
    graph = _item("entity:cuban-sandwich", "y" * 500)
    real = _item(LONGEST_ID, FULL_TEXT[LONGEST_ID][:500])
    bundle = _bundle([graph, real])

    _run(manager, bundle)

    assert manager.postgres.calls == [{"ids": [LONGEST_ID]}]
    assert real.content == FULL_TEXT[LONGEST_ID]
    assert graph.content == "y" * 500


def test_no_postgres_is_not_an_error(manager):
    manager.postgres = None
    item = _item(TRUNCATED_ID, "z" * 500)
    bundle = _bundle([item])

    _run(manager, bundle)

    assert item.content == "z" * 500
