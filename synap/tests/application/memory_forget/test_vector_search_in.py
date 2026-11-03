"""search_in: a collection nothing was written to is empty, not a failure."""

import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.agents.ingestion.storage.vector_wrapper import AsyncVectorWrapper

SCOPE = SimpleNamespace(client_id=uuid.uuid4(), customer_id=uuid.uuid4(),
                        user_id=uuid.uuid4(), instance_id=uuid.uuid4(), scope_level="user")


class Provider:
    def __init__(self, error=None):
        self.error = error

    def search(self, query):
        if self.error:
            raise self.error
        return SimpleNamespace(results=[SimpleNamespace(id="a", score=0.9, metadata={}, content="x")])


async def test_a_collection_that_does_not_exist_is_empty():
    w = AsyncVectorWrapper(Provider(RuntimeError("Collection [c_sentences_v2] does not exists")))
    assert await w.search_in([0.1], SCOPE, "sentences", 10, None) == []


async def test_any_other_error_still_propagates():
    w = AsyncVectorWrapper(Provider(RuntimeError("chroma down")))
    with pytest.raises(RuntimeError):
        await w.search_in([0.1], SCOPE, "sentences", 10, None)


async def test_hits_are_returned_as_before():
    w = AsyncVectorWrapper(Provider())
    out = await w.search_in([0.1], SCOPE, "sentences", 10, None)
    assert out[0]["id"] == "a" and out[0]["content"] == "x"
