"""Forget for a person who never stored anything answers "nothing matches", not a 500."""
import pytest

from synap.cloud.application.memory_forget.service import search_or_nothing


class _Vector:
    def __init__(self, error=None, hits=None):
        self.error, self.hits, self.calls = error, hits or [], []

    async def search(self, embedding, scope, top_k=10, filters=None):
        self.calls.append(filters)
        if self.error:
            raise self.error
        return self.hits


@pytest.mark.asyncio
async def test_a_user_with_no_collection_has_no_hits():
    v = _Vector(error=RuntimeError("Collection [synap_695f1c8f_v2] does not exist"))
    assert await search_or_nothing(v, [0.1], object(), user_id="u1") == []
    assert v.calls == [{"user_id": "u1"}]


@pytest.mark.asyncio
async def test_hits_pass_through():
    v = _Vector(hits=[{"id": "m1"}])
    assert await search_or_nothing(v, [0.1], object(), user_id="u1") == [{"id": "m1"}]


@pytest.mark.asyncio
async def test_any_other_error_still_raises():
    v = _Vector(error=RuntimeError("connection refused"))
    with pytest.raises(RuntimeError, match="connection refused"):
        await search_or_nothing(v, [0.1], object(), user_id="u1")
