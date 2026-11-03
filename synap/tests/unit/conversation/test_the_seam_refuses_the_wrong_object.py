"""The tracker refuses anything that is not the client's path dict.

WHY THIS FILE EXISTS

On 2026-08-31 a commit made the one-shot transcript route store the rung on
the conversation row, which was the right intent. It handed the row writer the
wrong object: the resolved `Scope` model built for the ingestion pipeline
rather than `request.scope`, the plain dict the client sent and the column
holds. The two are one attribute apart at the call site.

`ensure_conversation` types the parameter `Optional[Dict[str, str]]` and
nothing enforced that at runtime, so the first thing to notice was
`json.dumps`, which raised `TypeError: Object of type Scope is not JSON
serializable` on the FIRST write of the request. Every transcript ingest
returned 500, for every client, for about 24 hours, and stored nothing:
131 failed pushes, 130 of them one client's, across 98 of their leads.

WHY THE EXISTING TESTS DID NOT CATCH IT

Three tests each covered half the seam and none covered the join. The route
tests replaced the tracker with a mock, so nothing was ever serialised. The
tracker tests passed a dict directly, so the wrong type never appeared. And
the scoping guard asserted the source text `scope_path=scope,` was PRESENT,
which meant it required the bug to exist and would have gone red on the fix.

WHAT THIS PINS

The boundary itself, so the mistake is caught at the first call by any test
that uses the real tracker, with an error that names the mistake instead of
naming the serializer.
"""
from __future__ import annotations

import json

import pytest

from synap.cloud.application.conversation.tracker import _as_scope_path


class TestTheEncoderTakesOnlyAPathDict:
    def test_a_path_dict_is_encoded(self):
        out = _as_scope_path({"tenant": "acme", "practice": "high_street"})
        assert json.loads(out) == {"tenant": "acme", "practice": "high_street"}

    def test_none_stays_none(self):
        assert _as_scope_path(None) is None

    def test_empty_dict_is_none_not_an_empty_object(self):
        """`{}` must not become the string "{}".

        The read side treats a non-null column as "this conversation names a
        rung". An empty object would be a path that names nothing, which is a
        third state nobody handles.
        """
        assert _as_scope_path({}) is None

    def test_the_scope_model_is_refused_by_name(self):
        """The exact object that caused the outage."""
        pydantic = pytest.importorskip("pydantic")

        class Scope(pydantic.BaseModel):
            client_id: str = "c"
            instance_id: str = "i"

        with pytest.raises(TypeError) as exc:
            _as_scope_path(Scope())
        msg = str(exc.value)
        # The message has to point at the fix, not at the serializer.
        assert "request.scope" in msg
        assert "Scope" in msg

    @pytest.mark.parametrize("bad", ["a string", 42, ["a", "list"], object()])
    def test_anything_else_is_refused_too(self, bad):
        """Not just pydantic models. A string is the other plausible slip:
        somebody passes the rung's external id instead of the path."""
        with pytest.raises(TypeError):
            _as_scope_path(bad)


class TestTheTrackerUsesIt:
    @pytest.mark.asyncio
    async def test_ensure_conversation_refuses_a_model(self):
        """End of the seam: through the real method, not the helper alone.

        A helper can be correct and unused, which is how the original guard
        managed to be green while the route was broken.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        pydantic = pytest.importorskip("pydantic")

        class Scope(pydantic.BaseModel):
            client_id: str = "c"

        from synap.cloud.application.conversation.tracker import ConversationTracker

        session = AsyncMock()
        session.execute = AsyncMock()
        factory = MagicMock()
        factory.return_value.__aenter__ = AsyncMock(return_value=session)
        factory.return_value.__aexit__ = AsyncMock(return_value=False)

        tracker = ConversationTracker()
        with patch(
            "synap.cloud.application.conversation.tracker.get_session_factory",
            return_value=factory,
        ):
            with pytest.raises(TypeError) as exc:
                await tracker.ensure_conversation(
                    conversation_id="11111111-1111-1111-1111-111111111111",
                    instance_id="inst-1",
                    user_id="u1",
                    customer_id="c1",
                    scope_path=Scope(),
                )

        # ⚠ Asserting only `pytest.raises(TypeError)` here proves NOTHING, and
        # my own mutation run caught that: a raw `json.dumps(Scope())` raises
        # TypeError too, so the test stayed green with the guard removed. That
        # is the same shape of hole this whole file was written about. The
        # message is what distinguishes the guard from the bug, so the message
        # is what gets asserted.
        assert "request.scope" in str(exc.value), (
            "the tracker is not going through _as_scope_path: this is the raw "
            "json.dumps TypeError, which names the serializer and not the "
            "mistake, and is exactly what production showed for 24 hours"
        )
