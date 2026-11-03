"""The read half of the feature, against the object it is actually handed.

`_reveal_for_caller` read `bundle.items`. `ContextBundle` has no such field:
it groups by type in `items_by_type` and exposes `all_items`. So the getattr
returned None, the function returned before doing anything, and it had done
that for every caller since it was written. No `pii_reveal`
lines were ever logged.

The consequence is the one the page's warning badge exists to prevent, on the
setting that carries no badge. "Protect it" says, in as many words, that the
client's own app still receives the real value. Their app received
`[[DEVICE_IP_ADDRESS_3ckyatccgdgj]]`.

What hid it: the only tests of this wiring read the source of `fetch_context`
with `inspect.getsource` and asserted the call appears after the cache write.
It does appear there. Reading the source cannot tell you the function does
nothing when it arrives, so these call it instead.
"""

from __future__ import annotations

from synap.cloud.application.client_api.retrieval_manager import _revealable_items


def _bundle(**kwargs):
    from synap.cloud.agents.models.bundles import ContextBundle

    return ContextBundle(bundle_id="b", decision_id="d", **kwargs)


_seq = iter(range(1000))


def _item(content: str):
    from synap.cloud.agents.models.bundles import ContextItem

    return ContextItem(
        item_id=f"i{next(_seq)}",
        source="vector",
        content=content,
        context_type="facts",
    )


class TestTheRevealCanSeeTheItems:
    def test_the_real_bundle_shape_yields_its_items(self):
        bundle = _bundle(items_by_type={"facts": [_item("a"), _item("b")]})
        assert len(_revealable_items(bundle)) == 2

    def test_every_group_is_included_not_just_the_first(self):
        bundle = _bundle(items_by_type={
            "facts": [_item("a")],
            "preferences": [_item("b")],
            "episodes": [_item("c")],
        })
        assert len(_revealable_items(bundle)) == 3

    def test_the_field_it_used_to_read_does_not_exist(self):
        """The control, and the reason this file exists. If `items` ever
        becomes real, this test failing is the signal to simplify, not a
        reason to go back to reading it directly."""
        bundle = _bundle(items_by_type={"facts": [_item("a")]})
        assert getattr(bundle, "items", None) is None

    def test_a_flat_object_still_works(self):
        """A cached or reconstructed bundle may be a plain object with a list.
        Handling both is what stops this being a guess about a shape."""

        class _Flat:
            items = [_item("a")]

        assert len(_revealable_items(_Flat())) == 1

    def test_an_empty_bundle_is_not_an_error(self):
        assert _revealable_items(_bundle()) == []


class TestTheConversationContextIsRevealedToo:
    """It is carried beside the items rather than in them, and it is prose
    from the same conversation, so it holds the same labels. A bundle whose
    facts read correctly and whose recent turns still show placeholders is
    the same bug wearing a different field name."""

    def test_the_context_object_is_included(self):
        class _Ctx:
            content = "IP is [[DEVICE_IP_ADDRESS_x]]"

        bundle = _bundle(items_by_type={"facts": [_item("a")]})
        object.__setattr__(bundle, "conversation_context", _Ctx())
        assert len(_revealable_items(bundle)) == 2

    def test_recent_turns_inside_it_are_included(self):
        class _Ctx:
            content = ""
            recent_turns = [_item("turn one"), _item("turn two")]

        bundle = _bundle()
        object.__setattr__(bundle, "conversation_context", _Ctx())
        assert len(_revealable_items(bundle)) == 3
