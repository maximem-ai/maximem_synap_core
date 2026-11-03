"""R-14 must not rank the same shared fact two ways during the transition.

An audit finding, 2026-08-24. An unstamped record was scored at distance 0, the
requester's own node, on the reasoning that the legacy filter arm "is exactly the
requester's own customer and user scope". That holds for the `user_id = identity`
half and fails for the other half: a record carrying the CUSTOMER_SCOPE sentinel
is tenant-shared material one rung up.

So an identical shared record ranked ~30% differently depending only on whether
the backfill had reached it, in the same pool, on the same query:

    customer-shared, stamped    -> 0.56
    customer-shared, UNSTAMPED  -> 0.80

The direction was safe, shared material ranking too high is what happened before
the feature existed, so this is a precision fix rather than a leak. What it costs
is measurement: the sweep that moved the per-level cost to 0.30 ran on a mix
where many shared records were silently exempt from the term.
"""
from __future__ import annotations

import pytest

from synap.cloud.application.client_api.retrieval_manager import _closeness_scored
from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL

_CHAIN = ["n_client", "n_customer", "n_user"]
_RAW = 0.80
_COST = 0.30


def test_a_shared_record_ranks_the_same_stamped_or_not():
    """The finding itself. Both must land on the tenant rung's multiplier."""
    stamped = _closeness_scored(_RAW, {"node_id": "n_customer"}, _CHAIN, _COST)
    unstamped = _closeness_scored(
        _RAW, {"user_id": CUSTOMER_SCOPE_SENTINEL}, _CHAIN, _COST
    )
    assert stamped == pytest.approx(unstamped), (
        f"the same shared fact scores {stamped} stamped and {unstamped} unstamped, "
        f"so its rank depends on how far the backfill has got"
    )
    assert unstamped == pytest.approx(_RAW * (1 - _COST)), (
        "an unstamped shared record is not being scored one rung up"
    )


def test_an_unstamped_own_record_is_still_the_requesters_own():
    """R-14's original intent, which was right for this half and must survive.
    A record with a real user id and no node id is the requester's own."""
    v = _closeness_scored(_RAW, {"user_id": "a-real-user-uuid"}, _CHAIN, _COST)
    assert v == pytest.approx(_RAW), (
        "an unstamped own record is being demoted; R-14 exists to prevent that"
    )


def test_an_off_chain_record_is_still_dropped():
    """R-07 and R-08. Permission and ranking never blend, so a record the
    requester cannot read is not scored far away, it is refused."""
    assert _closeness_scored(_RAW, {"node_id": "n_someone_else"}, _CHAIN, _COST) is None


def test_a_collapsed_b2c_chain_does_not_go_negative():
    """In B2C the tenant IS the leaf, so 'one rung above the leaf' has nowhere
    to go. It must clamp to the requester's own node rather than compute a
    negative distance and raise."""
    two = ["n_client", "n_customer"]
    v = _closeness_scored(_RAW, {"user_id": CUSTOMER_SCOPE_SENTINEL}, two, _COST)
    assert v == pytest.approx(_RAW), (
        f"a collapsed B2C chain scored {v}; the tenant is the leaf there, so the "
        f"record is the requester's own"
    )


def test_a_single_rung_chain_is_safe():
    """Defensive: a one-rung chain has no tenant rung at all."""
    v = _closeness_scored(_RAW, {"user_id": CUSTOMER_SCOPE_SENTINEL}, ["n_client"], _COST)
    assert v == pytest.approx(_RAW)
