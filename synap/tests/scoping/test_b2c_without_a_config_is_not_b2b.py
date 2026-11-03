"""A B2C instance with no MACA config told one client two opposite things.

`b2c_contract.load_isolation` learned to read the instance's own
`user_relationship` when there is no config (e88cbe91). Scope resolution did
not: it kept a hardcoded `"strict"`, in `resolve_identity` and again in
`sdk_memories`. So the ingress guard said B2C while the resolver said B2B, and
between them no request shape was accepted at all.

Example: a client's instance created with `user_relationship = "b2c"`, whose
MACA config is written some minutes later. In the gap:

1st  create  user_id only            -> 400  "B2B instance: user_id
provided without customer_id"
2nd  create  user_id + customer_id   -> 400  customer_id_not_accepted_on_b2c
3rd  create  user_id only            -> 400  "B2B instance: ..."

Three rejections within about a minute, two of them contradicting each other,
on a caller's first contact with the API.

The tests below are behavioural on purpose. `test_b2c_holds_without_a_config.py`
asserts on `inspect.getsource`, which is why it passed all the way through this
defect: it can only see the half of the system it reads.
"""
from __future__ import annotations

import uuid as _uuid

import pytest

from synap.cloud.shared import b2c_contract
from synap.cloud.shared.scope import (
    B2C_ISOLATION,
    effective_user_context_isolation,
    resolve_identity,
)

CLIENT = "cli_00000000000000a1"
INSTANCE = "inst_00000000000000a1"
USER = "synap_ananya_1"


@pytest.fixture
def instance_says_b2c(monkeypatch):
    """No MACA config; the instance itself declares b2c, as theirs did."""
    async def _load(instance_id):
        assert instance_id == INSTANCE
        return B2C_ISOLATION
    monkeypatch.setattr(b2c_contract, "load_isolation", _load)


@pytest.fixture
def instance_says_nothing(monkeypatch):
    """No config and no declared relationship — the honest 'we do not know'."""
    async def _load(instance_id):
        return None
    monkeypatch.setattr(b2c_contract, "load_isolation", _load)


# ── the failure, as behaviour ────────────────────────────────────

async def test_user_id_only_is_not_rejected_as_b2b(instance_says_b2c):
    """The 10:27:56 and 10:28:58 requests. Both were correctly formed."""
    ident = await resolve_identity(
        CLIENT, INSTANCE, user_id=USER, customer_id=None,
        maca_scoping=None,
        derive_on_failure=True, resolve_chain=False,
    )
    assert ident.is_b2c is True
    assert ident.scope_level == "user"


async def test_the_guard_and_the_resolver_cannot_disagree(instance_says_b2c):
    """The actual defect: two sides, one instance, opposite answers.

    Pinning either side's value alone would have passed on the broken build.
    What matters is that they agree.
    """
    guard = await b2c_contract.load_isolation(INSTANCE)
    resolver = await effective_user_context_isolation(INSTANCE, None)
    assert guard == resolver, (
        "the ingress guard and scope resolution answered differently, which is "
        "how one client got 'customer_id is required' and 'customer_id is not "
        "accepted' inside 62 seconds"
    )


async def test_b2c_collapses_the_identity_even_with_no_config(instance_says_b2c):
    """Not just 'does not raise' — it must land on the B2C scope."""
    ident = await resolve_identity(
        CLIENT, INSTANCE, user_id=USER, customer_id=None,
        maca_scoping=None,
        derive_on_failure=True, resolve_chain=False,
    )
    assert ident.customer_uuid == ident.user_uuid, (
        "B2C means customer and user are one node"
    )
    assert isinstance(ident.user_uuid, _uuid.UUID)


# ── the fallback must not become a new guess ────────────────────────────────

async def test_a_caller_that_knows_still_wins(instance_says_b2c):
    """An explicit answer is never overridden by the lookup."""
    got = await effective_user_context_isolation(
        INSTANCE, {"user_context_isolation": "strict"},
    )
    assert got == "strict"


async def test_an_instance_that_says_nothing_stays_strict(instance_says_nothing):
    """Today's behaviour is preserved where there is genuinely no answer."""
    assert await effective_user_context_isolation(INSTANCE, None) == "strict"

    with pytest.raises(ValueError, match="B2B instance"):
        await resolve_identity(
            CLIENT, INSTANCE, user_id=USER, customer_id=None,
            maca_scoping=None,
            derive_on_failure=True, resolve_chain=False,
        )


async def test_the_lookup_can_never_fail_a_request(monkeypatch):
    """A fallback that can raise is not a fallback."""
    async def _boom(instance_id):
        raise RuntimeError("postgres is unreachable")
    monkeypatch.setattr(b2c_contract, "load_isolation", _boom)

    assert await effective_user_context_isolation(INSTANCE, None) == "strict"


# ── the write path must not keep its own copy ───────────────────────────────

def test_the_write_path_does_not_resolve_isolation_itself():
    """`sdk_memories` had its own MACA read and its own "strict" default.

    Two answers to one question is what produced the contradiction, so the
    write path must hand the question over rather than answer it again.

    Read off disk rather than imported: importing the route module pulls in the
    voice stack, so an `import` here would make this guard fail for a missing
    speech dependency and say nothing about scoping.
    """
    from pathlib import Path

    route = (
        Path(__file__).parents[2]          # .../synap
        / "cloud" / "application" / "sdk_api" / "sdk_memories.py"
    )
    assert route.is_file(), f"guard is pointing at nothing: {route}"
    src = route.read_text()
    assert 'user_context_isolation = "strict"' not in src, (
        "the write path is defaulting isolation again, independently of "
        "scope.effective_user_context_isolation"
    )
