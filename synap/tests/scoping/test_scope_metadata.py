"""What actually gets written onto a vector record.

The nested keys are additive, and these tests are mostly about proving that:
a caller who has not resolved a chain must get exactly the metadata they get
today, byte for byte, because that is what makes it safe to deploy the write
side long before anything reads it.

The rest is about the keys themselves. S-03 says level-keyed ancestors, S-04
says the legacy aliases stay, S-06 says no path ever reaches a record.
"""
from __future__ import annotations

import uuid as _uuid

import pytest

from synap.cloud.agents.ingestion.models.inputs import Scope, ScopeRung
from synap.cloud.agents.ingestion.storage.vector_wrapper import _scope_metadata
from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL

CLIENT = _uuid.uuid5(_uuid.NAMESPACE_URL, "cli_test")
INSTANCE = _uuid.uuid5(_uuid.NAMESPACE_URL, "inst_test")
CUST = _uuid.uuid5(CLIENT, "infosys")
USER = _uuid.uuid5(CUST, "d-42")
REGION = _uuid.uuid5(CLIENT, "emea")
TEAM = _uuid.uuid5(CUST, "payments")


def _legacy_scope(level: str) -> Scope:
    return Scope(
        client_id=CLIENT, instance_id=INSTANCE,
        customer_id=CUST if level != "client" else None,
        user_id=USER if level == "user" else None,
        scope_level=level,
    )


def _nested_scope(level: str) -> Scope:
    s = _legacy_scope(level)
    s.chain = [
        ScopeRung(node_id=CLIENT, level_key="client", role="root"),
        ScopeRung(node_id=CUST, level_key="customer", role="tenant_of_record"),
        ScopeRung(node_id=USER, level_key="user", role="identity"),
    ]
    return s


# ── the regression lock ─────────────────────────────────────────────────────

@pytest.mark.parametrize("level", ["user", "customer", "client"])
def test_without_a_chain_nothing_changes(level: str) -> None:
    """Every caller in the product today leaves `chain` unset. If this test
    ever fails, the write side has started emitting something new for callers
    who asked for nothing, and the deploy is no longer additive."""
    meta = _scope_metadata(_legacy_scope(level))
    expected = {"client_id": str(CLIENT), "instance_id": str(INSTANCE)}
    if level != "client":
        expected["customer_id"] = str(CUST)
        expected["user_id"] = str(USER) if level == "user" else CUSTOMER_SCOPE_SENTINEL
    assert meta == expected


# ── S-04: the aliases stay ──────────────────────────────────────────────────

@pytest.mark.parametrize("level", ["user", "customer", "client"])
def test_the_legacy_fields_are_untouched_by_the_new_ones(level: str) -> None:
    """S-04. Every existing query, dashboard, billing rollup and test reads
    customer_id and user_id. Adding node_id must not move them."""
    old = _scope_metadata(_legacy_scope(level))
    new = _scope_metadata(_nested_scope(level))
    for k, v in old.items():
        assert new[k] == v, f"{k} changed from {v!r} to {new[k]!r}"


# ── S-03: what the new keys are ─────────────────────────────────────────────

def test_a_user_level_record_is_stamped_with_the_identity_node() -> None:
    meta = _scope_metadata(_nested_scope("user"))
    assert meta["node_id"] == str(USER)
    assert meta["anc_client"] == str(CLIENT)
    assert meta["anc_customer"] == str(CUST)
    assert meta["anc_user"] == str(USER)


def test_a_customer_level_record_stops_at_the_tenant_node() -> None:
    """Shared material belongs to the tenant, not to whoever happened to be
    talking when it was extracted. Stamping the identity node here would file
    org-wide knowledge inside one person's scope, where nobody else can read
    it."""
    meta = _scope_metadata(_nested_scope("customer"))
    assert meta["node_id"] == str(CUST)
    assert "anc_user" not in meta
    assert meta["user_id"] == CUSTOMER_SCOPE_SENTINEL      # S-05, unchanged


def test_a_client_level_record_stops_at_the_root() -> None:
    meta = _scope_metadata(_nested_scope("client"))
    assert meta["node_id"] == str(CLIENT)
    assert set(meta) == {"client_id", "instance_id", "node_id", "anc_client"}


def test_an_explicit_override_beats_the_scope_level() -> None:
    """Callers downgrade the level per-write for shared extractions. The node
    has to follow the override, or the two disagree on the same record."""
    meta = _scope_metadata(_nested_scope("user"), "customer")
    assert meta["node_id"] == str(CUST)
    assert "anc_user" not in meta


# ── deeper ladders ──────────────────────────────────────────────────────────

def _deep_scope(level: str) -> Scope:
    s = Scope(
        client_id=CLIENT, instance_id=INSTANCE, customer_id=CUST, user_id=USER,
        scope_level=level,
    )
    s.chain = [
        ScopeRung(node_id=CLIENT, level_key="client", role="root"),
        ScopeRung(node_id=REGION, level_key="region"),
        ScopeRung(node_id=CUST, level_key="customer", role="tenant_of_record"),
        ScopeRung(node_id=TEAM, level_key="team"),
        ScopeRung(node_id=USER, level_key="user", role="identity"),
    ]
    return s


def test_a_five_level_record_carries_every_ancestor() -> None:
    meta = _scope_metadata(_deep_scope("user"))
    assert meta["node_id"] == str(USER)
    assert {k for k in meta if k.startswith("anc_")} == {
        "anc_client", "anc_region", "anc_customer", "anc_team", "anc_user"
    }


def test_a_record_can_be_written_at_a_level_the_product_has_never_heard_of() -> None:
    """I-03: one transcript produces memories at several levels in one pass, and
    the interesting ones are the levels the client invented. Nothing here knows
    what "team" means."""
    meta = _scope_metadata(_deep_scope("user"), "team")
    assert meta["node_id"] == str(TEAM)
    assert "anc_user" not in meta
    assert meta["anc_team"] == str(TEAM)


def test_ancestors_are_level_keyed_not_positional() -> None:
    """S-03's actual reason. With positional keys, inserting "region" would turn
    every pre-existing anc_2 from "customer" into something else, and a partial
    backfill would be wrong rather than merely incomplete."""
    shallow = _scope_metadata(_nested_scope("user"))
    deep = _scope_metadata(_deep_scope("user"))
    for key in ("anc_client", "anc_customer", "anc_user"):
        assert shallow[key] == deep[key], f"{key} means something different at depth 5"


# ── S-06 and the Chroma rules ───────────────────────────────────────────────

@pytest.mark.parametrize("level", ["user", "customer", "client"])
def test_no_path_is_ever_written_onto_a_record(level: str) -> None:
    """S-06. The path lives once, in the registry. On a record it would mean a
    shape change rewrites millions of documents rather than thousands of rows,
    and re-parenting would stop being a single UPDATE."""
    meta = _scope_metadata(_deep_scope(level))
    assert not any("path" in k for k in meta)
    assert not any("." in str(v) or "/" in str(v) for v in meta.values())


@pytest.mark.parametrize("level", ["user", "customer", "client"])
def test_no_value_is_ever_none(level: str) -> None:
    """Chroma drops None metadata values silently, so a key that is sometimes
    None is a key that sometimes does not exist, and an equality filter on it
    quietly matches nothing."""
    for meta in (_scope_metadata(_legacy_scope(level)), _scope_metadata(_deep_scope(level))):
        assert all(v is not None for v in meta.values())
        assert all(isinstance(v, str) for v in meta.values())


# ── the failure case ────────────────────────────────────────────────────────

def test_a_chain_that_does_not_reach_the_write_level_stamps_nothing() -> None:
    """Stamping the leaf instead would file the record one level narrower than
    the caller asked for, which reads as missing data later and has no trace
    back to here. Refusing to guess leaves the record on the legacy fields,
    which still work."""
    s = _legacy_scope("customer")
    s.chain = [ScopeRung(node_id=CLIENT, level_key="client", role="root")]
    meta = _scope_metadata(s)
    assert "node_id" not in meta
    assert meta["customer_id"] == str(CUST)       # legacy path still complete
    assert meta["user_id"] == CUSTOMER_SCOPE_SENTINEL


# ── B2C, where the chain reaches the level under a different name ──────────
#
# Found by audit 2026-08-23, and it was the more expensive half of a two-part
# data-loss bug. In B2C the customer node IS the identity node, so the chain is
# two rungs and neither has `level_key == "user"` nor `role == "identity"` on
# its own. `_nested_scope_metadata` matched roles by equality, found nothing,
# and returned {}. Nested scoping was switched on and every new record for
# every B2C client was written with no `node_id` at all.
#
# The test above, about a chain that genuinely does not reach the write level,
# is still correct and still passes. The difference is that this chain DOES
# reach it: the rung is there, holding two roles.


def _b2c_scope() -> Scope:
    """What `resolve_legacy_chain` emits for B2C: two rungs, tenant also identity."""
    s = Scope(
        client_id=CLIENT, instance_id=INSTANCE,
        customer_id=CUST, user_id=CUST,      # IdMapper collapses the two
        scope_level="user",
    )
    s.chain = [
        ScopeRung(node_id=CLIENT, level_key="client", role="root", roles=["root"]),
        ScopeRung(node_id=CUST, level_key="customer", role="tenant_of_record",
                  roles=["tenant_of_record", "identity"]),
    ]
    return s


def test_a_b2c_user_write_is_stamped():
    """The whole defect in one assertion. This returned {} for the entire build."""
    meta = _scope_metadata(_b2c_scope())
    assert meta.get("node_id") == str(CUST), (
        "a B2C user-scope write got no node_id, so nested scoping is on and "
        "every new record is invisible to it"
    )


def test_a_b2c_write_carries_its_ancestors():
    meta = _scope_metadata(_b2c_scope())
    assert meta.get("anc_client") == str(CLIENT)
    assert meta.get("anc_customer") == str(CUST)


def test_a_b2c_write_is_filed_at_the_customer_node_not_below_it():
    """There is no node below it. Filing at a narrower scope than exists is the
    failure the truncated-chain test above refuses to risk."""
    meta = _scope_metadata(_b2c_scope())
    assert meta["node_id"] == meta["anc_customer"]


def test_a_rung_carrying_one_role_still_behaves_exactly_as_before():
    """The B2B path must not move. Two roles on one rung is the special case."""
    meta = _scope_metadata(_nested_scope("user"))
    assert meta["node_id"] == str(USER)
    assert meta["anc_user"] == str(USER)


def test_a_chain_built_before_roles_existed_still_works():
    """`roles` is optional, so a chain persisted or constructed without it must
    fall back to `role` rather than stamping nothing."""
    s = _b2c_scope()
    for rung in s.chain:
        rung.roles = None
    meta = _scope_metadata(s)
    # Falls back to `role`, so the identity is not found and this is the old
    # behaviour: no stamp. Correct, and the reason the fix is on the resolver
    # rather than here.
    assert "node_id" not in meta
