"""Knowledge update must not compare one rung's memories against another's.

WHAT THIS IS

KU decides whether a brand-new memory is a duplicate of, or superseded by, an
existing one, and it can DROP the new memory on that basis. So the set of
candidates it compares against is a boundary, not an optimisation.

`_build_dedup_scope_filter`'s own docstring records this being found and fixed
one level down:

    "Without this filter the candidate search spans the entire
     synap_{customer_id} collection, so a new memory is compared against EVERY
     user's vectors in the customer. A second user ingesting content similar to
     a first user then has every memory classified NONE/SUPERSEDE against the
     other user's data and silently dropped from the vector index
     (multi-tenant isolation bug)."

The same bug existed one rung up and nobody had looked. A memory filed at an
intermediate rung, a team or a practice or a region, has no user of its own, so
`derive_scope_level` calls it "customer" and the write stamps
`user_id = SENTINEL`. Every team under one tenant then produces the identical
filter, and they share a collection. Team A's memory was compared against Team
B's and could be dropped.

Silent, in the worst way: no error, no 4xx, no log. A memory that stopped
existing.

WHY THE FIX REUSES THE READ PATH'S BUILDER

Two predicates written separately are two predicates that drift, which is what
S-08 exists to prevent and what this file would otherwise become a second copy
of. `build_scope_filter` compiles both, so dedup and retrieval agree by
construction.
"""
from __future__ import annotations

import uuid as _uuid

import pytest

from synap.cloud.agents.ingestion.models.inputs import Scope, ScopeRung
from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    _build_dedup_scope_filter,
)
from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL

CLIENT = _uuid.uuid4()
INSTANCE = _uuid.uuid4()
TENANT = _uuid.uuid4()
TEAM_A = _uuid.uuid4()
TEAM_B = _uuid.uuid4()
ALICE = _uuid.uuid4()


def _chain(*rungs):
    return [ScopeRung(node_id=n, level_key=k, role=r, roles=[r] if r else [])
            for n, k, r in rungs]


def _team_scope(team_node):
    """A write at the `team` rung: no user of its own, so the two-level model
    calls it customer-scope and stamps the sentinel."""
    return Scope(
        client_id=CLIENT, instance_id=INSTANCE, customer_id=TENANT,
        user_id=None, scope_level="customer",
        chain=_chain(
            (CLIENT, "client", "root"),
            (TENANT, "customer", "tenant_of_record"),
            (team_node, "team", None),
        ),
    )


def _matches(where, record) -> bool:
    """Compile the Chroma `where` KU produces and ask it about one record."""
    def ok(w, m):
        if "$and" in w:
            return all(ok(c, m) for c in w["$and"])
        if "$or" in w:
            return any(ok(c, m) for c in w["$or"])
        for field, cond in w.items():
            if isinstance(cond, dict) and "$in" in cond:
                if m.get(field) not in cond["$in"]:
                    return False
            elif m.get(field) != cond:
                return False
        return True
    return ok(where, record)


def test_one_teams_memory_is_not_a_dedup_candidate_for_another():
    """The bug, stated as the thing that must not happen.

    Both teams are customer-scope by the two-level model and share a
    collection, so before the fix this returned the identical filter for both
    and Team B's memory was a candidate for Team A's ingest.
    """
    where = _build_dedup_scope_filter(_team_scope(TEAM_A))
    team_b_memory = {
        "node_id": str(TEAM_B),
        "customer_id": str(TENANT),
        "user_id": CUSTOMER_SCOPE_SENTINEL,
        "anc_team": str(TEAM_B),
    }
    assert not _matches(where, team_b_memory), (
        "another team's memory is a dedup candidate, so KU can classify this "
        "team's new memory SUPERSEDE against it and drop it"
    )


def test_a_teams_own_memory_is_still_a_candidate():
    """The half that must keep working. A dedup that matches nothing is not
    safe, it just means duplicates accumulate forever."""
    where = _build_dedup_scope_filter(_team_scope(TEAM_A))
    own = {
        "node_id": str(TEAM_A),
        "customer_id": str(TENANT),
        "user_id": CUSTOMER_SCOPE_SENTINEL,
        "anc_team": str(TEAM_A),
    }
    assert _matches(where, own)


def test_material_from_the_rungs_above_is_still_a_candidate():
    """A team memory can legitimately supersede a tenant-wide one, because the
    tenant sits on the team's own chain. Upward is the direction that is
    allowed; sideways is not."""
    where = _build_dedup_scope_filter(_team_scope(TEAM_A))
    tenant_wide = {
        "node_id": str(TENANT),
        "customer_id": str(TENANT),
        "user_id": CUSTOMER_SCOPE_SENTINEL,
    }
    assert _matches(where, tenant_wide)


def test_legacy_records_are_candidates_at_a_rung_the_ids_can_express():
    """At the identity or tenant rung, `customer_id` and `user_id` say exactly
    where an unstamped record sits, so it is a fair candidate."""
    scope = Scope(
        client_id=CLIENT, instance_id=INSTANCE, customer_id=TENANT,
        user_id=ALICE, scope_level="user",
        chain=_chain(
            (CLIENT, "client", "root"),
            (TENANT, "customer", "tenant_of_record"),
            (ALICE, "user", "identity"),
        ),
    )
    where = _build_dedup_scope_filter(scope)
    assert _matches(where, {"customer_id": str(TENANT), "user_id": str(ALICE)})


def test_legacy_records_are_NOT_candidates_at_an_intermediate_rung():
    """The subtle half, and the one that broke the first version of the fix.

    An unstamped record carries no rung, and at an intermediate rung the two
    ids cannot express one: every team under a tenant writes
    `customer_id=<tenant>, user_id=SENTINEL`. So a legacy arm here matches every
    sibling team, and the sideways comparison this file exists to prevent comes
    straight back through it.

    The cost is that unstamped material at an intermediate rung does not dedup
    until the backfill stamps it, so duplicates accumulate there. That is the
    right way round: failing to dedup is recoverable and wrongly deduping is
    not.
    """
    where = _build_dedup_scope_filter(_team_scope(TEAM_A))
    unstamped = {"customer_id": str(TENANT), "user_id": CUSTOMER_SCOPE_SENTINEL}
    assert not _matches(where, unstamped), (
        "an unstamped record is a dedup candidate for a team write, and it "
        "could belong to any team under this tenant"
    )


def test_a_persons_private_memory_is_never_a_candidate_for_a_team_write():
    """L-08 in the dedup direction. A team-level write must not be able to drop
    something belonging to one person under it."""
    where = _build_dedup_scope_filter(_team_scope(TEAM_A))
    alices = {
        "node_id": str(ALICE),
        "customer_id": str(TENANT),
        "user_id": str(ALICE),
        "anc_team": str(TEAM_A),
    }
    assert not _matches(where, alices)


def test_the_three_rung_shape_is_unchanged():
    """Almost every client is on the default three rungs, and their filter must
    not move. A user-scope ingest compares against their own memories and the
    customer-shared ones, which is exactly what it did before."""
    scope = Scope(
        client_id=CLIENT, instance_id=INSTANCE, customer_id=TENANT,
        user_id=ALICE, scope_level="user",
        chain=_chain(
            (CLIENT, "client", "root"),
            (TENANT, "customer", "tenant_of_record"),
            (ALICE, "user", "identity"),
        ),
    )
    where = _build_dedup_scope_filter(scope)
    assert _matches(where, {"customer_id": str(TENANT), "user_id": str(ALICE)})
    assert _matches(where, {"customer_id": str(TENANT),
                            "user_id": CUSTOMER_SCOPE_SENTINEL})
    other = _uuid.uuid4()
    assert not _matches(where, {"customer_id": str(TENANT), "user_id": str(other)})


def test_no_chain_falls_back_to_the_two_level_filter():
    """Every caller without a ladder must get byte-identical behaviour."""
    scope = Scope(client_id=CLIENT, instance_id=INSTANCE, customer_id=TENANT,
                  user_id=ALICE, scope_level="user", chain=None)
    assert _build_dedup_scope_filter(scope) == {
        "user_id": {"$in": [str(ALICE), CUSTOMER_SCOPE_SENTINEL]}
    }


def test_dedup_never_honours_cross_user_even_where_reads_do():
    """The docstring's own rule, pinned.

    Dedup DROPS a brand-new memory. Doing that on the strength of another
    person's data is silent data loss, so it stays isolated even on an instance
    where cross-user READS have been granted.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(
        "synap/cloud/agents/ingestion/pipeline/knowledge_update.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_build_dedup_scope_filter")
    kwargs = {
        k.arg: getattr(k.value, "value", None)
        for c in ast.walk(fn) if isinstance(c, ast.Call)
        for k in c.keywords if k.arg
    }
    assert kwargs.get("cross_user") is False, (
        "dedup is honouring cross_user; a memory can now be dropped because "
        "somebody else's looks similar"
    )
    assert kwargs.get("cross_customer") is False
