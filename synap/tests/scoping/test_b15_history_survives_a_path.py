"""B-15. One correct request must not hide a person's earlier memories.

The failure, seen for a client with a five-rung ladder, user-61:

    1. read user-61, no registry node yet             -> 3 of 3 records
    2. ONE path-carrying request naming user-61       -> HTTP 200, creates the node
    3. repeat step 1 byte for byte                    -> 0 items

The trigger is not a mistake. It is a client using the new feature correctly,
once, for a person who already has history. A switched-off instance reads all
three records without difficulty, which is the control proving the data is
intact and indexed rather than lost.

WHY IT HAPPENED

Creating the node made the chain resolve, and both read arms then missed:

  * the nested arm looks for `node_id IN chain`, and records written without a
    path carry no `node_id` at all;
  * the legacy arm took its `customer_id` and `user_id` VALUES off that same
    chain, and the node had been minted with `gen_random_uuid()`.

The stored values against the resolved ones:

    chain identity node   <uuid A>   surrogate, created by a path
    stored user_id        <uuid B>   data-plane uuid, on all 3 records

WHY THE FIX IS NOT A BACKFILL

Two other repairs were on the table and both are worse. Stamping every old
record with a rung is a migration over live data. Repointing the surrogate nodes
to the data-plane uuids would strand the records that ARE stamped with those
surrogates, which for that client is a large number.

Feeding the legacy arm the request's own data-plane uuids fixes every affected
person at once, repairs nothing and migrates nothing, and where a node id
already equals the data-plane uuid it changes the filter not at all.

WHY IT LOOKS FIXED IF YOU TEST IT CASUALLY

On a fresh user it passes: they have no earlier records to lose. The
reproduction needs a user who already has history, which is why the fixture
below writes the record BEFORE the node exists. A test that creates the node
first cannot fail.
"""
from __future__ import annotations

import uuid as _uuid

import pytest

from synap.cloud.shared.scope_filter import to_python_predicate

pytestmark = []


class _Scope:
    """The slice of ScopeHierarchy the predicate reads."""

    def __init__(self, customer_id, user_id, chain_rungs):
        self.customer_id = customer_id
        self.user_id = user_id
        self.chain_rungs = chain_rungs


class _Maca:
    roll_up_to_customer = True
    allow_cross_user_retrieval = False
    allow_cross_customer_retrieval = False


# The failing shape: a five-rung ladder whose tenant and identity nodes were
# created by a scope path, so both carry a surrogate rather than the uuid the
# records carry.
CLIENT_NODE = str(_uuid.uuid4())
PRACTICE_NODE = str(_uuid.uuid4())
CARE_TEAM_NODE = str(_uuid.uuid4())
TENANT_SURROGATE = str(_uuid.uuid4())
IDENTITY_SURROGATE = str(_uuid.uuid4())

DATA_PLANE_CUSTOMER = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "bright-smile"))
DATA_PLANE_USER = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "patient-61"))

RUNGS = [
    {"node_id": CLIENT_NODE, "level_key": "client", "role": "root", "roles": ["root"]},
    {"node_id": TENANT_SURROGATE, "level_key": "customer",
     "role": "tenant_of_record", "roles": ["tenant_of_record"]},
    {"node_id": PRACTICE_NODE, "level_key": "practice", "role": None, "roles": []},
    {"node_id": CARE_TEAM_NODE, "level_key": "care_team", "role": None, "roles": []},
    {"node_id": IDENTITY_SURROGATE, "level_key": "user",
     "role": "identity", "roles": ["identity"]},
]

# What the person's earlier memories actually look like in the store: written
# before any path existed, so no node_id and no ancestors.
EARLIER_MEMORY = {
    "node_id": None,
    "customer_id": DATA_PLANE_CUSTOMER,
    "user_id": DATA_PLANE_USER,
}


def _predicate(scope, maca=None):
    from synap.cloud.application.client_api.retrieval_manager import (
        _nested_scope_predicate,
    )

    return to_python_predicate(_nested_scope_predicate(scope, maca or _Maca()))


def test_a_persons_earlier_memories_survive_the_first_path_request():
    """The headline. This is the whole bug in one assertion."""
    scope = _Scope(DATA_PLANE_CUSTOMER, DATA_PLANE_USER, RUNGS)
    assert _predicate(scope)(EARLIER_MEMORY), (
        "a person's earlier memories stopped being readable the moment their "
        "registry node existed. One correct request cost them their history."
    )


def test_a_record_stamped_with_the_surrogate_is_still_readable():
    """The other half, and the reason repointing the nodes was the wrong fix.

    Records written THROUGH the path carry the surrogate node id. They are
    matched by the nested arm and must keep working, which a repair that
    repointed the node to the data-plane uuid would have broken.
    """
    scope = _Scope(DATA_PLANE_CUSTOMER, DATA_PLANE_USER, RUNGS)
    stamped = {
        "node_id": IDENTITY_SURROGATE,
        "customer_id": DATA_PLANE_CUSTOMER,
        "user_id": DATA_PLANE_USER,
        "anc_care_team": CARE_TEAM_NODE,
    }
    assert _predicate(scope)(stamped)


def test_another_persons_memories_are_still_refused():
    """The fix must not widen. This is the assertion that says so.

    Feeding the legacy arm the request's own data-plane uuids scopes it to
    exactly the requester, which is what the same request reads with the
    feature switched off. A sibling under the same tenant stays out.
    """
    scope = _Scope(DATA_PLANE_CUSTOMER, DATA_PLANE_USER, RUNGS)
    someone_else = {
        "node_id": None,
        "customer_id": DATA_PLANE_CUSTOMER,
        "user_id": str(_uuid.uuid5(_uuid.NAMESPACE_URL, "patient-90")),
    }
    assert not _predicate(scope)(someone_else)


def test_another_tenants_memories_are_still_refused():
    """The cross-tenant half of the same question. R-01."""
    scope = _Scope(DATA_PLANE_CUSTOMER, DATA_PLANE_USER, RUNGS)
    other_tenant = {
        "node_id": None,
        "customer_id": str(_uuid.uuid5(_uuid.NAMESPACE_URL, "high-street")),
        "user_id": str(_uuid.uuid5(_uuid.NAMESPACE_URL, "patient-12")),
    }
    assert not _predicate(scope)(other_tenant)


def test_customer_shared_material_still_rolls_up():
    """S-05. Shared material stores a sentinel rather than a uuid in `user_id`."""
    from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL

    scope = _Scope(DATA_PLANE_CUSTOMER, DATA_PLANE_USER, RUNGS)
    shared = {
        "node_id": None,
        "customer_id": DATA_PLANE_CUSTOMER,
        "user_id": CUSTOMER_SCOPE_SENTINEL,
    }
    assert _predicate(scope)(shared)


def test_a_customer_scope_read_is_not_narrowed_to_a_person():
    """This is also B-09, which was reported separately as unexplained.

    B-09: on one instance, a customer-scope read with a path returned 4 items
    and the same read without one returned 0, and it was not chased to a root
    cause. It is this defect. The tenant node is a surrogate, so the legacy arm
    filtered on an id that the 16 customer-shared records do not carry, while
    the records written THROUGH the path carry the surrogate as their node_id
    and were matched by the nested arm. Hence 4 and 0.

    After the fix the same 16 records are
    matched now and were not before.

    The rest of what this test pins: a chain that reached no identity rung is
    an ordinary customer-scope read.

    Handing it the request's `user_id` would narrow it to one person, which is
    a different request from the one the caller made. The fallback must not do
    that, so it is refused explicitly rather than left to the `or`.
    """
    from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL

    customer_rungs = RUNGS[:2]
    # The request still carries a user_id; the CHAIN stopped at the tenant.
    scope = _Scope(DATA_PLANE_CUSTOMER, DATA_PLANE_USER, customer_rungs)
    pred = _predicate(scope)
    assert pred({"node_id": None, "customer_id": DATA_PLANE_CUSTOMER,
                 "user_id": CUSTOMER_SCOPE_SENTINEL})
    assert not pred({"node_id": None, "customer_id": DATA_PLANE_CUSTOMER,
                     "user_id": DATA_PLANE_USER}), (
        "a customer-scope read must not start returning one person's private "
        "memories just because the request happened to name them"
    )


def test_a_node_that_already_matches_produces_an_unchanged_filter():
    """The normal case: every node the flat-id path created.

    There the node id IS the data-plane uuid, so the fix must be a no-op. If
    this ever diverges, the change stopped being narrow.
    """
    from synap.cloud.application.client_api.retrieval_manager import (
        _nested_scope_predicate,
    )

    agreeing = [
        dict(RUNGS[0]),
        {**RUNGS[1], "node_id": DATA_PLANE_CUSTOMER},
        dict(RUNGS[2]), dict(RUNGS[3]),
        {**RUNGS[4], "node_id": DATA_PLANE_USER},
    ]
    scope = _Scope(DATA_PLANE_CUSTOMER, DATA_PLANE_USER, agreeing)
    with_ids = _nested_scope_predicate(scope, _Maca())
    without = _Scope(None, None, agreeing)
    off_the_chain = _nested_scope_predicate(without, _Maca())
    assert with_ids == off_the_chain


def test_a_missing_data_plane_id_falls_back_without_widening():
    """The fallback direction is load-bearing.

    Passing None where a data-plane id is absent would DROP the clause and
    widen the read to the whole store. The chain node at worst matches nothing.
    Never widen on a fallback.
    """
    from synap.cloud.shared.scope_filter import fields_referenced
    from synap.cloud.application.client_api.retrieval_manager import (
        _nested_scope_predicate,
    )

    scope = _Scope(None, None, RUNGS)
    pred = _nested_scope_predicate(scope, _Maca())
    assert "customer_id" in fields_referenced(pred), (
        "the customer clause vanished, so this read is no longer scoped to a "
        "tenant at all"
    )
    assert not to_python_predicate(pred)({
        "node_id": None,
        "customer_id": str(_uuid.uuid5(_uuid.NAMESPACE_URL, "high-street")),
        "user_id": str(_uuid.uuid5(_uuid.NAMESPACE_URL, "patient-12")),
    })


def test_the_reserve_scopes_the_same_way_as_the_fetch_it_supplements():
    """Two callers deriving the legacy arm separately is how they diverge.

    The leaf reserve only ever ADDS candidates to a pool ranked once
    afterwards, so if it scoped more widely than the main filter it would
    introduce records the main filter had excluded.
    """
    import ast
    import pathlib

    src = pathlib.Path(
        "synap/cloud/application/client_api/retrieval_manager.py"
    ).read_text()
    tree = ast.parse(src)
    callers = [
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == "_legacy_arm_ids"
            for c in ast.walk(n)
        )
    ]
    assert "_nested_scope_predicate" in callers
    assert len(callers) >= 2, (
        f"only {callers} derive the legacy arm through the shared helper; the "
        "leaf reserve must not build its own"
    )


def test_the_reserve_skip_still_compares_node_ids_to_node_ids():
    """The one place the CHAIN's tenant node is still the right answer.

    The reserve is skipped when the leaf IS the tenant, because a reserve there
    narrows nothing reachable and costs a third store call. That test is
    `chain[-1] == tenant`, so both sides have to be node ids.

    Swapping in the data-plane uuid there looks like consistency and is a bug:
    the two agree only for a node the flat-id path created, so on exactly the
    surrogate nodes this file is about, the skip would stop firing and every
    B2C and customer-scope request would pay for a reserve fetch that returns
    nothing. Measured at up to +19.9% p95 at 16 concurrent when it was wrong
    the other way (SHIP-15).

    A behavioural assertion would need a store double the reserve tests already
    own, and they cover the skip itself. What this pins is the thing those
    tests cannot see: which of the two values the comparison is fed.
    """
    import ast
    import pathlib as _p

    tree = ast.parse(
        _p.Path("synap/cloud/application/client_api/retrieval_manager.py").read_text()
    )
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == "_legacy_arm_ids"
            for c in ast.walk(n)
        )
        and any(
            isinstance(c, ast.Compare) and "chain" in ast.dump(c.left)
            for c in ast.walk(n)
        )
    )
    compares = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Compare) and "chain" in ast.dump(n.left)
    ]
    assert compares, "the leaf-is-the-tenant skip is gone"
    rhs = ast.dump(compares[0].comparators[0])
    assert "tenant" in rhs
    # The value must come from the chain roles, never from the data-plane ids.
    assigns = {
        t.id: ast.dump(n.value)
        for n in ast.walk(fn) if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Name)
    }
    assert "_roles_by_node" in assigns.get("by_role", ""), (
        "the reserve skip is no longer comparing a chain node against a chain "
        "node, so it will stop firing on exactly the nodes B-15 is about"
    )


def test_b16_switching_on_does_not_hide_what_was_written_while_off():
    """B-16, which was reported separately and is the same defect.

    An instance with the switch off writes no chain metadata at all: its
    records carry the plain id columns and no `node_id`. Turning the switch on
    for that instance was then measured to hide everything written while it was
    off, and a near-verbatim query for four such records returned two unrelated
    ones and none of the four.

    That is this bug from the other direction. The switched-off records are
    unstamped, the switched-on read resolves a chain, and the legacy arm that
    exists to bridge exactly that gap was filtering on the wrong ids. Closing
    B-15 closes this, and it is asserted here so the second finding has its own
    lock rather than being assumed covered.
    """
    scope = _Scope(DATA_PLANE_CUSTOMER, DATA_PLANE_USER, RUNGS)
    written_while_off = {
        "node_id": None,
        "customer_id": DATA_PLANE_CUSTOMER,
        "user_id": DATA_PLANE_USER,
    }
    assert _predicate(scope)(written_while_off), (
        "turning the switch on hid everything the instance wrote while it was "
        "off, which breaks continuity of the client's own data across a "
        "setting they were told was safe to toggle"
    )
