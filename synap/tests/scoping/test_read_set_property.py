"""X-01. Over randomly generated ladders, trees, grants and records, does any
read ever return something outside the permitted set?

R-01 is the security invariant of the whole feature, and hand-written cases
will not cover it: every test elsewhere in this directory is a shape someone
thought of. This generates shapes nobody thought of, thousands of them, and
checks the one thing that must never be false.

The oracle is deliberately not the code under test. The permitted set is
computed here, from the definitions in the spec, by walking parent pointers and
adding grant targets. The filter is compiled by the real builder. If the two
ever disagree the test says so, and neither one is trusted to be the reference.

No hypothesis dependency: a seeded RNG, a fixed iteration count, and the seed
printed on failure. Adding a library to the test suite for this would be a
bigger decision than the test.
"""
from __future__ import annotations

import random
import uuid as _uuid

import pytest

from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL
from synap.cloud.shared.scope_filter import (
    NothingReadable,
    build_scope_filter,
    to_python_predicate,
)

ITERATIONS = 400
BASE_SEED = 20260821


def _build_world(rng: random.Random):
    """A random client: a ladder, a tree, a requester, grants, and records."""
    depth = rng.randint(2, 7)
    levels = ["client"] + [f"lvl{i}" for i in range(1, depth - 1)] + ["user"]

    # nodes[level_index] -> list of (node_id, parent_id)
    nodes: list[list[tuple[str, str | None]]] = [[(str(_uuid.uuid4()), None)]]
    for _ in range(1, depth):
        parents = nodes[-1]
        layer = []
        for parent_id, _ in parents:
            for _ in range(rng.randint(1, 3)):
                layer.append((str(_uuid.uuid4()), parent_id))
        nodes.append(layer)

    parent_of = {nid: pid for layer in nodes for nid, pid in layer}
    all_ids = list(parent_of)

    # The requester sits at a random node, at any depth.
    requester = rng.choice(all_ids)
    chain: list[str] = []
    walk: str | None = requester
    while walk is not None:
        chain.append(walk)
        walk = parent_of[walk]
    chain.reverse()

    # Grants: random directed edges into the requester's chain. Only the target
    # node opens up, never its subtree (G-02), and never onward (G-04).
    granted: set[str] = set()
    for _ in range(rng.randint(0, 4)):
        viewer = rng.choice(chain)
        source = rng.choice(all_ids)
        if source in chain:
            continue
        if rng.random() < 0.8:            # some are denies
            granted.add(source)
        else:
            granted.discard(source)

    permitted = set(chain) | granted
    return levels, all_ids, chain, granted, permitted


def _records(rng: random.Random, all_ids: list[str], chain: list[str], include_legacy: bool):
    """A mix of new-style records carrying node_id and legacy ones that do not.

    Legacy records are only generated when the filter carries its legacy arm,
    because without it they are invisible by design: they have no `node_id` and
    the nested predicate is membership in a set of node ids. That is the whole
    reason the arm exists, so treating it as a leak would be testing the
    transition rather than the invariant.

    They are also only generated for a chain deep enough to have a real tenant
    and identity. A client-level requester reads the client collection, whose
    records carry neither a customer id nor a user id, so a customer-shaped
    record is not theirs to own. Attributing one to them was this generator's
    first bug and it looked exactly like a leak.
    """
    out = []
    can_be_legacy = include_legacy and len(chain) >= 2
    for _ in range(rng.randint(5, 40)):
        owner = rng.choice(all_ids)
        if not can_be_legacy or rng.random() < 0.7:
            out.append(({"node_id": owner}, owner))
        else:
            tenant, identity = chain[1], chain[-1]
            roll = rng.random()
            if roll < 0.4:
                out.append(({"customer_id": tenant, "user_id": identity}, identity))
            elif roll < 0.7:
                out.append((
                    {"customer_id": tenant, "user_id": CUSTOMER_SCOPE_SENTINEL}, tenant
                ))
            else:
                # Someone ELSE's personal record under the requester's own
                # tenant. This is the case the legacy arm's user clause exists
                # for, and without it in the generator the clause can be
                # deleted with every test still passing: every other legacy
                # record carries the requester's own tenant, so the customer
                # clause alone accepts them all correctly by accident.
                stranger = rng.choice([n for n in all_ids if n not in chain] or [identity])
                out.append(({"customer_id": tenant, "user_id": stranger}, stranger))
    return out


@pytest.mark.parametrize("include_legacy", [False, True])
def test_nothing_outside_the_permitted_set_is_ever_returned(include_legacy: bool) -> None:
    """The security invariant, and it is one-directional on purpose.

    R-01 forbids returning a record outside the permitted set. It does not
    promise that every permitted record is returned; that is completeness, it
    is checked separately below, and conflating the two hides which of them
    just broke. An over-strict filter is a missing result. An over-permissive
    one is a leak.
    """
    for i in range(ITERATIONS):
        seed = BASE_SEED + i
        rng = random.Random(seed)
        levels, all_ids, chain, granted, permitted = _build_world(rng)
        tenant = chain[1] if len(chain) >= 2 else None
        identity = chain[-1] if len(chain) >= 2 else None
        try:
            predicate = to_python_predicate(build_scope_filter(
                sorted(permitted), tenant_node_id=tenant, identity_node_id=identity,
                include_legacy=include_legacy,
            ))
        except NothingReadable:
            continue

        for metadata, owner in _records(rng, all_ids, chain, include_legacy):
            if predicate(metadata):
                assert owner in permitted, (
                    f"seed={seed} legacy={include_legacy}: LEAK. A record owned by "
                    f"{owner} was returned and is not in the permitted set.\n"
                    f"  metadata={metadata}\n  chain={chain}\n  granted={sorted(granted)}"
                )


@pytest.mark.parametrize("include_legacy", [False, True])
def test_every_permitted_record_that_carries_a_node_id_is_returned(
    include_legacy: bool,
) -> None:
    """Completeness, for the records the feature is actually about.

    A record stamped with `node_id` is one written since S-03 shipped, and the
    filter is membership in a set of node ids, so every permitted one must come
    back. This is the half that would catch an over-strict filter.
    """
    for i in range(ITERATIONS):
        seed = BASE_SEED + i
        rng = random.Random(seed)
        levels, all_ids, chain, granted, permitted = _build_world(rng)
        tenant = chain[1] if len(chain) >= 2 else None
        identity = chain[-1] if len(chain) >= 2 else None
        try:
            predicate = to_python_predicate(build_scope_filter(
                sorted(permitted), tenant_node_id=tenant, identity_node_id=identity,
                include_legacy=include_legacy,
            ))
        except NothingReadable:
            continue
        for node in permitted:
            assert predicate({"node_id": node}), (
                f"seed={seed}: {node} is permitted but its records are refused"
            )


def test_a_grant_does_not_reach_the_granted_node_legacy_records() -> None:
    """A real limitation, found by the property test above and pinned here so
    it is a known state rather than a surprise.

    The legacy arm can only express the requester's OWN tenant and identity,
    because those are the only two things `customer_id` and `user_id` can name
    for them. A granted node's pre-existing records carry that node's ids, not
    the requester's, so nothing matches them until the metadata backfill stamps
    `node_id` on them.

    The direction is the safe one: a grant returns less than the client asked
    for, never more. But a client who turns on sharing and sees nothing appear
    would reasonably call that broken, so it belongs in the rollout notes
    rather than in a debugging session six months from now.
    """
    chain = [str(_uuid.uuid4()) for _ in range(3)]
    granted_node = str(_uuid.uuid4())
    predicate = to_python_predicate(build_scope_filter(
        chain + [granted_node],
        tenant_node_id=chain[1],
        identity_node_id=chain[2],
        include_legacy=True,
    ))

    # Stamped since S-03: the grant works.
    assert predicate({"node_id": granted_node})
    # Written before it: invisible, and that is the current behaviour.
    assert not predicate({"customer_id": granted_node, "user_id": CUSTOMER_SCOPE_SENTINEL})
    # The requester's own legacy records are unaffected.
    assert predicate({"customer_id": chain[1], "user_id": chain[2]})


def test_a_grant_never_opens_the_granted_node_subtree() -> None:
    """G-02 as a property. The filter is membership in a set of node ids, so a
    descendant of a granted node has a different id and cannot match. Asserted
    over random trees because "cannot match" is the kind of claim that stops
    being true the moment someone adds a prefix match for convenience.
    """
    for i in range(ITERATIONS):
        rng = random.Random(BASE_SEED + 10_000 + i)
        levels, all_ids, chain, granted, permitted = _build_world(rng)
        if not granted:
            continue
        predicate = to_python_predicate(
            build_scope_filter(sorted(permitted), include_legacy=False)
        )
        for node in all_ids:
            if node in permitted:
                continue
            assert not predicate({"node_id": node}), (
                f"seed={BASE_SEED + 10_000 + i}: {node} is outside the permitted set "
                f"but the filter accepted it"
            )


def test_the_generator_actually_produces_the_shapes_being_tested() -> None:
    """A property test over a generator that only ever emits one trivial shape
    passes for the wrong reason, and there is no way to tell from the result.
    """
    depths, grant_counts, chain_lengths = set(), set(), set()
    for i in range(ITERATIONS):
        rng = random.Random(BASE_SEED + i)
        levels, all_ids, chain, granted, permitted = _build_world(rng)
        depths.add(len(levels))
        grant_counts.add(len(granted))
        chain_lengths.add(len(chain))
    assert len(depths) >= 4, f"ladder depth barely varies: {sorted(depths)}"
    assert max(grant_counts) >= 2, f"grants barely vary: {sorted(grant_counts)}"
    assert len(chain_lengths) >= 4, f"requesters all sit at one depth: {sorted(chain_lengths)}"
    assert 0 in grant_counts, "no iteration exercised the no-grant case"


# ── cross-instance isolation, which the tests above never touched ──────────
#
# The property tests above pass `visibility_allowed=None` on every iteration, so
# the clause that keeps one instance's records out of another instance's reads
# was never exercised by any of them. Deleting that clause from the builder
# entirely left all seven of them passing, which is how this gap was found.
#
# A leak across instances is worse than a leak across nodes: nodes belong to one
# client, instances can belong to different products of the same client and are
# the boundary a viewer policy is written against.

def _worlds_with_instances(rng: random.Random):
    """A tree, a chain, and a set of instances the viewer may see.

    ⚠ Two things this got wrong, both found by audit 2026-08-23.

    `allowed` was never empty, so the `NothingReadable` path was unreachable
    from here. A viewer restricted to no instance at all is the strictest case
    the clause has, and it was untested from this direction.

    And `allowed` was often the whole set: only 176 of 400 iterations had it as
    a proper subset, so barely half could catch a cross-instance leak at all.
    A generator that produces the trivial case half the time is a generator
    reporting twice the coverage it has.
    """
    levels, all_ids, chain, granted, permitted = _build_world(rng)
    n_inst = rng.randint(2, 5)
    instances = [f"inst_{i}_{rng.randint(1000, 9999)}" for i in range(n_inst)]
    # A proper subset every time, and empty one iteration in twenty so the
    # refusal path is exercised too.
    if rng.random() < 0.05:
        allowed: list[str] = []
    else:
        allowed = rng.sample(instances, rng.randint(1, len(instances) - 1))
    return all_ids, chain, permitted, instances, allowed


@pytest.mark.parametrize("include_legacy", [False, True])
def test_a_record_from_a_forbidden_instance_is_never_returned(
    include_legacy: bool,
) -> None:
    for i in range(ITERATIONS):
        seed = BASE_SEED + 90000 + i
        rng = random.Random(seed)
        all_ids, chain, permitted, instances, allowed = _worlds_with_instances(rng)
        tenant = chain[1] if len(chain) >= 2 else None
        identity = chain[-1] if len(chain) >= 2 else None
        try:
            predicate = to_python_predicate(build_scope_filter(
                sorted(permitted), tenant_node_id=tenant, identity_node_id=identity,
                include_legacy=include_legacy, visibility_allowed=allowed,
            ))
        except NothingReadable:
            continue

        for _ in range(12):
            owner = rng.choice(all_ids)
            inst = rng.choice(instances)
            meta = {"node_id": owner, "instance_id": inst}
            if predicate(meta):
                assert inst in allowed, (
                    f"seed={seed} legacy={include_legacy}: CROSS-INSTANCE LEAK. "
                    f"A record from instance {inst} was returned to a viewer "
                    f"restricted to {sorted(allowed)}.\n  metadata={meta}"
                )
                assert owner in permitted, (
                    f"seed={seed}: the instance clause admitted a record whose "
                    f"owner {owner} is outside the permitted set"
                )


def test_a_permitted_record_in_an_allowed_instance_still_comes_back() -> None:
    """The other direction. An isolation clause that returns nothing is not
    isolation, it is an outage, and only checking for leaks cannot tell them
    apart."""
    seen = 0
    for i in range(ITERATIONS):
        seed = BASE_SEED + 91000 + i
        rng = random.Random(seed)
        all_ids, chain, permitted, instances, allowed = _worlds_with_instances(rng)
        tenant = chain[1] if len(chain) >= 2 else None
        identity = chain[-1] if len(chain) >= 2 else None
        try:
            predicate = to_python_predicate(build_scope_filter(
                sorted(permitted), tenant_node_id=tenant, identity_node_id=identity,
                include_legacy=False, visibility_allowed=allowed,
            ))
        except NothingReadable:
            continue
        node = rng.choice(sorted(permitted))
        meta = {"node_id": node, "instance_id": rng.choice(allowed)}
        assert predicate(meta), (
            f"seed={seed}: a permitted record in an allowed instance was refused.\n"
            f"  metadata={meta}"
        )
        seen += 1
    assert seen > ITERATIONS // 2, (
        f"only {seen} of {ITERATIONS} iterations produced a readable case; the "
        "generator is not exercising what this test claims to check"
    )


def test_the_instance_generator_produces_the_shapes_it_claims_to() -> None:
    """A generator that quietly stops producing the interesting case turns a
    property test into a slow way of asserting nothing. This is the same check
    the node generator already has, which this one lacked."""
    proper_subsets = empties = 0
    for i in range(ITERATIONS):
        rng = random.Random(BASE_SEED + 90000 + i)
        _all_ids, _chain, _permitted, instances, allowed = _worlds_with_instances(rng)
        if not allowed:
            empties += 1
        elif set(allowed) < set(instances):
            proper_subsets += 1

    assert proper_subsets > ITERATIONS * 0.8, (
        f"only {proper_subsets} of {ITERATIONS} iterations can catch a "
        "cross-instance leak; the rest allow every instance there is"
    )
    assert empties > 0, (
        "no iteration restricts the viewer to nothing, so the NothingReadable "
        "path is never reached from here"
    )


def test_a_viewer_allowed_no_instance_at_all_is_refused() -> None:
    """The strictest case the clause has. Reachable from the generator now, and
    pinned directly here so it does not depend on the generator's dice."""
    rng = random.Random(BASE_SEED)
    _levels, _all_ids, chain, _granted, permitted = _build_world(rng)
    with pytest.raises(NothingReadable):
        build_scope_filter(
            sorted(permitted),
            tenant_node_id=chain[1] if len(chain) >= 2 else None,
            identity_node_id=chain[-1] if len(chain) >= 2 else None,
            include_legacy=False,
            visibility_allowed=[],
        )
