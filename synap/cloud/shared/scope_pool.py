"""Make sure the requester's own material is in the pool at all.

Closeness ranking decides the order of what was fetched. It cannot do anything
about what was never fetched, and that is the failure this module exists for.

THE FAILURE
-----------
A requester deep in a ladder can read everything above them, and everything
above them is usually a far larger pool than their own node holds. One
similarity-ordered fetch of top_k across that whole readable set can come back
with nothing from the requester's own node, not because their material is
irrelevant but because there is a thousand times more material one rung up and
some of it is fractionally closer to the query.

Re-ranking cannot recover a record that was never fetched. The answer that
comes back is well ordered and wrong, which is worse than one that is visibly
wrong, because nothing about it looks broken.

THE FIX, AND WHY IT IS SAFE
---------------------------
Fetch twice, rank once. The main fetch is unchanged. A second, narrow fetch
asks the store for the requester's own node alone, and the two are merged
before the single relevance pass (R-09).

Adding candidates to a pool cannot make the final answer worse. Ranking still
runs once over the union and still returns top_k, so every record that would
have won before is still there and still scores the same. The only thing that
changes is that records which never got the chance to compete now do. That is
the whole argument, and it is why this needs no tuning to be safe: the reserve
size affects how much extra is considered, never whether the result is correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from synap.cloud.shared.scope_filter import (
    Predicate,
    build_scope_filter,
)

# How many slots the requester's own node is guaranteed a chance at.
#
# NOT MEASURED, and unlike the closeness cost it does not need to be for the
# result to be correct: a bigger number considers more of the leaf's material
# and a smaller one considers less, but neither can make the ranking wrong.
# What it trades is cost against the depth of the guarantee, so the sensible
# default is small. Three is enough that a leaf with anything relevant at all
# is represented, and cheap enough to spend on every request.
DEFAULT_LEAF_RESERVE = 3

# Above this the second fetch stops being a reserve and becomes a second
# search, at which point the caller should be asking for a leaf-scoped query
# rather than a top-up.
_MAX_LEAF_RESERVE = 25


@dataclass(frozen=True)
class PoolPlan:
    """Two fetches and the sizes to ask for. One ranking pass follows, elsewhere."""
    main: Predicate
    main_top_k: int
    leaf: Optional[Predicate]
    leaf_top_k: int

    @property
    def fetches(self) -> int:
        return 1 if self.leaf is None else 2


def leaf_reserve(top_k: int, requested: Optional[int] = None) -> int:
    """How many slots to reserve, clamped to something that stays a reserve."""
    reserve = DEFAULT_LEAF_RESERVE if requested is None else int(requested)
    reserve = max(0, min(reserve, _MAX_LEAF_RESERVE))
    # Never ask the reserve to be most of the answer: at that point the main
    # fetch is the thing being starved, which is the failure inverted.
    return min(reserve, max(0, top_k // 2))


def plan_pool(
    chain_node_ids: Sequence[str],
    top_k: int,
    *,
    reserve: Optional[int] = None,
    **filter_kwargs: Any,
) -> PoolPlan:
    """The two predicates and sizes for one request.

    The leaf fetch is skipped when it cannot help: a chain of one means the
    requester IS the root, so the main fetch already reads only their node and
    a second identical query would be pure cost.
    """
    if not chain_node_ids:
        # scope_filter raises NothingReadable for this, and it should reach the
        # caller from there rather than being second-guessed here.
        return PoolPlan(build_scope_filter(chain_node_ids, **filter_kwargs), top_k, None, 0)

    main = build_scope_filter(chain_node_ids, **filter_kwargs)
    reserve_k = leaf_reserve(top_k, reserve)
    if len(set(chain_node_ids)) <= 1 or reserve_k <= 0:
        return PoolPlan(main, top_k, None, 0)

    # R-17. The reserve narrows the SCOPE arm and nothing else, so it is built
    # through the same builder as the main fetch rather than from
    # `nested_scope_predicate` directly.
    #
    # This is not tidiness. `nested_scope_predicate` is scope only; it is
    # `build_scope_filter` that ANDs the cross-instance visibility clause and
    # the memory_type clause on top. A reserve built from the bare scope
    # predicate would carry neither, so it would return records from instances
    # the viewer is not allowed to see, on every request from a requester more
    # than one rung deep, and hand them back labelled as the requester's own
    # material. That is a leak, not a slow query.
    #
    # `include_legacy=False` is deliberate rather than incidental. Before the
    # metadata backfill the legacy arm matches exactly the requester's own
    # records, and the main fetch already carries it, so a reserve carrying it
    # too is a duplicate query that finds nothing new. After the backfill the
    # node id arm is the only one that is needed. Either way the legacy arm
    # belongs in the main fetch and not here.
    leaf_kwargs = dict(filter_kwargs)
    leaf_kwargs.pop("include_legacy", None)
    leaf = build_scope_filter(
        [chain_node_ids[-1]], include_legacy=False, **leaf_kwargs
    )
    return PoolPlan(main, top_k, leaf, reserve_k)


def merge_pool(*result_sets: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Union the fetches, keeping each record once at its best score.

    NOT called by the retrieval read path, on purpose. That path already unions
    its tasks when it flattens them into `all_search_results`, and already keeps
    one copy of each id at the `seen[key]` dedup in `retrieval_manager.py`.
    Calling this as well would mean flattening a nested structure the
    surrounding result accounting depends on, and would leave two dedup
    mechanisms doing one job. It stays because the questions and sentences
    consolidation under S-01 needs a real merge once the three collections
    become one.

    Deliberately does not rank. R-09: relevance is computed once over the
    assembled pool, and scores drawn from separately ranked lists are not
    comparable, so any ordering imposed here would be a lie the ranker would
    then have to undo.

    Order is preserved from the first set a record appears in, purely so the
    output is stable and diffable. It carries no meaning.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for results in result_sets:
        for record in results or ():
            if record is None:
                continue
            key = str(record.get("id") or "")
            if not key:
                continue
            existing = seen.get(key)
            if existing is None:
                seen[key] = dict(record)
                continue
            # The same record can come back from both fetches with different
            # scores when a store reports distance for one query shape and
            # similarity for another. Keeping the higher is the safe direction:
            # the ranker can demote it, but it cannot recover a record that was
            # dropped for looking worse than it is.
            if _score(record) > _score(existing):
                seen[key] = dict(record)
    return list(seen.values())


def _score(record: Mapping[str, Any]) -> float:
    try:
        return float(record.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0
