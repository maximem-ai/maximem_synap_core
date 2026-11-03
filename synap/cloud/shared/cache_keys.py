"""
Cache key generation for multi-tier caching.

Key format: {prefix}:{client_id}:{customer_id}:{scope}:{hash}

Section 14 of the Anticipation Rebuild added the canonical ``ScopeKey``
dataclass below. New code should build a ``ScopeKey`` and call its derived
key methods; ``CacheKeyGenerator`` is preserved as a thin facade so existing
non-anticipation callers (e.g. ``retrieval_manager``) keep working.
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


def _chain_part(chain: Optional[Sequence[str]]) -> str:
    """The resolved chain as a key component, or "" when there is none.

    ⚠ The chain IS the scope, and neither key below carried it.

    `ScopeHierarchy.to_dict` used to argue that the rungs "are derived from ids
    already in the key, so carrying them would add length without
    distinguishing anything". That held while the legacy adapter was the only
    way to get a chain, because it derives one from exactly `customer_id` and
    `user_id`. It stopped holding when an explicit `scope` path arrived: a path
    resolves a chain those ids do NOT determine, so two requests with identical
    ids and different paths are two different people sharing one entry, and
    whichever arrives first serves the other for the whole TTL. No store is
    involved in that, so a correct filter underneath cannot prevent it.

    `RetrievalManager._build_cache_key` already folds the chain into the query
    component it passes here. This exists for the anticipation key, which has
    no query to fold anything into, and to make the property hold at the key
    builder itself rather than only at one of its callers.

    Empty when there is no chain, and that is deliberate rather than lazy: an
    empty component leaves the raw string byte-identical, so every L1 entry
    already in the fleet keeps its key. Almost no instance resolves a chain
    (only a handful of instances), so making this
    unconditional would cold-start the whole fleet's cache to distinguish
    requests that are not distinguishable.
    """
    if not chain:
        return ""
    return ":chain=" + ",".join(str(c) for c in chain)


def _payload_build() -> str:
    """The identity of the code that built a cached payload.

    An L1 entry holds a whole serialised bundle, every item's dates included,
    and the key had nothing in it that changes when the *meaning* of those
    fields changes. So a deploy reads back payloads its predecessor wrote and
    interprets them under the new rules.

    That is not hypothetical. When the read
    path stopped filling `event_date` with the conversation's own date,
    for the 300s TTL after the restart every cached bundle still carried one.
    The flag that used to mark such a date as borrowed had gone with the
    substitution, so the new reader saw an ordinary event date and printed
    "happened <date>" for a date nobody had ever stated: five minutes of
    asserting fabricated dates as fact, which is worse than the bug being
    fixed.

    Keyed on the running build rather than a schema number somebody has to
    remember to bump, because remembering is exactly what fails. The cost is
    one cold L1 per deploy, and L1 lives 300 seconds, so it is close to cold
    at all times anyway. Two builds serving at once during a rolling deploy
    simply keep separate namespaces.
    """
    return os.environ.get("SYNAP_BUILD_SHA") or "dev"


@dataclass(frozen=True)
class ScopeKey:
    """Canonical scope identifier used to derive cache keys.

    Section 14 — fixes the writer/reader fallback inconsistency that
    silently corrupted anticipation hit rates:
    - the writer in ``agent.warm_cache`` used ``""`` fallback for missing
      client/customer ids
    - the reader in ``graph.cache_check_node`` substituted ``instance_id``
    The two produced different Redis keys for the same logical entity.

    ``instance_id`` is required — including it in derived keys prevents
    collisions across multiple MACA-distinct instances under one client.
    ``mode`` defaults to ``"fast"``; that matches the dominant anticipation
    path. Empty strings are the canonical missing-value marker for every
    other field; ``conversation_id=None`` maps to the broad anticipation
    namespace where the explicit query is unknown.
    """

    instance_id: str
    client_id: str = ""
    customer_id: str = ""
    user_id: str = ""
    conversation_id: Optional[str] = None
    mode: str = "fast"
    # The resolved scope chain, root to leaf, as node ids. Empty for every
    # request that resolves no chain, which is nearly all of them. See
    # `_chain_part` for why an entry is a different entry when this differs.
    chain: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not self.instance_id:
            raise ValueError("ScopeKey requires instance_id")

    def l1_anticipation_key(self, context_types: List[str]) -> str:
        types_str = ",".join(sorted(context_types or []))
        conv_part = self.conversation_id or "any"
        user_part = self.user_id or "anonymous"
        raw = (
            f"b={_payload_build()}:"
            f"anticipation:{self.client_id}:{self.instance_id}:"
            f"{self.customer_id}:{user_part}:{conv_part}:{types_str}"
            f"{_chain_part(self.chain)}"
        )
        return f"synap:l1:anticipation:{CacheKeyGenerator._hash(raw)}"

    def l1_query_key(self, query: str, context_types: List[str]) -> str:
        types_str = ",".join(sorted(context_types or []))
        conv_part = self.conversation_id or "none"
        user_part = self.user_id or "anonymous"
        raw = (
            f"b={_payload_build()}:"
            f"{self.client_id}:{self.instance_id}:{self.customer_id}:"
            f"{user_part}:{conv_part}:{self.mode}:{query}:{types_str}"
            f"{_chain_part(self.chain)}"
        )
        return f"query:{self.client_id}:{self.customer_id}:{CacheKeyGenerator._hash(raw)}"


class CacheKeyGenerator:
    """
    Generate deterministic cache keys for retrieval caching.

    Tiers:
    - L0: Request cache (in-process)
    - L1: Query response cache (Redis, 5min)
    - L2: User data cache (Redis, 30min)
    - L3: Graph cache (Redis, 1hr)
    """

    # TTLs in seconds
    TTL_L0 = 60          # 1 minute
    TTL_L1 = 300         # 5 minutes
    TTL_L2 = 1800        # 30 minutes
    TTL_L3 = 3600        # 1 hour

    @staticmethod
    def _hash(value: str) -> str:
        """Generate short hash for cache key."""
        return hashlib.md5(value.encode()).hexdigest()[:12]

    @classmethod
    def l0_request_key(cls, request_id: str) -> str:
        """L0: Request deduplication key."""
        return f"request_cache:{request_id}"

    @classmethod
    def l1_query_key(
        cls,
        client_id: str,
        customer_id: str,
        user_id: Optional[str],
        query: str,
        context_types: List[str],
        conversation_id: Optional[str] = None,
        scope_level: Optional[str] = None,
        mode: Optional[str] = None,
        instance_id: str = "",
        chain: Optional[Sequence[str]] = None,
    ) -> str:
        """L1: Query response cache key.

        Same query + types + user + conversation + instance + mode = same
        key (cache hit). Section 14 added ``instance_id`` to the key
        composition; callers that don't pass it get the prior behavior
        (instance_id="") but lose the cross-instance collision guard. New
        callers should build a ``ScopeKey`` and call ``l1_query_key`` on it.
        """
        # Note: ``scope_level`` is preserved in the raw input for backwards
        # compatibility with cache entries pre-Section 14; new code should
        # leave it None.
        types_str = ",".join(sorted(context_types or []))
        user_part = user_id or "anonymous"
        conv_part = conversation_id or "none"
        scope_part = scope_level or "unknown"
        mode_part = mode or "fast"
        raw = (
            f"b={_payload_build()}:"
            f"{client_id}:{instance_id}:{customer_id}:{user_part}:{conv_part}:"
            f"{scope_part}:{mode_part}:{query}:{types_str}"
            f"{_chain_part(chain)}"
        )
        return f"query:{client_id}:{customer_id}:{cls._hash(raw)}"

    @classmethod
    def l1_anticipation_key(
        cls,
        client_id: str,
        customer_id: str,
        user_id: Optional[str],
        context_types: List[str],
        conversation_id: Optional[str] = None,
        instance_id: str = "",
        chain: Optional[Sequence[str]] = None,
    ) -> str:
        """L1: Broad anticipation cache key (no query component).

        Used for session-start prefetch and anticipatory bundles where the
        exact query won't match a subsequent explicit fetch. Section 14
        added ``instance_id`` to the key composition so two MACA-distinct
        instances under the same client/user/conv no longer collide.

        ``chain`` is the resolved scope chain. Pass it whenever the request
        resolved one: an anticipation bundle prefetched at one rung must not be
        served to a caller standing at another, and this key is the only thing
        between them. Omitting it reproduces the prior behaviour exactly.
        """
        types_str = ",".join(sorted(context_types or []))
        user_part = user_id or "anonymous"
        conv_part = conversation_id or "any"
        raw = (
            f"b={_payload_build()}:"
            f"anticipation:{client_id}:{instance_id}:{customer_id}:"
            f"{user_part}:{conv_part}:{types_str}"
            f"{_chain_part(chain)}"
        )
        return f"synap:l1:anticipation:{cls._hash(raw)}"

    @classmethod
    def l2_user_context_key(
        cls,
        client_id: str,
        customer_id: str,
        user_id: str,
        context_type: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        L2: User data cache key.

        Caches raw context items for a user, scoped by client + customer.
        Includes conversation_id so each conversation has its own
        short-term context cache (prevents cross-conversation contamination).
        """
        base = f"user_context:{client_id}:{customer_id}:{user_id}"
        if conversation_id:
            base = f"{base}:{conversation_id}"
        if context_type:
            return f"{base}:{context_type}"
        return base

    @classmethod
    def l3_graph_data_key(cls, client_id: str, customer_id: str, user_id: str) -> str:
        """L3: Graph data cache key (MessagePack binary)."""
        return f"graph_v2:data:{client_id}:{customer_id}:{user_id}"

    @classmethod
    def l3_graph_meta_key(cls, client_id: str, customer_id: str, user_id: str) -> str:
        """L3: Graph metadata cache key (for staleness check)."""
        return f"graph_v2:meta:{client_id}:{customer_id}:{user_id}"

    @classmethod
    def get_ttl(cls, tier: str) -> int:
        """Get TTL for cache tier."""
        return {
            "L0": cls.TTL_L0,
            "L1": cls.TTL_L1,
            "L2": cls.TTL_L2,
            "L3": cls.TTL_L3,
        }.get(tier, cls.TTL_L1)
