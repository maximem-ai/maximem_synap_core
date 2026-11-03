"""Tests for Section 14 — canonical ScopeKey.

Covers:
- Required instance_id rejection
- Same logical scope → same key (the writer/reader consistency bug fixed)
- Different instance_id → different keys (cross-instance collision guard)
- Different conversation_id → different keys
- Different mode → different keys
- Anticipation key vs query key are distinct
- Legacy CacheKeyGenerator class methods accept instance_id and behave consistently
"""

import pytest


class TestScopeKeyConstruction:
    def test_requires_instance_id(self):
        from synap.cloud.shared.cache_keys import ScopeKey
        with pytest.raises(ValueError, match="instance_id"):
            ScopeKey(instance_id="")

    def test_defaults(self):
        from synap.cloud.shared.cache_keys import ScopeKey
        s = ScopeKey(instance_id="inst-1")
        assert s.client_id == ""
        assert s.customer_id == ""
        assert s.user_id == ""
        assert s.conversation_id is None
        assert s.mode == "fast"


class TestKeyDeterminism:
    def test_same_scope_yields_same_keys(self):
        from synap.cloud.shared.cache_keys import ScopeKey
        a = ScopeKey(instance_id="i1", client_id="c", customer_id="cu",
                      user_id="u", conversation_id="conv", mode="fast")
        b = ScopeKey(instance_id="i1", client_id="c", customer_id="cu",
                      user_id="u", conversation_id="conv", mode="fast")
        assert a.l1_anticipation_key(["facts"]) == b.l1_anticipation_key(["facts"])
        assert a.l1_query_key("q", ["facts"]) == b.l1_query_key("q", ["facts"])

    def test_different_instance_yields_different_keys(self):
        from synap.cloud.shared.cache_keys import ScopeKey
        a = ScopeKey(instance_id="i1", user_id="u", conversation_id="conv")
        b = ScopeKey(instance_id="i2", user_id="u", conversation_id="conv")
        assert a.l1_anticipation_key(["facts"]) != b.l1_anticipation_key(["facts"])
        assert a.l1_query_key("q", ["facts"]) != b.l1_query_key("q", ["facts"])

    def test_different_conversation_yields_different_keys(self):
        from synap.cloud.shared.cache_keys import ScopeKey
        a = ScopeKey(instance_id="i1", user_id="u", conversation_id="conv-A")
        b = ScopeKey(instance_id="i1", user_id="u", conversation_id="conv-B")
        assert a.l1_anticipation_key(["facts"]) != b.l1_anticipation_key(["facts"])
        assert a.l1_query_key("q", ["facts"]) != b.l1_query_key("q", ["facts"])

    def test_different_mode_yields_different_query_key(self):
        from synap.cloud.shared.cache_keys import ScopeKey
        a = ScopeKey(instance_id="i1", user_id="u", mode="fast")
        b = ScopeKey(instance_id="i1", user_id="u", mode="accurate")
        # Anticipation key has no mode, so they should be equal there.
        assert a.l1_anticipation_key(["facts"]) == b.l1_anticipation_key(["facts"])
        # Query key includes mode.
        assert a.l1_query_key("q", ["facts"]) != b.l1_query_key("q", ["facts"])

    def test_anticipation_and_query_keys_have_distinct_prefixes(self):
        from synap.cloud.shared.cache_keys import ScopeKey
        s = ScopeKey(instance_id="i1", user_id="u")
        anti = s.l1_anticipation_key(["facts"])
        query = s.l1_query_key("q", ["facts"])
        assert anti.startswith("synap:l1:anticipation:")
        assert query.startswith("query:")

    def test_context_types_order_insensitive(self):
        from synap.cloud.shared.cache_keys import ScopeKey
        s = ScopeKey(instance_id="i1", user_id="u")
        k1 = s.l1_anticipation_key(["facts", "preferences"])
        k2 = s.l1_anticipation_key(["preferences", "facts"])
        assert k1 == k2


class TestLegacyMethods:
    def test_l1_anticipation_key_with_instance_id_param(self):
        from synap.cloud.shared.cache_keys import CacheKeyGenerator, ScopeKey
        legacy = CacheKeyGenerator.l1_anticipation_key(
            client_id="c", customer_id="cu", user_id="u",
            context_types=["facts"], conversation_id="conv",
            instance_id="i1",
        )
        modern = ScopeKey(instance_id="i1", client_id="c", customer_id="cu",
                          user_id="u", conversation_id="conv").l1_anticipation_key(["facts"])
        assert legacy == modern

    def test_legacy_calls_without_instance_id_get_blank_namespace(self):
        from synap.cloud.shared.cache_keys import CacheKeyGenerator
        # Two callers that both omit instance_id end up in the same namespace
        # (instance_id defaults to "") — backwards compatible.
        a = CacheKeyGenerator.l1_anticipation_key(
            client_id="c", customer_id="cu", user_id="u",
            context_types=["facts"], conversation_id="conv",
        )
        b = CacheKeyGenerator.l1_anticipation_key(
            client_id="c", customer_id="cu", user_id="u",
            context_types=["facts"], conversation_id="conv",
        )
        assert a == b

    def test_legacy_with_explicit_instance_diverges_from_default(self):
        from synap.cloud.shared.cache_keys import CacheKeyGenerator
        # Different instances under the same client/user/conv must not collide
        # (the original bug).
        a = CacheKeyGenerator.l1_anticipation_key(
            client_id="c", customer_id="cu", user_id="u",
            context_types=["facts"], conversation_id="conv",
            instance_id="i1",
        )
        b = CacheKeyGenerator.l1_anticipation_key(
            client_id="c", customer_id="cu", user_id="u",
            context_types=["facts"], conversation_id="conv",
            instance_id="i2",
        )
        assert a != b


class TestWriterReaderConsistency:
    """Regression test for the original bug:
    writer used '' fallback, reader used instance_id fallback,
    keys diverged silently."""

    def test_both_sides_use_same_fallback(self):
        from synap.cloud.shared.cache_keys import ScopeKey
        # Simulate the warm_cache writer's construction
        writer_scope = ScopeKey(
            instance_id="i1",
            client_id="" or "",        # the writer's '' fallback
            customer_id="" or "",
            user_id="u",
            conversation_id="conv",
            mode="fast",
        )
        # Simulate the cache_check_node reader's construction
        reader_scope = ScopeKey(
            instance_id="i1",
            client_id="" or "",
            customer_id="" or "",
            user_id="u",
            conversation_id="conv",
            mode="fast",
        )
        assert writer_scope.l1_anticipation_key(["facts"]) == reader_scope.l1_anticipation_key(["facts"])
        assert writer_scope.l1_query_key("q", ["facts"]) == reader_scope.l1_query_key("q", ["facts"])
