"""Regression tests for user-scoped dedup candidate filtering.

Guards the multi-tenant isolation fix: the knowledge-update candidate search
must restrict ChromaDB candidates to the new memory's own owner scope. Without
it, a second user ingesting content similar to a first user has every memory
classified NONE/SUPERSEDE against the *other* user's vectors and silently
dropped from the vector index.
"""

from types import SimpleNamespace

from synap.cloud.agents.ingestion.pipeline.knowledge_update import (
    _build_dedup_scope_filter,
)
from synap.cloud.shared.scope import CUSTOMER_SCOPE_SENTINEL


def test_user_scope_rolls_up_to_customer_shared():
    """User-scope ingest: own memories + customer-shared, never other users'."""
    scope = SimpleNamespace(user_id="user-uuid", customer_id="cust-uuid")
    assert _build_dedup_scope_filter(scope) == {
        "user_id": {"$in": ["user-uuid", CUSTOMER_SCOPE_SENTINEL]}
    }


def test_customer_scope_sees_only_shared():
    """Customer-scope ingest: only customer-shared docs, no individual users'."""
    scope = SimpleNamespace(user_id=None, customer_id="cust-uuid")
    assert _build_dedup_scope_filter(scope) == {"user_id": CUSTOMER_SCOPE_SENTINEL}


def test_client_scope_has_no_user_filter():
    """Client-scope ingest: namespace already isolates the client collection."""
    scope = SimpleNamespace(user_id=None, customer_id=None)
    assert _build_dedup_scope_filter(scope) is None


def test_two_users_get_disjoint_filters():
    """The core bug: two users with identical content must not share candidates."""
    a = _build_dedup_scope_filter(SimpleNamespace(user_id="alpha", customer_id="c"))
    b = _build_dedup_scope_filter(SimpleNamespace(user_id="bravo", customer_id="c"))
    assert a != b
    assert "alpha" in a["user_id"]["$in"] and "alpha" not in b["user_id"]["$in"]
