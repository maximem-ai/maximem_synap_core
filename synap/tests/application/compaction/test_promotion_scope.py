"""Compaction-promotion scope resolution (phantom-scope double-hash fix).

The retrieval-triggered compaction path
fires CompactionEvents with a MIXED identity pair — customer_id already
RESOLVED (uuid5) + user_id still external. The promotion callback re-hashed
the resolved customer (uuid5-of-uuid5) and, since external!=
resolved, failed the B2C user==customer check (two-level user uuid) — promoted memories
landed in a scope no retrieval ever reads.

Fix: resolve_compaction_promotion_scope prefers the conversation row's
EXTERNAL user_id/customer_id (the same source the tracker/worker trigger
paths pass) over the event fields, logging when they diverge.
"""

import uuid
from types import SimpleNamespace

import pytest

from synap.cloud.application.shared.providers import (
    resolve_compaction_promotion_scope,
)

CLIENT_EXT = "pg2_uber_client"
CLIENT_UUID = uuid.uuid5(uuid.NAMESPACE_URL, CLIENT_EXT)
EXT_ID = "cust_pgv2_0a2201ce23e04a72"           # external customer==user (B2C)
RESOLVED_CUST = uuid.uuid5(CLIENT_UUID, EXT_ID)  # what retrieval passes as customer_id
PHANTOM_CUST = uuid.uuid5(CLIENT_UUID, str(RESOLVED_CUST))  # the double-hash
PHANTOM_USER = uuid.uuid5(PHANTOM_CUST, EXT_ID)
CONV_ID = "pgv2_a96d562eeb504e3a842b3292e6a500c3"


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _FakeSession:
    """Dispatches on the SQL text: instance lookup vs conversation lookup."""

    def __init__(self, conversation_row=None, instance_client=None):
        self.conversation_row = conversation_row
        self.instance_client = instance_client

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        if "FROM instance" in sql:
            return _FakeResult(
                (self.instance_client,) if self.instance_client else None
            )
        if "FROM conversation" in sql:
            return _FakeResult(self.conversation_row)
        return _FakeResult(None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _wire_fakes(monkeypatch, session):
    import synap.cloud.database.connection as conn_mod
    import synap.cloud.shared.id_mapper as mapper_mod
    monkeypatch.setattr(conn_mod, "get_session_factory", lambda: (lambda: session))

    def _no_mapper():
        raise RuntimeError("id mapper unavailable in test")
    monkeypatch.setattr(mapper_mod, "get_id_mapper", _no_mapper)


def _event(customer_id, user_id, client_id=CLIENT_EXT):
    return SimpleNamespace(
        conversation_id=CONV_ID,
        instance_id="inst_pg2_uber_template",
        client_id=client_id,
        customer_id=customer_id,
        user_id=user_id,
    )


@pytest.mark.asyncio
class TestPromotionScopeResolution:
    async def test_prod_mixed_event_fixed_by_conversation_row(self, monkeypatch, caplog):
        """The mixed shape: event carries resolved customer + external
        user; the conversation row corrects both to external → single-hash
        customer, B2C user==customer. No phantom scope."""
        _wire_fakes(monkeypatch, _FakeSession(conversation_row=(EXT_ID, EXT_ID)))
        with caplog.at_level("WARNING"):
            scope = await resolve_compaction_promotion_scope(
                _event(customer_id=str(RESOLVED_CUST), user_id=EXT_ID)
            )
        assert str(scope.customer_id) == str(RESOLVED_CUST)
        assert str(scope.user_id) == str(RESOLVED_CUST)  # B2C: user == customer
        assert str(scope.customer_id) != str(PHANTOM_CUST)
        assert any("diverge" in r.message for r in caplog.records)

    async def test_consistent_event_unchanged_no_warning(self, monkeypatch, caplog):
        """Tracker/worker-shaped events (both ids external) resolve exactly as
        before and don't log divergence."""
        _wire_fakes(monkeypatch, _FakeSession(conversation_row=(EXT_ID, EXT_ID)))
        with caplog.at_level("WARNING"):
            scope = await resolve_compaction_promotion_scope(
                _event(customer_id=EXT_ID, user_id=EXT_ID)
            )
        assert str(scope.customer_id) == str(RESOLVED_CUST)
        assert str(scope.user_id) == str(RESOLVED_CUST)
        assert not any("diverge" in r.message for r in caplog.records)

    async def test_missing_conversation_row_falls_back_to_event_fields(self, monkeypatch):
        """No conversation row → legacy behavior (event fields verbatim). For
        a mixed event that still means the phantom scope — pinned here so the
        fallback shape is explicit and visible."""
        _wire_fakes(monkeypatch, _FakeSession(conversation_row=None))
        scope = await resolve_compaction_promotion_scope(
            _event(customer_id=str(RESOLVED_CUST), user_id=EXT_ID)
        )
        assert str(scope.customer_id) == str(PHANTOM_CUST)
        assert str(scope.user_id) == str(PHANTOM_USER)

    async def test_event_without_user_takes_row_user(self, monkeypatch):
        _wire_fakes(monkeypatch, _FakeSession(conversation_row=(EXT_ID, EXT_ID)))
        scope = await resolve_compaction_promotion_scope(
            _event(customer_id=str(RESOLVED_CUST), user_id=None)
        )
        assert str(scope.user_id) == str(RESOLVED_CUST)

    async def test_no_customer_anywhere_returns_none(self, monkeypatch):
        _wire_fakes(monkeypatch, _FakeSession(conversation_row=None))
        scope = await resolve_compaction_promotion_scope(
            _event(customer_id=None, user_id=None)
        )
        assert scope is None

    async def test_client_resolved_from_instance_when_event_lacks_it(self, monkeypatch):
        _wire_fakes(monkeypatch, _FakeSession(
            conversation_row=(EXT_ID, EXT_ID), instance_client=CLIENT_EXT,
        ))
        scope = await resolve_compaction_promotion_scope(
            _event(customer_id=EXT_ID, user_id=EXT_ID, client_id=None)
        )
        assert str(scope.client_id) == str(CLIENT_UUID)


class TestCallbackWiring:
    def test_promotion_uses_the_helper(self):
        """Promotion moved out of a post-compaction callback (compaction v3,
        plan §4.6); it still resolves scope through the one helper."""
        import inspect
        import synap.cloud.application.conversation.promotion as promotion
        src = inspect.getsource(promotion)
        assert "scope = await resolve_compaction_promotion_scope(event)" in src
        assert "resolve_customer" not in src
