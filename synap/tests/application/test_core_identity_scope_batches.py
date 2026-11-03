"""Scope refusal on the real batch request models.

Separate from test_core_identity.py because these import sdk_conversations and
sdk_memories, which do not import in a Core tree; the sync drops this file there
and keeps the Core-safe one.
"""
from __future__ import annotations

from fastapi import Depends
from fastapi.testclient import TestClient

from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.application.sdk_api.sdk_conversations import RecordBatchRequest
from synap.cloud.application.sdk_api.sdk_memories import BatchCreateRequest
from synap.tests.application.test_core_identity import _core_app, _no_token  # noqa: F401


def _batch_app(monkeypatch):
    app = _core_app(monkeypatch)

    @app.post("/v1/conversations/record/batch")
    async def rec(body: RecordBatchRequest, auth: SDKAuthContext = Depends(get_sdk_auth)):
        return {"n": len(body.messages)}

    @app.post("/api/v1/memories/batch")
    async def mem(body: BatchCreateRequest, auth: SDKAuthContext = Depends(get_sdk_auth)):
        return {"n": len(body.documents)}

    return TestClient(app)


def _msg(**kw):
    return {"conversation_id": "c", "role": "user", "content": "hi", "user_id": "u", **kw}


def _doc(**kw):
    return {"document": "text", "user_id": "u", **kw}


def test_scope_in_a_record_batch_item_is_refused(monkeypatch):
    c = _batch_app(monkeypatch)
    ok = c.post("/v1/conversations/record/batch", json={"messages": [_msg(), _msg(scope=None)]})
    assert ok.status_code == 200 and ok.json() == {"n": 2}
    bad = c.post("/v1/conversations/record/batch",
                 json={"messages": [_msg(), _msg(scope={"customer": "acme"})]})
    assert bad.status_code == 400


def test_scope_in_a_memory_batch_document_is_refused(monkeypatch):
    c = _batch_app(monkeypatch)
    assert c.post("/api/v1/memories/batch", json={"documents": [_doc()]}).status_code == 200
    bad = c.post("/api/v1/memories/batch",
                 json={"documents": [_doc(), _doc(scope={"team": "payments"})]})
    assert bad.status_code == 400


