"""The dashboard's "Deletion requests" queue (table ``deletion_requests``, migration 157).

A preview writes one pending row holding exactly the memories a person will be asked
to approve. Approving deletes those IDs and nothing else, so what was reviewed is what
goes. The candidate text is blanked once a request is decided, so the queue does not
keep a copy of what the user asked to forget.
"""

from __future__ import annotations

import json
import uuid as _uuid
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text

from synap.cloud.application.memory_forget.models import Candidate

PENDING, APPLIED, REJECTED, FAILED = "pending", "applied", "rejected", "failed"


async def create_pending(session, *, scope, query: str, candidates: Sequence[Candidate],
                         requested_by: Optional[str]) -> str:
    request_id = str(_uuid.uuid4())
    await session.execute(
        text("""
            INSERT INTO deletion_requests
                (id, client_id, instance_id, customer_id, user_id, scope_level,
                 query, candidates, status, requested_by)
            VALUES (CAST(:id AS uuid), CAST(:client_id AS uuid), CAST(:instance_id AS uuid),
                    CAST(:customer_id AS uuid), CAST(:user_id AS uuid), :scope_level,
                    :query, CAST(:candidates AS jsonb), 'pending', :requested_by)
        """),
        {"id": request_id, "client_id": str(scope.client_id),
         "instance_id": str(scope.instance_id) if scope.instance_id else None,
         "customer_id": str(scope.customer_id) if scope.customer_id else None,
         "user_id": str(scope.user_id) if scope.user_id else None,
         "scope_level": getattr(scope, "scope_level", None) or "user", "query": query,
         "candidates": json.dumps([c.as_dict() for c in candidates]),
         "requested_by": requested_by},
    )
    return request_id


async def record_direct(session, *, scope, query: str, candidates: Sequence[Candidate],
                        requested_by: Optional[str], status: str,
                        result: Dict[str, Any]) -> str:
    """File a direct forget as an already-decided request, so it has an id the
    caller gets back and shows in the dashboard queue beside approved ones.
    Like a decided request, it keeps the ids and stores, not the text."""
    request_id = str(_uuid.uuid4())
    await session.execute(
        text("""
            INSERT INTO deletion_requests
                (id, client_id, instance_id, customer_id, user_id, scope_level,
                 query, candidates, status, requested_by, decided_by, decided_at, result)
            VALUES (CAST(:id AS uuid), CAST(:client_id AS uuid), CAST(:instance_id AS uuid),
                    CAST(:customer_id AS uuid), CAST(:user_id AS uuid), :scope_level,
                    :query, CAST(:candidates AS jsonb), :status, :requested_by,
                    :requested_by, NOW(), CAST(:result AS jsonb))
        """),
        {"id": request_id, "client_id": str(scope.client_id),
         "instance_id": str(scope.instance_id) if scope.instance_id else None,
         "customer_id": str(scope.customer_id) if scope.customer_id else None,
         "user_id": str(scope.user_id) if scope.user_id else None,
         "scope_level": getattr(scope, "scope_level", None) or "user", "query": query,
         "candidates": json.dumps([{"memory_id": c.memory_id, "store": c.store}
                                   for c in candidates]),
         "status": status, "requested_by": requested_by, "result": json.dumps(result)},
    )
    return request_id


async def get_request(session, *, client_id: str, request_id: str) -> Optional[Dict[str, Any]]:
    row = (await session.execute(
        text("""
            SELECT id, client_id, instance_id, customer_id, user_id, scope_level, query,
                   candidates, status, requested_by, decided_by, decided_at, result, created_at
              FROM deletion_requests
             WHERE id = CAST(:id AS uuid) AND client_id = CAST(:client_id AS uuid)
        """),
        {"id": request_id, "client_id": client_id},
    )).mappings().first()
    return _shape(row) if row else None


async def list_requests(session, *, client_id: str, status: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
    rows = (await session.execute(
        text("""
            SELECT id, client_id, instance_id, customer_id, user_id, scope_level, query,
                   candidates, status, requested_by, decided_by, decided_at, result, created_at
              FROM deletion_requests
             WHERE client_id = CAST(:client_id AS uuid)
               AND (CAST(:status AS text) IS NULL OR status = :status)
             ORDER BY created_at DESC
             LIMIT :limit
        """),
        {"client_id": client_id, "status": status, "limit": limit},
    )).mappings().all()
    return [_shape(r) for r in rows]


async def claim_pending(session, *, client_id: str, request_id: str,
                        decided_by: str, new_status: str) -> bool:
    """Move a pending request to ``new_status`` once. False when it is not pending.

    The status check is in the UPDATE itself, so two people approving the same
    request cannot both win.
    """
    row = await session.execute(
        text("""
            UPDATE deletion_requests
               SET status = :new_status, decided_by = :decided_by, decided_at = NOW()
             WHERE id = CAST(:id AS uuid) AND client_id = CAST(:client_id AS uuid)
               AND status = 'pending'
         RETURNING id
        """),
        {"id": request_id, "client_id": client_id, "decided_by": decided_by,
         "new_status": new_status},
    )
    return row.first() is not None


async def finish(session, *, request_id: str, status: str, result: Dict[str, Any]) -> None:
    """Record the outcome and blank the candidate text."""
    await session.execute(
        text("""
            UPDATE deletion_requests
               SET status = :status, result = CAST(:result AS jsonb),
                   candidates = (
                       SELECT COALESCE(jsonb_agg(jsonb_build_object(
                                  'memory_id', c->>'memory_id',
                                  'store', COALESCE(c->>'store', 'memory'))), '[]'::jsonb)
                         FROM jsonb_array_elements(candidates) AS c)
             WHERE id = CAST(:id AS uuid)
        """),
        {"id": request_id, "status": status, "result": json.dumps(result)},
    )


def _shape(row) -> Dict[str, Any]:
    d = dict(row)
    for key in ("id", "client_id", "instance_id", "customer_id", "user_id"):
        if d.get(key) is not None:
            d[key] = str(d[key])
    cands = d.get("candidates")
    if isinstance(cands, str):
        cands = json.loads(cands)
    d["candidates"] = cands or []
    return d


def to_candidates(request: Dict[str, Any]) -> List[Candidate]:
    return [Candidate(memory_id=c["memory_id"], content=c.get("content") or "",
                      memory_type=c.get("memory_type"), store=c.get("store") or "memory")
            for c in request.get("candidates") or []]
