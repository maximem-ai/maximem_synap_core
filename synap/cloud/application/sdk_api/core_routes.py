"""Core versions of the credits, events and telemetry routes.

Same URLs as ``sdk_credits.py``, ``sdk_events.py`` and ``sdk_telemetry.py``, which
stay as they are for Cloud. Those three import the credits folder, the gRPC
package and the analytics folder at load time, none of which ship, so Core
includes this router instead of them.

* Credits and delete by scope: answer 404 "not available in the open-source server".
* Every other method and path under ``/v1/credits``, ``/v1/events`` and ``/v1/telemetry``
  answers the same 404 message, so a Cloud-only call never gets a bare "Not Found".
* Vendor import (`/v1/migrations`): answers 404 "not available yet".
* Events and telemetry: accept the request and discard it. Older SDK versions
  and the MCP server integration still send them, and a refusal would show up
  as repeated errors in the developer's logs. Nothing is stored or forwarded.

Every route still depends on ``get_sdk_auth``, so the optional token applies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth

NOT_AVAILABLE_CODE = "not_available_in_open_source"
CREDITS_MESSAGE = "Credits are not available in the open-source server."
DELETE_BY_SCOPE_MESSAGE = (
    "Delete by scope is not available in the open-source server. "
    "Delete one memory by id, or by description through forget."
)
MIGRATIONS_MESSAGE = "Vendor import is not available yet."
OTHER_CALL_MESSAGE = "This call is not available in the open-source server."

# Same cap as the Cloud route, so a batch that is refused there is refused here.
MAX_EVENTS_PER_BATCH = 500


def not_available_response(message: str = CREDITS_MESSAGE) -> JSONResponse:
    """The one answer for a feature the open-source server does not have.

    404, not 501: the Python SDK retries every 5xx and shows it as "service
    unavailable", and this is permanent."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": {"error": NOT_AVAILABLE_CODE, "message": message}},
    )


_ANY_METHOD = ["GET", "POST", "PUT", "PATCH", "DELETE"]


def _answer_every_other_call(router: APIRouter, message: str) -> None:
    """Make *router* answer every method on every path under its prefix with *message*.

    Must be called after the router's own routes are declared: a request goes to the
    first route that fits, so the routes above keep their own answers."""

    @router.api_route("", methods=_ANY_METHOD, include_in_schema=False)
    @router.api_route("/{path:path}", methods=_ANY_METHOD, include_in_schema=False)
    async def not_available(auth: SDKAuthContext = Depends(get_sdk_auth), path: str = ""):
        return not_available_response(message)


# ---------------------------------------------------------------------------
# Credits: /v1/credits
# ---------------------------------------------------------------------------

credits_router = APIRouter(prefix="/v1/credits", tags=["SDK Credits"])

_NOT_AVAILABLE_DOC = {404: {"description": "Not available in the open-source server"}}


@credits_router.get("/balance", responses=_NOT_AVAILABLE_DOC)
async def credits_balance(auth: SDKAuthContext = Depends(get_sdk_auth)):
    return not_available_response()


@credits_router.get("/ledger", responses=_NOT_AVAILABLE_DOC)
async def credits_ledger(auth: SDKAuthContext = Depends(get_sdk_auth)):
    return not_available_response()


@credits_router.post("/estimate", responses=_NOT_AVAILABLE_DOC)
async def credits_estimate(auth: SDKAuthContext = Depends(get_sdk_auth)):
    return not_available_response()


@credits_router.post("/redeem", responses=_NOT_AVAILABLE_DOC)
async def credits_redeem(auth: SDKAuthContext = Depends(get_sdk_auth)):
    return not_available_response()


_answer_every_other_call(credits_router, CREDITS_MESSAGE)


# ---------------------------------------------------------------------------
# Events: POST /v1/events/batch (accepted and discarded)
# ---------------------------------------------------------------------------

events_router = APIRouter(prefix="/v1/events", tags=["SDK Events"])


class _EventIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    event_id: Optional[str] = None


class EventBatchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    events: List[_EventIn] = Field(default_factory=list, max_length=MAX_EVENTS_PER_BATCH)


class EventResult(BaseModel):
    event_id: str = ""
    status: Literal["accepted", "duplicate", "rejected"]
    reason: str = ""


class EventBatchResponse(BaseModel):
    accepted: int
    duplicate: int
    rejected: int
    results: List[EventResult]


@events_router.post("/batch", response_model=EventBatchResponse)
async def ingest_event_batch(
    request: EventBatchRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
) -> EventBatchResponse:
    """Accept the batch and discard it. Every event is answered as accepted."""
    results = [
        EventResult(
            event_id=event.event_id or "",
            status="accepted",
            reason="discarded: the open-source server does not process stream events",
        )
        for event in request.events
    ]
    return EventBatchResponse(
        accepted=len(results), duplicate=0, rejected=0, results=results,
    )


_answer_every_other_call(events_router, OTHER_CALL_MESSAGE)


# ---------------------------------------------------------------------------
# Telemetry: POST /v1/telemetry/batch (accepted and discarded)
# ---------------------------------------------------------------------------

telemetry_router = APIRouter(prefix="/v1/telemetry", tags=["SDK Telemetry"])


class TelemetryBatch(BaseModel):
    model_config = ConfigDict(extra="allow")
    events: List[Dict[str, Any]] = Field(default_factory=list)
    batch_id: str = ""


class TelemetryBatchResponse(BaseModel):
    status: str = "accepted"
    batch_id: str
    events_received: int


@telemetry_router.post(
    "/batch",
    response_model=TelemetryBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_telemetry_batch(
    batch: TelemetryBatch,
    auth: SDKAuthContext = Depends(get_sdk_auth),
) -> TelemetryBatchResponse:
    """Accept the batch and discard it. Nothing is stored or forwarded."""
    return TelemetryBatchResponse(batch_id=batch.batch_id, events_received=len(batch.events))


_answer_every_other_call(telemetry_router, OTHER_CALL_MESSAGE)


# ---------------------------------------------------------------------------
# Delete by scope: /v1/data/delete (a paid feature)
# ---------------------------------------------------------------------------

data_router = APIRouter(prefix="/v1/data", tags=["SDK Data"])


@data_router.post("/delete", responses=_NOT_AVAILABLE_DOC)
async def delete_by_scope(auth: SDKAuthContext = Depends(get_sdk_auth)):
    return not_available_response(DELETE_BY_SCOPE_MESSAGE)


@data_router.get("/delete/{deletion_id}", responses=_NOT_AVAILABLE_DOC)
async def delete_by_scope_status(deletion_id: str, auth: SDKAuthContext = Depends(get_sdk_auth)):
    return not_available_response(DELETE_BY_SCOPE_MESSAGE)


# ---------------------------------------------------------------------------
# Vendor import: /v1/migrations/* (a fast-follower)
# ---------------------------------------------------------------------------

migrations_router = APIRouter(prefix="/v1/migrations", tags=["Migrations"])


@migrations_router.api_route("", methods=_ANY_METHOD, include_in_schema=False)
@migrations_router.api_route("/{path:path}", methods=_ANY_METHOD, include_in_schema=False)
async def vendor_import(auth: SDKAuthContext = Depends(get_sdk_auth), path: str = ""):
    return not_available_response(MIGRATIONS_MESSAGE)


# One router for all of them, for the file that assembles the Core route table.
router = APIRouter()
router.include_router(credits_router)
router.include_router(events_router)
router.include_router(telemetry_router)
router.include_router(data_router)
router.include_router(migrations_router)
