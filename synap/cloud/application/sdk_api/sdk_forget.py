"""SDK route: forget by description.

    POST /api/v1/memories/forget   {"description": "my phone number", "user_id": "...",
                                    "mode": "preview" | "direct"}

A request that names no ``mode`` gets the edition's default (``edition.forget_default_mode``).

``preview`` finds the matches and files a pending request in the dashboard's
Deletion requests; nothing is deleted. ``direct`` finds and soft-deletes in one call
and skips the review. Always one user: a request that resolves to anything wider is
refused.

DELETE /api/v1/memories/{memory_id} is a different route and is unchanged.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.application.sdk_api.sdk_memories import _b2c_check, _resolve_scope
from synap.cloud.application.memory_forget import service
from synap.cloud.application.memory_forget.judge import JudgeNotConfigured, JudgeUnavailable
from synap.cloud.database.connection import get_async_session
from synap.cloud.shared import edition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/memories", tags=["SDK Memories"])


class ForgetRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=500,
                             description="What to forget, in the user's own words.")
    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    scope: Optional[Dict[str, str]] = None
    # None: the edition's default applies (edition.forget_default_mode). A mode that is sent is always honoured.
    mode: Optional[Literal["preview", "direct"]] = None
    # The chat the request was made in. Optional and additive. With it, a request that finds
    # nothing is also matched to the batch of that chat when it is saved, instead of waiting
    # for a timed retry. Only used by mode "direct".
    conversation_id: Optional[str] = Field(None, min_length=1, max_length=200)


class ForgetCandidateOut(BaseModel):
    memory_id: str
    content: str
    memory_type: Optional[str] = None
    # Where it lives: "memory", or a copy of its text kept elsewhere ("chunk",
    # "sentence", "entity", "triple", "profile", "candidate").
    store: str = "memory"


class ForgetResponse(BaseModel):
    # nothing_matches | preview | deleted | partial
    status: str
    request_id: Optional[str] = None
    candidates: List[ForgetCandidateOut] = Field(default_factory=list)
    deleted_ids: List[str] = Field(default_factory=list)
    failed_ids: List[str] = Field(default_factory=list)
    still_present_in: List[str] = Field(default_factory=list)
    # True when a search limit stopped the loop while it was still finding matches.
    may_be_incomplete: bool = False
    # A direct forget that matched nothing is searched for again later (the note may
    # still be being saved). Additive: clients that do not know the field ignore it.
    retry_scheduled: bool = False


@router.post("/forget", response_model=ForgetResponse)
async def forget_memories(
    request: ForgetRequest,
    auth: SDKAuthContext = Depends(get_sdk_auth),
    session: AsyncSession = Depends(get_async_session),
):
    await _b2c_check(auth, request.customer_id, where="POST /api/v1/memories/forget",
                     user_id=request.user_id)
    try:
        scope = await _resolve_scope(auth, request.user_id, request.customer_id, request.scope)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    if scope.scope_level != "user" or scope.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Forget by description is scoped to one user. Send a user_id "
                   "(and a customer_id for a B2B instance).")

    mode = request.mode or edition.forget_default_mode
    deps = service.build_deps(session, scope=scope, correlation_id=auth.correlation_id)
    try:
        if mode == "direct":
            conversation_id = None
            if request.conversation_id:
                from synap.cloud.shared.conversation_id_norm import coerce_conversation_id
                conversation_id = coerce_conversation_id(request.conversation_id)
            outcome = await service.direct(session, scope=scope,
                                           description=request.description, deps=deps,
                                           requested_by=f"sdk:{auth.credential_id}",
                                           conversation_id=conversation_id,
                                           user_id=request.user_id,
                                           customer_id=request.customer_id)
        else:
            outcome = await service.preview(session, scope=scope,
                                            description=request.description, deps=deps,
                                            requested_by=f"sdk:{auth.credential_id}")
    except JudgeNotConfigured:
        logger.warning("forget by description was asked for, and JEV_API_KEY is not set on the server; "
                       "nothing was deleted")
        # A 4xx, not a 503: the Python SDK retries every 5xx and reports it as "service unavailable", and a missing
        # key does not clear by retrying. 424 reaches the caller as a permanent error carrying this message.
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail="Forget by description needs a Jev API key. Set JEV_API_KEY on the server, "
                   "or delete a memory by its id, which needs no key. Nothing was deleted.")
    except JudgeUnavailable:
        # Nothing was deleted and no request was filed. Say so, so the caller does
        # not tell the user it was removed.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not decide which memories match, so nothing was deleted. "
                   "Try again.")

    return ForgetResponse(
        status=outcome.status, request_id=outcome.request_id,
        candidates=[ForgetCandidateOut(**c.as_dict()) for c in outcome.candidates],
        deleted_ids=outcome.deleted_ids, failed_ids=outcome.failed_ids,
        still_present_in=outcome.still_present_in,
        may_be_incomplete=outcome.may_be_incomplete,
        retry_scheduled=outcome.retry_scheduled)
