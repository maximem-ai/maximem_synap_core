"""GET /api/v1/auth/whoami for Core.

Both SDKs call this when they start and read the client id, the instance id and
the scoping mode from the answer. Cloud's route lives in ``admin_api``, which
stays behind, so Core has its own here. The response has the same fields as
Cloud's.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from synap.cloud.application.sdk_api.core_identity import load_scoping_mode
from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth

router = APIRouter(tags=["keys"])


class WhoamiResponse(BaseModel):
    client_id: str
    instance_id: str
    credential_id: str
    # "equals_customer": send user_id only. "strict": customer_id is required.
    # null when it cannot be read, so the SDK does not assume one.
    user_context_isolation: Optional[str] = None


@router.get("/api/v1/auth/whoami", response_model=WhoamiResponse)
async def whoami(auth: SDKAuthContext = Depends(get_sdk_auth)) -> WhoamiResponse:
    """The fixed identity, and the scoping mode from the YAML."""
    return WhoamiResponse(
        client_id=auth.client_id,
        instance_id=auth.instance_id,
        credential_id=str(auth.credential_id),
        user_context_isolation=await load_scoping_mode(auth.instance_id),
    )
