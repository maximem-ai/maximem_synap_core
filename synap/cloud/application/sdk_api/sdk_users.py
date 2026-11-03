"""SDK user routes — user-profile read surface (spec §5.4).

``GET /v1/users/{user_id}/profile`` returns the profile document the
post-ingestion merge maintains (ProfileStore read; 404 when absent). The
SDK maps this to ``sdk.user.get_profile(...)`` (PR-D).

PII posture (spec §5.5): the path ``user_id`` is a raw external identifier
(often a phone number for voice tenants). Log lines here carry only its
UUID5 pseudonym, never the raw value.
"""

import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.application.profile.store import ProfileStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/users", tags=["SDK Users"])


def get_profile_store() -> ProfileStore:
    """Dependency seam (overridable in tests, mirrors get_retrieval_manager)."""
    return ProfileStore()


class UserProfileResponse(BaseModel):
    user_id: str
    profile: Dict[str, Any]
    retrieved_at: datetime


@router.get("/{user_id}/profile", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: str,
    customer_id: Optional[str] = Query(
        None,
        description=(
            "External customer id. Required for B2B callers. NOT accepted on a "
            "B2C instance (user_context_isolation=equals_customer), where the "
            "user id is the whole identity; sending one there is rejected. "
            "GET /api/v1/auth/whoami reports the mode."
        ),
    ),
    auth: SDKAuthContext = Depends(get_sdk_auth),
    store: ProfileStore = Depends(get_profile_store),
) -> UserProfileResponse:
    """Return the stored profile document for one end user.

    Owner resolution mirrors the dashboard's B2C fallback: user row by
    (client_id, user_id) if it exists, else the customer row (B2C creates
    no user row — IdMapper returns the customer UUID).
    """
    from synap.cloud.shared.b2c_contract import check_customer_id
    await check_customer_id(auth.instance_id, customer_id,
                            where="GET /v1/users/{user_id}/profile", user_id=user_id)

    # Log-safe pseudonym of the external id (spec §5.5) — never log user_id raw.
    user_log_id = _uuid.uuid5(_uuid.NAMESPACE_URL, user_id)

    profile = await store.read_profile(
        client_id=auth.client_id,
        external_user_id=user_id,
        external_customer_id=customer_id,
    )
    if profile is None:
        logger.info(
            "sdk_user_profile_not_found client_id=%s instance_id=%s user_uuid5=%s correlation_id=%s",
            auth.client_id,
            auth.instance_id,
            user_log_id,
            auth.correlation_id,
        )
        # Core builds a profile when a conversation ends or goes idle, so a user whose
        # memories all came from memories.create has none yet. Say so, so it is not read as a fault.
        from synap.cloud.shared import edition
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=("No profile yet for this user. Synap Core builds a profile when a conversation "
                    "ends or goes idle: record the user's messages in a conversation and let it end. "
                    "Memories added with memories.create do not build one."
                    if edition.profile_at_conversation_end else "Profile not found for this user"),
        )

    logger.info(
        "sdk_user_profile_read client_id=%s instance_id=%s user_uuid5=%s version=%s correlation_id=%s",
        auth.client_id,
        auth.instance_id,
        user_log_id,
        (profile.get("_meta") or {}).get("version"),
        auth.correlation_id,
    )
    return UserProfileResponse(
        user_id=user_id,
        profile=profile,
        retrieved_at=datetime.now(timezone.utc),
    )
