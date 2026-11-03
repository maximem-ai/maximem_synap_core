"""SDK authentication dependency for FastAPI routes.

Validates SDK requests using API keys. The Bearer token is hashed (SHA256)
and looked up in the api_key_hash column of the credential table.
"""

import hashlib

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.database.connection import get_async_session
from synap.cloud.logging_config import set_correlation_context

API_KEY_PREFIX = "synap_"

# ---------------------------------------------------------------------------
# In-process credential cache. Every SDK request paid one credential lookup
# (a cross-cloud round trip) for a row that effectively never
# changes between requests. Successful ACTIVE lookups are cached for a short
# TTL keyed by the token's SHA256; ``expires_at`` is still enforced on every
# request from the cached value. Consequence: a revoked/suspended key stays
# usable for up to the TTL on a given worker process — accepted trade-off,
# keep the TTL short. Failed lookups are never cached (an invalid-key flood
# still hits the DB, and a just-created key works immediately).
# ---------------------------------------------------------------------------
_AUTH_CACHE_TTL_SECONDS = 60.0
_AUTH_CACHE_MAX_ENTRIES = 10_000

# api_key_hash -> (cached_at_monotonic, client_id, instance_id, credential_id,
#                  expires_at, pii_reveal_grant)
#
# The grant is cached with the rest of the credential rather than looked up
# separately, because a fetch that read the grant from the database every time
# would put a query on the hot path to enforce something that changes when
# somebody edits a key. It goes stale for up to the same TTL as everything else
# here, and narrowing a key's grant therefore takes effect within a minute
# rather than instantly. That is stated so nobody is surprised by it.
_auth_cache: Dict[str, Tuple[float, str, str, uuid.UUID, Optional[datetime], str]] = {}


def clear_sdk_auth_cache() -> None:
    """Drop all cached credentials (tests; ops escape hatch)."""
    _auth_cache.clear()


def _auth_cache_get(api_key_hash: str):
    entry = _auth_cache.get(api_key_hash)
    if entry is None:
        return None
    if time.monotonic() - entry[0] > _AUTH_CACHE_TTL_SECONDS:
        _auth_cache.pop(api_key_hash, None)
        return None
    return entry


def _auth_cache_put(
    api_key_hash: str,
    client_id: str,
    instance_id: str,
    credential_id: uuid.UUID,
    expires_at: Optional[datetime],
    pii_reveal_grant: str = "full",
) -> None:
    if len(_auth_cache) >= _AUTH_CACHE_MAX_ENTRIES:
        _auth_cache.clear()
    _auth_cache[api_key_hash] = (
        time.monotonic(), client_id, instance_id, credential_id, expires_at,
        pii_reveal_grant,
    )


@dataclass
class SDKAuthContext:
    """Authenticated SDK context extracted from credential lookup."""
    client_id: str
    instance_id: str
    credential_id: uuid.UUID
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # What this key receives for protected field types: full, masked or none.
    # Can only narrow what the client's policy allows (PII Protection 5.1).
    pii_reveal_grant: str = "full"
    # The caller's address, resolved through Cloudflare and nginx at the one point
    # where the HTTP request is still in scope. A rate-limit rejection happens several
    # layers below this, holding an auth context and no Request, so without carrying it
    # here "50 rejections from one IP" could never be answered. Not used for any
    # decision, only for attribution in the log line the signal counts.
    client_ip: str = "unknown"


async def get_sdk_auth(
    request: Request,
    authorization: str = Header(..., description="Bearer {api_key}"),
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
    session: AsyncSession = Depends(get_async_session),
) -> SDKAuthContext:
    """FastAPI dependency that authenticates SDK requests via API key.

    After successful authentication, runs the credit gate inline
    (imported lazily to avoid a circular dependency). The gate is a
    no-op when ``SYNAP_CREDITS_ENABLED=false`` or the wallet's
    enforcement_mode is ``disabled``/``observe``.

    Returns:
        SDKAuthContext with client_id, instance_id, credential_id, correlation_id.

    Raises:
        HTTPException 401: Missing/invalid/expired/revoked credential.
        HTTPException 402: Credit balance insufficient (enforce mode only).
    """
    # Imported here, not at the top of the file: these folders do not ship in
    # Core, which never runs this function (it installs its own dependency).
    from synap.cloud.observability.request_fields import client_ip, log_auth_failure

    if not authorization.startswith("Bearer "):
        log_auth_failure(request, status=401, reason="no_credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected: Bearer {token}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[7:].strip()
    if not token:
        log_auth_failure(request, status=401, reason="no_credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing token in authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key_hash = hashlib.sha256(token.encode()).hexdigest()

    cached = _auth_cache_get(api_key_hash)
    if cached is not None:
        _, client_id, instance_id, credential_id, expires_at, pii_reveal_grant = cached
        if expires_at and expires_at < datetime.now(timezone.utc):
            _auth_cache.pop(api_key_hash, None)
            log_auth_failure(request, status=401, reason="expired",
                             key_id=api_key_hash[:12], client_id=client_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credential has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        from synap.cloud.auth.models.orm import CredentialDBStatus
        from synap.cloud.database.repositories.credential_repo import CredentialRepository

        repo = CredentialRepository(session)
        credential = await repo.get_by_api_key_hash(api_key_hash)

        if not credential:
            log_auth_failure(request, status=401, reason="unknown_key",
                             key_id=api_key_hash[:12])
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if credential.status != CredentialDBStatus.ACTIVE:
            log_auth_failure(request, status=401, reason="revoked",
                             key_id=api_key_hash[:12],
                             client_id=credential.client_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Credential is {credential.status.value}",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if credential.expires_at and credential.expires_at < datetime.now(timezone.utc):
            log_auth_failure(request, status=401, reason="expired",
                             key_id=api_key_hash[:12],
                             client_id=credential.client_id)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credential has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        client_id = credential.client_id
        instance_id = credential.instance_id
        credential_id = credential.credential_id
        pii_reveal_grant = getattr(credential, "pii_reveal_grant", None) or "full"
        try:
            from synap.cloud.application.lifecycle.instance_readiness import note_credential_age_hint

            # Free age hint for the readiness gate: the instance is at least as old as its key.
            note_credential_age_hint(instance_id, credential)
        except Exception:  # noqa: BLE001 - a hint, never a reason to fail auth
            pass
        _auth_cache_put(
            api_key_hash, client_id, instance_id, credential_id, credential.expires_at,
            pii_reveal_grant,
        )

    auth_context = SDKAuthContext(
        client_id=client_id,
        instance_id=instance_id,
        credential_id=credential_id,
        correlation_id=x_correlation_id or str(uuid.uuid4()),
        pii_reveal_grant=pii_reveal_grant,
        client_ip=client_ip(request),
    )
    # The read path reads this rather than being handed it, because the
    # retrieval manager is reached from several callers and threading a grant
    # through all of them is how one caller ends up not threading it.
    # Imported here, not at the top of the file: the privacy package does not
    # ship in Core, and Core never runs this function (it installs its own
    # dependency in its place), so the module must import without that package.
    # Deliberately not wrapped in try/except: in Cloud a broken privacy import
    # must stop the request, not quietly skip the grant.
    from synap.cloud.privacy.reveal_grants import set_credential_grant

    set_credential_grant(pii_reveal_grant)
    set_correlation_context(
        correlation_id=auth_context.correlation_id,
        instance_id=auth_context.instance_id,
        client_id=auth_context.client_id,
    )
    # Stash on request.state so downstream code (e.g. LoggingMiddleware's
    # request-payload capture) can read who the request was authenticated
    # as without rerunning the credential lookup.
    request.state.sdk_auth = auth_context

    # Credit gate. Lazy import breaks the import cycle between
    # sdk_auth and credits.enforcement; the gate itself decides whether
    # to raise based on per-wallet enforcement_mode + global config.
    try:
        from synap.cloud.application.credits.enforcement import (
            run_credit_gate,
        )
        await run_credit_gate(
            auth=auth_context, request=request, session=session,
        )
    except ImportError:  # pragma: no cover — credits module optional
        pass

    return auth_context


async def require_instance_ready(
    auth: SDKAuthContext = Depends(get_sdk_auth),
) -> SDKAuthContext:
    """Route dependency for runtime routes that write or read memory.

    Refuses with a retryable 409 (``instance_not_ready``) while the instance's
    memory architecture is still being built, instead of letting the request be
    processed with the default config. Rides on ``get_sdk_auth`` (cached per
    request), so a bad key still gets its 401 and the credit gate still runs
    first; this only adds the 409. See lifecycle/instance_readiness.py for when
    it does and does not refuse.
    """
    from synap.cloud.application.lifecycle.instance_readiness import check_instance_ready

    pending = await check_instance_ready(auth.instance_id)
    if pending is not None:
        raise pending
    return auth
