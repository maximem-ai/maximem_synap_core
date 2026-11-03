"""Core's identity: one fixed client, one fixed instance, no key lookup.

Cloud authenticates each SDK request by its API key (``sdk_auth.get_sdk_auth``).
Core has no keys: the server trusts whoever can reach it, and every
request is the same client and the same instance. This module supplies:

* ``get_core_sdk_auth``: a dependency with the same return shape as
  ``get_sdk_auth`` (``SDKAuthContext``), filled with fixed values, so routes and
  engine code do not change.
* ``install_edition_identity(app)``: swaps it in with FastAPI's
  ``dependency_overrides``. The choice is made by the ``identity`` value in the
  edition module. In Cloud it does nothing, so ``main.py`` can call it with no
  edition check of its own.
* The optional ``SYNAP_API_TOKEN``: one bearer token for the whole
  install, not a user identity. Unset means open, with a loud startup warning.
* The refusal of ``scope_path``.

Nothing here imports the credits, gRPC, analytics or privacy packages.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import uuid
from typing import Any, Optional

from fastapi import Header, HTTPException, Request, status

from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.shared import edition

logger = logging.getLogger(__name__)

API_TOKEN_ENV = "SYNAP_API_TOKEN"

# The one client and the one instance every Core request belongs to. The seeding
# function creates the rows with these ids. The ids are plain strings: the
# engine treats them as opaque and derives its internal uuid5 values from them.
DEFAULT_CLIENT_ID = "cli_core_default"
DEFAULT_INSTANCE_ID = "inst_core_default"
# A fixed credential id, so the "credential" in logs is stable across restarts.
DEFAULT_CREDENTIAL_ID = uuid.uuid5(uuid.NAMESPACE_URL, "synap-core:default-credential")

# Core has no PII package, so there is nothing to narrow.
CORE_PII_REVEAL_GRANT = "full"

# Request fields that carry a scope path.
_SCOPE_FIELDS = ("scope", "scope_path")


# ---------------------------------------------------------------------------
# The optional token
# ---------------------------------------------------------------------------

def configured_api_token() -> str:
    """The token the operator set, or an empty string when none is set."""
    return (os.environ.get(API_TOKEN_ENV) or "").strip()


def warn_if_no_api_token() -> bool:
    """Log a loud warning when no token is set. Returns True when it warned."""
    if configured_api_token():
        logger.info("%s is set: every request must carry it as a bearer token.", API_TOKEN_ENV)
        return False
    banner = "=" * 72
    logger.warning(
        "\n%s\n"
        "SYNAP CORE IS RUNNING WITH NO AUTHENTICATION.\n"
        "%s is not set, so anyone who can reach this port can read, write and\n"
        "delete every memory. Keep the server off the public network, or set\n"
        "%s to a long random value and send it as a bearer token.\n"
        "%s",
        banner, API_TOKEN_ENV, API_TOKEN_ENV, banner,
    )
    return True


def check_api_token(authorization: Optional[str]) -> None:
    """Raise 401 unless the request carries the configured token. No token set: allow."""
    expected = configured_api_token()
    if not expected:
        return
    header = authorization or ""
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"This server requires a bearer token. Send 'Authorization: Bearer <{API_TOKEN_ENV}>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Constant-time comparison, on bytes so a non-ASCII header cannot raise.
    if not hmac.compare_digest(presented.strip().encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# scope_path is refused
# ---------------------------------------------------------------------------

SCOPE_PATH_REFUSAL = (
    "scope_path is not supported by the open-source server. Core stores a memory "
    "at the level of the request (the user_id and customer_id you send), so a "
    "scope path cannot be honoured, and ignoring it would widen what the call "
    "reads or writes. Remove `scope` from the request. Nested scopes are a "
    "Synap Cloud feature."
)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, str)):
        return bool(value) and (not isinstance(value, str) or bool(value.strip()))
    return True


def _json_has_scope(body: Any) -> bool:
    """Top level, plus one level down into lists of objects (messages, documents, events)."""
    if not isinstance(body, dict):
        return False
    if any(_present(body.get(name)) for name in _SCOPE_FIELDS):
        return True
    for value in body.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and any(_present(item.get(n)) for n in _SCOPE_FIELDS):
                    return True
    return False


async def refuse_scope_path(request: Request) -> None:
    """Raise 400 when the request carries a non-empty scope path.

    Looks at the query string, at ``scope`` / ``scope_path`` in a JSON body (top
    level and inside each item of a list such as ``messages`` or ``documents``)
    and at the same names in a form body. The body is read through Starlette's cache, which
    FastAPI has already filled by the time dependencies run, so the route still
    sees it.
    """
    # Delete by scope has a `scope` argument of its own ("user", "conversation", "customer"), which is not a
    # scope path. Core answers that route "not available"; it must not be refused first for a
    # field it needs.
    if request.url.path.startswith("/v1/data/"):
        return
    content_type = (request.headers.get("content-type") or "").lower()
    found = any(_present(request.query_params.get(n)) for n in _SCOPE_FIELDS)
    try:
        if found:
            pass
        elif "application/json" in content_type:
            raw = await request.body()
            if raw:
                body = json.loads(raw)
                found = _json_has_scope(body)
        elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            form = await request.form()
            found = any(_present(form.get(name)) for name in _SCOPE_FIELDS)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - an unreadable body is the route's 4xx to give, not ours
        logger.debug("could not inspect the request body for a scope path", exc_info=True)
        return
    if found:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=SCOPE_PATH_REFUSAL)


# ---------------------------------------------------------------------------
# The dependency
# ---------------------------------------------------------------------------

async def get_core_sdk_auth(
    request: Request,
    authorization: Optional[str] = Header(None, description="Bearer {SYNAP_API_TOKEN}, only when the server has one"),
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
) -> SDKAuthContext:
    """Same return shape as ``get_sdk_auth``, with the fixed identity."""
    check_api_token(authorization)
    await refuse_scope_path(request)

    caller = request.client.host if request.client else "unknown"
    auth_context = SDKAuthContext(
        client_id=DEFAULT_CLIENT_ID,
        instance_id=DEFAULT_INSTANCE_ID,
        credential_id=DEFAULT_CREDENTIAL_ID,
        correlation_id=x_correlation_id or str(uuid.uuid4()),
        pii_reveal_grant=CORE_PII_REVEAL_GRANT,
        client_ip=caller,
    )
    try:
        from synap.cloud.logging_config import set_correlation_context

        set_correlation_context(
            correlation_id=auth_context.correlation_id,
            instance_id=auth_context.instance_id,
            client_id=auth_context.client_id,
        )
    except Exception:  # noqa: BLE001 - log context is never a reason to fail a request
        logger.debug("could not set the correlation context", exc_info=True)
    # Same as Cloud: downstream code reads who the request was from here.
    request.state.sdk_auth = auth_context
    return auth_context


def install_edition_identity(app: Any) -> bool:
    """Swap in the edition's identity dependency. Returns True when it did.

    Cloud (``identity == "key"``): does nothing, the route table and the auth
    dependency stay exactly as they are. Core (``identity == "fixed"``): every
    route that depends on ``get_sdk_auth`` gets ``get_core_sdk_auth`` instead,
    and the missing-token warning is logged once.
    """
    if edition.identity != "fixed":
        return False
    app.dependency_overrides[get_sdk_auth] = get_core_sdk_auth
    warn_if_no_api_token()
    return True


async def load_scoping_mode(instance_id: str = DEFAULT_INSTANCE_ID) -> Optional[str]:
    """The scoping mode (``user_context_isolation``), or None when it cannot be read.

    Reads it the way every other route does (``b2c_contract.load_isolation``):
    the instance's config first, which in Core is the YAML, then the seeded
    instance's record. None, never a guess: a wrong guess would refuse legal calls.
    """
    try:
        from synap.cloud.shared.b2c_contract import load_isolation

        return await load_isolation(instance_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "could not read the scoping mode for %s (%s: %s); answering null so "
            "the client does not assume one",
            instance_id, type(exc).__name__, exc,
        )
        return None
