"""``require_instance_ready``: the route dependency that refuses requests while an
instance's memory architecture is still being built.

The check itself is the ``instance_readiness_check`` hook. Cloud registers
``lifecycle.instance_readiness.check_instance_ready``; with nothing registered
(Core) the dependency does nothing and every instance is ready.
"""

from __future__ import annotations

from fastapi import Depends

from synap.cloud.application.sdk_api.sdk_auth import SDKAuthContext, get_sdk_auth
from synap.cloud.shared import edition


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
    pending = await edition.acall("instance_readiness_check", auth.instance_id)
    if pending is not None:
        raise pending
    return auth
