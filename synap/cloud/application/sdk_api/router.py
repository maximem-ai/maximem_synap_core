"""The SDK routes that ship in Core, as one router.

``main.py`` includes ``sdk_router`` when no Cloud hook has installed the full
``admin_api`` router (``edition.call("cloud_routers", app)``). In Cloud the hook
includes the full router, which already holds every route here, so this one is
not added there.

The routers are included in the order ``admin_api/routes/__init__.py`` includes
them, because two of them share the ``/v1/conversations`` prefix and a request is
matched to the first route that fits.
"""

from __future__ import annotations

from fastapi import APIRouter

from synap.cloud.application.sdk_api.core_browser import router as core_browser_router
from synap.cloud.application.sdk_api.core_routes import router as core_routes_router
from synap.cloud.application.sdk_api.core_whoami import router as core_whoami_router
from synap.cloud.application.sdk_api.maca_plugin import router as maca_plugin_router
from synap.cloud.application.sdk_api.sdk_compaction import router as sdk_compaction_router
from synap.cloud.application.sdk_api.sdk_context import router as sdk_context_router
from synap.cloud.application.sdk_api.sdk_conversations import router as sdk_conversations_router
from synap.cloud.application.sdk_api.sdk_forget import router as sdk_forget_router
from synap.cloud.application.sdk_api.sdk_memories import router as sdk_memories_router
from synap.cloud.application.sdk_api.sdk_users import router as sdk_users_router

sdk_router = APIRouter()
sdk_router.include_router(sdk_context_router)
sdk_router.include_router(sdk_memories_router)
sdk_router.include_router(sdk_forget_router)
sdk_router.include_router(sdk_conversations_router)
sdk_router.include_router(sdk_users_router)
sdk_router.include_router(sdk_compaction_router)

# Core's own routes, in place of Cloud's: credits ("not available"), events and
# telemetry (accept and discard), and GET /api/v1/auth/whoami.
sdk_router.include_router(core_whoami_router)
sdk_router.include_router(core_routes_router)

# The MACA plugin: POST /v1/maca/generate and GET /v1/maca/jobs/{id}.
sdk_router.include_router(maca_plugin_router)

# The memory browser: the page at /browser and its read-only routes under /browser/api.
sdk_router.include_router(core_browser_router)
