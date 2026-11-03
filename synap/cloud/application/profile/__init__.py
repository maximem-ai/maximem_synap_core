"""User-profile application module.

Client-configurable user profiles (spec: Synap_Async_Integration_and
_User_Profile_Spec_v1 §5): a per-owner ``profile`` document stored in the
existing ``"user".user_context`` / ``customer.customer_context`` JSONB
columns, maintained by a post-ingestion LLM merge and served via
``GET /v1/users/{user_id}/profile``.

All readers/writers go through :class:`ProfileStore`.
"""

from .store import ProfileStore

__all__ = ["ProfileStore"]
