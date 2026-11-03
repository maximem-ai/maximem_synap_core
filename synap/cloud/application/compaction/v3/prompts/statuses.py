"""Neutral interface for the v3 prompt module statuses.

The text is in ``cloud/statuses.py`` (Cloud) and ``core/statuses.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
