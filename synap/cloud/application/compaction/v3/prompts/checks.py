"""Neutral interface for the v3 prompt module checks.

The text is in ``cloud/checks.py`` (Cloud) and ``core/checks.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
