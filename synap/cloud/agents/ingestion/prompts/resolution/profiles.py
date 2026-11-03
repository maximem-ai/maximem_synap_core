"""Neutral interface for the prompt module profiles.

The text is in ``cloud/profiles.py`` (Cloud) and ``core/profiles.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
