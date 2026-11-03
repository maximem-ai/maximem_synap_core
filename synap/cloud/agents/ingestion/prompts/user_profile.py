"""Neutral interface for the prompt module user_profile.

The text is in ``cloud/user_profile.py`` (Cloud) and ``core/user_profile.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
