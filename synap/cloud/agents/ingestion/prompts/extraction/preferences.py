"""Neutral interface for the prompt module preferences.

The text is in ``cloud/preferences.py`` (Cloud) and ``core/preferences.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
