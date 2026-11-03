"""Neutral interface for the prompt module relationships.

The text is in ``cloud/relationships.py`` (Cloud) and ``core/relationships.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
