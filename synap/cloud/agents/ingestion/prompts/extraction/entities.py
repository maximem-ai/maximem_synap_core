"""Neutral interface for the prompt module entities.

The text is in ``cloud/entities.py`` (Cloud) and ``core/entities.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
