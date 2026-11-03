"""Neutral interface for the prompt module second_look.

The text is in ``cloud/second_look.py`` (Cloud) and ``core/second_look.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
