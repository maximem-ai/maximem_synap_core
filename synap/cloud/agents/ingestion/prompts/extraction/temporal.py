"""Neutral interface for the prompt module temporal.

The text is in ``cloud/temporal.py`` (Cloud) and ``core/temporal.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
