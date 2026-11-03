"""Neutral interface for the prompt module facts.

The text is in ``cloud/facts.py`` (Cloud) and ``core/facts.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
