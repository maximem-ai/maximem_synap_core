"""Neutral interface for the prompt module prompt_resolver.

The text is in ``cloud/prompt_resolver.py`` (Cloud) and ``core/prompt_resolver.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
