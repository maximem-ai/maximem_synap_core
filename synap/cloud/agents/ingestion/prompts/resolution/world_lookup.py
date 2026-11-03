"""Neutral interface for the prompt module world_lookup.

The text is in ``cloud/world_lookup.py`` (Cloud) and ``core/world_lookup.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
