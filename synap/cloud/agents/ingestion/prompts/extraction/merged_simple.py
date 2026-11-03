"""Neutral interface for the prompt module merged_simple.

The text is in ``cloud/merged_simple.py`` (Cloud) and ``core/merged_simple.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
