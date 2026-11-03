"""Neutral interface for the prompt module stage3_extraction.

The text is in ``cloud/stage3_extraction.py`` (Cloud) and ``core/stage3_extraction.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
