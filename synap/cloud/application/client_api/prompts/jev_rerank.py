"""Neutral interface for the prompt module jev_rerank.

The text is in ``cloud/jev_rerank.py`` (Cloud) and ``core/jev_rerank.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
