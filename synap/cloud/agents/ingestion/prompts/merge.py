"""Neutral interface for the prompt module merge.

The text is in ``cloud/merge.py`` (Cloud) and ``core/merge.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
