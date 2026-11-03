"""Neutral interface for the prompt module emotions.

The text is in ``cloud/emotions.py`` (Cloud) and ``core/emotions.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
