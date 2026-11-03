"""Neutral interface for the prompt module episodes.

The text is in ``cloud/episodes.py`` (Cloud) and ``core/episodes.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
