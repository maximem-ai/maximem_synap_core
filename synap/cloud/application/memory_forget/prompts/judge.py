"""Neutral interface for the prompt module judge.

The text is in ``cloud/judge.py`` (Cloud) and ``core/judge.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
