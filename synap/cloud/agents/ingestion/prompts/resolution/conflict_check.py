"""Neutral interface for the prompt module conflict_check.

The text is in ``cloud/conflict_check.py`` (Cloud) and ``core/conflict_check.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
