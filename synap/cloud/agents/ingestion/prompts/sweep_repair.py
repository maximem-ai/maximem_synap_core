"""Neutral interface for the prompt module sweep_repair.

The text is in ``cloud/sweep_repair.py`` (Cloud) and ``core/sweep_repair.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
