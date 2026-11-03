"""Neutral interface for the prompt module temporal_gate.

The text is in ``cloud/temporal_gate.py`` (Cloud) and ``core/temporal_gate.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
