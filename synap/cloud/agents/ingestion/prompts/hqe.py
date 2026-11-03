"""Neutral interface for the prompt module hqe.

The text is in ``cloud/hqe.py`` (Cloud) and ``core/hqe.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
