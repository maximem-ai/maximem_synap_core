"""Neutral interface for the prompt module earlier_reference.

The text is in ``cloud/earlier_reference.py`` (Cloud) and ``core/earlier_reference.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
