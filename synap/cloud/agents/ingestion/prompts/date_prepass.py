"""Neutral interface for the prompt module date_prepass.

The text is in ``cloud/date_prepass.py`` (Cloud) and ``core/date_prepass.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
