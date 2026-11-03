"""Neutral interface for the prompt module jev_adjudicator.

The text is in ``cloud/jev_adjudicator.py`` (Cloud) and ``core/jev_adjudicator.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
