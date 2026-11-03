"""Neutral interface for the prompt module temporal_addendum.

The text is in ``cloud/temporal_addendum.py`` (Cloud) and ``core/temporal_addendum.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
