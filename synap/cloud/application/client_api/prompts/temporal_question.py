"""Neutral interface for the prompt module temporal_question.

The text is in ``cloud/temporal_question.py`` (Cloud) and ``core/temporal_question.py`` (Core); the
edition module picks which one this name resolves to. See
``synap/cloud/shared/prompt_switch.py``.
"""

from synap.cloud.shared.prompt_switch import bind

bind(__name__)
