"""Expire memories older than ``guardrails.retention.max_memory_age_days``.

Off unless ``SYNAP_GUARDRAIL_RETENTION`` is on (Core sets it, Cloud does not).
"""
from synap.cloud.application.retention.sweep import FLAG, retention_wanted
from synap.cloud.application.retention.worker import RetentionWorker

__all__ = ["FLAG", "RetentionWorker", "retention_wanted"]
