"""Start-up messages for services that are not part of this build (Core).

``main.py`` starts its background services one by one, each in its own try block that logs a warning
(or, for the PII gate, an error) with a traceback when the start fails. In Core most of those services
are not shipped: the credits, billing, email, connector and PII packages stay behind, so the import at
the top of each block raises ``ModuleNotFoundError``. That is not a failure, and a developer reading the
log for the first time should not see thirty stack traces and a line saying documents are ingested
without PII detection.

``AbsentServiceFilter`` turns exactly those records into one INFO line that names the missing module. A
failure of any other kind (a shipped service that really cannot start) keeps its level and traceback.
Cloud does not install the filter (edition value ``quiet_absent_services``).
"""
from __future__ import annotations

import logging
import re

_CUT_AT = re.compile(r"\s+(?:failed to start|FAILED to start|failed)\b|[;—]")


class AbsentServiceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        info = record.exc_info
        if record.levelno < logging.WARNING or not info or not isinstance(info[1], ModuleNotFoundError):
            return True
        what = _CUT_AT.split(record.getMessage(), maxsplit=1)[0].strip() or "A service"
        record.levelno, record.levelname = logging.INFO, "INFO"
        record.msg = f"{what}: not started, {info[1].name or 'a module'} is not part of this build"
        record.args = ()
        record.exc_info = record.exc_text = None
        return True
