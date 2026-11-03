"""Setup Manager models."""

from .enums import ApplyMode, ReviewDecision, ConflictSeverity
from .conflicts import Conflict, ConflictResolution, ConflictDetectionResult
from .results import (
    InitResult,
    UpdateResult,
    ApplyResult,
    RollbackResult,
    ReviewResult,
    ConfigVersion,
)

__all__ = [
    "ApplyMode",
    "ReviewDecision",
    "ConflictSeverity",
    "Conflict",
    "ConflictResolution",
    "ConflictDetectionResult",
    "InitResult",
    "UpdateResult",
    "ApplyResult",
    "RollbackResult",
    "ReviewResult",
    "ConfigVersion",
]
