"""Setup Manager specific enums."""

from enum import Enum


class ApplyMode(str, Enum):
    """Mode for applying configurations."""
    STANDARD = "standard"      # Normal apply with validation
    DRY_RUN = "dry_run"        # Validate only, don't apply
    FORCE = "force"            # Skip conflict checks (admin only)


class ReviewDecision(str, Enum):
    """Decision on pending config review."""
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"


class ConflictSeverity(str, Enum):
    """Severity of detected conflicts."""
    LOW = "low"           # Warning only
    MEDIUM = "medium"     # Requires confirmation
    HIGH = "high"         # Blocks application
