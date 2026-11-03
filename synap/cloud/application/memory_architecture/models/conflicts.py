"""Conflict detection models."""

from dataclasses import dataclass, field
from typing import Optional

from synap.cloud.memory_config.enums import ConflictType
from .enums import ConflictSeverity


@dataclass
class Conflict:
    """Represents a detected conflict between configurations."""
    type: ConflictType
    severity: ConflictSeverity
    section: str                        # Config section with conflict
    message: str                        # Human-readable description
    current_value: Optional[str] = None
    proposed_value: Optional[str] = None
    suggested_resolution: Optional[str] = None
    requires_migration: bool = False

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "section": self.section,
            "message": self.message,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "suggested_resolution": self.suggested_resolution,
            "requires_migration": self.requires_migration
        }


@dataclass
class ConflictResolution:
    """Resolution for a conflict."""
    conflict_id: str
    resolution: str  # "accept_new" | "keep_current" | "migrate" | "archive"
    resolved_by: str  # User ID
    notes: Optional[str] = None


@dataclass
class ConflictDetectionResult:
    """Result of conflict detection."""
    has_conflicts: bool = False
    has_blocking_conflicts: bool = False
    conflicts: list[Conflict] = field(default_factory=list)
    requires_migration: bool = False

    def add(self, conflict: Conflict):
        self.conflicts.append(conflict)
        self.has_conflicts = True
        if conflict.severity == ConflictSeverity.HIGH:
            self.has_blocking_conflicts = True
        if conflict.requires_migration:
            self.requires_migration = True
