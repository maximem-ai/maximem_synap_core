"""Result types for Setup Manager operations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from synap.cloud.memory_config.enums import ConfigStatus
from .conflicts import Conflict


@dataclass
class InitResult:
    """Result of initializing memory architecture."""
    success: bool
    config_id: str
    status: ConfigStatus
    message: str
    warnings: list[str] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)


@dataclass
class UpdateResult:
    """Result of updating memory architecture."""
    success: bool
    config_id: str
    previous_config_id: Optional[str]
    status: ConfigStatus
    diff_summary: Optional[dict] = None
    conflicts: list[Conflict] = field(default_factory=list)
    requires_review: bool = True


@dataclass
class ApplyResult:
    """Result of applying a configuration."""
    success: bool
    status: ConfigStatus
    message: str
    applied_at: Optional[datetime] = None
    rollback_config_id: Optional[str] = None
    errors: list[str] = field(default_factory=list)


@dataclass
class RollbackResult:
    """Result of rolling back configuration."""
    success: bool
    rolled_back_from: str
    rolled_back_to: str
    message: str


@dataclass
class ReviewResult:
    """Result of reviewing a configuration."""
    success: bool
    config_id: str
    new_status: ConfigStatus
    message: str


@dataclass
class ConfigVersion:
    """Metadata for a config version."""
    config_id: str
    version: str
    status: ConfigStatus
    created_at: datetime
    created_by: str
    summary: str
    is_active: bool
