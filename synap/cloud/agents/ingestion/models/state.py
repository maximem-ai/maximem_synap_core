"""Ingestion pipeline state models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .inputs import Scope
from .pipeline import Stage1Output, Stage2Output, Stage3Output, Stage4Output


class IngestionPipelineState(BaseModel):
    """Full state of the ingestion pipeline for a single document."""
    ingestion_id: UUID
    request_id: str
    document_id: str

    # Input
    content: str
    content_hash: str
    scope: Scope

    # MACA config (loaded at start)
    maca_config_version: Optional[str] = None

    # Stage outputs
    stage1_output: Optional[Stage1Output] = None
    stage2_output: Optional[Stage2Output] = None
    stage3_output: Optional[Stage3Output] = None
    stage4_output: Optional[Stage4Output] = None

    # Current stage
    current_stage: str = "pending"

    # Traces (one per stage)
    traces: List[Dict[str, Any]] = Field(default_factory=list)

    # Errors
    error: Optional[str] = None
    error_stage: Optional[str] = None

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
