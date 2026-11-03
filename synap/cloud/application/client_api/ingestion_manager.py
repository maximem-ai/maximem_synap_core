"""Ingestion Manager for data ingestion operations - Client API facade."""

from typing import Any, Dict, Optional
from uuid import UUID

from synap.cloud.agents.ingestion.models import (
    IngestDocumentRequest,
    IngestBatchRequest,
    IngestResponse,
    IngestBatchResponse,
)
from synap.cloud.application.ingestion import IngestionManager as CoreIngestionManager


class IngestionManager:
    """
    Client-facing ingestion manager that delegates to core IngestionManager.
    
    This facade provides backward compatibility with SDK-facing methods
    while delegating to the new core implementation.
    """

    def __init__(self, core_manager: CoreIngestionManager):
        """Initialize client API ingestion manager.
        
        Args:
            core_manager: Core IngestionManager instance
        """
        self._core = core_manager

    async def ingest_conversation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest conversation turn data.
        
        Args:
            payload: Conversation turn payload with required fields
            
        Returns:
            Ingestion result with status and metadata
        """
        # Convert payload to IngestDocumentRequest
        request = IngestDocumentRequest(**payload)
        response = await self._core.ingest_document(request)
        return response.model_dump()

    async def ingest_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest arbitrary text (documents, notes).
        
        Args:
            payload: Text ingestion payload
            
        Returns:
            Ingestion result with status and metadata
        """
        # Convert payload to IngestDocumentRequest
        request = IngestDocumentRequest(**payload)
        response = await self._core.ingest_document(request)
        return response.model_dump()

    async def ingest_bootstrap(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One-time bulk ingestion for new instance setup.
        
        Args:
            payload: Bootstrap ingestion payload with 'documents' list
            
        Returns:
            Ingestion result with status and metadata
        """
        # Convert payload to IngestBatchRequest
        request = IngestBatchRequest(**payload)
        response = await self._core.ingest_batch(request)
        return response.model_dump()

    async def ingest_incremental_diff(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Incremental sync ingestion.
        
        Args:
            payload: Incremental diff payload
            
        Returns:
            Ingestion result with status and metadata
        """
        # Convert payload to IngestDocumentRequest
        request = IngestDocumentRequest(**payload)
        response = await self._core.ingest_document(request)
        return response.model_dump()

    async def get_status(self, ingestion_id: UUID) -> Optional[Dict[str, Any]]:
        """Get status of an ingestion job.
        
        Args:
            ingestion_id: Ingestion ID to query
            
        Returns:
            Status dict or None if not found
        """
        response = await self._core.get_status(ingestion_id)
        return response.model_dump() if response else None
