"""Protocol interfaces for ingestion pipeline dependencies."""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from synap.cloud.agents.ingestion.models import ExtractedEntity
from synap.cloud.memory_config.config_schema import DomainHints



@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM client interface."""

    async def complete(
        self,
        prompt: str = "",
        telemetry_scope: Optional[Any] = None,
        telemetry_operation: Optional[str] = None,
        telemetry_correlation_id: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Send a completion request to the LLM.

        Args:
            prompt: The prompt to send as a single user message (legacy)
            messages: Structured messages with role/content (preferred).
                      If provided, takes precedence over prompt.

        Returns:
            The LLM's response as a string
        """
        ...



@runtime_checkable
class EntityExtractor(Protocol):
    """Protocol for entity extraction interface."""

    async def extract(
        self,
        content: str,
        domain_hints: DomainHints,
        telemetry_scope: Optional[Any] = None,
        telemetry_correlation_id: Optional[str] = None,
    ) -> List[ExtractedEntity]:
        """
        Extract entities from content using domain hints.

        Args:
            content: The content to extract entities from
            domain_hints: Domain-specific hints for entity extraction

        Returns:
            List of extracted entities
        """
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding generation."""

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: The text to generate an embedding for

        Returns:
            List of floats representing the embedding vector
        """
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors, one per input text
        """
        ...
