"""Stage 1: Type/Source/Nature Qualification."""

import hashlib
import logging
import time
from typing import Dict, List, Optional

from synap.cloud.agents.ingestion.interfaces import LLMClient
from synap.cloud.observability.tracing import start_span, set_current_span_attributes

logger = logging.getLogger(__name__)
from synap.cloud.agents.ingestion.models import (
    ContentNature,
    DocumentSource,
    DocumentType,
    IngestDocumentRequest,
    Stage1Output,
)
from synap.cloud.memory_config.config_schema import MemoryArchitectureConfig


class Stage1Qualifier:
    """
    Qualifies incoming content by:
    - Document type (conversation, email, PDF, etc.)
    - Source (API, Slack, Gmail, etc.)
    - Nature (substantive, trivial, spam, duplicate)

    Mostly rule-based with LLM fallback for ambiguous cases.
    """

    # Minimum content lengths by type
    MIN_LENGTHS = {
        DocumentType.CONVERSATION: 10,
        DocumentType.EMAIL: 50,
        DocumentType.DOCUMENT: 100,
    }

    def __init__(
        self,
        config: MemoryArchitectureConfig,
        llm_client: Optional[LLMClient] = None,
    ):
        logger.debug("Initializing Stage1 qualifier")
        self._config = config
        self._llm = llm_client

    @staticmethod
    def content_hash(content: str) -> str:
        """The hash `qualify` compares against `existing_hashes`."""
        return hashlib.sha256(content.encode()).hexdigest()

    @classmethod
    def skip_reason(cls, doc_type: DocumentType, nature: ContentNature) -> str:
        """Why a text that was not processed stored nothing, in words the person who sent it can act on."""
        if nature == ContentNature.TRIVIAL:
            least = cls.MIN_LENGTHS.get(doc_type, 50)
            return (f"Nothing was stored: the text is shorter than {least} characters, the least that "
                    f"document_type '{doc_type.value}' is read for. Send more text, or send it as an "
                    f"ai-chat-conversation, which is read from 10 characters.")
        if nature == ContentNature.DUPLICATE:
            return "Nothing was stored: this exact text was already stored."
        if nature == ContentNature.SPAM:
            return "Nothing was stored: the text looks like spam."
        return "Nothing was stored: the text was not worth processing."

    async def qualify(
        self,
        request: IngestDocumentRequest,
        existing_hashes: Optional[set] = None,
    ) -> Stage1Output:
        """
        Qualify the input content.

        Steps:
        1. Detect document type (use hint or detect)
        2. Validate source
        3. Assess content nature
        4. Handle large documents (chunking)
        """
        # 1. Detect document type
        async with start_span("ingestion.stage1.qualify.type_detection", {
            "synap.content_len": len(request.content) if request.content else 0,
            "synap.has_type_hint": request.document_type is not None,
        }):
            t0 = time.monotonic()
            doc_type = await self._detect_type(request)
            set_current_span_attributes({
                "synap.detected_type": doc_type.value,
                "synap.detection_method": "hint" if request.document_type else (
                    "bytes" if isinstance(request.content, bytes) else "text"
                ),
                "synap.type_detection_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })

        # 2. Validate source
        source = self._validate_source(request.source, doc_type)

        # 3. Assess nature
        async with start_span("ingestion.stage1.qualify.nature_assessment", {
            "synap.doc_type": doc_type.value,
            "synap.content_len": len(request.content) if isinstance(request.content, str) else 0,
        }):
            t0 = time.monotonic()
            nature, should_process = await self._assess_nature(
                content=request.content,
                doc_type=doc_type,
                existing_hashes=existing_hashes,
            )
            set_current_span_attributes({
                "synap.nature": nature.value,
                "synap.should_process": should_process,
                "synap.is_duplicate": nature.value == "duplicate",
                "synap.nature_assessment_latency_ms": round((time.monotonic() - t0) * 1000, 2),
            })

        # Build output
        output = Stage1Output(
            input_id=request.request_id,
            document_type=doc_type,
            source=source,
            nature=nature,
            should_process=should_process,
            routing_hints=self._build_routing_hints(doc_type, source),
            is_chunked=False,  # Chunking not yet implemented
            total_chunks=1,
        )

        return output

    async def _detect_type(
        self,
        request: IngestDocumentRequest,
    ) -> DocumentType:
        """Detect document type from hints or content analysis."""
        # Trust hint if provided and valid
        if request.document_type:
            return request.document_type

        content = request.content
        if isinstance(content, bytes):
            return self._detect_from_bytes(content)

        # Text content analysis
        return self._detect_from_text(content)

    def _detect_from_text(self, content: str) -> DocumentType:
        """Detect type from text content patterns."""
        content_lower = content.lower()

        # Conversation patterns
        if self._has_conversation_pattern(content):
            return DocumentType.CONVERSATION

        # Email patterns
        if any(marker in content_lower for marker in ["from:", "to:", "subject:", "sent:"]):
            return DocumentType.EMAIL

        # Default to document
        return DocumentType.DOCUMENT

    def _has_conversation_pattern(self, content: str) -> bool:
        """Check for conversation turn patterns."""
        patterns = [
            "user:", "assistant:", "human:", "ai:",
            "[user]", "[assistant]",
            ">> ", "<< ",
        ]
        return any(p in content.lower() for p in patterns)

    def _detect_from_bytes(self, content: bytes) -> DocumentType:
        """Detect type from file magic bytes."""
        # PDF
        if content.startswith(b"%PDF"):
            return DocumentType.PDF

        # Images
        if content.startswith(b"\x89PNG") or content.startswith(b"\xff\xd8\xff"):
            return DocumentType.IMAGE

        # Audio (basic detection)
        if content.startswith(b"RIFF") or content.startswith(b"ID3"):
            return DocumentType.AUDIO

        return DocumentType.DOCUMENT

    def _validate_source(
        self,
        source: Optional[DocumentSource],
        doc_type: DocumentType,
    ) -> DocumentSource:
        """Validate and default source."""
        if source:
            return source

        # Default based on type
        return DocumentSource.API

    async def _assess_nature(
        self,
        content: str | bytes,
        doc_type: DocumentType,
        existing_hashes: Optional[set] = None,
    ) -> tuple[ContentNature, bool]:
        """Assess if content is worth processing."""
        if isinstance(content, bytes):
            # Binary content - assume substantive
            return ContentNature.SUBSTANTIVE, True

        # Check length
        min_length = self.MIN_LENGTHS.get(doc_type, 50)
        if len(content.strip()) < min_length:
            return ContentNature.TRIVIAL, False

        # Check for duplicate (content hash)
        if existing_hashes:
            if self.content_hash(content) in existing_hashes:
                return ContentNature.DUPLICATE, False

        # Check for spam patterns (basic)
        if self._is_spam(content):
            return ContentNature.SPAM, False

        return ContentNature.SUBSTANTIVE, True

    def _is_spam(self, content: str) -> bool:
        """Basic spam detection."""
        spam_indicators = [
            "click here", "unsubscribe", "act now",
            "limited time", "winner", "congratulations",
        ]
        content_lower = content.lower()
        spam_count = sum(1 for i in spam_indicators if i in content_lower)
        return spam_count >= 3

    def _build_routing_hints(
        self,
        doc_type: DocumentType,
        source: DocumentSource,
    ) -> Dict:
        """Build hints for downstream stages."""
        return {
            "doc_type": doc_type.value,
            "source": source.value,
            "requires_transcription": doc_type == DocumentType.AUDIO,
            "requires_vision": doc_type == DocumentType.IMAGE,
            "requires_pdf_parsing": doc_type == DocumentType.PDF,
        }


