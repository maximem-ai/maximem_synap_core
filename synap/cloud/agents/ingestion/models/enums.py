"""Ingestion enums."""

from enum import Enum

from synap.cloud.memory_config.enums import MemoryCategoryName


class DocumentType(str, Enum):
    """Type of document being ingested."""
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    EMAIL = "email"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    MEETING_TRANSCRIPT = "meeting_transcript"
    INTERNAL_COMM = "internal_comm"


class DocumentSource(str, Enum):
    """Source of the document."""
    API = "api"                    # Direct SDK call
    GDRIVE = "gdrive"
    UPLOADED = "uploaded"
    GMAIL = "gmail"
    SLACK = "slack"
    ZENDESK = "zendesk"
    SALESFORCE = "salesforce"
    LISTENING_AGENT = "listening_agent"  # From realtime conversation


class IngestPriority(str, Enum):
    """Priority level for queue processing."""
    REALTIME = "realtime"      # B1/B2 from Listening Agent
    EXPLICIT = "explicit"      # SDK sdk.memories.create()
    BOOTSTRAP = "bootstrap"    # Bulk import


class IngestStatus(str, Enum):
    """Status of an ingestion job."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


# Re-export MemoryCategoryName as MemoryType for ingestion-specific usage
MemoryType = MemoryCategoryName


class MemoryDestination(str, Enum):
    """Where memory is stored based on confidence."""
    MEMORIES = "memories"          # Above threshold
    CANDIDATES = "candidates"      # Below threshold
    CLIENT_KNOWLEDGE = "client_knowledge"
    CUSTOMER_KNOWLEDGE = "customer_knowledge"
    SHORT_TERM = "short_term"


class ContentNature(str, Enum):
    """Nature assessment of content."""
    SUBSTANTIVE = "substantive"    # Worth processing
    TRIVIAL = "trivial"            # Too short/empty
    SPAM = "spam"                  # Detected as spam
    DUPLICATE = "duplicate"        # Already processed


class PipelineStage(str, Enum):
    """Stages in the ingestion pipeline."""
    STAGE_1_QUALIFICATION = "stage_1_qualification"
    STAGE_2_CATEGORIZATION = "stage_2_categorization"
    STAGE_3_EXTRACTION = "stage_3_extraction"
    STAGE_4_ORGANIZATION = "stage_4_organization"
