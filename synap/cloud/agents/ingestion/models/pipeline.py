"""Ingestion pipeline stage models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .enums import ContentNature, DocumentSource, DocumentType, MemoryType
from .outputs import ExtractedEntity, ExtractedMemory, ExtractedRelationship


class Stage1Output(BaseModel):
    """Output of Stage 1: Type/Source/Nature Qualification."""
    input_id: str
    document_type: DocumentType
    source: DocumentSource
    nature: ContentNature
    should_process: bool
    routing_hints: Dict[str, Any] = Field(default_factory=dict)

    # Chunking info (if large document was chunked)
    is_chunked: bool = False
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    preceding_summary: Optional[str] = None


class CategoryAssignment(BaseModel):
    """A category/subcategory assignment."""
    category: str
    subcategory: str
    confidence: float = Field(ge=0.0, le=1.0)


class Stage2Output(BaseModel):
    """Output of Stage 2: Categorization."""
    input_id: str
    categories: List[CategoryAssignment]
    extraction_priorities: List[MemoryType]  # Derived from categories


class Stage3Output(BaseModel):
    """Output of Stage 3: Memory Extraction."""
    input_id: str
    memories: List[ExtractedMemory]
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]


class Stage4Output(BaseModel):
    """Output of Stage 4: Organization & Persistence."""
    input_id: str

    # What was persisted
    rdbms_memory_ids: List[str] = Field(default_factory=list)
    # customer_knowledge / client_knowledge ids; see RDBMSPersistResult.knowledge_ids
    rdbms_knowledge_ids: List[str] = Field(default_factory=list)
    rdbms_candidate_ids: List[str] = Field(default_factory=list)
    graph_entity_ids: List[str] = Field(default_factory=list)
    graph_relationship_ids: List[str] = Field(default_factory=list)
    vector_ids: List[str] = Field(default_factory=list)
    entity_vector_ids: List[str] = Field(default_factory=list)
    triple_vector_ids: List[str] = Field(default_factory=list)
    chunk_vector_ids: List[str] = Field(default_factory=list)
    chunk_count: int = 0
    question_vector_ids: List[str] = Field(default_factory=list)
    question_count: int = 0

    # S3 original
    s3_original_path: Optional[str] = None

    # Partial failure tracking (graph/vector persistence failures)
    partial_failure: bool = False
    failure_details: List[str] = Field(default_factory=list)
