"""Ingestion agent state schema."""

from typing import TypedDict, Optional, Dict, Any, List


class IngestionState(TypedDict):
    """State for Ingestion agent graph.
    
    Tracks the process of qualifying and classifying incoming data
    for persistence.
    """
    # Input
    raw_input: Dict[str, Any]                   # conversation turn, text, org knowledge
    input_type: str                             # "conversation" | "text" | "org_knowledge"
    conversation_id: Optional[str]
    auth_context: Dict[str, Any]

    # Processing
    qualification: Optional[Dict[str, Any]]     # Should persist? In what form?
    classification: Optional[str]               # fact | preference | episode | temporal | emotion

    # Output
    qualified_data: Optional[Dict[str, Any]]    # QualifiedData for persistence pipeline

    # Trace
    trace_id: str
    errors: Optional[List[str]]                 # Error messages if any
