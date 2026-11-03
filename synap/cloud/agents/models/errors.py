"""Agent error types."""


class AgentError(Exception):
    """Base exception for all agent errors."""

    def __init__(self, message: str, trace_id: str, retryable: bool = False):
        """Initialize agent error.
        
        Args:
            message: Error description
            trace_id: Trace/request ID for debugging
            retryable: Whether operation can be retried
        """
        super().__init__(message)
        self.message = message
        self.trace_id = trace_id
        self.retryable = retryable


class LLMUnavailableError(AgentError):
    """LLM provider is unavailable (429, 503, timeout)."""

    def __init__(self, message: str, trace_id: str, provider: str = None, model: str = None):
        """Initialize LLM unavailable error.
        
        Args:
            message: Error description
            trace_id: Trace ID
            provider: Optional provider name
            model: Optional model name
        """
        super().__init__(message, trace_id, retryable=True)
        self.provider = provider
        self.model = model


class MCPServerError(AgentError):
    """MCP server error (connection failed, tool not found, invocation failed)."""

    def __init__(self, message: str, trace_id: str, server_id: str = None, tool_name: str = None, retryable: bool = False):
        """Initialize MCP server error.
        
        Args:
            message: Error description
            trace_id: Trace ID
            server_id: Optional MCP server ID
            tool_name: Optional tool name
            retryable: Whether operation can be retried
        """
        super().__init__(message, trace_id, retryable)
        self.server_id = server_id
        self.tool_name = tool_name


class RetrievalError(AgentError):
    """Context retrieval failed (query failed, timeout, empty results)."""

    def __init__(self, message: str, trace_id: str, retryable: bool = True):
        """Initialize retrieval error.
        
        Args:
            message: Error description
            trace_id: Trace ID
            retryable: Whether operation can be retried
        """
        super().__init__(message, trace_id, retryable)


class IngestionError(AgentError):
    """Ingestion qualification or classification failed."""

    def __init__(self, message: str, trace_id: str, retryable: bool = False):
        """Initialize ingestion error.
        
        Args:
            message: Error description
            trace_id: Trace ID
            retryable: Whether operation can be retried
        """
        super().__init__(message, trace_id, retryable)


class ArchitectureDesignError(AgentError):
    """Memory architecture design failed."""

    def __init__(self, message: str, trace_id: str, retryable: bool = False):
        """Initialize architecture design error.
        
        Args:
            message: Error description
            trace_id: Trace ID
            retryable: Whether operation can be retried
        """
        super().__init__(message, trace_id, retryable)
