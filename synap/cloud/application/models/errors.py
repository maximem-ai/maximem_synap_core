"""Application layer error types."""


class ApplicationError(Exception):
    """Base exception for all application layer errors."""

    def __init__(self, message: str, code: str, retryable: bool = False):
        """Initialize application error.
        
        Args:
            message: Error description
            code: Error code for categorization
            retryable: Whether operation can be retried
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable


class ValidationError(ApplicationError):
    """Request validation failed."""

    def __init__(self, message: str, code: str = "VALIDATION_ERROR"):
        """Initialize validation error.
        
        Args:
            message: Error description
            code: Error code
        """
        super().__init__(message, code, retryable=False)


class NotFoundError(ApplicationError):
    """Requested resource not found."""

    def __init__(self, message: str, code: str = "NOT_FOUND"):
        """Initialize not found error.
        
        Args:
            message: Error description
            code: Error code
        """
        super().__init__(message, code, retryable=False)


class PermissionDeniedError(ApplicationError):
    """Permission denied for requested operation."""

    def __init__(self, message: str, code: str = "PERMISSION_DENIED"):
        """Initialize permission denied error.
        
        Args:
            message: Error description
            code: Error code
        """
        super().__init__(message, code, retryable=False)


class ConfigurationError(ApplicationError):
    """Configuration error."""

    def __init__(self, message: str, code: str = "CONFIGURATION_ERROR", retryable: bool = False):
        """Initialize configuration error.
        
        Args:
            message: Error description
            code: Error code
            retryable: Whether operation can be retried
        """
        super().__init__(message, code, retryable)


class IngestionError(ApplicationError):
    """Ingestion operation failed."""

    def __init__(self, message: str, code: str = "INGESTION_ERROR", retryable: bool = False):
        """Initialize ingestion error.
        
        Args:
            message: Error description
            code: Error code
            retryable: Whether operation can be retried
        """
        super().__init__(message, code, retryable)


class RetrievalError(ApplicationError):
    """Retrieval operation failed."""

    def __init__(self, message: str, code: str = "RETRIEVAL_ERROR", retryable: bool = True):
        """Initialize retrieval error.
        
        Args:
            message: Error description
            code: Error code
            retryable: Whether operation can be retried
        """
        super().__init__(message, code, retryable)


class RateLimitExceeded(ApplicationError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: int):
        """Initialize rate limit exceeded error.
        
        Args:
            retry_after: Seconds until retry is allowed
        """
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after} seconds",
            "RATE_LIMIT_EXCEEDED",
            retryable=True
        )
        self.retry_after = retry_after
