"""Step-by-step payload logging for retrieval and ingestion pipelines.

Provides detailed, formatted logging of request/response data at each step.
"""

import json
import logging
import time
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from synap.cloud.shared.logging_utils import truncate_dict, truncate_string


logger = logging.getLogger(__name__)


class StepLogger:
    """Logger for tracking data flow through pipeline steps."""

    def __init__(self, step_name: str, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize step logger.

        Args:
            step_name: Name of the step (e.g., "fetch_conversation_context", "stage1_qualification")
            logger_instance: Optional logger instance (defaults to module logger)
        """
        self.step_name = step_name
        self.logger = logger_instance or logger
        self.start_time: Optional[float] = None

    def log_request(
        self,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        truncate: bool = True,
    ) -> None:
        """
        Log the incoming request payload.

        Args:
            payload: Request payload data
            correlation_id: Optional correlation ID for tracking
            truncate: Whether to truncate large payloads (default: True)
        """
        self.start_time = time.time()

        # Prepare payload for logging
        log_payload = truncate_dict(
            payload,
            max_depth=5,
            max_items=20,
            max_str_len=500
        ) if truncate else payload

        # Format the log message
        separator = "=" * 100
        self.logger.info(separator)
        self.logger.info(f"{self.step_name} - REQUEST START")
        self.logger.info(separator)

        if correlation_id:
            self.logger.info(f"Correlation ID: {correlation_id}")

        self.logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

        # Log payload as key-value pairs
        if log_payload is None:
            self.logger.info("No payload data")
        else:
            try:
                for key, value in log_payload.items():
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, default=str)
                    else:
                        value_str = str(value)
                    self.logger.info(f"{key}: {value_str}")
            except Exception as e:
                self.logger.info(f"<Error serializing payload: {e}>")
                self.logger.info(str(log_payload))

        self.logger.info(separator)

    def log_response(
        self,
        response: Dict[str, Any],
        status_code: int = 200,
        success: bool = True,
        error: Optional[str] = None,
        correlation_id: Optional[str] = None,
        truncate: bool = True,
    ) -> None:
        """
        Log the response data.

        Args:
            response: Response data
            status_code: HTTP-style status code (default: 200)
            success: Whether the step succeeded
            error: Optional error message
            correlation_id: Optional correlation ID for tracking
            truncate: Whether to truncate large responses (default: True)
        """
        elapsed_ms = None
        if self.start_time:
            elapsed_ms = int((time.time() - self.start_time) * 1000)

        # Prepare response for logging
        log_response = truncate_dict(
            response,
            max_depth=5,
            max_items=20,
            max_str_len=500
        ) if truncate else response

        # Format the log message
        separator = "=" * 100
        status_text = "SUCCESS" if success else "FAILED"

        logger_fn = self.logger.info if success else self.logger.error

        logger_fn(separator)
        logger_fn(f"{self.step_name} - RESPONSE {status_text}")
        logger_fn(separator)

        logger_fn(f"Status Code: {status_code}")

        if elapsed_ms is not None:
            logger_fn(f"Duration: {elapsed_ms}ms")

        if correlation_id:
            logger_fn(f"Correlation ID: {correlation_id}")

        logger_fn(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

        if error:
            logger_fn(f"Error: {truncate_string(error, 300)}")

        # Log response as key-value pairs
        if log_response is None:
            logger_fn("No response data")
        else:
            try:
                for key, value in log_response.items():
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, default=str)
                    else:
                        value_str = str(value)
                    logger_fn(f"{key}: {value_str}")
            except Exception as e:
                logger_fn(f"<Error serializing response: {e}>")
                logger_fn(str(log_response))

        logger_fn(separator)


def log_step(
    step_name: str,
    payload: Dict[str, Any],
    response: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
    success: bool = True,
    error: Optional[str] = None,
    correlation_id: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
    truncate: bool = True,
) -> None:
    """
    Convenience function to log a complete step (request + response).

    Args:
        step_name: Name of the step
        payload: Request payload
        response: Response data (optional)
        status_code: HTTP-style status code
        success: Whether the step succeeded
        error: Optional error message
        correlation_id: Optional correlation ID
        logger_instance: Optional logger instance
        truncate: Whether to truncate large payloads
    """
    step_logger = StepLogger(step_name, logger_instance)
    step_logger.log_request(payload, correlation_id, truncate)

    if response is not None:
        step_logger.log_response(
            response,
            status_code,
            success,
            error,
            correlation_id,
            truncate,
        )


def log_substep(
    parent_step: str,
    substep_name: str,
    data: Dict[str, Any],
    data_type: str = "Data",
    correlation_id: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
    truncate: bool = True,
) -> None:
    """
    Log an intermediate substep within a larger step.

    Args:
        parent_step: Name of the parent step
        substep_name: Name of the substep
        data: Data to log
        data_type: Type label for the data (e.g., "Query", "Results", "Intermediate")
        correlation_id: Optional correlation ID
        logger_instance: Optional logger instance
        truncate: Whether to truncate large data
    """
    log_logger = logger_instance or logger

    # Prepare data for logging
    log_data = truncate_dict(
        data,
        max_depth=5,
        max_items=20,
        max_str_len=500
    ) if truncate else data

    # Format the log message
    separator = "-" * 100
    log_logger.info(separator)
    log_logger.info(f"{parent_step} -> {substep_name} ({data_type})")
    log_logger.info(separator)

    if correlation_id:
        log_logger.info(f"Correlation ID: {correlation_id}")

    log_logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Log data as key-value pairs
    if log_data is None:
        log_logger.info("No data")
    else:
        try:
            for key, value in log_data.items():
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, default=str)
                else:
                    value_str = str(value)
                log_logger.info(f"{key}: {value_str}")
        except Exception as e:
            log_logger.info(f"<Error serializing data: {e}>")
            log_logger.info(str(log_data))

    log_logger.info(separator)
