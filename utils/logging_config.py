"""
Structured logging configuration for Research Agent Hub.
Provides consistent logging across all modules.
"""

import logging
import os
import sys
from typing import Optional


# Log format with timestamp, level, module, and message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log level from environment or INFO
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the entire application.
    Call this once at application startup.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
    """
    log_level = getattr(logging, level or DEFAULT_LOG_LEVEL, logging.INFO)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance

    Example:
        from utils import get_logger
        logger = get_logger(__name__)
        logger.info("Starting research...")
    """
    return logging.getLogger(name)


# Convenience logger for quick imports
class LoggerMixin:
    """
    Mixin class that provides a logger property.

    Example:
        class MyAgent(LoggerMixin):
            def run(self):
                self.logger.info("Agent running")
    """
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(self.__class__.__name__)


# Module-level logger for this file
_module_logger = get_logger(__name__)


def log_api_call(
    api_name: str,
    endpoint: str,
    success: bool,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Log an API call with standardized format.

    Args:
        api_name: Name of the API (e.g., "finnhub", "yfinance")
        endpoint: API endpoint or method called
        success: Whether the call succeeded
        duration_ms: Optional duration in milliseconds
        error: Optional error message if failed
    """
    if success:
        if duration_ms:
            _module_logger.info(
                f"API:{api_name} | {endpoint} | OK | {duration_ms:.0f}ms"
            )
        else:
            _module_logger.info(f"API:{api_name} | {endpoint} | OK")
    else:
        _module_logger.warning(
            f"API:{api_name} | {endpoint} | FAILED | {error or 'Unknown error'}"
        )


def log_research_stage(
    stage: str,
    ticker_or_query: str,
    message: str,
) -> None:
    """
    Log a research pipeline stage.

    Args:
        stage: Stage name (e.g., "validate", "fetch", "analyze", "report")
        ticker_or_query: The ticker or query being researched
        message: Status message
    """
    _module_logger.info(f"STAGE:{stage} | {ticker_or_query} | {message}")
