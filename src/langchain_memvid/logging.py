"""Logging configuration for LangChain MemVid.

This module provides centralized logging configuration for the LangChain MemVid package.
All modules should import and use the logger from this module.
"""

import logging
import sys
from typing import Optional


LOGGER_PREFIX = "langchain_memvid"


class LogLevelFilter(logging.Filter):
    """Filter to only log messages at or above a certain level."""

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi

    def filter(self, record: logging.LogRecord) -> bool:
        return self.lo <= record.levelno <= self.hi


def setup_logger(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """Set up the LangChain MemVid logger.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
        date_format: Custom date format for log messages

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(LOGGER_PREFIX)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        # Set level
        logger.setLevel(level)

        # Create formatter
        if format_string is None:
            format_string = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        if date_format is None:
            date_format = "%Y-%m-%d %H:%M:%S"

        formatter = logging.Formatter(fmt=format_string, datefmt=date_format)

        # Create handlers
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(LogLevelFilter(logging.DEBUG, logging.WARNING))
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)
        stderr_handler.addFilter(LogLevelFilter(logging.ERROR, logging.CRITICAL))
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)

        # Don't propagate to root logger
        logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Module name (optional). If not provided, returns the root package logger.

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"{LOGGER_PREFIX}.{name}")
    return logger


# Create default logger instance
logger = setup_logger()
