"""Logging configuration for LangChain MemVid.

This module provides centralized logging configuration for the LangChain MemVid package.
All modules should import and use the logger from this module.
"""

import logging
import sys
from typing import Optional


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
    logger = logging.getLogger("langchain_memvid")

    # Prevent adding handlers multiple times
    if not logger.handlers:
        # Set level
        logger.setLevel(level)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        if format_string is None:
            format_string = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        if date_format is None:
            date_format = "%Y-%m-%d %H:%M:%S"

        formatter = logging.Formatter(fmt=format_string, datefmt=date_format)
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Don't propagate to root logger
        logger.propagate = False

    return logger


# Create default logger instance
logger = setup_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Module name (optional). If not provided, returns the root package logger.

    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"langchain_memvid.{name}")
    return logger
