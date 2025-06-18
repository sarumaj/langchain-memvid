"""Unit tests for logging configuration."""

import pytest  # noqa: F401
import logging
from unittest.mock import patch, MagicMock
from io import StringIO

from langchain_memvid.logging import (
    get_logger,
    setup_logger,
    LogLevelFilter,
    LOGGER_PREFIX
)


@pytest.fixture(autouse=True)
def clear_loggers():
    """Clear all loggers before each test to ensure clean state."""
    # Remove all handlers from the root logger
    root_logger = logging.getLogger(LOGGER_PREFIX)
    for handler in root_logger.handlers.copy():
        root_logger.removeHandler(handler)

    # Clear any child loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith(LOGGER_PREFIX):
            logger = logging.getLogger(name)
            for handler in logger.handlers.copy():
                logger.removeHandler(handler)

    yield


class TestLogLevelFilter:
    """Test the LogLevelFilter class."""

    def test_filter_within_range(self):
        """Test that filter accepts records within the specified range."""
        filter_obj = LogLevelFilter(logging.INFO, logging.WARNING)

        # Create mock records
        info_record = MagicMock()
        info_record.levelno = logging.INFO

        warning_record = MagicMock()
        warning_record.levelno = logging.WARNING

        # Test that both are accepted
        assert filter_obj.filter(info_record) is True
        assert filter_obj.filter(warning_record) is True

    def test_filter_outside_range(self):
        """Test that filter rejects records outside the specified range."""
        filter_obj = LogLevelFilter(logging.INFO, logging.WARNING)

        # Create mock records
        debug_record = MagicMock()
        debug_record.levelno = logging.DEBUG

        error_record = MagicMock()
        error_record.levelno = logging.ERROR

        # Test that both are rejected
        assert filter_obj.filter(debug_record) is False
        assert filter_obj.filter(error_record) is False


class TestSetupLogger:
    """Test the setup_logger function."""

    def test_setup_logger_defaults(self):
        """Test setup_logger with default parameters."""
        logger = setup_logger()

        assert logger.name == LOGGER_PREFIX
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2
        assert logger.propagate is False

    def test_setup_logger_custom_level(self):
        """Test setup_logger with custom level."""
        logger = setup_logger(level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_setup_logger_custom_format(self):
        """Test setup_logger with custom format."""
        custom_format = "%(levelname)s: %(message)s"
        logger = setup_logger(format_string=custom_format)

        # Check that the formatter uses our custom format
        for handler in logger.handlers:
            assert handler.formatter._fmt == custom_format

    def test_setup_logger_custom_date_format(self):
        """Test setup_logger with custom date format."""
        custom_date_format = "%H:%M:%S"
        logger = setup_logger(date_format=custom_date_format)

        # Check that the formatter uses our custom date format
        for handler in logger.handlers:
            assert handler.formatter.datefmt == custom_date_format

    def test_setup_logger_handler_configuration(self):
        """Test that handlers are properly configured."""
        logger = setup_logger()

        # Should have exactly 2 handlers
        assert len(logger.handlers) == 2

        # First handler should be for stdout (DEBUG to WARNING)
        stdout_handler = logger.handlers[0]
        assert isinstance(stdout_handler, logging.StreamHandler)
        assert stdout_handler.level == logging.DEBUG
        assert len(stdout_handler.filters) == 1

        # Second handler should be for stderr (ERROR to CRITICAL)
        stderr_handler = logger.handlers[1]
        assert isinstance(stderr_handler, logging.StreamHandler)
        assert stderr_handler.level == logging.ERROR
        assert len(stderr_handler.filters) == 1

    def test_setup_logger_no_duplicate_handlers(self):
        """Test that setup_logger doesn't add duplicate handlers."""
        logger1 = setup_logger()
        logger2 = setup_logger()

        # Both should reference the same logger instance
        assert logger1 is logger2
        assert len(logger1.handlers) == 2


class TestGetLogger:
    """Test the get_logger function."""

    def test_get_logger_no_name(self):
        """Test get_logger without a name returns root logger."""
        # Setup the root logger first
        setup_logger()
        logger = get_logger()

        assert logger.name == LOGGER_PREFIX
        assert len(logger.handlers) == 2

    def test_get_logger_with_name(self):
        """Test get_logger with a name returns child logger."""
        logger = get_logger("test_module")

        assert logger.name == f"{LOGGER_PREFIX}.test_module"
        # Child loggers don't have handlers by default, they inherit from parent
        assert len(logger.handlers) == 0

    def test_get_logger_multiple_calls_same_name(self):
        """Test that multiple calls with same name return same logger."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")

        assert logger1 is logger2


class TestLoggingIntegration:
    """Integration tests for the logging system."""

    def test_logging_output_format(self):
        """Test that log messages are formatted correctly."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # Create a fresh logger with mocked stdout
            logger = setup_logger()

            # Log a message
            logger.info("Test message")

            # Get the output
            output = mock_stdout.getvalue()

            # Check that the output contains expected elements
            assert "Test message" in output
            assert "INFO" in output
            assert LOGGER_PREFIX in output

    def test_logging_levels(self):
        """Test that different log levels work correctly."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # Create a fresh logger with mocked stdout
            logger = setup_logger()

            # Test different levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")

            output = mock_stdout.getvalue()

            # Debug should not appear (default level is INFO)
            assert "Debug message" not in output
            assert "Info message" in output
            assert "Warning message" in output

    def test_error_logging_to_stderr(self):
        """Test that error messages go to stderr."""
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            # Create a fresh logger with mocked stderr
            logger = setup_logger()

            # Log an error
            logger.error("Error message")

            # Get the output from stderr
            output = mock_stderr.getvalue()

            # Check that the error message appears in stderr
            assert "Error message" in output
            assert "ERROR" in output

    def test_logger_namespacing(self):
        """Test that logger namespacing works correctly."""
        root_logger = get_logger()
        module_logger = get_logger("encoder")
        submodule_logger = get_logger("encoder.submodule")

        assert root_logger.name == LOGGER_PREFIX
        assert module_logger.name == f"{LOGGER_PREFIX}.encoder"
        assert submodule_logger.name == f"{LOGGER_PREFIX}.encoder.submodule"

        # All should be different logger instances
        assert root_logger is not module_logger
        assert module_logger is not submodule_logger
        assert root_logger is not submodule_logger

    def test_child_logger_inheritance(self):
        """Test that child loggers inherit from parent logger."""
        # Setup root logger
        setup_logger(level=logging.DEBUG)

        # Create child logger
        child_logger = get_logger("test_module")

        # Child should inherit effective level from parent
        assert child_logger.getEffectiveLevel() == logging.DEBUG

        # Test that child logger can log messages by checking the parent logger's handlers
        parent_logger = logging.getLogger(LOGGER_PREFIX)
        assert len(parent_logger.handlers) == 2

        # Verify child logger has no handlers (inherits from parent)
        assert len(child_logger.handlers) == 0

        # Test that child logger name is correct
        assert child_logger.name == f"{LOGGER_PREFIX}.test_module"

    def test_child_logger_output(self):
        """Test that child loggers can actually output messages."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # Setup root logger with mocked stdout
            setup_logger()

            # Create child logger
            child_logger = get_logger("test_module")

            # Log a message through the child logger
            child_logger.info("Child logger message")

            # Check that the message appears in stdout
            output = mock_stdout.getvalue()
            assert "Child logger message" in output
            assert "test_module" in output
