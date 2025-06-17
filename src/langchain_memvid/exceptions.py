"""
Custom exceptions for the MemVid package.
"""


class MemVidError(Exception):
    """Base exception class for all MemVid-related errors."""
    pass


class EncodingError(MemVidError):
    """Raised when there is an error during the encoding process."""
    pass


class RetrievalError(MemVidError):
    """Raised when there is an error during the retrieval process."""
    pass


class MemVidIndexError(MemVidError):
    """Raised when there is an error with the index operations."""
    pass


class ConfigurationError(MemVidError):
    """Raised when there is an error in the configuration."""
    pass


class VideoProcessingError(MemVidError):
    """Raised when there is an error during video processing."""
    pass


class QRCodeError(MemVidError):
    """Raised when there is an error with QR code generation or decoding."""
    pass
