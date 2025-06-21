"""
Custom exceptions for the LangChain MemVid package.
"""


class MemVidError(Exception):
    """Base exception class for all LangChain MemVid-related errors."""


class EncodingError(MemVidError):
    """Raised when there is an error during the encoding process."""


class RetrievalError(MemVidError):
    """Raised when there is an error during the retrieval process."""


class MemVidIndexError(MemVidError):
    """Raised when there is an error with the index operations."""


class VideoProcessingError(MemVidError):
    """Raised when there is an error during video processing."""


class QRCodeError(MemVidError):
    """Raised when there is an error with QR code generation or decoding."""
