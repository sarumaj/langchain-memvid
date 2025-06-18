"""
LangChain MemVid - Video-based Vector Storage for LangChain

This package provides a video-based vector storage solution that is compatible
with the LangChain ecosystem. It uses QR codes to store document embeddings
in video format, enabling efficient storage and retrieval of document vectors.
"""

from .config import VideoConfig, QRCodeConfig, IndexConfig, VectorStoreConfig
from .vectorstore import VectorStore
from .retriever import Retriever
from .encoder import Encoder
from .index import IndexManager
from .exceptions import (
    MemVidError,
    EncodingError,
    RetrievalError,
    MemVidIndexError,
    VideoProcessingError,
    QRCodeError,
)

__all__ = [
    k for k, v in globals().items() if v in (
        # Core components
        VectorStore,
        Retriever,
        Encoder,
        IndexManager,

        # Configuration classes
        VideoConfig,
        QRCodeConfig,
        IndexConfig,
        VectorStoreConfig,

        # Exceptions
        MemVidError,
        EncodingError,
        RetrievalError,
        MemVidIndexError,
        VideoProcessingError,
        QRCodeError,
    )
]
