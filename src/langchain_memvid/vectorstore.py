"""
VectorStore implementation for Memvid.

This implementation is a wrapper around the Memvid vector store.
It uses the Memvid encoder to encode documents into QR codes and the Memvid retriever to search for similar documents.
"""

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import asyncio
import nest_asyncio
from pydantic import BaseModel, Field
import logging

from .index import IndexManager, IndexConfig
from .encoder import Encoder, EncoderConfig
from .retriever import Retriever, RetrieverConfig


logger = logging.getLogger(__name__)


class VectorStoreConfig(BaseModel):
    """Unified configuration for VectorStore.

    This configuration combines settings for the index, encoder, and retriever
    components into a single, organized structure.

    Attributes:
        index: Configuration for the FAISS index
        encoder: Configuration for the QR code encoder
        retriever: Configuration for the document retriever
    """
    index: IndexConfig = Field(
        default_factory=IndexConfig,
        description="Configuration for the FAISS index"
    )
    encoder: EncoderConfig = Field(
        default_factory=EncoderConfig,
        description="Configuration for the QR code encoder"
    )
    retriever: RetrieverConfig = Field(
        default_factory=RetrieverConfig,
        description="Configuration for the document retriever"
    )


class VectorStore(VectorStore):
    """Vector store that stores documents in a video format using QR codes.

    This vector store uses `memvid` to encode documents into QR codes and store them
    in a video file. It provides semantic search capabilities using FAISS index.

    Attributes:
        video_file (str): Path to the video file storing QR codes
        index_file (str): Path to the index file for semantic search
        encoder (Encoder): Encoder for converting documents to QR codes
        _retriever (Optional[Retriever]): Lazy-loaded retriever for searching and decoding QR codes
    """

    @staticmethod
    def _get_event_loop() -> asyncio.AbstractEventLoop:
        """Get the current event loop or create a new one if none exists.

        Returns:
            asyncio.AbstractEventLoop: The current or new event loop

        Note:
            This method handles both cases where:
            1. We're in an async context with a running event loop
            2. We're in a sync context and need to create a new event loop
            3. We're in a nested asyncio context (e.g., Jupyter notebook)
        """
        try:
            # Apply nest_asyncio to allow nested event loops
            nest_asyncio.apply()

            # Try to get the running loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Apply nest_asyncio to the new loop as well
            nest_asyncio.apply(loop)

        return loop

    def __init__(
        self, *,
        video_file: str,
        index_file: str,
        embedding: Embeddings,
        config: VectorStoreConfig,
    ):
        """Initialize VectorStore.

        Args:
            video_file: Path to store/load the video file
            index_file: Path to store/load the index file
            embedding: Embedding model for semantic search
            config: Optional unified configuration. If not provided, uses default configs
        """
        self.video_file = str(Path(video_file).absolute())
        self.index_file = str(Path(index_file).absolute())
        self.config = config

        # Initialize components with their respective configs
        self.index_manager = IndexManager(
            config=self.config.index,
            embeddings=embedding
        )

        self.encoder = Encoder(
            config=self.config.encoder,
            index_manager=self.index_manager
        )

        # Initialize retriever as None - will be created lazily
        self._retriever: Optional[Retriever] = None

    @property
    def retriever(self) -> Retriever:
        """Get the retriever instance, creating it if necessary.

        Returns:
            Retriever: The retriever instance

        Raises:
            RuntimeError: If video file doesn't exist when retriever is needed
        """
        if self._retriever is None:
            if not Path(self.video_file).exists():
                raise RuntimeError(
                    f"Video file {self.video_file} does not exist. "
                    "Add some texts first using add_texts() or add_documents()."
                )

            self._retriever = Retriever(
                video_file=self.video_file,
                index_file=self.index_file,
                config=self.config.retriever,
                index_manager=self.index_manager,
                load_index=False  # Don't load index during initialization
            )

        return self._retriever

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts for each text
            **kwargs: Additional arguments passed to the encoder

        Returns:
            List of chunk IDs

        Raises:
            ValueError: If no texts are provided
            RuntimeError: If video building fails
        """
        if not texts:
            raise ValueError("No texts provided to add")

        try:
            # Add texts to encoder
            self.encoder.add_chunks(texts)

            # Build video and index
            stats = self.encoder.build_video(
                output_file=self.video_file,
                index_file=self.index_file,
                **kwargs
            )

            # Reload index in index manager after building
            self.index_manager.load(self.index_file)

            # Reset retriever if it exists to force recreation
            self._retriever = None

            # Log build statistics
            logger.info(
                f"Built video with {stats.total_chunks} chunks "
                f"({stats.video_size_mb:.2f} MB) in {stats.encoding_time:.2f}s"
            )

            # Return chunk IDs
            return [str(i) for i in range(len(texts))]

        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            raise RuntimeError(f"Failed to add texts: {e}") from e

    async def aadd_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store asynchronously.

        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts for each text
            **kwargs: Additional arguments passed to the encoder

        Returns:
            List of chunk IDs

        Raises:
            ValueError: If no texts are provided
            RuntimeError: If video building fails
        """
        # Since Encoder doesn't support async operations,
        # we run the sync version in a thread pool
        loop = self._get_event_loop()
        return await loop.run_in_executor(
            None, self.add_texts, texts, metadatas, **kwargs
        )

    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            **kwargs: Additional arguments passed to add_texts

        Returns:
            List of chunk IDs
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas=metadatas, **kwargs)

    async def aadd_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vector store asynchronously.

        Args:
            documents: List of Document objects to add
            **kwargs: Additional arguments passed to add_texts

        Returns:
            List of chunk IDs
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await self.aadd_texts(texts, metadatas=metadatas, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments passed to the retriever

        Returns:
            List of Document objects
        """
        results = self.retriever.search_with_metadata(query, top_k=k)

        return [
            Document(
                page_content=result["text"],
                metadata=result["metadata"]
            )
            for result in results
        ]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents asynchronously.

        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments passed to the retriever

        Returns:
            List of Document objects
        """
        # Since Retriever doesn't support async operations,
        # we run the sync version in a thread pool
        loop = self._get_event_loop()
        return await loop.run_in_executor(
            None, self.similarity_search, query, k, **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores.

        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments passed to the retriever

        Returns:
            List of (Document, score) tuples
        """
        results = self.retriever.search_with_metadata(query, top_k=k)

        return [
            (
                Document(
                    page_content=result["text"],
                    metadata=result["metadata"]
                ),
                result["score"]
            )
            for result in results
        ]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores asynchronously.

        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments passed to the retriever

        Returns:
            List of (Document, score) tuples
        """
        # Since Retriever doesn't support async operations,
        # we run the sync version in a thread pool
        loop = self._get_event_loop()
        return await loop.run_in_executor(
            None, self.similarity_search_with_score, query, k, **kwargs
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        video_file: str,
        index_file: str,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """Create vector store from texts.

        Args:
            texts: List of text strings
            embedding: Embedding model
            video_file: Path to store the video file
            index_file: Path to store the index file
            metadatas: Optional list of metadata dicts
            **kwargs: Additional arguments passed to constructor

        Returns:
            VectorStore instance
        """
        vectorstore = cls(
            video_file=video_file,
            index_file=index_file,
            embedding=embedding,
            **kwargs
        )
        vectorstore.add_texts(texts, metadatas=metadatas)
        return vectorstore

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        video_file: str,
        index_file: str,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """Create vector store from texts asynchronously.

        Args:
            texts: List of text strings
            embedding: Embedding model
            video_file: Path to store the video file
            index_file: Path to store the index file
            metadatas: Optional list of metadata dicts
            **kwargs: Additional arguments passed to constructor

        Returns:
            VectorStore instance
        """
        vectorstore = cls(
            video_file=video_file,
            index_file=index_file,
            embedding=embedding,
            **kwargs
        )
        await vectorstore.aadd_texts(texts, metadatas=metadatas)
        return vectorstore

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        video_file: str,
        index_file: str,
        **kwargs: Any,
    ) -> "VectorStore":
        """Create vector store from documents.

        Args:
            documents: List of Document objects
            embedding: Embedding model
            video_file: Path to store the video file
            index_file: Path to store the index file
            **kwargs: Additional arguments passed to constructor

        Returns:
            VectorStore instance
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            video_file=video_file,
            index_file=index_file,
            metadatas=metadatas,
            **kwargs
        )

    @classmethod
    async def afrom_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        video_file: str,
        index_file: str,
        **kwargs: Any,
    ) -> "VectorStore":
        """Create vector store from documents asynchronously.

        Args:
            documents: List of Document objects
            embedding: Embedding model
            video_file: Path to store the video file
            index_file: Path to store the index file
            **kwargs: Additional arguments passed to constructor

        Returns:
            VectorStore instance
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await cls.afrom_texts(
            texts=texts,
            embedding=embedding,
            video_file=video_file,
            index_file=index_file,
            metadatas=metadatas,
            **kwargs
        )
