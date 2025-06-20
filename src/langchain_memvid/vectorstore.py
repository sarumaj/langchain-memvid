"""
VectorStore implementation for LangChain MemVid.

This implementation is a wrapper around the LangChain MemVid vector store.
It uses the LangChain MemVid encoder to encode documents into QR codes and
the LangChain MemVid retriever to search for similar documents.
"""

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List, Optional, Dict, Any, Tuple, Generator
from pathlib import Path
import asyncio
import nest_asyncio
from contextlib import contextmanager

from .index import IndexManager
from .encoder import Encoder
from .retriever import Retriever
from .config import VectorStoreConfig, LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE, LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR
from .utils import get_on_first_match
from .logging import get_logger

logger = get_logger("vectorstore")


class VectorStore(VectorStore):
    """Vector store that stores documents in a video format using QR codes.

    This vector store uses `memvid` to encode documents into QR codes and store them
    in a video file. It provides semantic search capabilities using FAISS index.

    Attributes:
        video_file (str): Path to the video file storing QR codes, defaults to LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE
        index_dir (str): Path to the index directory for semantic search, defaults to LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR
        encoder (Encoder): Encoder for converting documents to QR codes
        _retriever (Optional[Retriever]): Lazy-loaded retriever for searching and decoding QR codes
    """

    def __init__(
        self, *,
        embedding: Embeddings,
        video_file: Path = LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE,
        index_dir: Path = LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR,
        config: Optional[VectorStoreConfig] = None,
    ):
        """Initialize VectorStore.

        Args:
            embedding: Embedding model for semantic search
            video_file: Path to store/load the video file, defaults to LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE
            index_dir: Path to store/load the index file, defaults to LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR
            config: Optional unified configuration. If not provided, uses default configs
        """
        self.video_file = Path(video_file).absolute()
        self.index_dir = Path(index_dir).absolute()
        self.config = config or VectorStoreConfig()

        # Initialize components with their respective configs
        self.index_manager = IndexManager(
            config=self.config.index,
            embeddings=embedding
        )

        self.encoder = Encoder(
            config=self.config,
            index_manager=self.index_manager
        )

        # Initialize retriever as None - will be created lazily
        self._retriever: Optional[Retriever] = None

    @contextmanager
    def _access_retriever(self, k: int) -> Generator[Retriever, None, None]:
        """Context manager for temporarily setting retriever's k value.

        This avoids creating copies of the retriever while maintaining thread safety
        through Python's GIL, which ensures that the context manager's enter and exit
        operations are atomic.

        Args:
            k: The temporary k value to set
        """
        original_k, self.retriever.k = self.retriever.k, k
        try:
            yield self.retriever
        finally:
            self.retriever.k = original_k

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

        nest_asyncio.apply(loop)
        return loop

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
                index_dir=self.index_dir,
                config=self.config,
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
            **kwargs: Additional arguments (ignored)

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
            self.encoder.add_chunks(texts, metadatas=metadatas)

            # Build video and index
            stats = self.encoder.build_video(
                output_file=self.video_file,
                index_dir=self.index_dir,
            )

            # Reload index in index manager after building
            if self.index_dir.exists():
                self.index_manager.load(self.index_dir)

            # Reset retriever if it exists to force recreation
            self._retriever = None

            # Log build statistics
            logger.info(
                f"Built video with {stats.total_chunks} chunks "
                f"({stats.video_size_mb:.2f} MB)"
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
            **kwargs: Additional arguments (ignored)

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
            **kwargs: Additional arguments (ignored)

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
            **kwargs: Additional arguments (ignored)

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
            **kwargs: Additional arguments (ignored)

        Returns:
            List of Document objects
        """
        with self._access_retriever(k) as retriever:
            return retriever._get_relevant_documents(query)

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
            **kwargs: Additional arguments (ignored)

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
            **kwargs: Additional arguments (ignored)

        Returns:
            List of (Document, score) tuples
        """
        docs = self.similarity_search(query, k=k, **kwargs)
        return [(doc, get_on_first_match(
            doc.metadata,
            "similarity",
            "distance",
            expected_type=float,
            default=0.0
        )) for doc in docs]

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
            **kwargs: Additional arguments (ignored)

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
        video_file: Path = LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE,
        index_dir: Path = LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """Create vector store from texts.

        Args:
            texts: List of text strings
            embedding: Embedding model
            video_file: Path to store the video file, defaults to LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE
            index_dir: Path to store the index file, defaults to LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR
            metadatas: Optional list of metadata dicts
            **kwargs: Additional arguments passed to constructor

        Returns:
            VectorStore instance
        """
        vectorstore = cls(
            video_file=video_file,
            index_dir=index_dir,
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
        video_file: Path = LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE,
        index_dir: Path = LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """Create vector store from texts asynchronously.

        Args:
            texts: List of text strings
            embedding: Embedding model
            video_file: Path to store the video file, defaults to LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE
            index_dir: Path to store the index file, defaults to LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR
            metadatas: Optional list of metadata dicts
            **kwargs: Additional arguments passed to constructor

        Returns:
            VectorStore instance
        """
        vectorstore = cls(
            video_file=video_file,
            index_dir=index_dir,
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
        video_file: Path = LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE,
        index_dir: Path = LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR,
        **kwargs: Any,
    ) -> "VectorStore":
        """Create vector store from documents.

        Args:
            documents: List of Document objects
            embedding: Embedding model
            video_file: Path to store the video file, defaults to LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE
            index_dir: Path to store the index file, defaults to LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR
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
            index_dir=index_dir,
            metadatas=metadatas,
            **kwargs
        )

    @classmethod
    async def afrom_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        video_file: Path = LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE,
        index_dir: Path = LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR,
        **kwargs: Any,
    ) -> "VectorStore":
        """Create vector store from documents asynchronously.

        Args:
            documents: List of Document objects
            embedding: Embedding model
            video_file: Path to store the video file, defaults to LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE
            index_dir: Path to store the index file, defaults to LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR
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
            index_dir=index_dir,
            metadatas=metadatas,
            **kwargs
        )
