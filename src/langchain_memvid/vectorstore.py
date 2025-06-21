"""
VectorStore implementation for LangChain MemVid.

This implementation stores documents as QR codes in video frames with semantic search
capabilities using FAISS index.

The vector store implements a hybrid storage approach:

- FAISS Index: Stores essential metadata (text, source, category, doc_id, metadata_hash) for fast search
- Video Storage: Stores complete document data as QR codes with all metadata fields

Optimized deletion strategies avoid full video rebuilds by using frame index mapping
for selective frame removal.

Usage Examples

Fast search with essential metadata:
    results = vector_store.similarity_search("query", include_full_metadata=False)

Complete search with full metadata:
    results = vector_store.similarity_search("query", include_full_metadata=True)

Optimized deletion:
    vector_store.delete_by_ids(["0", "5", "10"])

Storage statistics:
    stats = vector_store.get_storage_stats()
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
from .types import StorageStats

logger = get_logger("vectorstore")


class VectorStore(VectorStore):
    """Vector store that stores documents in a video format using QR codes.

    This vector store uses memvid to encode documents into QR codes and store them
    in a video file. It provides semantic search capabilities using FAISS index.

    The vector store implements a hybrid storage approach:

    - FAISS Index: Stores essential metadata for fast search operations
    - Video Storage: Stores complete document data as QR codes with all metadata fields

    Optimized deletion strategies use frame index mapping to avoid full video rebuilds.

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

    def delete_by_ids(self, doc_ids: List[str]) -> bool:
        """Delete documents by their IDs.

        Uses optimized deletion strategies with frame index mapping to avoid full video rebuilds.
        Falls back to full rebuild if optimized deletion fails.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            True if any documents were deleted, False otherwise

        Raises:
            ValueError: If no document IDs are provided
            RuntimeError: If video file doesn't exist or deletion fails

        Example:
            vector_store.delete_by_ids(["0", "5", "10"])
        """
        if not doc_ids:
            raise ValueError("No document IDs provided to delete")

        if not self.video_file.exists():
            raise RuntimeError(
                f"Video file {self.video_file} does not exist. "
                "No documents to delete."
            )

        try:
            # Convert string IDs to integers
            try:
                int_ids = [int(doc_id) for doc_id in doc_ids]
            except ValueError as e:
                raise ValueError(f"Invalid document ID format: {e}")

            # Delete from index manager
            deleted = self.index_manager.delete_by_ids(int_ids)

            if deleted:
                # Try optimized frame deletion first
                try:
                    optimized_success = self._optimized_delete_frames(int_ids)
                    if optimized_success:
                        # Reset retriever to force recreation
                        self._retriever = None
                        logger.info(f"Optimized deletion successful for {len(doc_ids)} documents")
                        return True
                except Exception as e:
                    logger.warning(f"Optimized deletion failed, falling back to rebuild: {e}")

                # Fallback to full rebuild if optimized deletion fails
                self._rebuild_video_after_deletion()

                # Reset retriever to force recreation
                self._retriever = None

                logger.info(f"Deleted {len(doc_ids)} documents and rebuilt video")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise RuntimeError(f"Failed to delete documents: {e}") from e

    def delete_by_texts(self, texts: List[str]) -> bool:
        """Delete documents by their text content.

        Args:
            texts: List of text contents to delete

        Returns:
            True if any documents were deleted, False otherwise

        Raises:
            ValueError: If no texts are provided
            RuntimeError: If video file doesn't exist or deletion fails
        """
        if not texts:
            raise ValueError("No texts provided to delete")

        if not self.video_file.exists():
            raise RuntimeError(
                f"Video file {self.video_file} does not exist. "
                "No documents to delete."
            )

        try:
            # Delete from index manager
            deleted = self.index_manager.delete_by_texts(texts)

            if deleted:
                # Rebuild video with remaining documents
                self._rebuild_video_after_deletion()

                # Reset retriever to force recreation
                self._retriever = None

                logger.info("Deleted documents with specified texts and rebuilt video")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete documents by texts: {e}")
            raise RuntimeError(f"Failed to delete documents by texts: {e}") from e

    def delete_documents(self, documents: List[Document]) -> bool:
        """Delete documents by Document objects.

        Args:
            documents: List of Document objects to delete

        Returns:
            True if any documents were deleted, False otherwise

        Raises:
            ValueError: If no documents are provided
            RuntimeError: If video file doesn't exist or deletion fails
        """
        if not documents:
            raise ValueError("No documents provided to delete")

        texts = [doc.page_content for doc in documents]
        return self.delete_by_texts(texts)

    async def adelete_by_ids(self, doc_ids: List[str]) -> bool:
        """Delete documents by their IDs asynchronously.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            True if any documents were deleted, False otherwise

        Raises:
            ValueError: If no document IDs are provided
            RuntimeError: If video file doesn't exist or deletion fails
        """
        # Since deletion operations are not async in the underlying components,
        # we run the sync version in a thread pool
        loop = self._get_event_loop()
        return await loop.run_in_executor(None, self.delete_by_ids, doc_ids)

    async def adelete_by_texts(self, texts: List[str]) -> bool:
        """Delete documents by their text content asynchronously.

        Args:
            texts: List of text contents to delete

        Returns:
            True if any documents were deleted, False otherwise

        Raises:
            ValueError: If no texts are provided
            RuntimeError: If video file doesn't exist or deletion fails
        """
        # Since deletion operations are not async in the underlying components,
        # we run the sync version in a thread pool
        loop = self._get_event_loop()
        return await loop.run_in_executor(None, self.delete_by_texts, texts)

    async def adelete_documents(self, documents: List[Document]) -> bool:
        """Delete documents by Document objects asynchronously.

        Args:
            documents: List of Document objects to delete

        Returns:
            True if any documents were deleted, False otherwise

        Raises:
            ValueError: If no documents are provided
            RuntimeError: If video file doesn't exist or deletion fails
        """
        if not documents:
            raise ValueError("No documents provided to delete")

        texts = [doc.page_content for doc in documents]
        return await self.adelete_by_texts(texts)

    def _rebuild_video_after_deletion(self):
        """Rebuild the video file with remaining documents after deletion.

        This method rebuilds the video file using the remaining documents
        in the index manager.

        Raises:
            RuntimeError: If video rebuilding fails
        """
        try:
            # Get all remaining documents from index manager
            remaining_docs = self.index_manager.get_all_documents()

            if not remaining_docs:
                # If no documents remain, remove the video file
                if self.video_file.exists():
                    self.video_file.unlink()
                if self.index_dir.exists():
                    import shutil
                    shutil.rmtree(self.index_dir)
                logger.info("No documents remaining, removed video and index files")
                return

            # Extract texts and reconstruct metadata from essential metadata
            # Essential metadata contains fields directly, not in a nested 'metadata' field
            texts = []
            metadatas = []

            for doc in remaining_docs:
                texts.append(doc.get("text", ""))

                # Reconstruct metadata from essential metadata fields
                # Essential metadata has: text, id, source, category, metadata_hash
                # We need to preserve the essential fields and add any additional ones
                metadata = {}
                for key, value in doc.items():
                    if key not in ["text", "id", "metadata_hash"]:  # Skip internal fields
                        metadata[key] = value

                metadatas.append(metadata)

            # Clear encoder and add remaining chunks
            # The encoder expects chunks in the format {"text": "...", "metadata": {...}}
            self.encoder.clear()
            self.encoder.add_chunks(texts, metadatas=metadatas)

            # Clear the index manager to avoid deduplication issues during rebuild
            # We'll reload it after the video is built
            self.index_manager._metadata = []
            self.index_manager._index = None

            # Build new video
            stats = self.encoder.build_video(
                output_file=self.video_file,
                index_dir=self.index_dir,
            )

            # Reload index in index manager after building
            if self.index_dir.exists():
                self.index_manager.load(self.index_dir)

            logger.info(
                f"Rebuilt video with {stats.total_chunks} remaining chunks "
                f"({stats.video_size_mb:.2f} MB)"
            )

        except Exception as e:
            logger.error(f"Failed to rebuild video after deletion: {e}")
            raise RuntimeError(f"Failed to rebuild video after deletion: {e}") from e

    def _optimized_delete_frames(self, doc_ids: List[int]) -> bool:
        """Optimized deletion using frame removal instead of full rebuild.

        This method removes specific frames from the video instead of rebuilding
        the entire video, which is much more efficient.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            RuntimeError: If frame deletion fails
        """
        try:
            # Get frame numbers to delete
            frames_to_delete = self.index_manager.get_frames_to_delete(doc_ids)

            if not frames_to_delete:
                logger.info("No frames to delete")
                return False

            # Create temporary output file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(
                suffix=self.video_file.suffix,
                delete=False
            )
            temp_output = Path(temp_file.name)
            temp_file.close()

            try:
                # Remove frames from video
                new_video_path = self.encoder.video_processor.remove_frames_from_video(
                    video_path=self.video_file,
                    frame_numbers=frames_to_delete,
                    output_path=temp_output
                )

                # Replace original video with new one
                import shutil
                shutil.move(str(new_video_path), str(self.video_file))

                # Update frame mappings
                self.index_manager.delete_frames_from_mapping(frames_to_delete)

                # Save updated index
                self.index_manager.save(self.index_dir.with_suffix('.d'))

                logger.info(f"Optimized deletion: removed {len(frames_to_delete)} frames from video")
                return True

            except Exception as e:
                # Cleanup temporary file on error
                if temp_output.exists():
                    temp_output.unlink()
                raise e

        except Exception as e:
            logger.error(f"Failed to perform optimized frame deletion: {e}")
            raise RuntimeError(f"Failed to perform optimized frame deletion: {e}") from e

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        include_full_metadata: bool = False,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return
            include_full_metadata: Whether to fetch full metadata from video
            **kwargs: Additional arguments (ignored)

        Returns:
            List of Document objects

        Example:
            # Fast search with essential metadata
            results = vector_store.similarity_search("query", include_full_metadata=False)

            # Complete search with full metadata
            results = vector_store.similarity_search("query", include_full_metadata=True)
        """
        with self._access_retriever(k) as retriever:
            docs = retriever._get_relevant_documents(query)

            # If full metadata is requested, fetch it from video
            if include_full_metadata:
                docs = self._enhance_documents_with_full_metadata(docs)

            return docs

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
        include_full_metadata: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores.

        Args:
            query: Query text
            k: Number of results to return
            include_full_metadata: Whether to fetch full metadata from video
            **kwargs: Additional arguments (ignored)

        Returns:
            List of (Document, score) tuples
        """
        docs = self.similarity_search(query, k=k, include_full_metadata=include_full_metadata, **kwargs)
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

    def get_document_by_id(self, doc_id: str, include_full_metadata: bool = False) -> Optional[Document]:
        """Get a document by its ID.

        Args:
            doc_id: Document ID as string
            include_full_metadata: Whether to fetch full metadata from video

        Returns:
            Document if found, None otherwise

        Raises:
            ValueError: If document ID format is invalid
            RuntimeError: If video file doesn't exist

        Example:
            # Fast retrieval with essential metadata
            doc = vector_store.get_document_by_id("123", include_full_metadata=False)

            # Complete retrieval with full metadata
            doc_full = vector_store.get_document_by_id("123", include_full_metadata=True)
        """
        try:
            # Convert string ID to integer
            try:
                int_id = int(doc_id)
            except ValueError as e:
                raise ValueError(f"Invalid document ID format: {e}")

            if not self.video_file.exists():
                raise RuntimeError(
                    f"Video file {self.video_file} does not exist. "
                    "No documents to retrieve."
                )

            return self.retriever.get_document_by_id(int_id, include_full_metadata=include_full_metadata)

        except Exception as e:
            logger.error(f"Failed to get document by ID: {e}")
            raise RuntimeError(f"Failed to get document by ID: {e}") from e

    def get_documents_by_ids(self, doc_ids: List[str], include_full_metadata: bool = False) -> List[Document]:
        """Get documents by their IDs.

        Args:
            doc_ids: List of document IDs as strings
            include_full_metadata: Whether to fetch full metadata from video

        Returns:
            List of Document objects

        Raises:
            ValueError: If document ID format is invalid
            RuntimeError: If video file doesn't exist
        """
        try:
            # Convert string IDs to integers
            try:
                int_ids = [int(doc_id) for doc_id in doc_ids]
            except ValueError as e:
                raise ValueError(f"Invalid document ID format: {e}")

            if not self.video_file.exists():
                raise RuntimeError(
                    f"Video file {self.video_file} does not exist. "
                    "No documents to retrieve."
                )

            return self.retriever.get_documents_by_ids(int_ids, include_full_metadata=include_full_metadata)

        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            raise RuntimeError(f"Failed to get documents by IDs: {e}") from e

    def _enhance_documents_with_full_metadata(self, docs: List[Document]) -> List[Document]:
        """Enhance documents with full metadata from video storage.

        Args:
            docs: List of documents with essential metadata

        Returns:
            List of documents with full metadata
        """
        enhanced_docs = []
        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            if doc_id is not None:
                # Try to get full metadata from video
                full_metadata = self.retriever._get_full_metadata_from_video(doc_id)
                if full_metadata:
                    # Merge essential and full metadata
                    merged_metadata = {**doc.metadata, **full_metadata}
                    merged_metadata["metadata_type"] = "full"
                    enhanced_doc = Document(
                        page_content=doc.page_content,
                        metadata=merged_metadata
                    )
                    enhanced_docs.append(enhanced_doc)
                else:
                    # Keep original document if full metadata not available
                    enhanced_docs.append(doc)
            else:
                # Keep original document if no doc_id
                enhanced_docs.append(doc)

        return enhanced_docs

    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics for the hybrid approach.

        Returns:
            StorageStats: Comprehensive storage statistics for the hybrid approach.

        Raises:
            RuntimeError: If video file doesn't exist

        Example:
            stats = vector_store.get_storage_stats()
            print(f"Total documents: {stats.total_documents}")
            print(f"Video file size: {stats.video_file_size_mb:.2f} MB")
            print(f"Index size: {stats.index_size_mb:.2f} MB")
            print(f"Redundancy percentage: {stats.redundancy_percentage:.1f}%")
            print(f"Storage efficiency: {stats.storage_efficiency}")

            # Frame mapping statistics
            frame_stats = stats.frame_mapping_stats
            print(f"Mapped documents: {frame_stats.mapped_documents}")
            print(f"Mapping coverage: {frame_stats.mapping_coverage:.1f}%")
        """
        try:
            if not self.video_file.exists():
                raise RuntimeError(
                    f"Video file {self.video_file} does not exist."
                )

            # Get index statistics
            index_stats = self.index_manager.get_frame_mapping_stats()

            # Get video file size
            video_size_mb = self.video_file.stat().st_size / (1024 * 1024)

            # Get index directory size
            index_size_mb = 0
            if self.index_dir.exists():
                index_size_mb = sum(
                    f.stat().st_size for f in self.index_dir.rglob('*') if f.is_file()
                ) / (1024 * 1024)

            # Calculate redundancy metrics
            total_docs = index_stats.total_documents
            essential_metadata_size = total_docs * 0.001  # Rough estimate per document
            full_metadata_size = video_size_mb * 0.8  # Rough estimate of metadata portion

            redundancy_percentage = (
                (essential_metadata_size / full_metadata_size * 100)
                if full_metadata_size > 0 else 0
            )

            return StorageStats(
                total_documents=total_docs,
                video_file_size_mb=video_size_mb,
                index_size_mb=index_size_mb,
                essential_metadata_size_mb=essential_metadata_size,
                full_metadata_size_mb=full_metadata_size,
                redundancy_percentage=redundancy_percentage,
                storage_efficiency="hybrid",
                frame_mapping_stats=index_stats
            )

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            raise RuntimeError(f"Failed to get storage stats: {e}") from e

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
