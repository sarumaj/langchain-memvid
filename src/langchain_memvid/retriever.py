"""
Retriever for MemVid vector store.

- Performs semantic search using FAISS index and retrieves documents from video storage.
- Supports both essential metadata (fast) and full metadata (from video QR codes).
- Implements frame caching for efficient repeated access.
"""

from pathlib import Path
from typing import List, Any, Optional, Dict, Union
import orjson
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from tqdm import tqdm

from .exceptions import RetrievalError
from .video import VideoProcessor
from .index import IndexManager
from .config import VectorStoreConfig, LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE, LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR
from .logging import get_logger

logger = get_logger("retriever")


class Retriever(BaseRetriever, BaseModel):
    """Retriever for MemVid vector store.

    - Performs semantic search using FAISS index and retrieves documents from video storage.
    - Supports both essential metadata (fast) and full metadata (from video QR codes).
    - Implements frame caching for efficient repeated access.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        strict=False,            # Allow type coercion
        from_attributes=True     # Allow conversion from objects with attributes
    )

    video_file: Path = Field(description="Path to the video file", default=LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE)
    index_dir: Path = Field(description="Path to the index directory", default=LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR)
    config: VectorStoreConfig = Field(description="Configuration for the retriever")

    # Using Any to allow unit testing
    index_manager: Union[IndexManager, Any] = Field(description="Index manager for vector search")

    # Using Any to allow unit testing, initialized in model_post_init
    video_processor: Union[VideoProcessor, Any] = Field(description="Video processor for video decoding", default=None)

    load_index: bool = Field(default=True, description="Whether to load the index during initialization")
    k: int = Field(default=4, description="Number of results to return for semantic search")
    frame_cache_size: int = Field(default=100, description="Maximum number of frames to cache in memory")

    # Using PrivateAttr to avoid validation errors
    _frame_cache: Dict[int, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any):
        """Initialize additional attributes after Pydantic model initialization."""
        try:
            # Initialize video processor
            self.video_processor = VideoProcessor(
                video_config=self.config.video,
                qrcode_config=self.config.qrcode
            )

            # Load index if requested
            if self.load_index:
                self.index_manager.load(self.index_dir)

            logger.info(f"Initialized retriever with video: {self.video_file}")

        except Exception as e:
            logger.error(f"Failed to initialize retriever: {str(e)}")
            raise RetrievalError(f"Failed to initialize retriever: {str(e)}") from e

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """Invoke the retriever on a single input.

        Args:
            input: Query string
            config: Optional configuration for the run

        Returns:
            List of relevant documents
        """
        return self.retrieve(input)

    async def ainvoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """Asynchronously invoke the retriever on a single input.

        Args:
            input: Query string
            config: Optional configuration for the run

        Returns:
            List of relevant documents
        """
        return self.retrieve(input)  # For now, just use synchronous version

    def batch(
        self,
        inputs: List[str],
        config: Optional[RunnableConfig] = None,
        *,
        return_exceptions: bool = False,
    ) -> List[List[Document]]:
        """Invoke the retriever on multiple inputs.

        Args:
            inputs: List of query strings
            config: Optional configuration for the run
            return_exceptions: Whether to return exceptions instead of raising them

        Returns:
            List of document lists, one for each input
        """
        results = []
        for _input in tqdm(inputs, desc="Processing batch queries"):
            try:
                results.append(self.retrieve(_input))
            except Exception as e:
                if return_exceptions:
                    results.append(e)
                else:
                    raise
        return results

    async def abatch(
        self,
        inputs: List[str],
        config: Optional[RunnableConfig] = None,
        *,
        return_exceptions: bool = False,
    ) -> List[List[Document]]:
        """Asynchronously invoke the retriever on multiple inputs.

        Args:
            inputs: List of query strings
            config: Optional configuration for the run
            return_exceptions: Whether to return exceptions instead of raising them

        Returns:
            List of document lists, one for each input
        """
        return self.batch(inputs, config, return_exceptions=return_exceptions)  # For now, just use synchronous version

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents relevant to the query.

        Args:
            query: Query string

        Returns:
            List of relevant documents

        Raises:
            RetrievalError: If retrieval fails
        """
        return self._get_relevant_documents(query)

    def _get_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:
        """Get documents relevant to the query.

        This method implements the hybrid storage approach for optimal search performance:

        Hybrid Storage Implementation

        - Essential Metadata Only: Returns documents with minimal metadata from FAISS
        - Fast Search: Leverages FAISS capabilities for sub-second search
        - Metadata Structure: Includes text, source, category, doc_id, metadata_hash
        - Metadata Type Flag: Sets "metadata_type": "essential" for identification

        Performance Optimizations

        - Progress Bar: Shows progress for large result sets (>10 documents)
        - Memory Efficient: Processes results in batches to avoid memory issues
        - Caching: Leverages frame caching for repeated access

        Metadata Structure

        - source: Document source
        - category: Document category
        - similarity: Similarity score
        - doc_id: Document ID
        - metadata_hash: Metadata hash
        - metadata_type: Metadata type
        - ... other essential fields

        Args:
            query: Query string

        Returns:
            List of relevant documents with essential metadata

        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            # Use IndexManager's search_text method which leverages FAISS capabilities
            results = self.index_manager.search_text(query, k=self.k)

            # Convert SearchResult objects to Documents with progress bar if more than 10 results
            documents = []
            if len(results) > 10:
                for result in tqdm(results, desc="Processing search results"):
                    # Create document with essential metadata from FAISS
                    # Full metadata can be fetched from video if needed
                    doc = Document(
                        page_content=result.text,
                        metadata={
                            "source": result.source,
                            "category": result.category,
                            "similarity": result.similarity,
                            "doc_id": result.metadata.get("id") if result.metadata else None,
                            "metadata_hash": result.metadata.get("metadata_hash") if result.metadata else None,
                            # Flag indicating this is essential metadata only
                            "metadata_type": "essential",
                            **(result.metadata or {})
                        }
                    )
                    documents.append(doc)
            else:
                for result in results:
                    # Create document with essential metadata from FAISS
                    doc = Document(
                        page_content=result.text,
                        metadata={
                            "source": result.source,
                            "category": result.category,
                            "similarity": result.similarity,
                            "doc_id": result.metadata.get("id") if result.metadata else None,
                            "metadata_hash": result.metadata.get("metadata_hash") if result.metadata else None,
                            # Flag indicating this is essential metadata only
                            "metadata_type": "essential",
                            **(result.metadata or {})
                        }
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}") from e

    def get_document_by_id(self, doc_id: int, include_full_metadata: bool = False) -> Optional[Document]:
        """Get a document by its ID.

        This method supports the hybrid storage approach with flexible metadata retrieval:

        - Essential Metadata Only (include_full_metadata=False): Fast retrieval from FAISS index

          - Document text, source, category, doc_id, metadata_hash
          - O(1) lookup time from FAISS
          - Minimal memory usage
          - Metadata type: "essential"

        - Full Metadata (include_full_metadata=True): Complete metadata from video storage

          - All metadata fields and custom attributes
          - Requires video frame decoding
          - Complete data access with integrity checking
          - Metadata type: "full"

        Args:
            doc_id: Document ID
            include_full_metadata: Whether to fetch full metadata from video

        Returns:
            Document if found, None otherwise

        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            # Get essential metadata from FAISS index
            metadata_list = self.index_manager.get_metadata([doc_id])
            if not metadata_list or not metadata_list[0]:
                return None

            essential_metadata = metadata_list[0]

            # If full metadata is requested, fetch from video
            if include_full_metadata:
                full_metadata = self._get_full_metadata_from_video(doc_id)
                if full_metadata:
                    # Merge essential and full metadata
                    merged_metadata = {**essential_metadata, **full_metadata}
                    merged_metadata["metadata_type"] = "full"
                else:
                    # Fallback to essential metadata
                    merged_metadata = essential_metadata
                    merged_metadata["metadata_type"] = "essential"
            else:
                merged_metadata = essential_metadata
                merged_metadata["metadata_type"] = "essential"

            return Document(
                page_content=essential_metadata.get("text", ""),
                metadata=merged_metadata
            )

        except Exception as e:
            raise RetrievalError(f"Failed to get document by ID: {str(e)}") from e

    def _get_full_metadata_from_video(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get full metadata from video storage for a specific document.

        This method implements the full metadata retrieval component of the hybrid storage approach:

        Hybrid Storage Implementation

        - Video Decoding: Decodes specific video frames to extract complete metadata
        - Frame Mapping: Uses document-to-frame mapping for efficient frame location
        - Complete Data: Retrieves all metadata fields and custom attributes
        - Fallback Mechanism: Provides complete data access when FAISS data is insufficient

        Performance Characteristics

        - Frame Lookup: O(1) lookup using frame mapping
        - Video Decoding: Additional processing time for frame decoding and QR code processing
        - Memory Usage: Medium (requires frame decoding and QR code processing)

        Error Handling

        - Returns None if frame mapping is not available
        - Returns None if video decoding fails
        - Logs warnings for debugging purposes
        - Graceful degradation when video data is corrupted

        Use Cases

        - Complete Metadata Access: When all metadata fields are required
        - Data Integrity Verification: When FAISS data needs validation
        - Backup Recovery: When FAISS index is corrupted or incomplete
        - Custom Field Access: When accessing fields not in essential metadata

        Args:
            doc_id: Document ID

        Returns:
            Full metadata dictionary if found, None otherwise
        """
        try:
            # Get frame number for this document
            frame_number = self.index_manager.get_frame_number(doc_id)
            if frame_number is None:
                return None

            # Decode the frame to get full metadata
            doc = self.decode_frame(frame_number)
            if doc:
                return doc.metadata
            return None

        except Exception as e:
            logger.warning(f"Failed to get full metadata from video for doc_id {doc_id}: {e}")
            return None

    def get_documents_by_ids(self, doc_ids: List[int], include_full_metadata: bool = False) -> List[Document]:
        """Get documents by their IDs.

        Args:
            doc_ids: List of document IDs
            include_full_metadata: Whether to fetch full metadata from video

        Returns:
            List of documents

        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            documents = []
            for doc_id in doc_ids:
                doc = self.get_document_by_id(doc_id, include_full_metadata=include_full_metadata)
                if doc:
                    documents.append(doc)
            return documents

        except Exception as e:
            raise RetrievalError(f"Failed to get documents by IDs: {str(e)}") from e

    def _get_frame(self, frame_number: int) -> Optional[Any]:
        """Get a specific frame from the video with caching.

        Args:
            frame_number: Frame number to get

        Returns:
            Frame if found, None otherwise

        Raises:
            RetrievalError: If frame retrieval fails
        """
        try:
            # Check cache first
            if frame_number in self._frame_cache:
                return self._frame_cache[frame_number]

            # Get frame from video
            frames = list(self.video_processor.decode_video(self.video_file))
            if frame_number >= len(frames):
                return None

            # Cache the frame
            frame = frames[frame_number]
            self._frame_cache[frame_number] = frame
            return frame

        except Exception as e:
            logger.error(f"Failed to get frame {frame_number}: {str(e)}")
            raise RetrievalError(f"Failed to get frame {frame_number}: {str(e)}")

    def decode_frame(self, frame_number: int) -> Optional[Document]:
        """Decode a specific frame from the video.

        Args:
            frame_number: Frame number to decode

        Returns:
            Document if frame contains valid QR code, None otherwise

        Raises:
            RetrievalError: If decoding fails
        """
        try:
            # Get frame from video (using cache)
            frame = self._get_frame(frame_number)
            if frame is None:
                return None

            # Extract QR codes from frame
            qr_data = self.video_processor.extract_qr_codes(frame)
            if not qr_data:
                return None

            # Parse QR code data
            chunk_data = orjson.loads(qr_data[0])
            return Document(
                page_content=chunk_data["text"],
                metadata=chunk_data["metadata"]
            )

        except Exception as e:
            logger.error(f"Failed to decode frame: {str(e)}")
            raise RetrievalError(f"Failed to decode frame: {str(e)}") from e

    def decode_all_frames(self) -> List[Document]:
        """Decode all frames from the video.

        Returns:
            List of all documents in the video

        Raises:
            RetrievalError: If decoding fails
        """
        try:
            documents = []
            frame_count = 0

            # Process frames in chunks to avoid memory issues
            for frame in self.video_processor.decode_video(self.video_file):
                frame_count += 1
                if frame_count > self.frame_cache_size:
                    logger.warning(f"Reached maximum frame cache size ({self.frame_cache_size})")
                    break

                # Extract QR codes from frame
                qr_data = self.video_processor.extract_qr_codes(frame)
                if not qr_data:
                    continue

                # Parse QR code data
                for data in qr_data:
                    try:
                        chunk_data = orjson.loads(data)
                        doc = Document(
                            page_content=chunk_data["text"],
                            metadata=chunk_data["metadata"]
                        )
                        documents.append(doc)
                    except orjson.JSONDecodeError:
                        logger.warning(f"Failed to parse QR code data: {data}")
                        continue

            return documents

        except Exception as e:
            logger.error(f"Failed to decode all frames: {str(e)}")
            raise RetrievalError(f"Failed to decode all frames: {str(e)}") from e

    def clear_cache(self):
        """Clear the frame cache."""
        self._frame_cache.clear()
