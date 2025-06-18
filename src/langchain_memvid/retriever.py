"""
Retriever module for LangChain MemVid.

This module provides functionality for retrieving documents from video storage
using semantic search and QR code decoding.
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
from .config import VectorStoreConfig
from .logging import get_logger

logger = get_logger("retriever")


class Retriever(BaseRetriever, BaseModel):
    """Retriever for LangChain MemVid vector store."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        strict=False,            # Allow type coercion
        from_attributes=True     # Allow conversion from objects with attributes
    )

    video_file: Path = Field(description="Path to the video file")
    index_dir: Path = Field(description="Path to the index directory")
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

    def model_post_init(self, __context: Any) -> None:
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

        Args:
            query: Query string

        Returns:
            List of relevant documents

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
                    # Create document with the text and metadata
                    doc = Document(
                        page_content=result.text,
                        metadata={
                            "source": result.source,
                            "category": result.category,
                            "similarity": result.similarity,
                            **(result.metadata or {})
                        }
                    )
                    documents.append(doc)
            else:
                for result in results:
                    doc = Document(
                        page_content=result.text,
                        metadata={
                            "source": result.source,
                            "category": result.category,
                            "similarity": result.similarity,
                            **(result.metadata or {})
                        }
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}") from e

    def get_document_by_id(self, doc_id: int) -> Optional[Document]:
        """Get a document by its ID.

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise

        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            # Get metadata
            metadata_list = self.index_manager.get_metadata([doc_id])
            if not metadata_list:
                return None

            metadata = metadata_list[0]
            return Document(
                page_content=metadata["text"],
                metadata=metadata["metadata"]
            )

        except Exception as e:
            logger.error(f"Failed to get document by ID: {str(e)}")
            raise RetrievalError(f"Failed to get document by ID: {str(e)}") from e

    def get_documents_by_ids(self, doc_ids: List[int]) -> List[Document]:
        """Get multiple documents by their IDs.

        Args:
            doc_ids: List of document IDs

        Returns:
            List of documents

        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            # Get metadata for all IDs
            metadata_list = self.index_manager.get_metadata(doc_ids)

            # Create documents with progress bar if more than 10 documents
            documents = []
            if len(metadata_list) > 10:
                for metadata in tqdm(metadata_list, desc="Processing documents"):
                    doc = Document(
                        page_content=metadata["text"],
                        metadata=metadata["metadata"]
                    )
                    documents.append(doc)
            else:
                for metadata in metadata_list:
                    doc = Document(
                        page_content=metadata["text"],
                        metadata=metadata["metadata"]
                    )
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {str(e)}")
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

    def clear_cache(self) -> None:
        """Clear the frame cache."""
        self._frame_cache.clear()
