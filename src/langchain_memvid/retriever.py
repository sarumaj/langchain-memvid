"""
Retriever - Fast semantic search, QR frame extraction, and context assembly

Original code from https://github.com/memvid/memvid/blob/main/memvid/retriever.py
"""

import orjson as json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Set
from dataclasses import dataclass, asdict
import time
import cv2
from pydantic import BaseModel, Field

from .utils import batch_extract_and_decode, extract_and_decode_cached
from .index import IndexManager

logger = logging.getLogger(__name__)


class RetrieverConfig(BaseModel):
    """Configuration for Retriever."""
    # Cache settings
    cache_size: int = Field(default=1000, description="Maximum number of frames to cache")
    max_workers: int = Field(default=4, description="Maximum number of parallel workers for frame decoding")

    # Search settings
    default_top_k: int = Field(default=5, description="Default number of results to return from search")
    default_window_size: int = Field(default=2, description="Default context window size")

    # Performance settings
    prefetch_batch_size: int = Field(default=10, description="Number of frames to prefetch in one batch")
    cache_prefetch_threshold: int = Field(default=5, description="Minimum number of frames to trigger prefetch")
    search_batch_size: int = Field(default=50, description="Batch size for parallel search operations")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")


@dataclass
class VideoMetadata:
    """Metadata about the video file."""
    total_frames: int
    fps: float
    file_path: str

    def dict(self) -> Dict[str, Any]:
        """Convert VideoMetadata to a dictionary."""
        return asdict(self)


@dataclass
class RetrieverStats:
    """Statistics about the retriever state.

    Attributes:
        video_file: Path to the video file
        total_frames: Total number of frames in the video
        fps: Video frames per second
        cache_size: Current number of cached frames
        cache_ttl: Cache time-to-live in seconds
        cache_age: Age of the cache in seconds
        config: Current configuration
        index_stats: Statistics from the index manager
    """
    video_file: str
    total_frames: int
    fps: float
    cache_size: int
    cache_ttl: int
    cache_age: float
    config: Dict[str, Any]
    index_stats: Dict[str, Any]

    def dict(self) -> Dict[str, Any]:
        """Convert RetrieverStats to a dictionary."""
        return asdict(self)


class SearchResult(TypedDict):
    """Type definition for search results with metadata."""
    text: str
    score: float
    chunk_id: int
    frame: int
    metadata: Dict[str, Any]


class Retriever:
    """
    Fast retrieval from QR code videos using semantic search.

    This class provides functionality for:
    - Semantic search in video content
    - QR code frame extraction and decoding
    - Context window retrieval
    - Frame caching for performance

    Attributes:
        video_file (str): Path to QR code video
        index_file (str): Path to index file
        config (RetrieverConfig): Configuration settings
        index_manager (IndexManager): Index manager for semantic search
        _frame_cache (Dict[int, str]): Cache for decoded frames
        video_metadata (VideoMetadata): Metadata about the video file
    """

    def __init__(
        self,
        video_file: str,
        index_file: str,
        config: RetrieverConfig,
        index_manager: IndexManager,
        load_index: bool = True
    ):
        """
        Initialize Retriever.

        Args:
            video_file: Path to QR code video
            index_file: Path to index file
            config: Configuration settings
            index_manager: Index manager for semantic search
            load_index: Whether to load the index immediately (default: True)

        Raises:
            ValueError: If video file cannot be opened
        """
        self.video_file = str(Path(video_file).absolute())
        self.index_file = str(Path(index_file).absolute())
        self.config = config

        # Store index manager
        self.index_manager = index_manager

        # Load index if requested
        if load_index:
            self.index_manager.load(self.index_file)

        # Initialize cache with TTL
        self._frame_cache: Dict[int, tuple[str, float]] = {}
        self._cache_creation_time = time.time()

        # Verify video file and get metadata
        self.video_metadata = self._verify_video()

        logger.info(f"Initialized retriever with {self.index_manager.get_stats().total_chunks} chunks")

    def _verify_video(self) -> VideoMetadata:
        """
        Verify video file is accessible and get metadata.

        Returns:
            VideoMetadata: Metadata about the video file

        Raises:
            ValueError: If video file cannot be opened
        """
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_file}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        logger.info(f"Video has {total_frames} frames at {fps} fps")
        return VideoMetadata(total_frames=total_frames, fps=fps, file_path=self.video_file)

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid based on TTL."""
        return time.time() - self._cache_creation_time < self.config.cache_ttl

    def _clean_expired_cache(self):
        """Remove expired cache entries."""
        if not self._is_cache_valid():
            self._frame_cache.clear()
            self._cache_creation_time = time.time()

    def _extract_frame_numbers(self, search_results: List[tuple]) -> Set[int]:
        """Extract unique frame numbers from search results."""
        return {result[2]["frame"] for result in search_results}

    def _extract_text_from_decoded(self, decoded_frames: Dict[int, str], metadata: Dict[str, Any]) -> str:
        """Extract text from decoded frame or fallback to metadata."""
        frame_num = metadata["frame"]
        if frame_num in decoded_frames:
            try:
                chunk_data = json.loads(decoded_frames[frame_num])
                return chunk_data["text"]
            except (json.JSONDecodeError, KeyError):
                pass
        return metadata["text"]

    def search(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Search for relevant chunks using semantic search.

        Args:
            query: Search query string
            top_k: Number of results to return (defaults to config.default_top_k)

        Returns:
            List[str]: List of relevant text chunks

        Note:
            Results are ordered by relevance, with most relevant chunks first.
        """
        start_time = time.time()
        top_k = top_k or self.config.default_top_k

        # Ensure index is loaded before search
        if not self.index_manager._is_loaded:
            self.index_manager.load(self.index_file)

        # Semantic search in index
        search_results = self.index_manager.search(query, top_k)

        # Extract unique frame numbers and decode frames
        frame_numbers = self._extract_frame_numbers(search_results)
        decoded_frames = self._decode_frames_parallel(frame_numbers)

        # Extract text from decoded data
        results = [
            self._extract_text_from_decoded(decoded_frames, metadata)
            for _, _, metadata in search_results
        ]

        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed:.3f}s for query: '{query[:50]}...'")

        return results

    def get_chunk_by_id(self, chunk_id: int) -> Optional[str]:
        """
        Get specific chunk by ID.

        Args:
            chunk_id: Chunk ID to retrieve

        Returns:
            Optional[str]: Chunk text if found, None otherwise
        """
        # Ensure index is loaded before retrieval
        if not self.index_manager._is_loaded:
            self.index_manager.load(self.index_file)

        metadata = self.index_manager.get_chunk_by_id(chunk_id)
        if metadata:
            frame_num = metadata["frame"]
            decoded = self._decode_single_frame(frame_num)
            if decoded:
                try:
                    chunk_data = json.loads(decoded)
                    return chunk_data["text"]
                except (json.JSONDecodeError, KeyError):
                    pass
            return metadata["text"]
        return None

    def _decode_single_frame(self, frame_number: int) -> Optional[str]:
        """
        Decode single frame with caching.

        Args:
            frame_number: Frame number to decode

        Returns:
            Optional[str]: Decoded frame data if successful, None otherwise
        """
        self._clean_expired_cache()

        # Check cache
        if frame_number in self._frame_cache:
            data, _ = self._frame_cache[frame_number]
            return data

        # Decode frame
        result = extract_and_decode_cached(self.video_file, frame_number)

        # Update cache
        if result and len(self._frame_cache) < self.config.cache_size:
            self._frame_cache[frame_number] = (result, time.time())

        return result

    def _decode_frames_parallel(self, frame_numbers: List[int]) -> Dict[int, str]:
        """
        Decode multiple frames in parallel.

        Args:
            frame_numbers: List of frame numbers to decode

        Returns:
            Dict[int, str]: Dictionary mapping frame numbers to decoded data
        """
        self._clean_expired_cache()

        # Check cache first
        results = {}
        uncached_frames = []

        for frame_num in frame_numbers:
            if frame_num in self._frame_cache:
                data, _ = self._frame_cache[frame_num]
                results[frame_num] = data
            else:
                uncached_frames.append(frame_num)

        if not uncached_frames:
            return results

        # Process uncached frames in batches
        for i in range(0, len(uncached_frames), self.config.prefetch_batch_size):
            batch = uncached_frames[i:i + self.config.prefetch_batch_size]

            # Decode batch in parallel
            decoded = batch_extract_and_decode(
                self.video_file,
                batch,
                max_workers=self.config.max_workers
            )

            # Update results and cache
            for frame_num, data in decoded.items():
                results[frame_num] = data
                if len(self._frame_cache) < self.config.cache_size:
                    self._frame_cache[frame_num] = (data, time.time())

        return results

    def search_with_metadata(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Search with full metadata.

        Args:
            query: Search query string
            top_k: Number of results to return (defaults to config.default_top_k)

        Returns:
            List[SearchResult]: List of result dictionaries with text, score, and metadata

        Note:
            Results include similarity scores and additional metadata for each chunk.
        """
        start_time = time.time()
        top_k = top_k or self.config.default_top_k

        # Ensure index is loaded before search
        if not self.index_manager._is_loaded:
            self.index_manager.load(self.index_file)

        # Semantic search
        search_results = self.index_manager.search(query, top_k)

        # Extract frame numbers and decode frames
        frame_numbers = self._extract_frame_numbers(search_results)
        decoded_frames = self._decode_frames_parallel(frame_numbers)

        # Build results with metadata
        results = [
            {
                "text": self._extract_text_from_decoded(decoded_frames, metadata),
                "score": 1.0 / (1.0 + distance),  # Convert distance to similarity score
                "chunk_id": chunk_id,
                "frame": metadata["frame"],
                "metadata": metadata
            }
            for chunk_id, distance, metadata in search_results
        ]

        elapsed = time.time() - start_time
        logger.info(f"Search with metadata completed in {elapsed:.3f}s")

        return results

    def get_context_window(self, chunk_id: int, window_size: Optional[int] = None) -> List[str]:
        """
        Get chunk with surrounding context.

        Args:
            chunk_id: Central chunk ID
            window_size: Number of chunks before/after to include (defaults to config.default_window_size)

        Returns:
            List[str]: List of chunks in context window

        Note:
            The context window includes the specified number of chunks before and after
            the central chunk, if available.
        """
        window_size = window_size or self.config.default_window_size

        # Generate range of chunk IDs to retrieve
        chunk_ids = range(chunk_id - window_size, chunk_id + window_size + 1)

        # Get chunks using list comprehension
        return [
            chunk for chunk_id in chunk_ids
            if (chunk := self.get_chunk_by_id(chunk_id)) is not None
        ]

    def prefetch_frames(self, frame_numbers: List[int]):
        """
        Prefetch frames into cache for faster retrieval.

        Args:
            frame_numbers: List of frame numbers to prefetch

        Note:
            This method is useful for optimizing performance when you know which frames
            will be needed in advance.
        """
        # Only prefetch frames not in cache
        to_prefetch = [f for f in frame_numbers if f not in self._frame_cache]

        if len(to_prefetch) >= self.config.cache_prefetch_threshold:
            logger.info(f"Prefetching {len(to_prefetch)} frames...")
            # Process in batches
            for i in range(0, len(to_prefetch), self.config.prefetch_batch_size):
                batch = to_prefetch[i:i + self.config.prefetch_batch_size]
                decoded = batch_extract_and_decode(
                    self.video_file,
                    batch,
                    max_workers=self.config.max_workers
                )
                # Update cache
                for frame_num, data in decoded.items():
                    if len(self._frame_cache) < self.config.cache_size:
                        self._frame_cache[frame_num] = (data, time.time())
                logger.info(f"Prefetched {len(decoded)} frames")

    def clear_cache(self):
        """
        Clear frame cache.

        Note:
            This method clears both the internal frame cache and the cached function results
            from extract_and_decode_cached.
        """
        self._frame_cache.clear()
        extract_and_decode_cached.cache_clear()
        self._cache_creation_time = time.time()
        logger.info("Cleared frame cache")

    def get_stats(self) -> RetrieverStats:
        """
        Get retriever statistics.

        Returns:
            RetrieverStats: Statistics about the retriever state including:
                - video_file: Path to the video file
                - total_frames: Total number of frames in the video
                - fps: Video frames per second
                - cache_size: Current number of cached frames
                - cache_ttl: Cache time-to-live in seconds
                - cache_age: Age of the cache in seconds
                - config: Current configuration
                - index_stats: Statistics from the index manager
        """
        return RetrieverStats(
            video_file=self.video_file,
            total_frames=self.video_metadata.total_frames,
            fps=self.video_metadata.fps,
            cache_size=len(self._frame_cache),
            cache_ttl=self.config.cache_ttl,
            cache_age=time.time() - self._cache_creation_time,
            config=self.config.model_dump(),
            index_stats=self.index_manager.get_stats().dict()
        )
