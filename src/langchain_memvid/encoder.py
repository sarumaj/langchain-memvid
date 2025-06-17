"""
Encoder module for MemVid.

This module provides functionality for encoding text chunks into QR codes
and managing the encoding process.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .exceptions import EncodingError
from .video import VideoProcessor
from .index import IndexManager
from .config import VectorStoreConfig

logger = logging.getLogger(__name__)


@dataclass
class BuildStats:
    """Statistics for video build process."""
    total_chunks: int
    video_size_mb: float
    encoding_time: float
    index_path: Path
    video_path: Path


class Encoder:
    """Handles encoding of text chunks into QR codes and video frames."""

    def __init__(
        self,
        config: VectorStoreConfig,
        index_manager: IndexManager,
    ):
        """Initialize the encoder.

        Args:
            config: Configuration for the encoder
            index_manager: Index manager for storing embeddings

        Example:
            >>> config = VectorStoreConfig(...)
            >>> index_manager = IndexManager(...)
            >>> encoder = Encoder(config, index_manager)
        """
        self.config = config
        self.index_manager = index_manager
        self.video_processor = VideoProcessor(
            video_config=config.video,
            qrcode_config=config.qrcode
        )
        self._chunks: List[Dict[str, Any]] = []

    def add_chunks(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add text chunks for encoding.

        Args:
            texts: List of text chunks to encode
            metadatas: Optional list of metadata dictionaries for each chunk

        Raises:
            EncodingError: If adding chunks fails

        Example:
            >>> encoder.add_chunks(["text1", "text2"], [{"source": "doc1"}, {"source": "doc2"}])
        """
        try:
            if metadatas is None:
                metadatas = [{} for _ in texts]

            if len(texts) != len(metadatas):
                raise EncodingError("Number of texts must match number of metadata entries")

            # Add chunks with metadata
            for text, metadata in zip(texts, metadatas):
                self._chunks.append({
                    "text": text,
                    "metadata": metadata
                })

            logger.info(f"Added {len(texts)} chunks for encoding")

        except Exception as e:
            raise EncodingError(f"Failed to add chunks: {str(e)}")

    def build_video(
        self,
        output_file: Path,
        index_file: Path,
        **kwargs: Any,
    ) -> BuildStats:
        """Build video from added chunks.

        Args:
            output_file: Path to save the video file
            index_file: Path to save the index file
            **kwargs: Additional arguments for video building

        Returns:
            Dictionary with build statistics

        Raises:
            EncodingError: If video building fails
            ValueError: If output paths are invalid

        Example:
            >>> stats = encoder.build_video(Path("output.mp4"), Path("index.json"))
            >>> print(f"Encoded {stats['total_chunks']} chunks in {stats['encoding_time']:.2f}s")
        """
        try:
            # Validate paths
            if not output_file.parent.exists():
                raise ValueError(f"Output directory does not exist: {output_file.parent}")
            if not index_file.parent.exists():
                raise ValueError(f"Index directory does not exist: {index_file.parent}")

            if not self._chunks:
                raise EncodingError("No chunks to encode")

            start_time = time.time()

            # Get texts from chunks
            texts = [chunk["text"] for chunk in self._chunks]

            # Add vectors to index
            self.index_manager.add_texts(
                texts=texts,
                metadata=self._chunks
            )

            # Create QR codes for each chunk
            qr_frames = []
            for chunk in self._chunks:
                # Convert chunk to JSON string
                chunk_data = json.dumps(chunk)
                # Create QR code
                qr_frame = self.video_processor.create_qr_code(chunk_data)
                qr_frames.append(qr_frame)

            # Encode video
            self.video_processor.encode_video(
                frames=qr_frames,
                output_path=output_file
            )

            # Save index
            index_dir = index_file.with_suffix('.d')
            self.index_manager.save(index_dir)

            # Calculate statistics
            encoding_time = time.time() - start_time
            stats: BuildStats = BuildStats(
                total_chunks=len(self._chunks),
                video_size_mb=output_file.stat().st_size / (1024 * 1024),
                encoding_time=encoding_time,
                index_path=index_dir,
                video_path=output_file
            )

            # Clear chunks after successful build
            self._chunks = []

            logger.info(f"Built video with {stats.total_chunks} chunks in {stats.encoding_time:.2f}s")
            return stats

        except Exception as e:
            raise EncodingError(f"Failed to build video: {str(e)}")

    def clear(self) -> None:
        """Clear all added chunks."""
        self._chunks = []
        logger.info("Cleared all chunks")
