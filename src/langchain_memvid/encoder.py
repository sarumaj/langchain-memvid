"""
Encodes text chunks and metadata as QR codes in video frames for MemVid.

- Adds new documents and builds video storage with QR codes.
- Maintains mapping between document IDs and video frames for efficient deletion.
"""

import orjson
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .exceptions import EncodingError
from .video import VideoProcessor
from .index import IndexManager
from .config import VectorStoreConfig, LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE, LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR
from .logging import get_logger
from .types import BuildStats

logger = get_logger("encoder")


class Encoder:
    """Encodes text chunks and metadata as QR codes in video frames for MemVid.

    - Adds new documents and builds video storage with QR codes.
    - Maintains mapping between document IDs and video frames for efficient deletion.
    """

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
    ):
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
        output_file: Path = LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE,
        index_dir: Path = LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR,
    ) -> BuildStats:
        """Build video from added chunks.

        This method implements the hybrid storage approach and optimization strategies
        for efficient video building and frame mapping.

        Hybrid Storage Implementation

        - Essential Metadata: Stores only essential metadata in FAISS for efficiency

          - Document text, source, category, doc_id, metadata_hash
          - Significant reduction in FAISS index size
          - Fast search operations with minimal memory usage

        - Full Metadata: Stores complete metadata in video QR codes

          - All metadata fields and custom attributes
          - Complete backup and archive functionality
          - On-demand retrieval when needed
          - Metadata is stored in the video QR codes

        Optimization Strategies

        Frame Index Mapping

        - Bidirectional Mapping: Establishes doc_id frame_number mapping
        - O(1) Lookup: Enables constant-time frame number retrieval
        - Deletion Optimization: Allows precise frame-level deletion without full video rebuilds
        - Consistency: Maintains synchronization between FAISS index and video frames

        Performance Characteristics

        - Encoding Time: Optimized for large document collections
        - Memory Usage: Efficient processing of chunks and frames
        - Storage Efficiency: Hybrid approach reduces overall storage requirements
        - Quality: Maintains video quality while optimizing storage

        Process Flow

        1. Text Processing: Extract texts from chunks for FAISS indexing
        2. FAISS Indexing: Add essential metadata to FAISS index
        3. QR Code Generation: Create QR codes with full metadata
        4. Frame Mapping: Establish bidirectional document-to-frame mapping
        5. Video Encoding: Encode QR codes into video frames
        6. Index Saving: Save FAISS index with frame mappings

        Args:
            output_file: Path to save the video file, defaults to LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE
            index_dir: Path to save the index directory, defaults to LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR

        Returns:
            BuildStats: Statistics for the video build process including:

            - total_chunks: Number of chunks encoded
            - video_size_mb: Size of the video file in MB
            - encoding_time: Time taken for encoding in seconds
            - index_path: Path to the saved index
            - video_path: Path to the saved video

        Raises:
            EncodingError: If video building fails
            ValueError: If output paths are invalid

        Example:
            # Build video with hybrid storage approach
            stats = encoder.build_video(Path("output.mp4"), Path("index.d"))
            print(f"Encoded {stats.total_chunks} chunks in {stats.encoding_time:.2f}s")
            print(f"Video size: {stats.video_size_mb:.2f} MB")

            # Check frame mapping statistics
            frame_stats = encoder.index_manager.get_frame_mapping_stats()
            print(f"Frame mapping coverage: {frame_stats['mapping_coverage']:.1f}%")
        """
        try:
            # Validate paths
            if not output_file.parent.exists():
                raise ValueError(f"Output directory does not exist: {output_file.parent}")
            if not index_dir.parent.exists():
                raise ValueError(f"Index directory does not exist: {index_dir.parent}")

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
            for i, chunk in enumerate(self._chunks):
                # Convert chunk to JSON string
                chunk_data = orjson.dumps(chunk, option=orjson.OPT_NON_STR_KEYS).decode()
                # Create QR code
                qr_frame = self.video_processor.create_qr_code(chunk_data)
                qr_frames.append(qr_frame)

                # Set frame mapping for efficient deletion
                self.index_manager.set_frame_mapping(i, len(qr_frames) - 1)

            # Encode video
            self.video_processor.encode_video(
                frames=qr_frames,
                output_path=output_file
            )

            # Save index
            index_dir = index_dir.with_suffix('.d')
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

    def clear(self):
        """Clear all added chunks."""
        self._chunks = []
        logger.info("Cleared all chunks")
