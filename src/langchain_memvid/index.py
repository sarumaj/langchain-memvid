"""
Manages the FAISS vector index and essential metadata for MemVid.

- Stores essential metadata for fast search.
- Maintains bidirectional mapping between document IDs and video frame numbers for efficient deletion.

This module provides functionality for managing vector indices used in LangChain MemVid,
including FAISS index creation, updating, and searching.
"""

import faiss
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import orjson

from .exceptions import MemVidIndexError
from .config import IndexConfig, LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR
from .utils import ProgressDisplay
from .logging import get_logger
from .types import FrameMappingStats

logger = get_logger("index")


@dataclass
class SearchResult:
    """Represents a search result with metadata and similarity score."""
    text: str
    source: Optional[str] = None
    category: Optional[str] = None
    similarity: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any], similarity: float) -> 'SearchResult':
        """Create a SearchResult from metadata dictionary and similarity score."""
        # Create a copy of metadata to avoid modifying the original
        metadata = metadata.copy()

        # Extract text from the main metadata or nested metadata
        text = metadata.get('text', '')
        if not text and 'metadata' in metadata:
            text = metadata['metadata'].get('text', '')

        # Create a clean metadata dict without nested structures
        clean_metadata = {}
        for key, value in metadata.items():
            if key != 'metadata':  # Skip nested metadata
                clean_metadata[key] = value
            elif isinstance(value, dict):  # If we have nested metadata, merge it
                clean_metadata.update(value)

        return cls(
            text=text,
            source=clean_metadata.get('source'),
            category=clean_metadata.get('category'),
            similarity=similarity,
            metadata=clean_metadata
        )


class IndexManager:
    """Manages vector indices for MemVid.

    This index manager implements a hybrid storage approach that optimizes storage efficiency
    while maintaining performance and data integrity.

    Hybrid Storage Approach

    - Essential Metadata Only: Stores only essential metadata in FAISS for efficiency

      - Document text, source, category, doc_id, metadata_hash
      - Significant reduction in FAISS index size compared to full metadata storage
      - Fast search operations with minimal memory usage

    - Full Metadata in Video: Complete metadata stored in video QR codes

      - All metadata fields and custom attributes
      - Complete backup and archive functionality
      - On-demand retrieval when needed

    Optimization Strategies for Document Deletion

    The index manager implements optimized deletion strategies to avoid full video rebuilds:

    Frame Index Mapping

    - Maintains bidirectional mapping between document IDs and frame numbers
    - Enables O(1) lookup for frame numbers given document IDs
    - Allows precise frame-level deletion without full video rebuilds

    Performance Characteristics

    - Search Performance: Sub-second search with essential metadata
    - Storage Efficiency: Significant reduction in FAISS index size
    - Deletion Performance: O(k) time complexity where k = frames to delete
    - Memory Usage: Optimized for large-scale operations

    Best Practices

    - Batch Operations: Add or delete multiple documents at once for better efficiency
    - Frame Mapping: Monitor frame mapping integrity for optimal deletion performance
    - Metadata Management: Use essential metadata for search, full metadata for details
    - Error Handling: Implement fallback mechanisms for corrupted data
    """

    def __init__(
        self,
        config: IndexConfig,
        embeddings: Any,
    ):
        """Initialize the index manager.

        Args:
            config: Configuration for the index
            embeddings: LangChain embeddings interface
        """
        self.config = config
        self.embeddings = embeddings
        self._index: Optional[faiss.Index] = None
        self._metadata: List[Dict[str, Any]] = []
        self._is_trained: bool = False
        self._dimension: Optional[int] = None
        self._min_points: Optional[int] = None
        self._progress = ProgressDisplay(show_progress=config.show_progress)

        # Frame mapping for efficient deletion
        self._frame_mapping: Dict[int, int] = {}  # doc_id -> frame_number
        self._reverse_frame_mapping: Dict[int, int] = {}  # frame_number -> doc_id

    def create_index(self):
        """Create a new FAISS index based on the embeddings model."""
        try:
            # Get dimension from embeddings
            test_vector = self.embeddings.embed_query("test")
            self._dimension = len(test_vector)

            match self.config.index_type:
                case "faiss":
                    match self.config.metric:
                        case "cosine" | "ip":
                            self._index = faiss.IndexFlatIP(self._dimension)
                        case "l2":
                            self._index = faiss.IndexFlatL2(self._dimension)
                        case _:
                            raise MemVidIndexError(f"Unsupported metric: {self.config.metric}")

                    # If using IVF index
                    if self.config.nlist > 0:
                        # FAISS requires at least 30 * nlist points for training
                        self._min_points = 30 * self.config.nlist

                        # Use flat index if minimum points is too high
                        if self._min_points > 1000:
                            logger.warning(
                                f"Minimum points required ({self._min_points}) is too high. "
                                "Falling back to flat index."
                            )
                            self._is_trained = True
                        else:
                            # We'll convert to IVF when we have enough points
                            self._is_trained = True
                    else:
                        self._is_trained = True

                case _:
                    raise MemVidIndexError(f"Unsupported index type: {self.config.index_type}")

            logger.info(f"Created {self.config.index_type} index with {self.config.metric} metric")

        except Exception as e:
            raise MemVidIndexError(f"Failed to create index: {str(e)}")

    def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add texts and essential metadata to the index."""
        try:
            if self._index is None:
                self.create_index()

            # Convert texts to vectors using embeddings with progress bar
            logger.info(f"Embedding {len(texts)} texts...")
            with self._progress.progress(total=1, desc="Embedding texts") as pbar:
                vectors = np.array(self.embeddings.embed_documents(texts), dtype='float32')
                pbar.update(1)

            # Use empty metadata if none provided
            if metadata is None:
                metadata = [{"text": text} for text in texts]
            else:
                # Ensure each metadata dict has the original text
                for i, text in enumerate(texts):
                    if "text" not in metadata[i]:
                        metadata[i]["text"] = text

            # Create a mapping of text to index for deduplication
            text_to_idx = {m["text"]: i for i, m in enumerate(self._metadata)}

            # Filter out duplicates and keep track of which texts to add
            unique_indices = []
            for i, text in self._progress.tqdm(enumerate(texts), desc="Deduplicating texts", total=len(texts)):
                if text not in text_to_idx:
                    unique_indices.append(i)
                    text_to_idx[text] = len(self._metadata) + len(unique_indices) - 1

            if not unique_indices:
                logger.info("No new texts to add - all were duplicates")
                return

            # Filter vectors and metadata to only include unique texts
            unique_vectors = vectors[unique_indices]
            unique_metadata = [metadata[i] for i in unique_indices]

            # Store only essential metadata in FAISS for efficiency
            # Full metadata will be stored in video QR codes
            essential_metadata = []
            for i, meta in enumerate(unique_metadata):
                # Create a stable hash of the metadata for integrity checking
                metadata_str = str(sorted(meta.items()))
                metadata_hash = hashlib.sha256(metadata_str.encode('utf-8')).hexdigest()

                essential_meta = {
                    "text": meta.get("text", ""),
                    "id": len(self._metadata) + i,  # Document ID for mapping
                    # Store only frequently accessed fields
                    "source": meta.get("source"),
                    "category": meta.get("category"),
                    # Store a hash of the full metadata for integrity checking
                    "metadata_hash": metadata_hash
                }
                essential_metadata.append(essential_meta)

            # Check if we should convert to IVF index
            if (
                self.config.nlist > 0
                and not isinstance(self._index, faiss.IndexIVFFlat)
                and self._index.ntotal + len(unique_vectors) >= self._min_points
            ):
                logger.info("Converting to IVF index...")
                # Create IVF index
                quantizer = self._index
                metric = (
                    faiss.METRIC_INNER_PRODUCT if self.config.metric == "cosine"
                    else faiss.METRIC_L2
                )
                self._index = faiss.IndexIVFFlat(
                    quantizer,
                    self._dimension,
                    self.config.nlist,
                    metric
                )

                # Get all existing vectors with progress bar
                batch_size = 1000
                all_vectors = np.zeros(
                    (self._index.ntotal, self._dimension),
                    dtype='float32'
                )
                for i in self._progress.tqdm(
                    range(0, self._index.ntotal, batch_size),
                    desc="Reconstructing vectors"
                ):
                    end_idx = min(i + batch_size, self._index.ntotal)
                    self._index.reconstruct_n(i, end_idx - i, all_vectors[i:end_idx])

                # Train the index
                logger.info("Training IVF index...")
                self._index.train(all_vectors)
                self._is_trained = True

                # Add back the vectors with progress bar
                for i in self._progress.tqdm(
                    range(0, len(all_vectors), batch_size),
                    desc="Adding vectors to IVF index"
                ):
                    batch = all_vectors[i:i + batch_size]
                    self._index.add(batch)
                logger.info(f"Converted to IVF index and trained with {self._index.ntotal} points")

            # Check if IVF index needs training
            if isinstance(self._index, faiss.IndexIVFFlat) and not self._is_trained:
                # Train the index with these vectors
                logger.info("Training IVF index...")
                self._index.train(unique_vectors)
                self._is_trained = True

            # Normalize vectors for cosine similarity
            if self.config.metric == "cosine":
                faiss.normalize_L2(unique_vectors)

            # Add vectors to index in batches with progress bar
            batch_size = 1000
            for i in self._progress.tqdm(range(0, len(unique_vectors), batch_size), desc="Adding vectors to index"):
                batch = unique_vectors[i:i + batch_size]
                self._index.add(batch)
            self._metadata.extend(essential_metadata)

            logger.info(f"Added {len(unique_vectors)} unique texts to index")

        except Exception as e:
            raise MemVidIndexError(f"Failed to add texts: {str(e)}")

    def search_text(
        self,
        query_text: str,
        k: int = 4,
    ) -> List[SearchResult]:
        """Search for similar texts using a text query."""
        try:
            if self._index is None:
                raise MemVidIndexError("Index not initialized")

            # Convert query text to vector
            query_vector = np.array(self.embeddings.embed_query(query_text), dtype='float32').reshape(1, -1)

            # Normalize query vector for cosine similarity
            if self.config.metric == "cosine":
                faiss.normalize_L2(query_vector)

            # Search using the vector
            distances, indices = self._index.search(query_vector, k)

            # Create SearchResult objects
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                metadata = self._metadata[idx]
                results.append(SearchResult.from_metadata(metadata, float(distance)))

            return results

        except Exception as e:
            raise MemVidIndexError(f"Failed to search text: {str(e)}")

    def get_metadata(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get metadata for given indices."""
        try:
            return [self._metadata[i] for i in indices]
        except Exception as e:
            raise MemVidIndexError(f"Failed to get metadata: {str(e)}")

    def delete_by_ids(self, doc_ids: List[int]) -> bool:
        """Delete documents by their IDs and update index and mappings."""
        try:
            if self._index is None:
                raise MemVidIndexError("Index not initialized")

            if not doc_ids:
                return False

            # Validate IDs
            max_id = len(self._metadata) - 1
            invalid_ids = [doc_id for doc_id in doc_ids if doc_id < 0 or doc_id > max_id]
            if invalid_ids:
                raise MemVidIndexError(f"Invalid document IDs: {invalid_ids}")

            # Sort IDs in descending order to avoid index shifting issues
            doc_ids = sorted(doc_ids, reverse=True)

            # Remove from metadata first
            for doc_id in doc_ids:
                del self._metadata[doc_id]

            # Rebuild the index without the deleted vectors
            self._rebuild_index_without_deleted(doc_ids)

            logger.info(f"Deleted {len(doc_ids)} documents from index")
            return True

        except Exception as e:
            raise MemVidIndexError(f"Failed to delete documents: {str(e)}")

    def delete_by_texts(self, texts: List[str]) -> bool:
        """Delete documents by their text content."""
        try:
            if self._index is None:
                raise MemVidIndexError("Index not initialized")

            if not texts:
                return False

            # Find document IDs by text content
            doc_ids = []
            for text in texts:
                for i, metadata in enumerate(self._metadata):
                    if metadata.get("text") == text:
                        doc_ids.append(i)
                        break  # Only delete first occurrence of each text

            if not doc_ids:
                logger.info("No documents found with the specified texts")
                return False

            return self.delete_by_ids(doc_ids)

        except Exception as e:
            raise MemVidIndexError(f"Failed to delete documents by texts: {str(e)}")

    def _rebuild_index_without_deleted(self, deleted_ids: List[int]):
        """Rebuild the index after deleting specified document IDs.

        Args:
            deleted_ids: List of document IDs that were deleted (in descending order)

        Raises:
            MemVidIndexError: If rebuilding fails
        """
        try:
            # Create a new index of the same type
            self._index = None
            self.create_index()

            # If we have remaining documents, rebuild the index
            if self._metadata:
                # Get all remaining texts
                remaining_texts = [metadata.get("text", "") for metadata in self._metadata]

                # Convert texts to vectors
                vectors = np.array(self.embeddings.embed_documents(remaining_texts), dtype='float32')

                # Normalize vectors for cosine similarity
                if self.config.metric == "cosine":
                    faiss.normalize_L2(vectors)

                # Add vectors to the new index
                self._index.add(vectors)

                # Reassign document IDs to be sequential starting from 0
                for i, metadata in enumerate(self._metadata):
                    metadata["id"] = i

            logger.info(f"Rebuilt index with {len(self._metadata)} remaining documents")

        except Exception as e:
            raise MemVidIndexError(f"Failed to rebuild index: {str(e)}")

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in the index.

        Returns:
            List of all document metadata dictionaries

        Raises:
            MemVidIndexError: If retrieval fails
        """
        try:
            return self._metadata.copy()
        except Exception as e:
            raise MemVidIndexError(f"Failed to get all documents: {str(e)}")

    def get_document_count(self) -> int:
        """Get the total number of documents in the index.

        Returns:
            Number of documents in the index
        """
        return len(self._metadata) if self._metadata else 0

    def set_frame_mapping(self, doc_id: int, frame_number: int):
        """Set the frame mapping for a document.

        This method establishes the bidirectional mapping between document IDs and frame numbers,
        which is essential for optimized deletion strategies.

        Frame Index Mapping

        - Bidirectional Mapping: doc_id frame_number for efficient lookups
        - O(1) Lookup: Enables constant-time frame number retrieval
        - Deletion Optimization: Allows precise frame-level deletion without full video rebuilds
        - Consistency: Maintains synchronization between FAISS index and video frames

        Performance Benefits

        - Fast Deletion: O(k) time complexity where k = frames to delete
        - Memory Efficient: Minimal memory overhead for mapping storage
        - Scalable: Efficient for large document collections
        - Reliable: Provides fallback mechanisms when mappings are corrupted

        Use Cases

        - Optimized Deletion: Enables selective frame removal from videos
        - Frame Lookup: Fast retrieval of frame numbers for document IDs
        - Document Lookup: Fast retrieval of document IDs for frame numbers
        - Statistics: Provides mapping coverage statistics for monitoring

        Args:
            doc_id: Document ID
            frame_number: Frame number in the video

        Example:
            # Set frame mapping for a document
            index_manager.set_frame_mapping(123, 5)

            # Retrieve frame number
            frame_num = index_manager.get_frame_number(123)  # Returns 5

            # Retrieve document ID
            doc_id = index_manager.get_document_id(5)  # Returns 123
        """
        self._frame_mapping[doc_id] = frame_number
        self._reverse_frame_mapping[frame_number] = doc_id

    def get_frame_number(self, doc_id: int) -> Optional[int]:
        """Get the frame number for a document.

        Args:
            doc_id: Document ID

        Returns:
            Frame number if found, None otherwise
        """
        return self._frame_mapping.get(doc_id)

    def get_document_id(self, frame_number: int) -> Optional[int]:
        """Get the document ID for a frame.

        Args:
            frame_number: Frame number

        Returns:
            Document ID if found, None otherwise
        """
        return self._reverse_frame_mapping.get(frame_number)

    def get_frames_to_delete(self, doc_ids: List[int]) -> List[int]:
        """Get frame numbers that need to be deleted for given document IDs.

        This method is a key component of the optimized deletion strategy, enabling
        precise frame-level deletion without full video rebuilds.

        Optimization Strategy

        - Frame Mapping Lookup: Uses O(1) lookup to find frame numbers for document IDs
        - Safe Deletion Order: Returns frames in reverse order for safe deletion
        - Efficient Processing: Processes multiple document IDs in a single operation
        - Error Handling: Gracefully handles missing frame mappings

        Performance Characteristics

        - Lookup Time: O(k) where k = number of document IDs
        - Memory Usage: Minimal temporary storage for frame numbers
        - Scalability: Efficient for large-scale deletions
        - Reliability: Handles missing mappings gracefully

        Use Cases

        - Video Frame Removal: Provides frame numbers for selective video editing
        - Optimized Deletion: Enables efficient document removal without full rebuilds
        - Batch Processing: Supports deletion of multiple documents at once
        - Statistics: Provides data for deletion performance analysis

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            List of frame numbers to delete, sorted in reverse order for safe deletion

        Example:
            # Get frames to delete for multiple documents
            doc_ids = [0, 5, 10]
            frames_to_delete = index_manager.get_frames_to_delete(doc_ids)
            print(f"Frames to delete: {frames_to_delete}")  # e.g., [10, 5, 0]

            # Use frames for video editing
            video_processor.remove_frames_from_video(video_path, frames_to_delete)
        """
        frames = []
        for doc_id in doc_ids:
            frame_number = self._frame_mapping.get(doc_id)
            if frame_number is not None:
                frames.append(frame_number)
        return sorted(frames, reverse=True)  # Sort in descending order for safe deletion

    def delete_frames_from_mapping(self, frame_numbers: List[int]):
        """Remove frame mappings for deleted frames.

        Args:
            frame_numbers: List of frame numbers that were deleted
        """
        for frame_number in frame_numbers:
            doc_id = self._reverse_frame_mapping.pop(frame_number, None)
            if doc_id is not None:
                self._frame_mapping.pop(doc_id, None)

    def get_frame_mapping_stats(self) -> FrameMappingStats:
        """Get statistics about frame mappings for monitoring and optimization.

        Returns:
            FrameMappingStats: Statistics about frame mappings.
        """
        return FrameMappingStats(
            total_documents=len(self._metadata),
            mapped_documents=len(self._frame_mapping),
            mapping_coverage=(len(self._frame_mapping) / len(self._metadata)) * 100 if self._metadata else 0,
            mapping_efficiency={
                "total_frames": len(self._reverse_frame_mapping),
                "frame_range": {
                    "min": min(self._reverse_frame_mapping.keys()) if self._reverse_frame_mapping else None,
                    "max": max(self._reverse_frame_mapping.keys()) if self._reverse_frame_mapping else None
                }
            }
        )

    def save(self, path: Path = LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR):
        """Save the index and metadata to disk.

        Args:
            path: Path to save the index, defaults to LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR

        Raises:
            MemVidIndexError: If saving fails
        """
        try:
            if self._index is None:
                raise MemVidIndexError("No index to save")

            # Create directory if it doesn't exist
            path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self._index, str(path / "index.faiss"))

            # Save metadata
            with open(path / "metadata.json", "wb") as f:
                f.write(orjson.dumps(self._metadata, option=orjson.OPT_NON_STR_KEYS))

            logger.info(f"Saved index to {path}")

        except Exception as e:
            raise MemVidIndexError(f"Failed to save index: {str(e)}")

    def load(self, path: Path = LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR):
        """Load the index and metadata from disk.

        Args:
            path: Path to load the index from, defaults to LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR

        Raises:
            MemVidIndexError: If loading fails
        """
        try:
            if not path.exists() and not path.is_dir():
                raise FileNotFoundError(f"Path {path} does not exist or is not a directory")

            index_file = path / "index.faiss"
            metadata_file = path / "metadata.json"

            if not index_file.exists():
                raise FileNotFoundError(f"Index file not found at {index_file}")

            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

            # Load FAISS index
            self._index = faiss.read_index(str(index_file))
            self._dimension = self._index.d

            # Load metadata
            with open(metadata_file, "rb") as f:
                self._metadata = orjson.loads(f.read())

            logger.info(f"Loaded index from {path}")

        except Exception as e:
            raise MemVidIndexError(f"Failed to load index: {str(e)}")
