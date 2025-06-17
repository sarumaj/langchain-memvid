"""
Vector index management for MemVid.

This module provides functionality for managing vector indices used in MemVid,
including FAISS index creation, updating, and searching.
"""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import orjson
import logging

from .exceptions import MemVidIndexError
from .config import IndexConfig

logger = logging.getLogger(__name__)


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
    """Manages vector indices for MemVid."""

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

    def create_index(self) -> None:
        """Create a new FAISS index.

        The dimension is automatically determined from the embeddings model.
        If using IVF index and there aren't enough points for training,
        falls back to a flat index.

        Raises:
            MemVidIndexError: If index creation fails
        """
        try:
            # Get dimension from embeddings
            test_vector = self.embeddings.embed_query("test")
            self._dimension = len(test_vector)

            if self.config.index_type == "faiss":
                match self.config.metric:
                    case "cosine":
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

            logger.info(f"Created {self.config.index_type} index with {self.config.metric} metric")

        except Exception as e:
            raise MemVidIndexError(f"Failed to create index: {str(e)}")

    def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add texts to the index by converting them to vectors using embeddings.

        Args:
            texts: List of texts to add
            metadata: Optional list of metadata dictionaries for each text

        Raises:
            MemVidIndexError: If adding texts fails
        """
        try:
            if self._index is None:
                self.create_index()

            # Convert texts to vectors using embeddings
            vectors = np.array(self.embeddings.embed_documents(texts), dtype='float32')

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
            for i, text in enumerate(texts):
                if text not in text_to_idx:
                    unique_indices.append(i)
                    text_to_idx[text] = len(self._metadata) + len(unique_indices) - 1

            if not unique_indices:
                logger.info("No new texts to add - all were duplicates")
                return

            # Filter vectors and metadata to only include unique texts
            unique_vectors = vectors[unique_indices]
            unique_metadata = [metadata[i] for i in unique_indices]

            # Check if we should convert to IVF index
            if (
                self.config.nlist > 0
                and not isinstance(self._index, faiss.IndexIVFFlat)
                and self._index.ntotal + len(unique_vectors) >= self._min_points
            ):
                # Create IVF index
                quantizer = self._index
                self._index = faiss.IndexIVFFlat(
                    quantizer,
                    self._dimension,
                    self.config.nlist,
                    faiss.METRIC_INNER_PRODUCT if self.config.metric == "cosine"
                    else faiss.METRIC_L2
                )

                # Get all existing vectors
                all_vectors = np.zeros((self._index.ntotal, self._dimension), dtype='float32')
                self._index.reconstruct_n(0, self._index.ntotal, all_vectors)

                # Train the index
                self._index.train(all_vectors)
                self._is_trained = True

                # Add back the vectors
                self._index.add(all_vectors)
                logger.info(f"Converted to IVF index and trained with {self._index.ntotal} points")

            # Check if IVF index needs training
            if isinstance(self._index, faiss.IndexIVFFlat) and not self._is_trained:
                # Train the index with these vectors
                self._index.train(unique_vectors)
                self._is_trained = True

            # Normalize vectors for cosine similarity
            if self.config.metric == "cosine":
                faiss.normalize_L2(unique_vectors)

            # Add vectors to index
            self._index.add(unique_vectors)
            self._metadata.extend(unique_metadata)

            logger.info(f"Added {len(unique_vectors)} unique texts to index")

        except Exception as e:
            raise MemVidIndexError(f"Failed to add texts: {str(e)}")

    def search_text(
        self,
        query_text: str,
        k: int = 4,
    ) -> List[SearchResult]:
        """Search for similar texts using a text query.

        Args:
            query_text: Text to search for
            k: Number of results to return

        Returns:
            List of SearchResult objects containing the results and their similarity scores

        Raises:
            MemVidIndexError: If search fails
        """
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
        """Get metadata for given indices.

        Args:
            indices: List of indices to get metadata for

        Returns:
            List of metadata dictionaries

        Raises:
            MemVidIndexError: If getting metadata fails
        """
        try:
            return [self._metadata[i] for i in indices]
        except Exception as e:
            raise MemVidIndexError(f"Failed to get metadata: {str(e)}")

    def save(self, path: Path) -> None:
        """Save the index and metadata to disk.

        Args:
            path: Path to save the index

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

    def load(self, path: Path) -> None:
        """Load the index and metadata from disk.

        Args:
            path: Path to load the index from

        Raises:
            MemVidIndexError: If loading fails
        """
        try:
            # Load FAISS index
            self._index = faiss.read_index(str(path / "index.faiss"))
            self._dimension = self._index.d

            # Load metadata
            with open(path / "metadata.json", "rb") as f:
                self._metadata = orjson.loads(f.read())

            logger.info(f"Loaded index from {path}")

        except Exception as e:
            raise MemVidIndexError(f"Failed to load index: {str(e)}")
