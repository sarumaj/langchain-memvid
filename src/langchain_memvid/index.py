"""
Index management for embeddings and vector search.

This module provides functionality for managing embeddings, FAISS index, and metadata
for fast retrieval of text chunks from video frames. It supports both Flat and IVF
index types, with automatic training and fallback mechanisms.

Example:
    ```python
    from langchain_memvid.index import IndexManager, IndexConfig
    from langchain.embeddings import HuggingFaceEmbeddings

    # Initialize
    config = IndexConfig(index_type="Flat")
    embeddings = HuggingFaceEmbeddings()
    index_manager = IndexManager(config=config, embeddings=embeddings)

    # Add chunks
    chunks = ["text chunk 1", "text chunk 2"]
    frame_numbers = [1, 2]
    chunk_ids = index_manager.add_chunks(chunks, frame_numbers)

    # Search
    results = index_manager.search("query", top_k=5)
    ```

Original source: https://github.com/Olow304/memvid/blob/main/memvid/index.py
"""

import orjson as json
import numpy as np
import faiss
from langchain_core.embeddings import Embeddings
from typing import List, Dict, Tuple, Optional, Literal, TypedDict, Set, Union, Any
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict
from collections import defaultdict
import msgpack
import msgpack_numpy as m
m.patch()  # Enable numpy array serialization

logger = logging.getLogger(__name__)


class IndexConfig(BaseModel):
    """Configuration for FAISS index.

    Attributes:
        index_type: Type of index to use. "Flat" for exact search, "IVF" for approximate search.
        nlist: Number of clusters for IVF index. Only used when index_type is "IVF".
        serialization_format: Format to use for metadata serialization ('json' or 'msgpack').
    """
    index_type: Literal["Flat", "IVF"] = Field(
        default="Flat",
        description="Can be \"IVF\" for larger datasets, otherwise use Flat"
    )
    nlist: int = Field(default=100, description="Number of clusters for IVF index")
    serialization_format: Literal["json", "msgpack"] = Field(
        default="msgpack",
        description="Format to use for metadata serialization"
    )


class ChunkMetadata(TypedDict):
    """Metadata for a text chunk.

    Attributes:
        id: Unique identifier for the chunk.
        text: The actual text content of the chunk.
        frame: Frame number this chunk belongs to.
        length: Length of the text chunk in characters.
    """
    id: int
    text: str
    frame: int
    length: int


@dataclass
class IndexStats:
    """Statistics about the index.

    Attributes:
        total_chunks: Total number of chunks in the index.
        total_frames: Total number of frames with chunks.
        index_type: Type of FAISS index being used.
        embedding_model: Name of the embedding model.
        dimension: Dimension of the embeddings.
        avg_chunks_per_frame: Average number of chunks per frame.
    """
    total_chunks: int
    total_frames: int
    index_type: str
    embedding_model: str
    dimension: int
    avg_chunks_per_frame: float
    config: Dict[str, Any]

    def dict(self) -> Dict[str, Any]:
        """Convert IndexStats to a dictionary."""
        return asdict(self)


class IndexManager:
    """Manages embeddings, FAISS index, and metadata for fast retrieval.

    This class handles the creation and management of FAISS indices for efficient
    similarity search of text chunks. It supports both exact (Flat) and approximate
    (IVF) search methods, with automatic training and fallback mechanisms.

    Attributes:
        config: Index configuration.
        embedding_model: LangChain embedding model for generating embeddings.
        index: FAISS index for similarity search.
        metadata: Dictionary mapping chunk IDs to their metadata.
        frame_to_chunks: Dictionary mapping frame numbers to sets of chunk IDs.
        _index_path: Optional path to the index file for lazy loading (as string).
        _is_loaded: Whether the index has been loaded.
    """

    def __init__(self, *, config: IndexConfig, embeddings: Embeddings):
        """Initialize IndexManager.

        Args:
            config: Index configuration specifying index type and parameters.
            embeddings: LangChain embedding model to use for generating embeddings.

        Raises:
            ValueError: If embedding dimension cannot be determined.
        """
        self.config = config
        self.embedding_model = embeddings
        self._dimension: Optional[int] = None
        self._index_path: Optional[str] = None
        self._is_loaded = False

        # Initialize empty containers
        self.index: Optional[faiss.Index] = None
        self.metadata: Dict[int, ChunkMetadata] = {}
        self.frame_to_chunks: Dict[int, Set[int]] = defaultdict(set)

    @property
    def dimension(self) -> int:
        """Get the embedding dimension, initializing it if needed.

        Returns:
            int: The dimension of the embeddings.

        Raises:
            ValueError: If embedding dimension cannot be determined.
        """
        if self._dimension is None:
            try:
                # Try to get dimension from model attributes first
                if hasattr(self.embedding_model, "embedding_dimension"):
                    self._dimension = self.embedding_model.embedding_dimension
                elif hasattr(self.embedding_model, "dimensions"):
                    self._dimension = self.embedding_model.dimensions
                else:
                    # Fallback to test embedding
                    test_embedding = self.embedding_model.embed_query("test")
                    self._dimension = len(test_embedding)

                logger.info(f"Initialized embedding dimension: {self._dimension}")
            except Exception as e:
                logger.error(f"Failed to determine embedding dimension: {e}")
                raise ValueError(
                    "Could not determine embedding dimension. "
                    "Please ensure the embedding model is properly initialized."
                ) from e

        return self._dimension

    def _ensure_loaded(self) -> None:
        """Ensure the index is loaded before performing operations.

        This method implements lazy loading - the index is only loaded when needed.
        """
        if not self._is_loaded:
            if self._index_path is not None:
                self.load(self._index_path)
            else:
                # Create new index if no path is set
                self.index = self._create_index()
                self._is_loaded = True

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration.

        Returns:
            faiss.Index: Initialized FAISS index with ID mapping.

        Raises:
            ValueError: If index type is invalid.
        """
        dim = self.dimension

        match self.config.index_type:
            case "Flat":
                index = faiss.IndexFlatL2(dim)

            case "IVF":
                # Create a quantizer first
                quantizer = faiss.IndexFlatL2(dim)
                # Create IVF index with the quantizer
                index = faiss.IndexIVFFlat(quantizer, dim, self.config.nlist)

            case _:
                raise ValueError(f"Invalid index type: {self.config.index_type}")

        return faiss.IndexIDMap(index)

    def add_chunks(
        self,
        chunks: List[Union[str, Any]],
        frame_numbers: List[int],
        show_progress: bool = True,
    ) -> List[int]:
        """Add chunks to index with robust error handling and validation.

        This method processes text chunks, generates embeddings, and adds them to the
        FAISS index. It includes validation, error handling, and batch processing
        capabilities.

        Args:
            chunks: List of text chunks to add.
            frame_numbers: Corresponding frame numbers for each chunk.
            show_progress: Whether to show progress bar during processing.

        Returns:
            List[int]: List of successfully added chunk IDs.

        Raises:
            ValueError: If number of chunks doesn't match number of frame numbers.
        """
        self._ensure_loaded()
        if len(chunks) != len(frame_numbers):
            raise ValueError("Number of chunks must match number of frame numbers")

        logger.info(f"Processing {len(chunks)} chunks for indexing...")

        # Phase 1: Validate and filter chunks
        valid_chunks = []
        valid_frames = []
        skipped_count = 0

        for chunk, frame_num in zip(chunks, frame_numbers):
            if self._is_valid_chunk(chunk):
                valid_chunks.append(chunk)
                valid_frames.append(frame_num)
            else:
                skipped_count += 1
                chunk_length = len(str(chunk)) if chunk is not None else 0
                logger.warning(f"Skipping invalid chunk at frame {frame_num}: length={chunk_length}")

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid chunks out of {len(chunks)} total")

        if not valid_chunks:
            logger.error("No valid chunks to process")
            return []

        logger.info(f"Processing {len(valid_chunks)} valid chunks")

        # Phase 2: Generate embeddings
        try:
            embeddings = self._generate_embeddings(valid_chunks, show_progress)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

        if embeddings is None or len(embeddings) == 0:
            logger.error("No embeddings generated")
            return []

        # Phase 3: Add to FAISS index
        try:
            chunk_ids = self._add_to_index(embeddings, valid_chunks, valid_frames)
            logger.info(f"Successfully added {len(chunk_ids)} chunks to index")
            return chunk_ids
        except Exception as e:
            logger.error(f"Failed to add chunks to index: {e}")
            return []

    def _is_valid_chunk(self, chunk: Union[str, Any]) -> bool:
        """Validate chunk for processing.

        Args:
            chunk: Text chunk to validate.

        Returns:
            bool: True if chunk is valid, False otherwise.
        """
        if not isinstance(chunk, str):
            return False

        chunk = chunk.strip()
        if not chunk or len(chunk) > 8192:  # SentenceTransformer limit
            return False

        try:
            chunk.encode('utf-8')
            return True
        except UnicodeEncodeError:
            return False

    def _generate_embeddings(self, chunks: List[str], show_progress: bool) -> np.ndarray:
        """Generate embeddings with error handling and batch processing.

        Args:
            chunks: List of text chunks to embed.
            show_progress: Whether to show progress bar.

        Returns:
            np.ndarray: Array of embeddings with shape (n_chunks, dimension).

        Raises:
            RuntimeError: If no embeddings could be generated.
        """
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks (full batch)")
            embeddings = self.embedding_model.embed_documents(chunks)
            return np.array(embeddings).astype('float32')
        except Exception as e:
            logger.warning(f"Full batch embedding failed: {e}. Trying batch processing...")
            return self._generate_embeddings_batched(chunks, show_progress)

    def _generate_embeddings_batched(self, chunks: List[str], show_progress: bool) -> np.ndarray:
        """Generate embeddings in smaller batches with individual error handling.

        Args:
            chunks: List of text chunks to embed.
            show_progress: Whether to show progress bar.

        Returns:
            np.ndarray: Array of embeddings with shape (n_chunks, dimension).

        Raises:
            RuntimeError: If no embeddings could be generated.
        """
        all_embeddings = []
        valid_chunks = []
        batch_size = 100

        total_batches = (len(chunks) + batch_size - 1) // batch_size

        if show_progress:
            from tqdm import tqdm
            batch_iter = tqdm(
                range(0, len(chunks), batch_size),
                desc="Processing chunks in batches",
                total=total_batches
            )
        else:
            batch_iter = range(0, len(chunks), batch_size)

        for i in batch_iter:
            batch_chunks = chunks[i:i + batch_size]
            try:
                batch_embeddings = self.embedding_model.embed_documents(batch_chunks)
                all_embeddings.extend(batch_embeddings)
                valid_chunks.extend(batch_chunks)
            except Exception as e:
                logger.warning(f"Batch {i//batch_size} failed: {e}. Processing individually...")
                for chunk in batch_chunks:
                    try:
                        embedding = self.embedding_model.embed_documents([chunk])
                        all_embeddings.extend(embedding)
                        valid_chunks.append(chunk)
                    except Exception as chunk_error:
                        logger.error(f"Failed to embed individual chunk (length={len(chunk)}): {chunk_error}")
                        continue

        if not all_embeddings:
            raise RuntimeError("No embeddings could be generated")

        logger.info(f"Generated embeddings for {len(valid_chunks)} out of {len(chunks)} chunks")
        return np.array(all_embeddings).astype('float32')

    def _add_to_index(self, embeddings: np.ndarray, chunks: List[str], frame_numbers: List[int]) -> List[int]:
        """Add embeddings to FAISS index with error handling.

        Args:
            embeddings: Array of embeddings to add.
            chunks: List of text chunks corresponding to embeddings.
            frame_numbers: List of frame numbers corresponding to chunks.

        Returns:
            List[int]: List of assigned chunk IDs.

        Raises:
            Exception: If adding to index fails.
        """
        if len(embeddings) != len(chunks) or len(embeddings) != len(frame_numbers):
            min_len = min(len(embeddings), len(chunks), len(frame_numbers))
            embeddings = embeddings[:min_len]
            chunks = chunks[:min_len]
            frame_numbers = frame_numbers[:min_len]
            logger.warning(f"Trimmed to {min_len} items due to length mismatch")

        # Assign IDs
        start_id = len(self.metadata)
        chunk_ids = list(range(start_id, start_id + len(chunks)))

        # Train index if needed (for IVF)
        self._train_index_if_needed(embeddings)

        # Add to index
        try:
            self.index.add_with_ids(embeddings, np.array(chunk_ids, dtype=np.int64))
        except Exception as e:
            logger.error(f"Failed to add embeddings to FAISS index: {e}")
            raise

        # Store metadata
        for chunk, frame_num, chunk_id in zip(chunks, frame_numbers, chunk_ids):
            try:
                metadata: ChunkMetadata = {
                    "id": chunk_id,
                    "text": chunk,
                    "frame": frame_num,
                    "length": len(chunk)
                }
                self.metadata[chunk_id] = metadata
                self.frame_to_chunks[frame_num].add(chunk_id)
            except Exception as e:
                logger.error(f"Failed to store metadata for chunk {chunk_id}: {e}")
                continue

        return chunk_ids

    def _train_index_if_needed(self, embeddings: np.ndarray) -> None:
        """Train the index if needed (for IVF).

        This method handles the training of IVF indexes, including automatic
        fallback to Flat index if training data is insufficient.

        Args:
            embeddings: Array of embeddings to use for training.
        """
        try:
            underlying_index = self.index.index

            if isinstance(underlying_index, faiss.IndexIVFFlat):
                nlist = underlying_index.nlist

                if not underlying_index.is_trained:
                    if len(embeddings) < nlist:
                        logger.warning(
                            f"Insufficient training data: need at least {nlist} embeddings, "
                            f"got {len(embeddings)}"
                        )
                        logger.info("Auto-switching to IndexFlatL2 for reliable operation")
                        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                    else:
                        recommended_min = nlist * 10
                        if len(embeddings) < recommended_min:
                            logger.warning(
                                f"Suboptimal training data: {len(embeddings)} embeddings "
                                f"(recommended: {recommended_min}+)"
                            )

                        training_data = embeddings[:min(50000, len(embeddings))]
                        underlying_index.train(training_data)
                        logger.info("FAISS IVF training completed successfully")
                else:
                    logger.info(f"FAISS IVF index already trained (nlist={nlist})")
            else:
                logger.info(f"Using {type(underlying_index).__name__} (no training required)")

        except Exception as e:
            logger.error(f"Index training failed: {e}")
            logger.info("Falling back to IndexFlatL2 for reliability")
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, ChunkMetadata]]:
        """Search for similar chunks.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List[Tuple[int, float, ChunkMetadata]]: List of (chunk_id, distance, metadata) tuples,
                sorted by distance (lower is better).
        """
        self._ensure_loaded()
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self.metadata:  # Valid result
                metadata = self.metadata[idx]
                results.append((int(idx), float(dist), metadata))

        return results

    def get_chunks_by_frame(self, frame_number: int) -> List[ChunkMetadata]:
        """Get all chunks associated with a frame.

        Args:
            frame_number: Frame number to get chunks for.

        Returns:
            List[ChunkMetadata]: List of chunk metadata for the specified frame.
        """
        self._ensure_loaded()
        chunk_ids = self.frame_to_chunks.get(frame_number, set())
        return [self.metadata[chunk_id] for chunk_id in chunk_ids if chunk_id in self.metadata]

    def get_chunk_by_id(self, chunk_id: int) -> Optional[ChunkMetadata]:
        """Get chunk metadata by ID.

        Args:
            chunk_id: ID of the chunk to retrieve.

        Returns:
            Optional[ChunkMetadata]: Chunk metadata if found, None otherwise.
        """
        self._ensure_loaded()
        return self.metadata.get(chunk_id)

    def save(self, path: str) -> None:
        """Save index to disk.

        This method saves both the FAISS index and associated metadata to disk.
        The index is saved with .faiss extension and metadata with .json or .msgpack extension.

        Args:
            path: Base path to save index (without extension).
        """
        self._ensure_loaded()
        path = Path(path)
        self._index_path = str(path.absolute())

        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix('.faiss')))

        # Prepare data for serialization
        data = {
            "metadata": self.metadata,
            "frame_to_chunks": {k: list(v) for k, v in self.frame_to_chunks.items()},
            "config": self.config.model_dump()
        }

        # Save metadata using selected format
        if self.config.serialization_format == "msgpack":
            with open(path.with_suffix('.msgpack'), 'wb') as f:
                f.write(msgpack.packb(data))
        else:  # json
            with open(path.with_suffix('.json'), 'w') as f:
                json.dump(data, f, indent=2)

        logger.info(f"Saved index to {path} using {self.config.serialization_format} format")

    def load(self, path: str) -> None:
        """Load index from disk.

        This method loads both the FAISS index and associated metadata from disk.
        The index is loaded from .faiss extension and metadata from .json or .msgpack extension.

        Args:
            path: Base path to load index from (without extension).
        """
        path = Path(path)
        self._index_path = str(path.absolute())

        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix('.faiss')))

        # Try to load metadata using msgpack first, fall back to json if not found
        msgpack_path = path.with_suffix('.msgpack')
        json_path = path.with_suffix('.json')

        if msgpack_path.exists():
            with open(msgpack_path, 'rb') as f:
                data = msgpack.unpackb(f.read(), strict_map_key=False)
            self.config.serialization_format = "msgpack"
        elif json_path.exists():
            with open(json_path, 'r') as f:
                data = json.loads(f.read())
            self.config.serialization_format = "json"
        else:
            raise FileNotFoundError(f"No metadata file found at {path}")

        self.metadata = {int(k): v for k, v in data["metadata"].items()}
        self.frame_to_chunks = defaultdict(
            set,
            {int(k): set(v) for k, v in data["frame_to_chunks"].items()}
        )

        # Update config if available
        if "config" in data:
            self.config = IndexConfig(**data["config"])

        self._is_loaded = True
        logger.info(f"Loaded index from {path} using {self.config.serialization_format} format")

    def get_stats(self) -> IndexStats:
        """Get index statistics.

        Returns:
            IndexStats: Statistics about the index including total chunks,
                frames, and average chunks per frame.
        """
        self._ensure_loaded()
        return IndexStats(
            total_chunks=len(self.metadata),
            total_frames=len(self.frame_to_chunks),
            index_type=self.config.index_type,
            embedding_model=getattr(self.embedding_model, "model_name", "unknown"),
            dimension=self.dimension,
            avg_chunks_per_frame=(
                np.mean([len(chunks) for chunks in self.frame_to_chunks.values()])
                if self.frame_to_chunks else 0
            ),
            config=self.config.model_dump()
        )
