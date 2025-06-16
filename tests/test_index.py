"""Unit tests for the IndexManager class."""

import pytest
import faiss
import numpy as np
from pathlib import Path
import tempfile

from langchain_memvid.index import IndexManager, IndexConfig


class MockEmbeddings:
    """Mock embedding model for testing."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.embedding_dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        """Generate deterministic embeddings for testing."""
        if not isinstance(text, str):
            return [0.0] * self.dimension
        # Simple hash-based embedding for testing
        return [hash(text) % 100 / 100.0 for _ in range(self.dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings for multiple texts."""
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def mock_embeddings():
    """Fixture providing a mock embedding model."""
    return MockEmbeddings(dimension=4)  # Small dimension for testing


@pytest.fixture
def index_manager(mock_embeddings):
    """Fixture providing an initialized IndexManager."""
    config = IndexConfig(index_type="Flat")
    return IndexManager(config=config, embeddings=mock_embeddings)


def test_init(index_manager, mock_embeddings):
    """Test IndexManager initialization."""
    config = IndexConfig(nlist=2)
    manager = IndexManager(config=config, embeddings=mock_embeddings)

    # Add some data to initialize the Flat index
    chunks = ["test chunk 1", "test chunk 2", "test chunk 3"]
    frame_numbers = [1, 2, 3]
    manager.add_chunks(chunks, frame_numbers)

    # Verify the index type
    assert isinstance(manager.index, faiss.IndexIDMap)
    assert hasattr(manager.index, 'index')


def test_init_lazy(index_manager, mock_embeddings):
    """Test lazy loading of IndexManager."""
    assert index_manager.embedding_model == mock_embeddings
    assert index_manager.index is None
    assert index_manager.dimension == 4
    assert len(index_manager.metadata) == 0
    assert len(index_manager.frame_to_chunks) == 0


def test_init_ivf(mock_embeddings):
    """Test IndexManager initialization with IVF index."""
    config = IndexConfig(index_type="IVF", nlist=2)
    manager = IndexManager(config=config, embeddings=mock_embeddings)

    # Add some data to initialize the IVF index
    chunks = ["test chunk 1", "test chunk 2", "test chunk 3"]
    frame_numbers = [1, 2, 3]
    manager.add_chunks(chunks, frame_numbers)

    # Verify the index type
    assert isinstance(manager.index, faiss.IndexIDMap)
    assert hasattr(manager.index, 'index')


def test_add_chunks(index_manager):
    """Test adding chunks to the index."""
    chunks = ["test chunk 1", "test chunk 2"]
    frame_numbers = [1, 2]

    chunk_ids = index_manager.add_chunks(chunks, frame_numbers)

    assert len(chunk_ids) == 2
    assert len(index_manager.metadata) == 2
    assert len(index_manager.frame_to_chunks) == 2

    # Verify metadata
    for chunk_id, chunk, frame in zip(chunk_ids, chunks, frame_numbers):
        metadata = index_manager.metadata[chunk_id]
        assert metadata["text"] == chunk
        assert metadata["frame"] == frame
        assert metadata["id"] == chunk_id
        assert metadata["length"] == len(chunk)

    # Verify frame mappings
    assert set(index_manager.frame_to_chunks[1]) == {chunk_ids[0]}
    assert set(index_manager.frame_to_chunks[2]) == {chunk_ids[1]}


def test_add_chunks_validation(index_manager):
    """Test chunk validation during addition."""
    # Test with invalid chunk (empty string)
    chunks = ["valid chunk", ""]
    frame_numbers = [1, 2]

    chunk_ids = index_manager.add_chunks(chunks, frame_numbers)

    assert len(chunk_ids) == 1
    assert len(index_manager.metadata) == 1
    assert index_manager.metadata[chunk_ids[0]]["text"] == "valid chunk"


def test_add_chunks_length_mismatch(index_manager):
    """Test handling of mismatched chunk and frame numbers."""
    chunks = ["chunk 1", "chunk 2"]
    frame_numbers = [1]  # Mismatched length

    with pytest.raises(ValueError, match="Number of chunks must match number of frame numbers"):
        index_manager.add_chunks(chunks, frame_numbers)


def test_search(index_manager):
    """Test searching the index."""
    # Add some test data
    chunks = ["apple pie", "banana bread", "cherry tart"]
    frame_numbers = [1, 2, 3]
    index_manager.add_chunks(chunks, frame_numbers)

    # Search for similar content
    results = index_manager.search("apple", top_k=2)

    assert len(results) > 0  # Should find at least one result
    assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
    assert all(isinstance(r[0], (int, np.int64)) for r in results)  # chunk_id
    assert all(isinstance(r[1], float) for r in results)  # distance
    assert all(isinstance(r[2], dict) for r in results)  # metadata


def test_get_chunks_by_frame(index_manager):
    """Test retrieving chunks by frame number."""
    # Add test data
    chunks = ["frame 1 chunk 1", "frame 1 chunk 2", "frame 2 chunk 1"]
    frame_numbers = [1, 1, 2]
    index_manager.add_chunks(chunks, frame_numbers)

    # Get chunks for frame 1
    frame1_chunks = index_manager.get_chunks_by_frame(1)
    assert len(frame1_chunks) == 2
    assert all(chunk["frame"] == 1 for chunk in frame1_chunks)

    # Get chunks for frame 2
    frame2_chunks = index_manager.get_chunks_by_frame(2)
    assert len(frame2_chunks) == 1
    assert all(chunk["frame"] == 2 for chunk in frame2_chunks)

    # Get chunks for non-existent frame
    empty_chunks = index_manager.get_chunks_by_frame(999)
    assert len(empty_chunks) == 0


def test_get_chunk_by_id(index_manager):
    """Test retrieving chunk by ID."""
    # Add test data
    chunks = ["test chunk"]
    frame_numbers = [1]
    chunk_ids = index_manager.add_chunks(chunks, frame_numbers)

    # Get existing chunk
    chunk = index_manager.get_chunk_by_id(chunk_ids[0])
    assert chunk is not None
    assert chunk["text"] == "test chunk"

    # Get non-existent chunk
    assert index_manager.get_chunk_by_id(999) is None


def test_save_load(index_manager):
    """Test saving and loading the index."""
    # Add test data
    chunks = ["test chunk 1", "test chunk 2"]
    frame_numbers = [1, 2]
    index_manager.add_chunks(chunks, frame_numbers)

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "test_index"

        # Save index
        index_manager.save(str(path))
        assert path.with_suffix('.faiss').exists()
        assert path.with_suffix('.msgpack').exists()

        # Create new manager and load index
        new_manager = IndexManager(
            config=IndexConfig(index_type="Flat"),
            embeddings=MockEmbeddings(dimension=4)
        )
        new_manager.load(str(path))

        # Verify loaded data
        assert len(new_manager.metadata) == len(index_manager.metadata)
        assert len(new_manager.frame_to_chunks) == len(index_manager.frame_to_chunks)

        # Verify content
        for chunk_id, metadata in index_manager.metadata.items():
            loaded_metadata = new_manager.metadata[chunk_id]
            assert loaded_metadata["text"] == metadata["text"]
            assert loaded_metadata["frame"] == metadata["frame"]


def test_get_stats(index_manager):
    """Test getting index statistics."""
    # Add test data
    chunks = ["chunk 1", "chunk 2", "chunk 3"]
    frame_numbers = [1, 1, 2]  # 2 chunks in frame 1, 1 chunk in frame 2
    index_manager.add_chunks(chunks, frame_numbers)

    stats = index_manager.get_stats()

    assert stats.total_chunks == 3
    assert stats.total_frames == 2
    assert stats.index_type == "Flat"
    assert stats.dimension == 4
    assert stats.avg_chunks_per_frame == 1.5  # (2 + 1) / 2


def test_ivf_training(mock_embeddings):
    """Test IVF index training behavior."""
    config = IndexConfig(index_type="IVF", nlist=2)
    manager = IndexManager(config=config, embeddings=mock_embeddings)

    # Test with insufficient training data
    chunks = ["chunk 1"]  # Only one chunk, less than nlist
    frame_numbers = [1]
    manager.add_chunks(chunks, frame_numbers)

    # Should fall back to Flat index
    assert isinstance(manager.index, faiss.IndexIDMap)
    assert hasattr(manager.index, 'index')

    # Test with sufficient training data
    config = IndexConfig(index_type="IVF", nlist=2)
    manager = IndexManager(config=config, embeddings=mock_embeddings)

    chunks = ["chunk 1", "chunk 2", "chunk 3", "chunk 4"]  # More than nlist
    frame_numbers = [1, 2, 3, 4]
    manager.add_chunks(chunks, frame_numbers)

    # Should remain as IVF index
    assert isinstance(manager.index, faiss.IndexIDMap)
    assert hasattr(manager.index, 'index')


def test_batch_processing(index_manager):
    """Test batch processing of chunks."""
    # Create many chunks to test batch processing
    chunks = [f"chunk {i}" for i in range(150)]  # More than default batch size
    frame_numbers = list(range(150))

    chunk_ids = index_manager.add_chunks(chunks, frame_numbers)

    assert len(chunk_ids) == 150
    assert len(index_manager.metadata) == 150
    assert all(chunk_id in index_manager.metadata for chunk_id in chunk_ids)


def test_error_handling(index_manager):
    """Test error handling in various scenarios."""
    # Test with invalid chunk type
    chunks = ["valid chunk", 123]  # Invalid type
    frame_numbers = [1, 2]

    chunk_ids = index_manager.add_chunks(chunks, frame_numbers)
    assert len(chunk_ids) == 1  # Only valid chunk should be added

    # Test with very long chunk
    long_chunk = "x" * 10000  # Exceeds typical model limits
    chunks = ["valid chunk", long_chunk]
    frame_numbers = [1, 2]

    chunk_ids = index_manager.add_chunks(chunks, frame_numbers)
    assert len(chunk_ids) == 1  # Only valid chunk should be added
