import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, call
import faiss

from langchain_memvid.index import IndexManager
from langchain_memvid.config import IndexConfig
from langchain_memvid.exceptions import MemVidIndexError


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings object."""
    embeddings = MagicMock()
    # Mock the embedding methods to return random vectors
    mock_vectors = [np.random.rand(128).tolist() for _ in range(10)]
    embeddings.embed_documents.return_value = mock_vectors
    embeddings.embed_query.return_value = np.random.rand(128).tolist()
    return embeddings


@pytest.fixture
def index_config():
    """Create a test index configuration."""
    return IndexConfig(
        index_type="faiss",
        metric="cosine",
        nlist=2  # Using IVF index
    )


@pytest.fixture
def index_manager(index_config, mock_embeddings):
    """Create an IndexManager instance for testing."""
    return IndexManager(config=index_config, embeddings=mock_embeddings)


@pytest.fixture
def test_texts():
    """Create test texts for testing."""
    return [f"test text {i}" for i in range(10)]


@pytest.fixture
def test_metadata():
    """Create test metadata for testing."""
    return [{"id": i, "text": f"test_{i}"} for i in range(10)]


def test_add_texts(index_manager, test_texts, mock_embeddings):
    """Test adding texts to the index."""
    # Add texts
    index_manager.add_texts(test_texts)

    # Verify embeddings were called
    mock_embeddings.embed_documents.assert_called_once_with(test_texts)

    # Verify vectors were added
    assert index_manager._index.ntotal == len(test_texts)
    assert len(index_manager._metadata) == len(test_texts)
    assert index_manager._is_trained  # Should be trained automatically


def test_add_texts_with_metadata(index_manager, test_texts, test_metadata):
    """Test adding texts with custom metadata."""
    # Add texts with metadata
    index_manager.add_texts(test_texts, test_metadata)

    # Verify metadata was preserved
    assert len(index_manager._metadata) == len(test_texts)
    assert all(m["id"] == i for m, i in zip(index_manager._metadata, range(len(test_texts))))


def test_search_text(index_manager, test_texts):
    """Test searching using text query."""
    # Add texts
    index_manager.add_texts(test_texts)

    # Search using text
    results = index_manager.search_text("test query", k=3)

    # Verify embeddings were called
    index_manager.embeddings.embed_query.assert_has_calls([
        call("test"),
        call("test query"),
    ])

    # Verify results
    assert len(results) == 3
    assert all(isinstance(r.similarity, float) for r in results)


def test_search_text_uninitialized_index(index_manager):
    """Test searching with text on uninitialized index."""
    with pytest.raises(MemVidIndexError, match="Index not initialized"):
        index_manager.search_text("test query")


def test_create_index(index_manager, mock_embeddings):
    """Test index creation."""
    # Create index
    index_manager.create_index()

    # Verify dimension was set correctly
    assert index_manager._dimension == len(mock_embeddings.embed_query.return_value)
    assert index_manager._index is not None
    assert isinstance(index_manager._index, faiss.IndexFlatIP)


def test_create_index_invalid_metric(index_config, mock_embeddings):
    """Test index creation with invalid metric."""
    index_config.metric = "invalid"
    index_manager = IndexManager(config=index_config, embeddings=mock_embeddings)
    with pytest.raises(MemVidIndexError, match="Unsupported metric"):
        index_manager.create_index()


def test_save_and_load(index_manager, test_texts, test_metadata):
    """Test saving and loading the index."""
    # Add texts with metadata
    index_manager.add_texts(test_texts, test_metadata)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir)

        # Save index
        index_manager.save(path)

        # Create new index manager
        new_index_manager = IndexManager(
            config=index_manager.config,
            embeddings=index_manager.embeddings
        )

        # Load index
        new_index_manager.load(path)

        # Verify loaded data
        assert new_index_manager._index.ntotal == index_manager._index.ntotal
        assert len(new_index_manager._metadata) == len(index_manager._metadata)
        assert new_index_manager._dimension == index_manager._dimension


def test_get_metadata(index_manager, test_texts, test_metadata):
    """Test retrieving metadata for indices."""
    # Add texts with metadata
    index_manager.add_texts(test_texts, test_metadata)

    # Get metadata for some indices
    indices = [0, 2, 4]
    metadata = index_manager.get_metadata(indices)

    assert len(metadata) == len(indices)
    assert all(m["id"] == i for m, i in zip(metadata, indices))


def test_get_metadata_invalid_index(index_manager, test_texts, test_metadata):
    """Test getting metadata with invalid indices."""
    # Add texts with metadata
    index_manager.add_texts(test_texts, test_metadata)

    with pytest.raises(MemVidIndexError):
        index_manager.get_metadata([100])  # Index out of range
