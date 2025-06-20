"""Unit tests for the IndexManager class."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, call
import faiss

from langchain_memvid.index import IndexManager
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
def index_manager(index_config, mock_embeddings):
    """Create an IndexManager instance for testing."""
    return IndexManager(config=index_config, embeddings=mock_embeddings)


class TestIndexManagerTextOperations:
    """Test cases for text addition and management."""

    def test_add_texts(self, index_manager, test_texts, mock_embeddings):
        """Test adding texts to the index."""
        # Add texts
        index_manager.add_texts(test_texts)

        # Verify embeddings were called
        mock_embeddings.embed_documents.assert_called_once_with(test_texts)

        # Verify vectors were added
        assert index_manager._index.ntotal == len(test_texts)
        assert len(index_manager._metadata) == len(test_texts)
        assert index_manager._is_trained  # Should be trained automatically

    def test_add_texts_with_metadata(self, index_manager, test_texts, test_metadata):
        """Test adding texts with custom metadata."""
        # Add texts with metadata
        index_manager.add_texts(test_texts, test_metadata)

        # Verify metadata was preserved
        assert len(index_manager._metadata) == len(test_texts)
        assert all(m["id"] == i for m, i in zip(index_manager._metadata, range(len(test_texts))))


class TestIndexManagerSearchOperations:
    """Test cases for search functionality."""

    def test_search_text(self, index_manager, test_texts):
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

    def test_search_text_uninitialized_index(self, index_manager):
        """Test searching with text on uninitialized index."""
        with pytest.raises(MemVidIndexError, match="Index not initialized"):
            index_manager.search_text("test query")


class TestIndexManagerIndexCreation:
    """Test cases for index creation and configuration."""

    def test_create_index(self, index_manager, mock_embeddings):
        """Test index creation."""
        # Create index
        index_manager.create_index()

        # Verify dimension was set correctly
        assert index_manager._dimension == len(mock_embeddings.embed_query.return_value)
        assert index_manager._index is not None
        assert isinstance(index_manager._index, faiss.IndexFlatIP)

    def test_create_index_invalid_metric(self, index_config, mock_embeddings):
        """Test index creation with invalid metric."""
        index_config.metric = "invalid"
        index_manager = IndexManager(config=index_config, embeddings=mock_embeddings)
        with pytest.raises(MemVidIndexError, match="Unsupported metric"):
            index_manager.create_index()


class TestIndexManagerPersistence:
    """Test cases for saving and loading index data."""

    def test_save_and_load(self, index_manager, test_texts, test_metadata):
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


class TestIndexManagerMetadataOperations:
    """Test cases for metadata retrieval and management."""

    def test_get_metadata(self, index_manager, test_texts, test_metadata):
        """Test retrieving metadata for indices."""
        # Add texts with metadata
        index_manager.add_texts(test_texts, test_metadata)

        # Get metadata for some indices
        indices = [0, 2, 4]
        metadata = index_manager.get_metadata(indices)

        assert len(metadata) == len(indices)
        assert all(m["id"] == i for m, i in zip(metadata, indices))

    def test_get_metadata_invalid_index(self, index_manager, test_texts, test_metadata):
        """Test getting metadata with invalid indices."""
        # Add texts with metadata
        index_manager.add_texts(test_texts, test_metadata)

        with pytest.raises(MemVidIndexError):
            index_manager.get_metadata([100])  # Index out of range


class TestIndexManagerDeletionOperations:
    """Test cases for deletion operations."""

    def test_delete_by_ids_success(self, index_manager):
        """Test deleting documents by IDs successfully."""
        # First add some documents
        texts = ["test1", "test2", "test3"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
        index_manager.add_texts(texts, metadatas)

        # Delete document with ID 1
        result = index_manager.delete_by_ids([1])

        assert result is True
        assert len(index_manager._metadata) == 2
        assert index_manager._metadata[0]["text"] == "test1"
        assert index_manager._metadata[1]["text"] == "test3"

    def test_delete_by_ids_multiple(self, index_manager):
        """Test deleting multiple documents by IDs."""
        # First add some documents
        texts = ["test1", "test2", "test3", "test4"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}, {"source": "doc4"}]
        index_manager.add_texts(texts, metadatas)

        # Delete documents with IDs 1 and 3
        result = index_manager.delete_by_ids([1, 3])

        assert result is True
        assert len(index_manager._metadata) == 2
        assert index_manager._metadata[0]["text"] == "test1"
        assert index_manager._metadata[1]["text"] == "test3"

    def test_delete_by_ids_empty_list(self, index_manager):
        """Test deleting with empty IDs list."""
        # Initialize the index first
        index_manager.create_index()
        result = index_manager.delete_by_ids([])
        assert result is False

    def test_delete_by_ids_invalid_ids(self, index_manager):
        """Test deleting with invalid IDs."""
        # Add some documents first
        texts = ["test1", "test2"]
        index_manager.add_texts(texts)

        with pytest.raises(MemVidIndexError):
            index_manager.delete_by_ids([100])  # Invalid ID

    def test_delete_by_ids_index_not_initialized(self, index_manager):
        """Test deleting when index is not initialized."""
        with pytest.raises(MemVidIndexError):
            index_manager.delete_by_ids([0])

    def test_delete_by_texts_success(self, index_manager):
        """Test deleting documents by texts successfully."""
        # First add some documents
        texts = ["test1", "test2", "test3"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
        index_manager.add_texts(texts, metadatas)

        # Delete document with text "test2"
        result = index_manager.delete_by_texts(["test2"])

        assert result is True
        assert len(index_manager._metadata) == 2
        assert index_manager._metadata[0]["text"] == "test1"
        assert index_manager._metadata[1]["text"] == "test3"

    def test_delete_by_texts_multiple(self, index_manager):
        """Test deleting multiple documents by texts."""
        # First add some documents
        texts = ["test1", "test2", "test3", "test4"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}, {"source": "doc4"}]
        index_manager.add_texts(texts, metadatas)

        # Delete documents with texts "test2" and "test4"
        result = index_manager.delete_by_texts(["test2", "test4"])

        assert result is True
        assert len(index_manager._metadata) == 2
        assert index_manager._metadata[0]["text"] == "test1"
        assert index_manager._metadata[1]["text"] == "test3"

    def test_delete_by_texts_empty_list(self, index_manager):
        """Test deleting with empty texts list."""
        # Initialize the index first
        index_manager.create_index()
        result = index_manager.delete_by_texts([])
        assert result is False

    def test_delete_by_texts_not_found(self, index_manager):
        """Test deleting texts that don't exist."""
        # Add some documents first
        texts = ["test1", "test2"]
        index_manager.add_texts(texts)

        result = index_manager.delete_by_texts(["nonexistent"])
        assert result is False

    def test_delete_by_texts_index_not_initialized(self, index_manager):
        """Test deleting texts when index is not initialized."""
        with pytest.raises(MemVidIndexError):
            index_manager.delete_by_texts(["test"])

    def test_get_all_documents(self, index_manager):
        """Test getting all documents."""
        # Add some documents
        texts = ["test1", "test2", "test3"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]
        index_manager.add_texts(texts, metadatas)

        # Get all documents
        documents = index_manager.get_all_documents()

        assert len(documents) == 3
        assert all(doc["text"] == text for doc, text in zip(documents, texts))

    def test_get_all_documents_empty(self, index_manager):
        """Test getting all documents when index is empty."""
        documents = index_manager.get_all_documents()
        assert len(documents) == 0

    def test_get_document_count(self, index_manager):
        """Test getting document count."""
        # Add some documents
        texts = ["test1", "test2", "test3"]
        index_manager.add_texts(texts)

        count = index_manager.get_document_count()
        assert count == 3
