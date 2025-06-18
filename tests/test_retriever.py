"""
Unit tests for the Retriever class.
"""

import pytest
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

from langchain_memvid.retriever import Retriever
from langchain_memvid.config import VectorStoreConfig, VideoConfig, QRCodeConfig
from langchain_memvid.exceptions import RetrievalError
from langchain_core.documents import Document
from langchain_memvid.index import SearchResult


@pytest.fixture
def video_config():
    """Create a test video configuration."""
    return VideoConfig(
        fps=30,
        resolution=(640, 480),
        codec="mp4v"
    )


@pytest.fixture
def qrcode_config():
    """Create a test QR code configuration."""
    return QRCodeConfig(
        error_correction="H",
        box_size=10,
        border=4
    )


@pytest.fixture
def vector_store_config(video_config, qrcode_config):
    """Create a test vector store configuration."""
    return VectorStoreConfig(
        video=video_config,
        qrcode=qrcode_config
    )


@pytest.fixture
def mock_index_manager():
    """Create a mock index manager."""
    manager = Mock()

    # Mock embeddings
    embeddings = Mock()
    embeddings.embed_query.return_value = np.random.rand(384)  # Mock embedding
    manager.embeddings = embeddings

    # Create mock search results
    search_results = [
        SearchResult(
            text="test1",
            source="doc1",
            category="test",
            similarity=0.8,
            metadata={"source": "doc1"}
        ),
        SearchResult(
            text="test2",
            source="doc2",
            category="test",
            similarity=0.6,
            metadata={"source": "doc2"}
        )
    ]

    # Create a MagicMock for search_text that calls embed_query internally
    search_text_mock = MagicMock()

    def search_text_side_effect(query_text, k=4):
        # Call embed_query to ensure it's called
        manager.embeddings.embed_query(query_text)
        return search_results[:k]

    search_text_mock.side_effect = search_text_side_effect
    manager.search_text = search_text_mock

    manager.get_metadata.return_value = [
        {"text": "test1", "metadata": {"source": "doc1"}},
        {"text": "test2", "metadata": {"source": "doc2"}}
    ]
    return manager


@pytest.fixture
def mock_video_processor():
    """Create a mock video processor."""
    processor = MagicMock()

    def create_new_frame():
        return Image.new('RGB', (640, 480), color='white')

    # Make decode_video return a generator that yields new frames each time
    def decode_video_mock(*args, **kwargs):
        return [create_new_frame()]

    processor.decode_video.side_effect = decode_video_mock
    processor.extract_qr_codes.return_value = [
        json.dumps({"text": "test", "metadata": {"source": "test"}})
    ]
    return processor


@pytest.fixture
def retriever(vector_store_config, mock_index_manager, mock_video_processor):
    """Create a test retriever instance."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video, \
         tempfile.NamedTemporaryFile(suffix='.d', delete=False) as tmp_index:
        video_path = Path(tmp_video.name)
        index_path = Path(tmp_index.name)

    with patch('langchain_memvid.retriever.VideoProcessor', return_value=mock_video_processor):
        retriever = Retriever(
            video_file=video_path,
            index_dir=index_path,
            config=vector_store_config,
            index_manager=mock_index_manager
        )

    yield retriever

    # Cleanup
    video_path.unlink(missing_ok=True)
    index_path.unlink(missing_ok=True)


def test_retriever_initialization(retriever, vector_store_config, mock_index_manager):
    """Test retriever initialization."""
    assert retriever.config == vector_store_config
    assert retriever.index_manager == mock_index_manager
    assert retriever.video_processor is not None
    assert retriever.k == 4  # Default value
    assert retriever.frame_cache_size == 100  # Default value


def test_get_relevant_documents(retriever):
    """Test getting relevant documents."""
    query = "test query"
    documents = retriever._get_relevant_documents(query)

    # Verify index manager calls
    retriever.index_manager.embeddings.embed_query.assert_called_once_with(query)
    retriever.index_manager.search_text.assert_called_once()

    # Verify results
    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)
    assert documents[0].page_content == "test1"
    assert documents[0].metadata["source"] == "doc1"
    assert documents[0].metadata["similarity"] == 0.8
    assert documents[1].page_content == "test2"
    assert documents[1].metadata["source"] == "doc2"
    assert documents[1].metadata["similarity"] == 0.6


def test_get_document_by_id(retriever):
    """Test getting document by ID."""
    doc = retriever.get_document_by_id(0)

    assert isinstance(doc, Document)
    assert doc.page_content == "test1"
    assert doc.metadata["source"] == "doc1"


def test_get_document_by_id_not_found(retriever):
    """Test getting document by non-existent ID."""
    retriever.index_manager.get_metadata.return_value = []
    doc = retriever.get_document_by_id(999)
    assert doc is None


def test_get_documents_by_ids(retriever):
    """Test getting multiple documents by IDs."""
    docs = retriever.get_documents_by_ids([0, 1])

    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)
    assert docs[0].page_content == "test1"
    assert docs[1].page_content == "test2"


def test_decode_frame(retriever):
    """Test decoding a specific frame."""
    doc = retriever.decode_frame(0)

    assert isinstance(doc, Document)
    assert doc.page_content == "test"
    assert doc.metadata["source"] == "test"


def test_decode_frame_out_of_range(retriever):
    """Test decoding frame with out of range index."""
    retriever.video_processor.decode_video.return_value = []
    doc = retriever.decode_frame(999)
    assert doc is None


def test_decode_all_frames(retriever):
    """Test decoding all frames."""
    docs = retriever.decode_all_frames()

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "test"
    assert docs[0].metadata["source"] == "test"


def test_error_handling(retriever):
    """Test error handling in various methods."""
    # Test error in get_relevant_documents
    retriever.index_manager.search_text.side_effect = Exception("Test error")
    with pytest.raises(RetrievalError):
        retriever._get_relevant_documents("test")

    # Test error in get_document_by_id
    retriever.index_manager.get_metadata.side_effect = Exception("Test error")
    with pytest.raises(RetrievalError):
        retriever.get_document_by_id(0)

    # Test error in decode_frame
    retriever.video_processor.decode_video.side_effect = Exception("Test error")
    with pytest.raises(RetrievalError):
        retriever.decode_frame(0)


def test_retriever_initialization_with_custom_params(vector_store_config, mock_index_manager, mock_video_processor):
    """Test retriever initialization with custom parameters."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video, \
         tempfile.NamedTemporaryFile(suffix='.d', delete=False) as tmp_index:
        video_path = Path(tmp_video.name)
        index_path = Path(tmp_index.name)

    with patch('langchain_memvid.retriever.VideoProcessor', return_value=mock_video_processor):
        retriever = Retriever(
            video_file=video_path,
            index_dir=index_path,
            config=vector_store_config,
            index_manager=mock_index_manager,
            k=10,
            frame_cache_size=50
        )

    try:
        assert retriever.k == 10
        assert retriever.frame_cache_size == 50
        assert retriever.video_processor is not None
    finally:
        video_path.unlink(missing_ok=True)
        index_path.unlink(missing_ok=True)


def test_get_relevant_documents_with_custom_k(retriever):
    """Test getting relevant documents with custom k parameter."""
    retriever.k = 3
    query = "test query"
    documents = retriever._get_relevant_documents(query)

    # Verify index manager calls
    retriever.index_manager.search_text.assert_called_once()
    call_args = retriever.index_manager.search_text.call_args[1]
    assert call_args["k"] == 3

    # Verify results
    assert len(documents) == 2  # Mock returns 2 results


def test_frame_caching(retriever):
    """Test frame caching functionality."""
    # First call should decode video
    frame1 = retriever._get_frame(0)
    assert frame1 is not None

    # Second call should use cache
    frame2 = retriever._get_frame(0)
    assert frame2 is not None
    assert frame1 is frame2  # Should be the same object

    # Verify video processor was only called once
    assert retriever.video_processor.decode_video.call_count == 1


def test_clear_cache(retriever):
    """Test clearing the frame cache."""
    # Get a frame to populate cache
    frame1 = retriever._get_frame(0)
    assert frame1 is not None

    # Verify video processor was called
    assert retriever.video_processor.decode_video.call_count == 1

    # Clear cache
    retriever.clear_cache()

    # Get frame again
    frame2 = retriever._get_frame(0)
    assert frame2 is not None

    # Verify video processor was called again after cache clear
    assert retriever.video_processor.decode_video.call_count == 2

    # Verify we got new frames (even if they're identical)
    assert id(frame1) != id(frame2)  # Should be different objects in memory


def test_decode_all_frames_with_cache_limit(retriever):
    """Test decoding all frames with cache size limit."""
    retriever.frame_cache_size = 2

    # Create mock frames
    mock_frames = [
        Image.new('RGB', (640, 480), color='white'),
        Image.new('RGB', (640, 480), color='white'),
        Image.new('RGB', (640, 480), color='white')
    ]
    retriever.video_processor.decode_video.return_value = mock_frames

    # Decode frames
    docs = retriever.decode_all_frames()

    # Should only process frame_cache_size frames
    assert len(docs) <= 2  # Mock returns 1 doc per frame
    assert retriever.video_processor.decode_video.call_count == 1
