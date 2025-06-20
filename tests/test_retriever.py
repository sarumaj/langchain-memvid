"""Unit tests for the Retriever class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

import pytest

from langchain_core.documents import Document

from langchain_memvid.retriever import Retriever
from langchain_memvid.exceptions import RetrievalError


@pytest.fixture
def mock_video_processor():
    """Create a mock video processor."""
    processor = MagicMock()

    def create_new_frame():
        return Image.new('RGB', (640, 480), color='white')

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


class TestRetrieverInitialization:
    """Test cases for Retriever initialization."""

    def test_retriever_initialization(self, retriever, vector_store_config, mock_index_manager):
        """Test retriever initialization."""
        assert retriever.config == vector_store_config
        assert retriever.index_manager == mock_index_manager
        assert retriever.video_processor is not None
        assert retriever.k == 4  # Default value
        assert retriever.frame_cache_size == 100  # Default value

    def test_retriever_initialization_with_custom_params(
        self, vector_store_config, mock_index_manager, mock_video_processor
    ):
        """Test retriever initialization with custom parameters."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video, \
             tempfile.NamedTemporaryFile(suffix='.d', delete=False) as tmp_index:
            video_path = Path(tmp_video.name)
            index_path = Path(tmp_index.name)

        try:
            with patch('langchain_memvid.retriever.VideoProcessor', return_value=mock_video_processor):
                retriever = Retriever(
                    video_file=video_path,
                    index_dir=index_path,
                    config=vector_store_config,
                    index_manager=mock_index_manager,
                    k=10,
                    frame_cache_size=200
                )

            assert retriever.k == 10
            assert retriever.frame_cache_size == 200

        finally:
            video_path.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)


class TestRetrieverDocumentRetrieval:
    """Test cases for document retrieval operations."""

    def test_get_relevant_documents(self, retriever):
        """Test getting relevant documents."""
        query = "test query"
        documents = retriever._get_relevant_documents(query)

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].page_content == "test1"
        assert documents[1].page_content == "test2"

        # Verify embeddings were called
        retriever.index_manager.embeddings.embed_query.assert_called_once_with(query)

    def test_get_document_by_id(self, retriever):
        """Test getting a document by ID."""
        doc = retriever.get_document_by_id(0)
        assert doc.page_content == "test1"
        assert doc.metadata["source"] == "doc1"

    def test_get_document_by_id_not_found(self, retriever):
        """Test getting a document by non-existent ID."""
        # Mock the get_metadata method to return empty list for non-existent ID
        retriever.index_manager.get_metadata.return_value = []

        # The method should return None when document is not found
        result = retriever.get_document_by_id(999)
        assert result is None

    def test_get_document_by_id_with_full_metadata(self, retriever):
        """Test getting a document by ID with full metadata."""
        doc = retriever.get_document_by_id(0, include_full_metadata=True)
        assert doc.page_content == "test1"
        assert doc.metadata["source"] == "doc1"
        assert doc.metadata["category"] == "test_category"

    def test_get_documents_by_ids(self, retriever):
        """Test getting multiple documents by IDs."""
        docs = retriever.get_documents_by_ids([0, 1])
        assert len(docs) == 2
        assert docs[0].page_content == "test1"
        assert docs[1].page_content == "test2"


class TestRetrieverFrameDecoding:
    """Test cases for frame decoding operations."""

    def test_decode_frame(self, retriever):
        """Test decoding a single frame."""
        # Mock the video processor to return a Document
        retriever.video_processor.extract_qr_codes.return_value = [
            '{"text": "test", "metadata": {"source": "test"}}'
        ]

        doc = retriever.decode_frame(0)
        assert isinstance(doc, Document)
        assert doc.page_content == "test"

    def test_decode_all_frames(self, retriever):
        """Test decoding all frames."""
        # Mock the video processor to return Documents
        retriever.video_processor.extract_qr_codes.return_value = [
            '{"text": "test", "metadata": {"source": "test"}}'
        ]

        docs = retriever.decode_all_frames()
        assert len(docs) == 1
        assert isinstance(docs[0], Document)


class TestRetrieverCaching:
    """Test cases for caching functionality."""

    def test_frame_caching(self, retriever):
        """Test frame caching behavior."""
        # Mock the video processor
        retriever.video_processor.extract_qr_codes.return_value = [
            '{"text": "test", "metadata": {"source": "test"}}'
        ]

        # First call should decode from video
        doc1 = retriever.decode_frame(0)
        assert isinstance(doc1, Document)

        # Second call should use cache
        doc2 = retriever.decode_frame(0)
        assert isinstance(doc2, Document)

        # Verify video processor was called only once for frame decoding
        assert retriever.video_processor.decode_video.call_count == 1

    def test_clear_cache(self, retriever):
        """Test clearing the frame cache."""
        # Mock the video processor
        retriever.video_processor.extract_qr_codes.return_value = [
            '{"text": "test", "metadata": {"source": "test"}}'
        ]

        # Decode a frame to populate cache
        retriever.decode_frame(0)
        assert len(retriever._frame_cache) > 0

        # Clear cache
        retriever.clear_cache()
        assert len(retriever._frame_cache) == 0


class TestRetrieverErrorHandling:
    """Test cases for error handling."""

    def test_error_handling(self, retriever):
        """Test error handling during retrieval."""
        # Mock index manager to raise an error
        retriever.index_manager.search_text.side_effect = Exception("Test error")

        with pytest.raises(RetrievalError):
            retriever._get_relevant_documents("test query")
