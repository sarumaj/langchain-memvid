import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np

from langchain_memvid.encoder import (
    Encoder,
    EncoderConfig,
    EncodingStats,
    EncoderStats
)
from langchain_memvid.index import IndexManager, IndexConfig


@pytest.fixture
def config():
    """Create a test configuration."""
    return EncoderConfig(
        chunk_size=100,
        overlap=10,
        codec="mp4v"
    )


@pytest.fixture
def mock_index_manager(config):
    """Create a mock index manager."""
    manager = Mock(spec=IndexManager)

    # Mock embedding model
    mock_embedding_model = Mock()
    mock_embedding_model.embed_query.return_value = [0.0] * 384  # Return a fixed-size embedding
    mock_embedding_model.embedding_dimension = 384  # Set dimension attribute

    # Set up manager attributes
    manager.embedding_model = mock_embedding_model
    manager.config = IndexConfig(index_type="Flat", nlist=100)
    manager._dimension = 384  # Set dimension directly

    # Mock get_stats
    manager.get_stats.return_value = Mock(
        total_chunks=2,
        total_frames=2,
        index_type="Flat",
        embedding_model="test-model",
        dimension=384,
        avg_chunks_per_frame=1.0,
        config={}
    )
    return manager


@pytest.fixture
def encoder(config, mock_index_manager):
    """Create a test encoder instance."""
    return Encoder(config=config, index_manager=mock_index_manager)


def test_encoder_initialization(config, mock_index_manager):
    """Test encoder initialization with config."""
    encoder = Encoder(config=config, index_manager=mock_index_manager)
    assert encoder.config == config
    assert encoder.chunks == []
    assert encoder.index_manager == mock_index_manager


def test_add_chunks(encoder):
    """Test adding chunks to encoder."""
    chunks = ["chunk1", "chunk2", "chunk3"]
    encoder.add_chunks(chunks)
    assert encoder.chunks == chunks


def test_add_text(encoder):
    """Test adding and chunking text."""
    text = "This is a test text that should be chunked into multiple pieces."
    encoder.add_text(text)
    assert len(encoder.chunks) > 0
    assert all(isinstance(chunk, str) for chunk in encoder.chunks)


def test_add_text_with_custom_params(encoder):
    """Test adding text with custom chunk size and overlap."""
    text = "This is a test text that should be chunked into multiple pieces."
    custom_chunk_size = 20
    custom_overlap = 5
    encoder.add_text(text, chunk_size=custom_chunk_size, overlap=custom_overlap)
    assert len(encoder.chunks) > 0
    assert all(len(chunk) <= custom_chunk_size for chunk in encoder.chunks)


def test_add_pdf(encoder):
    """Test adding PDF content."""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write(b"%PDF-1.4\nTest PDF content")
        temp_pdf_path = temp_pdf.name

    try:
        with patch('pypdf.PdfReader') as mock_pdf_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test PDF content"
            mock_pdf_reader.return_value.pages = [mock_page]

            encoder.add_pdf(temp_pdf_path)
            assert len(encoder.chunks) > 0
    finally:
        os.unlink(temp_pdf_path)


def test_add_epub(encoder):
    """Test adding EPUB content."""
    with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as temp_epub:
        temp_epub.write(b"PK\x03\x04Test EPUB content")
        temp_epub_path = temp_epub.name

    try:
        with patch('ebooklib.epub.read_epub') as mock_epub_reader:
            # Mock EPUB reader
            mock_item = Mock()
            mock_item.get_type.return_value = 9  # ITEM_DOCUMENT
            mock_item.get_content.return_value = b"<html><body>Test content</body></html>"
            mock_epub_reader.return_value.get_items.return_value = [mock_item]

            # Add EPUB content
            encoder.add_epub(temp_epub_path)

            # Verify chunks were added
            assert len(encoder.chunks) > 0
            assert any("Test content" in chunk for chunk in encoder.chunks)
    finally:
        os.unlink(temp_epub_path)


def test_create_video_writer(encoder):
    """Test video writer creation."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video_path = temp_video.name

    try:
        with patch('cv2.VideoWriter') as mock_writer:
            mock_writer.return_value = Mock()
            writer = encoder.create_video_writer(temp_video_path, "mp4v")
            assert writer is not None
            mock_writer.assert_called_once()
    finally:
        os.unlink(temp_video_path)


def test_create_video_writer_invalid_codec(encoder):
    """Test video writer creation with invalid codec."""
    with pytest.raises(ValueError, match="Unsupported codec"):
        encoder.create_video_writer("test.mp4", "invalid_codec")


def test_build_video(encoder):
    """Test video building process."""
    # Add some test chunks
    encoder.add_chunks(["test chunk 1", "test chunk 2"])

    with tempfile.TemporaryDirectory() as temp_dir:
        output_video = Path(temp_dir) / "output.mp4"
        output_index = Path(temp_dir) / "index.json"

        with (
            patch('cv2.VideoWriter') as mock_writer,
            patch('cv2.imread') as mock_imread,
            patch('langchain_memvid.utils.encode_to_qr') as mock_encode_qr
        ):
            # Mock QR code generation
            mock_qr = Mock()
            mock_qr.save = Mock()
            mock_encode_qr.return_value = mock_qr

            # Mock frame reading with numpy array
            mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Create a black frame
            mock_imread.return_value = mock_frame

            # Mock video writer
            mock_writer_instance = Mock()
            mock_writer.return_value = mock_writer_instance

            # Build video
            stats = encoder.build_video(
                str(output_video),
                str(output_index),
                codec="mp4v",
                show_progress=False
            )

            assert isinstance(stats, EncodingStats)
            assert stats.total_chunks == 2
            assert stats.codec == "mp4v"
            assert stats.video_file == str(output_video)
            assert stats.index_file == str(output_index)

            # Verify index manager was called with correct arguments
            encoder.index_manager.add_chunks.assert_called_once_with(
                ["test chunk 1", "test chunk 2"],
                [0, 1],  # Frame numbers
                False    # Show Progress
            )
            encoder.index_manager.save.assert_called_once()


def test_build_video_empty_chunks(encoder):
    """Test building video with no chunks."""
    with pytest.raises(ValueError, match="No chunks to encode"):
        encoder.build_video("output.mp4", "index.json")


def test_build_video_fallback(encoder):
    """Test video building with codec fallback."""
    encoder.add_chunks(["test chunk"])

    with tempfile.TemporaryDirectory() as temp_dir:
        output_video = Path(temp_dir) / "output.mp4"
        output_index = Path(temp_dir) / "index.json"

        with (
            patch('cv2.VideoWriter') as mock_writer,
            patch('cv2.imread') as mock_imread,
            patch('langchain_memvid.utils.encode_to_qr') as mock_encode_qr,
            patch.object(encoder, '_encode_with_ffmpeg') as mock_ffmpeg
        ):
            # Mock FFmpeg failure
            mock_ffmpeg.side_effect = RuntimeError("FFmpeg failed")

            # Mock QR code generation
            mock_qr = Mock()
            mock_qr.save = Mock()
            mock_encode_qr.return_value = mock_qr

            # Mock frame reading with numpy array
            mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # Create a black frame
            mock_imread.return_value = mock_frame

            # Mock video writer
            mock_writer_instance = Mock()
            mock_writer.return_value = mock_writer_instance

            # Build video with fallback
            stats = encoder.build_video(
                str(output_video),
                str(output_index),
                codec="h265",
                show_progress=False,
                allow_fallback=True
            )

            assert isinstance(stats, EncodingStats)
            assert stats.codec == "mp4v"  # Should fall back to mp4v

            # Verify index manager was called with correct arguments
            encoder.index_manager.add_chunks.assert_called_once_with(
                ["test chunk"],
                [0],    # Frame number
                False   # Show Progress
            )
            encoder.index_manager.save.assert_called_once()


def test_get_stats(encoder):
    """Test getting encoder statistics."""
    encoder.add_chunks(["test chunk 1", "test chunk 2"])
    stats = encoder.get_stats()

    assert isinstance(stats, EncoderStats)
    assert stats.total_chunks == 2
    assert stats.total_characters == len("test chunk 1") + len("test chunk 2")
    assert isinstance(stats.avg_chunk_size, float)
    assert isinstance(stats.supported_codecs, list)
    assert isinstance(stats.config, dict)


def test_clear(encoder):
    """Test clearing encoder state."""
    encoder.add_chunks(["test chunk"])
    assert len(encoder.chunks) > 0
    encoder.clear()
    assert len(encoder.chunks) == 0


def test_from_file(config, mock_index_manager):
    """Test creating encoder from file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("Test content for file")
        temp_file_path = temp_file.name

    try:
        encoder = Encoder.from_file(
            temp_file_path,
            config=config,
            index_manager=mock_index_manager
        )
        assert isinstance(encoder, Encoder)
        assert len(encoder.chunks) > 0
    finally:
        os.unlink(temp_file_path)


def test_from_documents(config, mock_index_manager):
    """Test creating encoder from documents."""
    documents = ["Document 1", "Document 2", "Document 3"]
    encoder = Encoder.from_documents(
        documents,
        config=config,
        index_manager=mock_index_manager
    )
    assert isinstance(encoder, Encoder)
    assert len(encoder.chunks) > 0


def test_from_file_missing_index_manager(config):
    """Test creating encoder from file without index manager."""
    with pytest.raises(ValueError, match="index_manager parameter is required"):
        Encoder.from_file("test.txt", config=config)


def test_from_documents_missing_index_manager(config):
    """Test creating encoder from documents without index manager."""
    with pytest.raises(ValueError, match="index_manager parameter is required"):
        Encoder.from_documents(["test"], config=config)
