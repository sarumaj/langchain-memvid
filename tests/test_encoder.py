"""Unit tests for the Encoder class."""

import pytest
from unittest.mock import Mock
from PIL import Image

from langchain_memvid.encoder import Encoder
from langchain_memvid.exceptions import EncodingError
from langchain_memvid.config import VectorStoreConfig, VideoConfig, QRCodeConfig


@pytest.fixture
def mock_config():
    return VectorStoreConfig(
        video=VideoConfig(
            width=100,
            height=100,
            fps=30,
            duration=1.0
        ),
        qrcode=QRCodeConfig(
            version=1,
            box_size=10,
            border=4
        )
    )


@pytest.fixture
def encoder(vector_store_config, mock_index_manager):
    """Create an Encoder instance for testing."""
    return Encoder(config=vector_store_config, index_manager=mock_index_manager)


class TestEncoderInitialization:
    """Test cases for Encoder initialization."""

    def test_encoder_initialization(self, vector_store_config, mock_index_manager):
        """Test encoder initialization."""
        encoder = Encoder(config=vector_store_config, index_manager=mock_index_manager)
        assert encoder.config == vector_store_config
        assert encoder.index_manager == mock_index_manager
        assert encoder._chunks == []


class TestEncoderChunkManagement:
    """Test cases for chunk management operations."""

    def test_add_chunks_success(self, encoder):
        """Test adding chunks successfully."""
        texts = ["test1", "test2"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}]

        encoder.add_chunks(texts, metadatas)

        assert len(encoder._chunks) == 2
        assert encoder._chunks[0]["text"] == "test1"
        assert encoder._chunks[0]["metadata"] == {"source": "doc1"}
        assert encoder._chunks[1]["text"] == "test2"
        assert encoder._chunks[1]["metadata"] == {"source": "doc2"}

    def test_add_chunks_without_metadata(self, encoder):
        """Test adding chunks without metadata."""
        texts = ["test1", "test2"]

        encoder.add_chunks(texts)

        assert len(encoder._chunks) == 2
        assert encoder._chunks[0]["text"] == "test1"
        assert encoder._chunks[0]["metadata"] == {}
        assert encoder._chunks[1]["text"] == "test2"
        assert encoder._chunks[1]["metadata"] == {}

    def test_add_chunks_mismatched_lengths(self, encoder):
        """Test adding chunks with mismatched lengths."""
        texts = ["test1", "test2"]
        metadatas = [{"source": "doc1"}]  # Only one metadata entry

        with pytest.raises(EncodingError, match="Number of texts must match number of metadata entries"):
            encoder.add_chunks(texts, metadatas)

    def test_clear_chunks(self, encoder):
        """Test clearing chunks."""
        texts = ["test1", "test2"]
        encoder.add_chunks(texts)
        assert len(encoder._chunks) == 2

        encoder.clear()
        assert len(encoder._chunks) == 0


class TestEncoderVideoBuilding:
    """Test cases for video building functionality."""

    def test_build_video_success(self, encoder, tmp_path):
        """Test successful video building."""
        # Setup
        texts = ["test1", "test2"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}]
        encoder.add_chunks(texts, metadatas)

        output_file = tmp_path / "output.mp4"
        index_dir = tmp_path / "index.d"

        # Create mock output file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            f.write(b'mock video content' * 1024)  # Write 1KB of mock data

        # Create a mock QR code image
        mock_qr_image = Image.new('RGB', (100, 100), color='white')

        # Mock video processor methods
        encoder.video_processor.create_qr_code = Mock(return_value=mock_qr_image)

        def mock_encode_video(frames, output_path):
            # Ensure the output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Create a mock video file
            with open(output_path, 'wb') as f:
                f.write(b'mock video content' * 1024)

        encoder.video_processor.encode_video = Mock(side_effect=mock_encode_video)

        # Execute
        stats = encoder.build_video(output_file, index_dir)

        # Verify
        assert stats.total_chunks == 2
        assert isinstance(stats.video_size_mb, float)
        assert stats.video_size_mb > 0  # Should be non-zero since we wrote mock data
        assert isinstance(stats.encoding_time, float)
        assert len(encoder._chunks) == 0  # Chunks should be cleared after build

        # Verify video processor calls
        assert encoder.video_processor.create_qr_code.call_count == 2  # Called once for each chunk
        encoder.video_processor.encode_video.assert_called_once()

        # Verify index manager calls
        encoder.index_manager.add_texts.assert_called_once()
        encoder.index_manager.save.assert_called_once_with(index_dir.with_suffix(".d"))

    def test_build_video_no_chunks(self, encoder, tmp_path):
        """Test building video with no chunks."""
        output_file = tmp_path / "output.mp4"
        index_dir = tmp_path / "index.d"

        with pytest.raises(EncodingError, match="No chunks to encode"):
            encoder.build_video(output_file, index_dir)

    def test_build_video_handles_errors(self, encoder, tmp_path):
        """Test error handling during video building."""
        # Setup
        texts = ["test1"]
        encoder.add_chunks(texts)

        output_file = tmp_path / "output.mp4"
        index_dir = tmp_path / "index.d"

        # Mock video processor to raise an error
        encoder.video_processor.create_qr_code = Mock(side_effect=Exception("Test error"))

        with pytest.raises(EncodingError, match="Failed to build video"):
            encoder.build_video(output_file, index_dir)
