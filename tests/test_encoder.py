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
def mock_index_manager():
    manager = Mock()
    manager.embeddings = Mock()
    manager.embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    return manager


@pytest.fixture
def encoder(mock_config, mock_index_manager):
    return Encoder(config=mock_config, index_manager=mock_index_manager)


def test_encoder_initialization(mock_config, mock_index_manager):
    encoder = Encoder(config=mock_config, index_manager=mock_index_manager)
    assert encoder.config == mock_config
    assert encoder.index_manager == mock_index_manager
    assert encoder._chunks == []


def test_add_chunks_success(encoder):
    texts = ["test1", "test2"]
    metadatas = [{"source": "doc1"}, {"source": "doc2"}]

    encoder.add_chunks(texts, metadatas)

    assert len(encoder._chunks) == 2
    assert encoder._chunks[0]["text"] == "test1"
    assert encoder._chunks[0]["metadata"] == {"source": "doc1"}
    assert encoder._chunks[1]["text"] == "test2"
    assert encoder._chunks[1]["metadata"] == {"source": "doc2"}


def test_add_chunks_without_metadata(encoder):
    texts = ["test1", "test2"]

    encoder.add_chunks(texts)

    assert len(encoder._chunks) == 2
    assert encoder._chunks[0]["text"] == "test1"
    assert encoder._chunks[0]["metadata"] == {}
    assert encoder._chunks[1]["text"] == "test2"
    assert encoder._chunks[1]["metadata"] == {}


def test_add_chunks_mismatched_lengths(encoder):
    texts = ["test1", "test2"]
    metadatas = [{"source": "doc1"}]  # Only one metadata entry

    with pytest.raises(EncodingError, match="Number of texts must match number of metadata entries"):
        encoder.add_chunks(texts, metadatas)


def test_clear_chunks(encoder):
    texts = ["test1", "test2"]
    encoder.add_chunks(texts)
    assert len(encoder._chunks) == 2

    encoder.clear()
    assert len(encoder._chunks) == 0


def test_build_video_success(encoder, tmp_path):
    # Setup
    texts = ["test1", "test2"]
    metadatas = [{"source": "doc1"}, {"source": "doc2"}]
    encoder.add_chunks(texts, metadatas)

    output_file = tmp_path / "output.mp4"
    index_file = tmp_path / "index.json"

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
    stats = encoder.build_video(output_file, index_file)

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
    encoder.index_manager.save.assert_called_once_with(index_file.with_suffix(".d"))


def test_build_video_no_chunks(encoder, tmp_path):
    output_file = tmp_path / "output.mp4"
    index_file = tmp_path / "index.json"

    with pytest.raises(EncodingError, match="No chunks to encode"):
        encoder.build_video(output_file, index_file)


def test_build_video_handles_errors(encoder, tmp_path):
    # Setup
    texts = ["test1"]
    encoder.add_chunks(texts)

    output_file = tmp_path / "output.mp4"
    index_file = tmp_path / "index.json"

    # Mock video processor to raise an error
    encoder.video_processor.create_qr_code = Mock(side_effect=Exception("Test error"))

    with pytest.raises(EncodingError, match="Failed to build video"):
        encoder.build_video(output_file, index_file)
