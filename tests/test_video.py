"""
Unit tests for the VideoProcessor class.
"""

import pytest
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import cv2
import qrcode

from langchain_memvid.video import VideoProcessor
from langchain_memvid.config import VideoConfig, QRCodeConfig, VideoBackend
from langchain_memvid.exceptions import VideoProcessingError, QRCodeError


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
        error_correction="M",
        box_size=10,
        border=4
    )


@pytest.fixture
def video_processor(video_config, qrcode_config):
    """Create a VideoProcessor instance for testing."""
    return VideoProcessor(
        video_config=video_config,
        qrcode_config=qrcode_config
    )


def test_backend_selection():
    """Test automatic backend selection based on codec."""
    # OpenCV codecs
    config = VideoConfig(codec="mp4v")
    assert config.backend == VideoBackend.OPENCV

    config = VideoConfig(codec="mjpg")
    assert config.backend == VideoBackend.OPENCV

    config = VideoConfig(codec="xvid")
    assert config.backend == VideoBackend.OPENCV

    # FFmpeg codecs
    config = VideoConfig(codec="libx264")
    assert config.backend == VideoBackend.FFMPEG

    config = VideoConfig(codec="libx265")
    assert config.backend == VideoBackend.FFMPEG

    config = VideoConfig(codec="h265")
    assert config.backend == VideoBackend.FFMPEG

    config = VideoConfig(codec="av1")
    assert config.backend == VideoBackend.FFMPEG


def test_backend_override():
    """Test manual backend override."""
    # Override to FFmpeg
    config = VideoConfig(
        codec="mp4v",
        backend=VideoBackend.FFMPEG
    )
    assert config.backend == VideoBackend.FFMPEG

    # Override to OpenCV
    config = VideoConfig(
        codec="libx264",
        backend=VideoBackend.OPENCV
    )
    assert config.backend == VideoBackend.OPENCV


def test_create_qr_code(video_processor):
    """Test QR code creation."""
    # Test with simple data
    data = "test data"
    qr_image = video_processor.create_qr_code(data)

    assert isinstance(qr_image, qrcode.image.pil.PilImage)
    assert qr_image.mode == "1"  # Should be binary
    assert qr_image.size[0] > 0
    assert qr_image.size[1] > 0


def test_create_qr_code_invalid_data(video_processor):
    """Test QR code creation with invalid data."""
    with pytest.raises(QRCodeError):
        video_processor.create_qr_code(None)


def test_encode_video(video_processor):
    """Test video encoding."""
    # Create test frames
    frames = []
    for _ in range(3):
        # Create a simple test image
        img = Image.new('RGB', (640, 480), color='white')
        frames.append(img)

    # Create temporary file for output
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        # Encode video
        video_processor.encode_video(frames, output_path)

        # Verify video file exists and has content
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify video can be read
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()

        # Verify frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count == len(frames)

        cap.release()

    finally:
        # Cleanup
        output_path.unlink(missing_ok=True)


def test_encode_video_empty_frames(video_processor):
    """Test video encoding with empty frame list."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        with pytest.raises(VideoProcessingError):
            video_processor.encode_video([], output_path)
    finally:
        output_path.unlink(missing_ok=True)


def test_decode_video(video_processor):
    """Test video decoding."""
    # Create test video
    frames = []
    for i in range(3):
        # Create test image with different colors
        img = Image.new('RGB', (640, 480), color=(i * 50, i * 50, i * 50))
        frames.append(img)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        video_path = Path(tmp.name)

    try:
        # Encode test video
        video_processor.encode_video(frames, video_path)

        # Decode video
        decoded_frames = list(video_processor.decode_video(video_path))

        # Verify frame count
        assert len(decoded_frames) == len(frames)

        # Verify frame dimensions
        for frame in decoded_frames:
            assert isinstance(frame, Image.Image)
            assert frame.size == (640, 480)

    finally:
        video_path.unlink(missing_ok=True)


def test_decode_video_nonexistent(video_processor):
    """Test video decoding with nonexistent file."""
    with pytest.raises(VideoProcessingError):
        list(video_processor.decode_video(Path("nonexistent.mp4")))


def test_extract_qr_codes(video_processor):
    """Test QR code extraction from frame."""
    # Create test data
    test_data = "test qr code data"

    # Create QR code
    qr_image = video_processor.create_qr_code(test_data)

    # Convert to numpy array for OpenCV
    frame = np.array(qr_image)

    # Extract QR codes
    extracted_data = video_processor.extract_qr_codes(Image.fromarray(frame))

    # Verify extracted data
    assert len(extracted_data) == 1
    assert extracted_data[0] == test_data


def test_extract_qr_codes_no_qr(video_processor):
    """Test QR code extraction from frame without QR code."""
    # Create empty frame
    frame = Image.new('RGB', (640, 480), color='white')

    # Extract QR codes
    extracted_data = video_processor.extract_qr_codes(frame)

    # Verify no QR codes found
    assert len(extracted_data) == 0


def test_extract_qr_codes_invalid_frame(video_processor):
    """Test QR code extraction with invalid frame."""
    with pytest.raises(QRCodeError):
        video_processor.extract_qr_codes(None)
