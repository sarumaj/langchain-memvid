"""
Unit tests for the FFmpeg video processor.
"""

import pytest
from pathlib import Path
import tempfile
from PIL import Image
import subprocess

from langchain_memvid.video.ffmpeg import FFmpegProcessor
from langchain_memvid.exceptions import VideoProcessingError


# Test if FFmpeg is available
def is_ffmpeg_available():
    """Check if FFmpeg is available on the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# Skip all tests if FFmpeg is not installed
pytestmark = pytest.mark.skipif(
    not is_ffmpeg_available(),
    reason="FFmpeg is not available"
)


@pytest.fixture
def ffmpeg_processor():
    """Create a test FFmpeg processor."""
    return FFmpegProcessor(
        fps=30,
        resolution=(640, 480),
        codec="libx264",
        ffmpeg_options={"preset": "ultrafast"}
    )


class TestFFmpegVideoEncoding:
    """Test cases for video encoding functionality."""

    def test_encode_video(self, ffmpeg_processor):
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
            ffmpeg_processor.encode_video(frames, output_path)

            # Verify video file exists and has content
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify video can be read
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "json", str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            assert result.returncode == 0

        finally:
            # Cleanup
            output_path.unlink(missing_ok=True)

    def test_encode_video_empty_frames(self, ffmpeg_processor):
        """Test video encoding with empty frame list."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            output_path = Path(tmp.name)

        try:
            with pytest.raises(VideoProcessingError):
                ffmpeg_processor.encode_video([], output_path)
        finally:
            output_path.unlink(missing_ok=True)


class TestFFmpegVideoDecoding:
    """Test cases for video decoding functionality."""

    def test_decode_video(self, ffmpeg_processor):
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
            ffmpeg_processor.encode_video(frames, video_path)

            # Decode video
            decoded_frames = list(ffmpeg_processor.decode_video(video_path))

            # Verify frame count
            assert len(decoded_frames) == len(frames)

            # Verify frame dimensions
            for frame in decoded_frames:
                assert isinstance(frame, Image.Image)
                assert frame.size == (640, 480)

        finally:
            video_path.unlink(missing_ok=True)

    def test_decode_video_nonexistent(self, ffmpeg_processor):
        """Test video decoding with nonexistent file."""
        with pytest.raises(VideoProcessingError):
            list(ffmpeg_processor.decode_video(Path("nonexistent.mp4")))
