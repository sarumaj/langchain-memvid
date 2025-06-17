"""
FFmpeg video processing backend for MemVid.

This module provides FFmpeg-based video processing functionality.
"""

import subprocess
from pathlib import Path
from typing import List, Generator, Optional, Dict, Any
import logging
from PIL import Image
import tempfile
import numpy as np

from ..exceptions import VideoProcessingError
from .codecs import get_codec_parameters, CodecParameters

logger = logging.getLogger(__name__)


class FFmpegProcessor:
    """FFmpeg-based video processor."""

    def __init__(
        self,
        fps: int,
        resolution: tuple[int, int],
        codec: str,
        ffmpeg_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize FFmpeg processor.

        Args:
            fps: Frames per second
            resolution: Video resolution (width, height)
            codec: Video codec to use
            ffmpeg_options: Additional FFmpeg options
        """
        self.fps = fps
        self.width, self.height = resolution
        self.codec = codec.lower()
        self.ffmpeg_options = ffmpeg_options or {}

        # Get codec-specific parameters
        self.codec_params, is_supported = get_codec_parameters(self.codec)
        if not is_supported:
            logger.warning(f"Codec {self.codec} is not supported, using default parameters")

        # Override with provided options
        if ffmpeg_options:
            # Convert dict to CodecParameters for validation
            override_params = CodecParameters(**ffmpeg_options)
            # Update only the provided fields
            for field in override_params.model_fields:
                if field in ffmpeg_options:
                    setattr(self.codec_params, field, ffmpeg_options[field])

    def _get_ffmpeg_command(self, input_path: Optional[Path] = None) -> List[str]:
        """Get FFmpeg command with configured options.

        Args:
            input_path: Optional input file path

        Returns:
            List of command arguments
        """
        cmd = ["ffmpeg", "-y"]  # -y to overwrite output files

        if input_path:
            cmd.extend(["-i", str(input_path)])

        return cmd

    def encode_video(
        self,
        frames: List[Image.Image],
        output_path: Path,
    ) -> None:
        """Encode frames into a video file using FFmpeg.

        Args:
            frames: List of PIL Images to encode
            output_path: Path to save the video file

        Raises:
            VideoProcessingError: If video encoding fails
        """
        try:
            if not frames:
                raise VideoProcessingError("No frames to encode")

            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save frames as PNG files
                frame_paths = []
                for i, frame in enumerate(frames):
                    frame_path = Path(temp_dir) / f"frame_{i:04d}.png"
                    frame.save(frame_path, "PNG")
                    frame_paths.append(frame_path)

                # Create FFmpeg command
                cmd = self._get_ffmpeg_command()
                cmd.extend([
                    "-framerate", str(self.fps),
                    "-i", str(Path(temp_dir) / "frame_%04d.png"),
                    "-c:v", self.codec,
                    "-pix_fmt", self.codec_params.pix_fmt,
                    "-vf", f"scale={self.width}:{self.height}"
                ])

                # Add codec-specific options
                if self.codec_params.video_crf is not None:
                    cmd.extend(["-crf", str(self.codec_params.video_crf)])
                if self.codec_params.video_preset is not None:
                    cmd.extend(["-preset", self.codec_params.video_preset])
                if self.codec_params.video_profile is not None:
                    cmd.extend(["-profile:v", self.codec_params.video_profile])

                # Add extra FFmpeg arguments if specified
                if self.codec_params.extra_ffmpeg_args:
                    cmd.extend(self.codec_params.extra_ffmpeg_args.split())

                # Add output path
                cmd.append(str(output_path))

                # Run FFmpeg
                _ = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )

                logger.info(f"Video encoded successfully to {output_path}")

        except subprocess.CalledProcessError as e:
            raise VideoProcessingError(
                f"FFmpeg encoding failed: {e.stderr}"
            ) from e
        except Exception as e:
            raise VideoProcessingError(f"Failed to encode video: {str(e)}") from e

    def decode_video(
        self,
        video_path: Path,
    ) -> Generator[Image.Image, None, None]:
        """Decode frames from a video file using FFmpeg.

        Args:
            video_path: Path to the video file

        Yields:
            PIL Images from the video frames

        Raises:
            VideoProcessingError: If video decoding fails
        """
        try:
            if not video_path.exists():
                raise VideoProcessingError(f"Video file not found: {video_path}")

            # Create FFmpeg command for frame extraction
            cmd = self._get_ffmpeg_command(video_path)
            cmd.extend([
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-vf", f"scale={self.width}:{self.height}",
                "-"
            ])

            # Run FFmpeg and process output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )

            frame_size = self.width * self.height * 3  # RGB24 format
            while True:
                # Read raw frame data
                frame_data = process.stdout.read(frame_size)
                if not frame_data or len(frame_data) != frame_size:
                    break

                # Convert to numpy array and create PIL Image
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame_array = frame_array.reshape((self.height, self.width, 3))
                yield Image.fromarray(frame_array)

            # Check for errors
            process.wait()
            if process.returncode != 0:
                error = process.stderr.read().decode()
                raise VideoProcessingError(f"FFmpeg decoding failed: {error}")

        except Exception as e:
            raise VideoProcessingError(f"Failed to decode video: {str(e)}") from e
