"""
FFmpeg-based video processing implementation.

This module provides video processing capabilities using FFmpeg as the backend.
"""

import subprocess
import tempfile
import numpy as np
import json
from pathlib import Path
from typing import List, Generator, Optional, Dict, Any
from PIL import Image

from ..exceptions import VideoProcessingError
from .codecs import get_codec_parameters, CodecParameters
from ..logging import get_logger
from ..types import VideoInfo

logger = get_logger("video")


class FFmpegProcessor:
    """FFmpeg-based video processor for MemVid.

    - Encodes and decodes video frames using FFmpeg.
    - Used for efficient frame-level operations and video manipulation.
    """

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
            for field in override_params.__class__.model_fields:
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
                subprocess.run(
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

    def remove_frames_from_video(
        self,
        video_path: Path,
        frame_numbers: List[int],
        output_path: Path,
    ) -> Path:
        """Remove specific frames from a video file using FFmpeg.

        This method uses FFmpeg's select filter to efficiently remove frames
        without re-encoding the entire video.

        Args:
            video_path: Path to the input video file
            frame_numbers: List of frame numbers to remove (0-indexed)
            output_path: Path to the output video file

        Returns:
            Path to the new video file without the specified frames

        Raises:
            VideoProcessingError: If frame removal fails
        """
        try:
            if not video_path.exists():
                raise VideoProcessingError(f"Video file not found: {video_path}")

            if not frame_numbers:
                # No frames to remove, copy original file
                import shutil
                shutil.copy2(video_path, output_path)
                return output_path

            # Sort frame numbers in ascending order for FFmpeg filter
            frame_numbers = sorted(frame_numbers)

            # Create FFmpeg select filter expression
            # The filter keeps frames where 'not(n)' is true for frames to remove
            select_expr = "not(" + "+".join(f"eq(n,{frame})" for frame in frame_numbers) + ")"

            # Build FFmpeg command
            cmd = self._get_ffmpeg_command(video_path)
            cmd.extend([
                "-vf", f"select='{select_expr}'",
                "-c:v", self.codec,
                "-pix_fmt", self.codec_params.pix_fmt,
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts"
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
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"Removed {len(frame_numbers)} frames from video using FFmpeg")
            return output_path

        except subprocess.CalledProcessError as e:
            raise VideoProcessingError(
                f"FFmpeg frame removal failed: {e.stderr}"
            ) from e
        except Exception as e:
            raise VideoProcessingError(f"Failed to remove frames from video: {str(e)}") from e

    def get_video_info(self, video_path: Path) -> VideoInfo:
        """Get information about a video file using FFmpeg.

        Args:
            video_path: Path to the video file

        Returns:
            VideoInfo: Information about the video file

        Raises:
            VideoProcessingError: If getting video info fails
        """
        try:
            if not video_path.exists():
                raise VideoProcessingError(f"Video file not found: {video_path}")

            # Use FFmpeg to get video information
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            # Find video stream
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if not video_stream:
                raise VideoProcessingError("No video stream found")

            # Extract video information
            frame_count = int(video_stream.get("nb_frames", 0))
            fps_str = video_stream.get("r_frame_rate", "0/1")
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 0
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            duration_seconds = float(data.get("format", {}).get("duration", 0))
            file_size_mb = video_path.stat().st_size / (1024 * 1024)

            return VideoInfo(
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
                duration_seconds=duration_seconds,
                file_size_mb=file_size_mb
            )

        except subprocess.CalledProcessError as e:
            raise VideoProcessingError(f"FFmpeg command failed: {e}")
        except Exception as e:
            raise VideoProcessingError(f"Failed to get video info: {str(e)}")
