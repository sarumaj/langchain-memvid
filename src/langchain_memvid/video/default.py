"""
Video processing utilities for MemVid.

This module provides utilities for video processing, including encoding and decoding
of QR codes in video format.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Generator, NamedTuple, Iterable
import qrcode
from qrcode.image.base import BaseImage
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from ..exceptions import VideoProcessingError, QRCodeError
from ..config import VideoConfig, QRCodeConfig, VideoBackend
from ..utils import ProgressDisplay
from ..logging import get_logger
from .ffmpeg import FFmpegProcessor
from .codecs import get_codec_parameters

logger = get_logger("video")


class QRCodeDetection(NamedTuple):
    retval: bool
    decoded_info: List[str]
    points: List[List[int]]
    straight_qrcode: np.ndarray


class VideoProcessor:
    """Handles video processing operations for MemVid."""

    def __init__(
        self,
        video_config: VideoConfig,
        qrcode_config: QRCodeConfig,
    ):
        """Initialize the video processor.

        Args:
            video_config: Configuration for video processing
            qrcode_config: Configuration for QR code generation
        """
        self.video_config = video_config
        self.qrcode_config = qrcode_config
        self._qr = qrcode.QRCode(
            version=qrcode_config.version,
            error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{qrcode_config.error_correction}"),
            box_size=qrcode_config.box_size,
            border=qrcode_config.border
        )
        # Pre-calculate video dimensions
        self.width, self.height = video_config.resolution
        # Pre-calculate scaling factor for QR codes
        self._scale_factor = min(self.width, self.height) * 0.8

        # Get codec parameters for file type validation
        self._codec_params, is_supported = get_codec_parameters(video_config.codec)
        if not is_supported:
            logger.warning(f"Codec {video_config.codec} is not supported, using default parameters")

        # Initialize video backend
        if video_config.backend == VideoBackend.FFMPEG:
            self._video_processor = FFmpegProcessor(
                fps=video_config.fps,
                resolution=video_config.resolution,
                codec=video_config.codec,
                ffmpeg_options=video_config.ffmpeg_options
            )
        else:
            self._video_processor = None  # Use OpenCV methods directly

        # Initialize progress display
        self._progress = ProgressDisplay(show_progress=video_config.show_progress)

    def _validate_output_path(self, output_path: Path) -> Path:
        """Validate and adjust output path based on codec file type.

        Args:
            output_path: Original output path

        Returns:
            Adjusted output path with correct extension

        Raises:
            VideoProcessingError: If the codec doesn't support the requested extension
        """
        # Get expected extension from codec parameters
        expected_ext = f".{self._codec_params.video_file_type}"
        current_ext = output_path.suffix.lower()

        if current_ext != expected_ext:
            # Try to change the extension
            new_path = output_path.with_suffix(expected_ext)
            logger.warning(
                f"Changing output extension from {current_ext} to {expected_ext} "
                f"to match codec {self.video_config.codec} requirements"
            )
            return new_path

        return output_path

    def create_qr_code(self, data: str) -> BaseImage:
        """Create a QR code image from data.

        Args:
            data: Data to encode in QR code

        Returns:
            QR code image in binary mode

        Raises:
            QRCodeError: If QR code generation fails
        """
        try:
            if data is None:
                raise QRCodeError("Data cannot be None")

            self._qr.clear()
            self._qr.add_data(data)
            self._qr.make(fit=True)
            return self._qr.make_image(fill_color="black", back_color="white")

        except Exception as e:
            raise QRCodeError(f"Failed to create QR code: {str(e)}")

    def _prepare_frame(self, frame: Image.Image) -> np.ndarray:
        """Prepare a frame for video encoding.

        Args:
            frame: PIL Image to prepare

        Returns:
            OpenCV-compatible numpy array
        """
        # Calculate new size based on pre-calculated scale factor
        scale = self._scale_factor / max(frame.size)
        new_size = tuple(int(dim * scale) for dim in frame.size)

        # Resize the frame
        frame = frame.resize(new_size, Image.Resampling.LANCZOS)

        # Calculate position to center the frame
        x = (self.width - new_size[0]) // 2
        y = (self.height - new_size[1]) // 2

        # Create frame array directly in BGR format
        frame_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame_array.fill(255)  # White background

        # Convert frame to numpy array and paste it
        frame_np = np.array(frame)
        match frame.mode:
            case '1':
                frame_np = frame_np.astype(np.uint8) * 255
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
            case 'RGB':
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            case _:
                raise ValueError(f"Unsupported frame mode: {frame.mode}")

        # Paste the frame onto the background
        frame_array[y:y+new_size[1], x:x+new_size[0]] = frame_np
        return frame_array

    def _prepare_frames_batch(self, frames: Iterable[Image.Image]) -> Generator[np.ndarray, None, None]:
        """Prepare a batch of frames for video encoding.

        Args:
            frames: List of PIL Images to prepare

        Returns:
            List of OpenCV-compatible numpy arrays
        """
        with ThreadPoolExecutor() as executor:
            # Convert to list to get length for progress bar
            frames_list = list(frames)
            futures = list(executor.map(self._prepare_frame, frames_list))
            for future in self._progress.tqdm(futures, desc="Preparing frames", total=len(frames_list)):
                yield future

    def encode_video(
        self,
        frames: Iterable[Image.Image],
        output_path: Path,
    ) -> None:
        """Encode frames into a video file.

        Args:
            frames: List of PIL Images to encode
            output_path: Path to save the video file

        Raises:
            VideoProcessingError: If video encoding fails
        """
        try:
            if not frames:
                raise VideoProcessingError("No frames to encode")

            # Validate and adjust output path
            output_path = self._validate_output_path(output_path)

            if self._video_processor is not None:
                self._video_processor.encode_video(frames, output_path)
                return

            # Convert frames to list to get total count
            frames_list = list(frames)
            total_frames = len(frames_list)
            logger.info(f"Encoding {total_frames} frames to video...")

            # Use OpenCV backend
            # Prepare all frames in parallel
            prepared_frames = self._prepare_frames_batch(frames_list)

            # Set up video writer with optimized parameters
            fourcc = cv2.VideoWriter_fourcc(*self.video_config.codec)
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.video_config.fps,
                (self.width, self.height),
                isColor=True
            )

            # Write frames with progress bar
            for frame in self._progress.tqdm(prepared_frames, desc="Writing video", total=total_frames):
                out.write(frame)

            out.release()
            logger.info(f"Video encoded successfully to {output_path}")

        except Exception as e:
            raise VideoProcessingError(f"Failed to encode video: {str(e)}")

    def decode_video(
        self,
        video_path: Path,
    ) -> Generator[Image.Image, None, None]:
        """Decode frames from a video file.

        Args:
            video_path: Path to the video file

        Yields:
            PIL Images from the video frames

        Raises:
            VideoProcessingError: If video decoding fails
        """
        try:
            if self._video_processor is not None:
                yield from self._video_processor.decode_video(video_path)
                return

            # Use OpenCV backend
            if not (cap := cv2.VideoCapture(str(video_path))).isOpened():
                raise VideoProcessingError(f"Failed to open video file: {video_path}")

            # Get total frame count for progress bar
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Decoding {total_frames} frames from video...")

            with self._progress.progress(total=total_frames, desc="Decoding video") as pbar:
                while True:
                    retval, frame = cap.read()
                    if not retval:
                        break

                    # Convert OpenCV frame to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield Image.fromarray(frame_rgb)
                    pbar.update(1)

            cap.release()

        except Exception as e:
            raise VideoProcessingError(f"Failed to decode video: {str(e)}")

    def extract_qr_codes(
        self,
        frame: Image.Image,
    ) -> List[str]:
        """Extract QR codes from a frame.

        Args:
            frame: PIL Image to extract QR codes from

        Returns:
            List of decoded QR code data

        Raises:
            QRCodeError: If QR code extraction fails
        """
        try:
            # Convert PIL Image to OpenCV format
            match frame.mode:
                case '1':       # Binary image
                    frame = frame.convert('RGB')
                case 'RGB':     # RGB image
                    pass
                case _:
                    raise ValueError(f"Unsupported frame mode: {frame.mode}")

            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

            # Detect QR codes
            detector = cv2.QRCodeDetector()
            detection = QRCodeDetection(*detector.detectAndDecodeMulti(frame_cv))

            return [info for info in detection.decoded_info if info] if detection.retval else []

        except Exception as e:
            raise QRCodeError(f"Failed to extract QR codes: {str(e)}")
