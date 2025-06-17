"""
Video processing utilities for MemVid.

This module provides utilities for video processing, including encoding and decoding
of QR codes in video format.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Generator
import qrcode
from qrcode.image.base import BaseImage
from PIL import Image
import logging
from concurrent.futures import ThreadPoolExecutor

from .exceptions import VideoProcessingError, QRCodeError
from .config import VideoConfig, QRCodeConfig

logger = logging.getLogger(__name__)


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
            error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{qrcode_config.error_correction}"),
            box_size=qrcode_config.box_size,
            border=qrcode_config.border
        )
        # Pre-calculate video dimensions
        self.width, self.height = video_config.resolution
        # Pre-calculate scaling factor for QR codes
        self._scale_factor = min(self.width, self.height) * 0.8

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
        if frame.mode == '1':
            frame_np = frame_np.astype(np.uint8) * 255
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
        elif frame.mode == 'RGB':
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Paste the frame onto the background
        frame_array[y:y+new_size[1], x:x+new_size[0]] = frame_np

        return frame_array

    def _prepare_frames_batch(self, frames: List[Image.Image]) -> List[np.ndarray]:
        """Prepare a batch of frames for video encoding.

        Args:
            frames: List of PIL Images to prepare

        Returns:
            List of OpenCV-compatible numpy arrays
        """
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self._prepare_frame, frames))

    def encode_video(
        self,
        frames: List[Image.Image],
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

            # Prepare all frames in parallel
            prepared_frames = self._prepare_frames_batch(frames)

            # Set up video writer with optimized parameters
            fourcc = cv2.VideoWriter_fourcc(*self.video_config.codec)
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.video_config.fps,
                (self.width, self.height),
                isColor=True
            )

            # Write frames
            for frame in prepared_frames:
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
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise VideoProcessingError(f"Failed to open video file: {video_path}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert OpenCV frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(frame_rgb)

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
            if frame.mode == '1':  # Binary image
                frame = frame.convert('RGB')
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

            # Detect QR codes
            detector = cv2.QRCodeDetector()
            retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(frame_cv)

            # Keep for debugging
            _ = (points, straight_qrcode)

            if not retval:
                return []

            return [info for info in decoded_info if info]

        except Exception as e:
            raise QRCodeError(f"Failed to extract QR codes: {str(e)}")
