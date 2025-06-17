"""
Optimized utility functions for Memvid.

This module provides core utility functions for QR code generation, video frame processing,
and text chunking operations used in the Memvid application.
"""

import base64
import gzip
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import qrcode
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def encode_to_qr(
    data: str | bytes,
    version: int = 35,
    error_correction: str = "M",
    box_size: int = 5,
    border: int = 3,
) -> Image.Image:
    """Encode data into a QR code image with automatic compression for large data.

    Args:
        data: The string or bytes data to encode into the QR code
        version: QR code version (1-40). Higher versions support more data
        error_correction: Error correction level ('L', 'M', 'Q', 'H')
        box_size: Size of each QR code box in pixels
        border: Width of the border around the QR code

    Returns:
        PIL Image object containing the generated QR code
    """
    # Compress data if it's large
    if len(data) > 100:
        compressed = gzip.compress(data.encode() if isinstance(data, str) else data)
        data = "GZ:" + base64.b64encode(compressed).decode()

    qr = qrcode.QRCode(
        version=version,
        error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{error_correction}"),
        box_size=box_size,
        border=border,
    )

    qr.add_data(data.decode() if isinstance(data, bytes) else data)
    qr.make(fit=True)

    return qr.make_image(fill_color="black", back_color="white")


def _decode_qr(image: np.ndarray) -> Optional[str]:
    """Internal function to decode QR code from an image array."""
    try:
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(image)
        if data:
            if data.startswith("GZ:"):
                return gzip.decompress(base64.b64decode(data[3:])).decode()
            return data
    except Exception as e:
        logger.debug(f"QR decode failed: {e}")
    return None


def _extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """Internal function to extract a single frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


@lru_cache(maxsize=1000)
def extract_and_decode_cached(video_path: str, frame_number: int) -> Optional[str]:
    """Extract and decode a single frame with caching.

    Args:
        video_path: Path to the video file
        frame_number: Zero-based index of the frame to extract

    Returns:
        Decoded string data if successful, None if decoding fails
    """
    frame = _extract_frame(video_path, frame_number)
    return _decode_qr(frame) if frame is not None else None


def _process_frame_batch(args: Tuple[str, List[int], int, int]) -> List[Tuple[int, Optional[str]]]:
    """Internal function to process a batch of frames."""
    video_path, frame_nums, start_idx, batch_size = args
    results = []
    cap = cv2.VideoCapture(video_path)

    try:
        for frame_num in frame_nums[start_idx:start_idx + batch_size]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                data = _decode_qr(frame)
                results.append((frame_num, data))
            else:
                results.append((frame_num, None))
    finally:
        cap.release()

    return results


def batch_extract_and_decode(
    video_path: str,
    frame_numbers: List[int],
    max_workers: int = 4,
    show_progress: bool = False,
    batch_size: int = 50,
) -> Dict[int, str]:
    """Extract and decode multiple QR code frames from a video efficiently.

    Args:
        video_path: Path to the video file
        frame_numbers: List of frame indices to process
        max_workers: Number of parallel workers for processing frames
        show_progress: Whether to show a progress bar
        batch_size: Number of frames to process in each batch

    Returns:
        Dictionary mapping frame numbers to their decoded data
    """
    result = {}
    sorted_frames = sorted(frame_numbers)
    total_frames = len(sorted_frames)

    # Create batches for parallel processing
    batches = [(video_path, sorted_frames, i, batch_size)
               for i in range(0, total_frames, batch_size)]

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(executor.map(_process_frame_batch, batches))

        # Flatten results and show progress if requested
        if show_progress:
            futures = tqdm(futures, desc="Processing frames", total=len(batches))

        for batch_results in futures:
            for frame_num, data in batch_results:
                if data is not None:
                    result[frame_num] = data

    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks while preserving sentence boundaries.

    Args:
        text: The text to split into chunks
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks, each containing complete sentences where possible
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # Handle the last chunk
        if end >= text_len:
            chunks.append(text[start:].strip())
            break

        # Try to break at sentence boundary in the last 20% of chunk
        chunk = text[start:end]
        last_period = chunk.rfind('.')
        if last_period > chunk_size * 0.8:
            end = start + last_period + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks
