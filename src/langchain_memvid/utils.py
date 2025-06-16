"""
Shared utility functions for Memvid.

This module provides utility functions for QR code generation, video frame processing,
and text chunking operations used in the Memvid application.

Original source: https://github.com/Olow304/memvid/blob/main/memvid/utils.py
"""

import orjson as json
import qrcode
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import logging
from tqdm import tqdm
import base64
import gzip

logger = logging.getLogger(__name__)


def encode_to_qr(
    data: str,
    version: int = 35,
    error_correction: str = "M",
    box_size: int = 5,
    border: int = 3,
    fill_color: str = "black",
    back_color: str = "white",
) -> Image.Image:
    """Encode data into a QR code image with optional compression for large data.

    Args:
        data (str): The string data to encode into the QR code.
        version (int, optional): QR code version (1-40). Higher versions support more data.
            Defaults to 35.
        error_correction (str, optional): Error correction level ('L', 'M', 'Q', 'H').
            Defaults to "M".
        box_size (int, optional): Size of each QR code box in pixels. Defaults to 5.
        border (int, optional): Width of the border around the QR code. Defaults to 3.
        fill_color (str, optional): Color of the QR code. Defaults to "black".
        back_color (str, optional): Background color. Defaults to "white".

    Returns:
        Image.Image: A PIL Image object containing the generated QR code.

    Note:
        For data longer than 100 characters, automatic compression is applied using gzip
        and base64 encoding. The compressed data is prefixed with "GZ:" to indicate
        compression.
    """

    # Compress data if it's large
    if len(data) > 100:
        compressed = gzip.compress(data.encode())
        data = base64.b64encode(compressed).decode()
        data = "GZ:" + data  # Prefix to indicate compression

    qr = qrcode.QRCode(
        version=version,
        error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{error_correction}"),
        box_size=box_size,
        border=border,
    )

    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color=fill_color, back_color=back_color)
    return img


def decode_qr(image: np.ndarray) -> Optional[str]:
    """Decode QR code from an image array.

    Args:
        image (np.ndarray): OpenCV image array (BGR format) containing the QR code.

    Returns:
        Optional[str]: The decoded string data if successful, None if decoding fails.

    Note:
        Handles both regular and compressed QR codes (prefixed with "GZ:").
        Compressed data is automatically decompressed using gzip.
    """
    try:
        # Initialize OpenCV QR code detector
        detector = cv2.QRCodeDetector()

        # Detect and decode
        data, bbox, straight_qrcode = detector.detectAndDecode(image)

        if data:
            # Check if data was compressed
            if data.startswith("GZ:"):
                compressed_data = base64.b64decode(data[3:])
                data = gzip.decompress(compressed_data).decode()

            return data

    except Exception as e:
        logger.warning(f"QR decode failed: {e}")
    return None


def qr_to_frame(qr_image: Image.Image, frame_size: Tuple[int, int]) -> np.ndarray:
    """Convert a QR code PIL Image to a video frame array.

    Args:
        qr_image (Image.Image): PIL Image containing the QR code.
        frame_size (Tuple[int, int]): Target frame dimensions (width, height).

    Returns:
        np.ndarray: OpenCV-compatible frame array in BGR format.

    Note:
        The QR code is resized to fit the target frame size while maintaining
        aspect ratio using Lanczos resampling.
    """
    # Resize to fit frame while maintaining aspect ratio
    qr_image = qr_image.resize(frame_size, Image.Resampling.LANCZOS)

    # Convert to RGB mode if necessary (handles L, P, etc. modes)
    if qr_image.mode != 'RGB':
        qr_image = qr_image.convert('RGB')

    # Convert to numpy array and ensure proper dtype
    img_array = np.array(qr_image, dtype=np.uint8)

    # Convert to OpenCV format
    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return frame


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """Extract a single frame from a video file.

    Args:
        video_path (str): Path to the video file.
        frame_number (int): Zero-based index of the frame to extract.

    Returns:
        Optional[np.ndarray]: OpenCV frame array if successful, None if frame
            extraction fails or frame number is out of range.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            return frame
    finally:
        cap.release()
    return None


@lru_cache(maxsize=1000)
def extract_and_decode_cached(video_path: str, frame_number: int) -> Optional[str]:
    """
    Extract and decode frame with caching
    """
    frame = extract_frame(video_path, frame_number)
    if frame is not None:
        return decode_qr(frame)
    return None


def batch_extract_frames(
    video_path: str,
    frame_numbers: List[int],
    max_workers: int = 4,
    show_progress: bool = False
) -> List[Tuple[int, Optional[np.ndarray]]]:
    """Extract multiple frames from a video file efficiently.

    Args:
        video_path (str): Path to the video file.
        frame_numbers (List[int]): List of frame indices to extract.
        max_workers (int, optional): Number of parallel workers. Defaults to 4.
        show_progress (bool, optional): Whether to display a progress bar.
            Defaults to False.

    Returns:
        List[Tuple[int, Optional[np.ndarray]]]: List of tuples containing
            (frame_number, frame_array) pairs. Frame array is None if extraction fails.
    """
    results = []

    # Sort frame numbers for sequential access
    sorted_frames = sorted(frame_numbers)

    cap = cv2.VideoCapture(video_path)
    try:
        for frame_num in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            results.append((frame_num, frame if ret else None))
    finally:
        cap.release()

    return results


def parallel_decode_qr(
    frames: List[Tuple[int, np.ndarray]],
    max_workers: int = 4,
    show_progress: bool = False
) -> List[Tuple[int, Optional[str]]]:
    """Decode multiple QR code frames in parallel.

    Args:
        frames (List[Tuple[int, np.ndarray]]): List of (frame_number, frame_array) pairs.
        max_workers (int, optional): Number of parallel workers. Defaults to 4.
        show_progress (bool, optional): Whether to display a progress bar.
            Defaults to False.

    Returns:
        List[Tuple[int, Optional[str]]]: List of (frame_number, decoded_data) pairs.
            decoded_data is None if decoding fails.
    """
    def decode_frame(item):
        frame_num, frame = item
        if frame is not None:
            data = decode_qr(frame)
            return (frame_num, data)
        return (frame_num, None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(decode_frame, frames))

    return results


def batch_extract_and_decode(
    video_path: str,
    frame_numbers: List[int],
    max_workers: int = 4,
    show_progress: bool = False
) -> Dict[int, str]:
    """Extract and decode multiple QR code frames from a video efficiently.

    Args:
        video_path (str): Path to the video file.
        frame_numbers (List[int]): List of frame indices to process.
        max_workers (int, optional): Number of parallel workers. Defaults to 4.
        show_progress (bool, optional): Whether to display a progress bar.
            Defaults to False.

    Returns:
        Dict[int, str]: Dictionary mapping frame numbers to their decoded data.
            Only includes frames that were successfully decoded.
    """
    # Extract frames
    frames = batch_extract_frames(video_path, frame_numbers)

    # Decode in parallel
    if show_progress:
        frames = tqdm(frames, desc="Decoding QR frames")

    decoded = parallel_decode_qr(frames, max_workers)

    # Build result dict
    result = {}
    for frame_num, data in decoded:
        if data is not None:
            result[frame_num] = data

    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks while preserving sentence boundaries.

    Args:
        text (str): The text to split into chunks.
        chunk_size (int, optional): Target size of each chunk in characters.
            Defaults to 500.
        overlap (int, optional): Number of characters to overlap between chunks.
            Defaults to 50.

    Returns:
        List[str]: List of text chunks, each containing complete sentences where possible.

    Note:
        Attempts to break chunks at sentence boundaries (periods) when possible,
        but only if the break point is within the last 20% of the chunk size.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.8:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def save_index(index_data: Dict[str, Any], output_path: str) -> None:
    """Save index data to a JSON file.

    Args:
        index_data (Dict[str, Any]): The index data to save.
        output_path (str): Path where the JSON file will be saved.

    Note:
        Uses orjson for efficient JSON serialization with proper indentation.
    """
    with open(output_path, 'w') as f:
        json.dump(index_data, f, indent=2)


def load_index(index_path: str) -> Dict[str, Any]:
    """Load index data from a JSON file.

    Args:
        index_path (str): Path to the JSON file containing the index data.

    Returns:
        Dict[str, Any]: The loaded index data.

    Note:
        Uses orjson for efficient JSON deserialization.
    """
    with open(index_path, 'r') as f:
        return json.load(f)
