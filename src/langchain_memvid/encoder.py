"""
Encoder - Unified encoding with native OpenCV and FFmpeg (Docker/native) support

Original source: https://github.com/Olow304/memvid/blob/main/memvid/encoder.py
"""

import orjson as json
import logging
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Generator
from tqdm import tqdm
import cv2
import numpy as np
from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import time

from .utils import encode_to_qr, chunk_text
from .config import CODEC_PARAMETERS
from .index import IndexManager

logger = logging.getLogger(__name__)


class EncoderConfig(BaseModel):
    """Configuration for Encoder."""
    # Chunking settings
    chunk_size: int = Field(default=1024, description="Chunk size for text encoding")
    overlap: int = Field(default=32, description="Overlap between chunks")

    # Video settings
    codec: str = Field(default="h265", description="Video codec")
    video_fps: int = Field(default=30, description="Video frames per second")
    frame_width: int = Field(default=1280, description="Video frame width")
    frame_height: int = Field(default=720, description="Video frame height")

    # Encoding settings
    show_progress: bool = Field(default=True, description="Show progress bars during encoding")
    allow_fallback: bool = Field(default=True, description="Allow fallback to MP4V if advanced codec fails")

    # Performance settings
    max_workers: int = Field(default=4, description="Maximum number of parallel workers")
    batch_size: int = Field(default=100, description="Batch size for processing chunks")
    qr_cache_size: int = Field(default=1000, description="Maximum number of QR codes to cache")
    frame_cache_size: int = Field(default=100, description="Maximum number of frames to cache in memory")


@dataclass
class EncodingStats:
    """Statistics about the encoding process."""
    backend: str
    codec: str
    total_frames: int
    video_size_mb: float
    fps: int
    duration_seconds: float
    total_chunks: int
    video_file: str
    index_file: str
    index_stats: Dict[str, Any]
    encoding_time: float

    def dict(self) -> Dict[str, Any]:
        """Convert EncodingStats to a dictionary."""
        return asdict(self)


@dataclass
class EncoderStats:
    """Statistics about the encoder state."""
    total_chunks: int
    total_characters: int
    avg_chunk_size: float
    supported_codecs: List[str]
    config: Dict[str, Any]
    cache_stats: Dict[str, Any]


class Encoder:
    """
    Unified Encoder with clean separation between encoding logic and Docker management.
    Supports both native OpenCV encoding and FFmpeg encoding (native or Docker-based).

    This class handles the creation and management of video encoding from text chunks,
    with support for multiple codecs and both native and Docker-based encoding.

    Attributes:
        config: Encoder configuration
        chunks: List of text chunks to encode
        index_manager: Index manager for storing and retrieving chunks
        dcker_mngr: Optional Docker manager for containerized encoding
    """

    def __init__(self, config: EncoderConfig, index_manager: IndexManager):
        """
        Initialize Encoder.

        Args:
            config: Configuration for the encoder
            index_manager: Index manager for storing and retrieving chunks

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.chunks = []
        self.index_manager = index_manager

        # Initialize caches
        self._qr_cache: Dict[str, Any] = {}
        self._frame_cache: Dict[int, np.ndarray] = {}

    def add_chunks(self, chunks: List[str]):
        """
        Add text chunks to be encoded

        Args:
            chunks: List of text chunks
        """
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")

    def add_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        """
        Add text and automatically chunk it

        Args:
            text: Text to chunk and add
            chunk_size: Target chunk size (defaults to config.chunk_size)
            overlap: Overlap between chunks (defaults to config.overlap)
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.overlap

        chunks = chunk_text(text, chunk_size, overlap)
        self.add_chunks(chunks)

    def add_pdf(self, pdf_path: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        """
        Extract text from PDF and add as chunks

        Args:
            pdf_path: Path to PDF file
            chunk_size: Target chunk size (defaults to config.chunk_size)
            overlap: Overlap between chunks (defaults to config.overlap)
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDF support. Install with: pip install pypdf")

        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.overlap

        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            logger.info(f"Extracting text from {num_pages} pages of {Path(pdf_path).name}")

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n\n"

        if text.strip():
            self.add_text(text, chunk_size, overlap)
            logger.info(f"Added PDF content: {len(text)} characters from {Path(pdf_path).name}")
        else:
            logger.warning(f"No text extracted from PDF: {pdf_path}")

    def add_epub(self, epub_path: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        """
        Extract text from EPUB and add as chunks

        Args:
            epub_path: Path to EPUB file
            chunk_size: Target chunk size (defaults to config.chunk_size)
            overlap: Overlap between chunks (defaults to config.overlap)
        """
        try:
            from ebooklib import epub, ITEM_DOCUMENT as ebooklib_ITEM_DOCUMENT
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "ebooklib and beautifulsoup4 are required for EPUB support. "
                "Install with: pip install ebooklib beautifulsoup4"
            )

        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.overlap

        if not Path(epub_path).exists():
            raise FileNotFoundError(f"EPUB file not found: {epub_path}")

        try:
            book = epub.read_epub(epub_path)
            text_content = []

            logger.info(f"Extracting text from EPUB: {Path(epub_path).name}")

            # Extract text from all document items
            for item in book.get_items():
                if item.get_type() == ebooklib_ITEM_DOCUMENT:
                    # Parse HTML content
                    soup = BeautifulSoup(item.get_content(), 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Get text and clean it up
                    text = soup.get_text()

                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)

                    if text.strip():
                        text_content.append(text)

            # Combine all text
            full_text = "\n\n".join(text_content)

            if full_text.strip():
                self.add_text(full_text, chunk_size, overlap)
                logger.info(f"Added EPUB content: {len(full_text)} characters from {Path(epub_path).name}")
            else:
                logger.warning(f"No text extracted from EPUB: {epub_path}")

        except Exception as e:
            logger.error(f"Error processing EPUB {epub_path}: {e}")
            raise

    def create_video_writer(self, output_path: str, codec: Optional[str] = None) -> cv2.VideoWriter:
        """
        Create OpenCV video writer for native encoding

        Args:
            output_path: Path to output video file
            codec: Video codec for OpenCV

        Returns:
            cv2.VideoWriter instance
        """
        if codec is None:
            codec = self.config.codec

        if codec not in CODEC_PARAMETERS.keys():
            raise ValueError(f"Unsupported codec: {codec}")

        codec_config = CODEC_PARAMETERS[codec]

        # OpenCV codec mapping
        opencv_codec_map = {
            "mp4v": "mp4v",
            "xvid": "XVID",
            "mjpg": "MJPG"
        }

        opencv_codec = opencv_codec_map.get(codec, codec)
        fourcc = cv2.VideoWriter_fourcc(*opencv_codec)

        return cv2.VideoWriter(
            output_path,
            fourcc,
            codec_config["video_fps"],
            (codec_config["frame_width"], codec_config["frame_height"])
        )

    def _batch_iterator(self, items: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
        """Generate batches of items."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def _generate_qr_batch(self, chunks: List[Tuple[int, str]], show_progress: bool = True) -> List[Tuple[int, Any]]:
        """
        Generate QR codes for a batch of chunks in parallel.

        Args:
            chunks: List of (frame_num, chunk) tuples
            show_progress: Show progress bar

        Returns:
            List of (frame_num, QR image) tuples
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Create futures for uncached chunks
            futures = {
                frame_num: executor.submit(self._generate_single_qr, frame_num, chunk)
                for frame_num, chunk in chunks
                if f"{frame_num}:{chunk}" not in self._qr_cache
            }

            # Add cached results
            results.extend(
                (frame_num, self._qr_cache[f"{frame_num}:{chunk}"])
                for frame_num, chunk in chunks
                if f"{frame_num}:{chunk}" in self._qr_cache
            )

            # Process futures
            if show_progress:
                futures_iter = tqdm(futures.items(), total=len(futures), desc="Generating QR codes")
            else:
                futures_iter = futures.items()

            for frame_num, future in futures_iter:
                try:
                    qr_image = future.result()
                    cache_key = f"{frame_num}:{chunks[frame_num][1]}"
                    self._qr_cache[cache_key] = qr_image
                    results.append((frame_num, qr_image))
                except Exception as e:
                    logger.error(f"Failed to generate QR code for frame {frame_num}: {e}")

        return results

    def _generate_single_qr(self, frame_num: int, chunk: str) -> Any:
        """Generate single QR code with error handling."""
        try:
            chunk_data = {"id": frame_num, "text": chunk, "frame": frame_num}
            return encode_to_qr(json.dumps(chunk_data))
        except Exception as e:
            logger.error(f"Error generating QR code for frame {frame_num}: {e}")
            raise

    def _generate_qr_frames(self, temp_dir: Path, show_progress: bool = True) -> Path:
        """
        Generate QR code frames to temporary directory

        Args:
            temp_dir: Temporary directory for frame storage
            show_progress: Show progress bar

        Returns:
            Path to frames directory
        """
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()

        # Prepare chunks for batch processing
        chunks = list(enumerate(self.chunks))
        total_batches = (len(chunks) + self.config.batch_size - 1) // self.config.batch_size

        if show_progress:
            logger.info(f"Generating {len(chunks)} QR frames in {total_batches} batches")

        # Process in batches
        for batch in self._batch_iterator(chunks, self.config.batch_size):
            qr_results = self._generate_qr_batch(batch, show_progress)

            # Save frames
            for frame_num, qr_image in qr_results:
                frame_path = frames_dir / f"frame_{frame_num:06d}.png"
                qr_image.save(frame_path)

        created_frames = list(frames_dir.glob("frame_*.png"))
        logger.info(f"Generated {len(created_frames)} QR frames in {frames_dir}")
        return frames_dir

    def _build_ffmpeg_command(self, frames_dir: Path, output_file: Path, codec: str) -> List[str]:
        """Build optimized FFmpeg command using codec configuration"""

        # Get codec-specific configuration
        if codec not in CODEC_PARAMETERS:
            raise ValueError(f"Unsupported codec: {codec}")
        codec_config = CODEC_PARAMETERS[codec]

        # FFmpeg codec mapping
        ffmpeg_codec_map = {
            "h265": "libx265", "hevc": "libx265",
            "h264": "libx264", "avc": "libx264",
            "av1": "libaom-av1", "vp9": "libvpx-vp9"
        }

        ffmpeg_codec = ffmpeg_codec_map.get(codec, codec)

        # Ensure output file has correct extension
        expected_ext = codec_config["video_file_type"]
        if not str(output_file).endswith(expected_ext):
            output_file = output_file.with_suffix(expected_ext)

        # Build base command using config
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(codec_config["video_fps"]),
            '-i', str(frames_dir / 'frame_%06d.png'),
            '-c:v', ffmpeg_codec,
            '-preset', codec_config["video_preset"],
            '-crf', str(codec_config["video_crf"]),
        ]

        # Apply scaling and pixel format based on codec
        if ffmpeg_codec in ['libx265', 'libx264']:
            # Scale to config dimensions for advanced codecs
            target_width = codec_config["frame_width"]
            target_height = codec_config["frame_height"]
            cmd.extend(['-vf', f'scale={target_width}:{target_height}'])
            cmd.extend(['-pix_fmt', codec_config["pix_fmt"]])

            # Add profile if specified in config
            if codec_config.get("video_profile"):
                cmd.extend(['-profile:v', codec_config["video_profile"]])
        else:
            # Use pixel format from config for other codecs
            cmd.extend(['-pix_fmt', codec_config["pix_fmt"]])

        # Threading (limit to 16 max)
        import os
        thread_count = min(os.cpu_count() or 4, 16)
        cmd.extend(['-threads', str(thread_count)])

        print("🎬 FFMPEG ENCODING SUMMARY:")
        print("   🎥 Codec Config:")
        print(f"      • codec: {codec}")
        print(f"      • file_type: {codec_config.get('video_file_type', 'unknown')}")
        print(f"      • fps: {codec_config.get('fps', 'default')}")
        print(f"      • crf: {codec_config.get('crf', 'default')}")
        print(f"      • height: {codec_config.get('frame_height', 'default')}")
        print(f"      • width: {codec_config.get('frame_width', 'default')}")
        print(f"      • preset: {codec_config.get('video_preset', 'default')}")
        print(f"      • pix_fmt: {codec_config.get('pix_fmt', 'default')}")
        print(f"      • extra_ffmpeg_args: {codec_config.get('extra_ffmpeg_args', 'default')}")

        # Add codec-specific parameters from config
        if codec_config.get("extra_ffmpeg_args"):
            extra_args = codec_config["extra_ffmpeg_args"]
            if isinstance(extra_args, str):
                # Parse string args and add thread count for x264/x265
                if ffmpeg_codec == 'libx265':
                    extra_args = f"{extra_args}:threads={thread_count}"
                    cmd.extend(['-x265-params', extra_args])
                elif ffmpeg_codec == 'libx264':
                    extra_args = f"{extra_args}:threads={thread_count}"
                    cmd.extend(['-x264-params', extra_args])
            else:
                # Direct args list
                cmd.extend(extra_args)

        # General optimizations
        cmd.extend(['-movflags', '+faststart', '-avoid_negative_ts', 'make_zero'])

        cmd.append(str(output_file))
        return cmd

    def _encode_with_opencv(self, frames_dir: Path, output_file: Path, codec: str,
                            show_progress: bool = True) -> Dict[str, Any]:
        """
        Encode video using native OpenCV with optimized frame loading and caching.

        Args:
            frames_dir: Directory containing PNG frames
            output_file: Output video file path
            codec: Video codec
            show_progress: Show progress bar

        Returns:
            Encoding statistics
        """
        if codec not in CODEC_PARAMETERS:
            raise ValueError(f"Unsupported codec: {codec}")

        codec_config = CODEC_PARAMETERS[codec]
        start_time = time.time()

        if show_progress:
            logger.info(f"Encoding with OpenCV using {codec} codec...")

        # Create video writer
        writer = self.create_video_writer(str(output_file), codec)
        frame_numbers = []

        try:
            # Load and write frames
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            frame_iter = enumerate(frame_files)

            if show_progress:
                frame_iter = tqdm(frame_iter, total=len(frame_files), desc="Writing video frames")

            # Process frames in batches
            for batch_files in self._batch_iterator(frame_files, self.config.batch_size):
                # Load batch of frames
                frames = []
                for frame_file in batch_files:
                    frame_num = int(frame_file.stem.split('_')[1])

                    # Check cache
                    if frame_num in self._frame_cache:
                        frame = self._frame_cache[frame_num]
                    else:
                        frame = cv2.imread(str(frame_file))
                        if frame is not None and len(self._frame_cache) < self.config.frame_cache_size:
                            self._frame_cache[frame_num] = frame

                    if frame is not None:
                        # Resize if needed
                        target_size = (codec_config["frame_width"], codec_config["frame_height"])
                        if frame.shape[:2][::-1] != target_size:
                            frame = cv2.resize(frame, target_size)
                        frames.append((frame_num, frame))

                # Write frames
                for frame_num, frame in frames:
                    writer.write(frame)
                    frame_numbers.append(frame_num)

            encoding_time = time.time() - start_time
            return {
                "backend": "opencv",
                "codec": codec,
                "total_frames": len(frame_numbers),
                "video_size_mb": output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0,
                "fps": codec_config["video_fps"],
                "duration_seconds": len(frame_numbers) / codec_config["video_fps"],
                "encoding_time": encoding_time
            }

        finally:
            writer.release()
            self._frame_cache.clear()  # Clear frame cache after encoding

    def _encode_with_ffmpeg(
        self, frames_dir: Path,
        output_file: Path,
        codec: str,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Encode video using FFmpeg (native or Docker)

        Args:
            frames_dir: Directory containing PNG frames
            output_file: Output video file path
            codec: Video codec
            show_progress: Show progress bar

        Returns:
            Encoding statistics
        """
        cmd = self._build_ffmpeg_command(frames_dir, output_file, codec)

        if show_progress:
            logger.info(f"Encoding with native FFmpeg using {codec} codec...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Native FFmpeg failed: {result.stderr}")

        frame_count = len(list(frames_dir.glob("frame_*.png")))
        return {
            "backend": "native_ffmpeg",
            "codec": codec,
            "total_frames": frame_count,
            "video_size_mb": output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0,
            "fps": CODEC_PARAMETERS[codec]["video_fps"],
            "duration_seconds": frame_count / CODEC_PARAMETERS[codec]["video_fps"]
        }

    def build_video(
        self,
        output_file: str,
        index_file: str,
        codec: Optional[str] = None,
        show_progress: Optional[bool] = None,
        allow_fallback: Optional[bool] = None
    ) -> EncodingStats:
        """
        Build QR code video and index from chunks with unified codec handling.

        Args:
            output_file: Path to output video file
            index_file: Path to output index file
            codec: Video codec ('mp4v', 'h265', 'h264', etc.)
            show_progress: Show progress bar (defaults to config.show_progress)
            allow_fallback: Whether to fall back to MP4V if advanced codec fails (defaults to config.allow_fallback)

        Returns:
            EncodingStats: Statistics about the encoding process

        Raises:
            ValueError: If no chunks to encode
            RuntimeError: If encoding fails and fallback is not allowed
        """
        if not self.chunks:
            raise ValueError("No chunks to encode. Use add_chunks() first.")

        output_path = Path(output_file)
        index_path = Path(index_file)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Use config values if not specified
        codec = codec or self.config.codec
        show_progress = show_progress if show_progress is not None else self.config.show_progress
        allow_fallback = allow_fallback if allow_fallback is not None else self.config.allow_fallback

        logger.info(f"Building video with {len(self.chunks)} chunks using {codec} codec")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate QR frames (always local)
            frames_dir = self._generate_qr_frames(temp_path, show_progress)

            try:
                # Choose encoding method based on codec
                match codec:
                    case "mp4v":
                        # Always use OpenCV for MP4V
                        stats = self._encode_with_opencv(
                            frames_dir,
                            output_path,
                            codec,
                            show_progress
                        )
                    case _:
                        # Use FFmpeg for advanced codecs
                        stats = self._encode_with_ffmpeg(
                            frames_dir,
                            output_path,
                            codec,
                            show_progress,
                        )

            except Exception as e:
                if allow_fallback and codec != "mp4v":
                    warnings.warn(f"{codec} encoding failed: {e}. Falling back to MP4V.", UserWarning)
                    stats = self._encode_with_opencv(frames_dir, output_path, "mp4v", show_progress)
                else:
                    raise RuntimeError(f"Encoding failed: {e}") from e

            # Build search index
            if show_progress:
                logger.info("Building search index...")

            frame_numbers = list(range(len(self.chunks)))
            self.index_manager.add_chunks(self.chunks, frame_numbers, show_progress)

            # Save index
            self.index_manager.save(str(index_path.with_suffix('')))

            # Create stats object
            return EncodingStats(
                backend=stats["backend"],
                codec=stats["codec"],
                total_frames=stats["total_frames"],
                video_size_mb=stats["video_size_mb"],
                fps=stats["fps"],
                duration_seconds=stats["duration_seconds"],
                total_chunks=len(self.chunks),
                video_file=str(output_path),
                index_file=str(index_path),
                index_stats=self.index_manager.get_stats().dict(),
                encoding_time=stats.get("encoding_time", 0)
            )

    def clear(self) -> None:
        """
        Clear all chunks and reset the index manager.

        This method resets the encoder to its initial state by:
        1. Clearing all stored chunks
        2. Resetting the index manager
        """
        self.chunks = []
        self.index_manager = IndexManager(
            config=self.index_manager.config,
            embeddings=self.index_manager.embedding_model
        )
        logger.info("Cleared all chunks")

    def get_stats(self) -> EncoderStats:
        """
        Get encoder statistics.

        Returns:
            EncoderStats: Statistics about the encoder state including:
                - total_chunks: Number of chunks
                - total_characters: Total character count
                - avg_chunk_size: Average chunk size
                - supported_codecs: List of supported codecs
                - config: Current configuration
                - cache_stats: Cache statistics
        """
        return EncoderStats(
            total_chunks=len(self.chunks),
            total_characters=sum(len(chunk) for chunk in self.chunks),
            avg_chunk_size=np.mean([len(chunk) for chunk in self.chunks]) if self.chunks else 0,
            supported_codecs=list(CODEC_PARAMETERS.keys()),
            config=self.config.model_dump(),
            cache_stats={
                "qr_cache_size": len(self._qr_cache),
                "frame_cache_size": len(self._frame_cache),
                "qr_cache_limit": self.config.qr_cache_size,
                "frame_cache_limit": self.config.frame_cache_size
            }
        )

    @classmethod
    def from_file(
        cls,
        file_path: str,
        config: Optional[EncoderConfig] = None,
        index_manager: Optional[IndexManager] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> 'Encoder':
        """
        Create encoder from text file.

        Args:
            file_path: Path to text file
            config: Optional configuration. If not provided, uses default config
            index_manager: Optional index manager. If not provided, raises ValueError
            chunk_size: Optional chunk size override
            overlap: Optional overlap override

        Returns:
            Encoder: Instance with chunks loaded from file

        Raises:
            ValueError: If index_manager is not provided
            FileNotFoundError: If file_path does not exist
        """
        if config is None:
            config = EncoderConfig()
        if index_manager is None:
            raise ValueError("index_manager parameter is required")

        encoder = cls(config=config, index_manager=index_manager)

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        encoder.add_text(text, chunk_size, overlap)
        return encoder

    @classmethod
    def from_documents(
        cls,
        documents: List[str],
        config: Optional[EncoderConfig] = None,
        index_manager: Optional[IndexManager] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> 'Encoder':
        """
        Create encoder from list of documents.

        Args:
            documents: List of document strings to encode
            config: Optional configuration. If not provided, uses default config
            index_manager: Optional index manager. If not provided, raises ValueError
            chunk_size: Optional chunk size override
            overlap: Optional overlap override

        Returns:
            Encoder: Instance with chunks loaded from documents

        Raises:
            ValueError: If index_manager is not provided
        """
        if config is None:
            config = EncoderConfig()
        if index_manager is None:
            raise ValueError("index_manager parameter is required")

        encoder = cls(config=config, index_manager=index_manager)

        for doc in documents:
            encoder.add_text(doc, chunk_size, overlap)

        return encoder
