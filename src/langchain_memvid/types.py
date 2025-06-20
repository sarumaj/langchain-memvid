"""
Type definitions for LangChain MemVid.

This module contains dataclasses for structured return types used across the codebase.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class FrameMappingStats:
    """Statistics about frame mappings for monitoring and optimization."""
    total_documents: int
    mapped_documents: int
    mapping_coverage: float
    mapping_efficiency: Dict[str, Any]


@dataclass
class VideoInfo:
    """Information about a video file."""
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float
    file_size_mb: float


@dataclass
class StorageStats:
    """Statistics for the hybrid storage approach."""
    total_documents: int
    video_file_size_mb: float
    index_size_mb: float
    essential_metadata_size_mb: float
    full_metadata_size_mb: float
    redundancy_percentage: float
    storage_efficiency: str
    frame_mapping_stats: FrameMappingStats


@dataclass
class BuildStats:
    """Statistics for the video build process."""
    total_chunks: int
    video_size_mb: float
    encoding_time: float
    index_path: Path
    video_path: Path
