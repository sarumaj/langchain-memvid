"""
Configuration management for LangChain MemVid.

This module provides configuration classes for different components of the LangChain MemVid system.
Each configuration class is a Pydantic model that provides validation and documentation.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from enum import Enum
from typing import Optional, Dict, Any, Tuple, Literal
from pathlib import Path


LANGCHAIN_MEMVID_DEFAULT_VIDEO_FILE = Path("kb_data.mp4")
LANGCHAIN_MEMVID_DEFAULT_INDEX_DIR = Path("kb_index.d")


class VideoBackend(str, Enum):
    """Video processing backend to use."""
    OPENCV = "opencv"
    FFMPEG = "ffmpeg"


class VideoConfig(BaseModel):
    """Configuration for video processing."""

    codec: str = Field(
        default="mp4v",
        description="Video codec to use. Backend will be automatically selected based on codec."
    )

    fps: int = Field(
        default=30,
        description="Frames per second for video encoding",
        ge=1,
        le=60
    )

    resolution: Tuple[int, int] = Field(
        default=(1920, 1080),
        description="Video resolution (width, height)"
    )

    backend: Optional[VideoBackend] = Field(
        default=None,
        description="Video processing backend to use"
    )

    ffmpeg_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional FFmpeg options"
    )

    show_progress: bool = Field(
        default=True,
        description="Whether to show progress bars during operations"
    )

    @field_validator("resolution", mode="before")
    def validate_resolution(cls, v):
        width, height = v
        if width < 640 or height < 480:     # at least SD (16:9)
            raise ValueError("Resolution too low")
        if width > 3840 or height > 2160:   # at most 4K (16:9)
            raise ValueError("Resolution too high")
        return v

    @model_validator(mode="after")
    def set_backend_from_codec(self):
        """Set backend based on codec if not explicitly set."""
        if self.backend is None:
            self.backend = (
                VideoBackend.FFMPEG
                if self.codec.lower().startswith(("lib", "h265", "av1"))
                else VideoBackend.OPENCV
            )
        return self


class QRCodeConfig(BaseModel):
    """Configuration for QR code generation."""

    error_correction: str = Field(
        default="M",
        description="Error correction level (L, M, Q, H)",
        pattern="^[LMQH]$"
    )

    box_size: int = Field(
        default=5,
        description="Size of each QR code box in pixels",
        ge=1,
        le=50
    )

    border: int = Field(
        default=3,
        description="Border size in boxes",
        ge=0,
        le=10
    )

    version: int = Field(
        default=35,
        description="Version of the QR code (the higher the version, the more data can be encoded)",
        ge=1,
        le=40
    )


class IndexConfig(BaseModel):
    """Configuration for the vector index."""

    index_type: Literal["faiss"] = Field(
        default="faiss",
        description="Type of vector index to use"
    )

    metric: str = Field(
        default="cosine",
        description="Distance metric to use",
        pattern="^(cosine|l2|ip)$"
    )

    nlist: int = Field(
        default=100,
        description="Number of clusters for FAISS index",
        ge=1
    )

    show_progress: bool = Field(
        default=True,
        description="Whether to show progress bars during operations"
    )


class VectorStoreConfig(BaseModel):
    """Unified configuration for VectorStore."""

    video: VideoConfig = Field(
        default_factory=VideoConfig,
        description="Video processing configuration"
    )

    qrcode: QRCodeConfig = Field(
        default_factory=QRCodeConfig,
        description="QR code generation configuration"
    )

    index: IndexConfig = Field(
        default_factory=IndexConfig,
        description="Vector index configuration"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        strict=False,            # Allow type coercion
        from_attributes=True     # Allow conversion from objects with attributes
    )
