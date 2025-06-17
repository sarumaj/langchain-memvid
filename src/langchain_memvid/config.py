"""
Configuration management for MemVid.

This module provides configuration classes for different components of the MemVid system.
Each configuration class is a Pydantic model that provides validation and documentation.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict


class VideoConfig(BaseModel):
    """Configuration for video processing."""

    fps: int = Field(
        default=30,
        description="Frames per second for the output video",
        ge=1,
        le=60
    )

    resolution: tuple[int, int] = Field(
        default=(1920, 1080),
        description="Video resolution (width, height)"
    )

    codec: str = Field(
        default="h264",
        description="Video codec to use"
    )

    @field_validator("resolution", mode="before")
    def validate_resolution(cls, v):
        width, height = v
        if width < 640 or height < 480:
            raise ValueError("Resolution too low")
        if width > 3840 or height > 2160:
            raise ValueError("Resolution too high")
        return v


class QRCodeConfig(BaseModel):
    """Configuration for QR code generation."""

    error_correction: str = Field(
        default="M",
        description="Error correction level (L, M, Q, H)",
        pattern="^[LMQH]$"
    )

    box_size: int = Field(
        default=10,
        description="Size of each QR code box in pixels",
        ge=1,
        le=50
    )

    border: int = Field(
        default=4,
        description="Border size in boxes",
        ge=0,
        le=10
    )


class IndexConfig(BaseModel):
    """Configuration for the vector index."""

    index_type: str = Field(
        default="faiss",
        description="Type of index to use (faiss, annoy, etc.)"
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

    model_config = ConfigDict(arbitrary_types_allowed=True)
