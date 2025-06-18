"""
Video codec configurations for MemVid.

This module provides optimized configurations for different video codecs.
"""

from typing import Dict, Optional, Tuple
from pydantic import BaseModel, Field


class CodecParameters(BaseModel):
    """Base configuration for video codecs."""

    video_file_type: str = Field(
        default="mp4",
        description="Output video file type/container format"
    )
    video_fps: int = Field(
        default=30,
        description="Frames per second",
        ge=1,
        le=120
    )
    frame_height: int = Field(
        default=1080,
        description="Frame height in pixels",
        ge=1
    )
    frame_width: int = Field(
        default=1920,
        description="Frame width in pixels",
        ge=1
    )
    video_crf: int = Field(
        default=23,
        description="Constant Rate Factor (0-51, lower = better quality)",
        ge=0,
        le=51
    )
    video_preset: str = Field(
        default="medium",
        description="Encoding preset (ultrafast to veryslow)",
        pattern="^(ultrafast|superfast|veryfast|faster|fast|medium|slow|slower|veryslow)$"
    )
    video_profile: str = Field(
        default="high",
        description="Video profile (baseline, main, high, etc.)"
    )
    pix_fmt: str = Field(
        default="yuv420p",
        description="Pixel format"
    )
    extra_ffmpeg_args: Optional[str] = Field(
        default=None,
        description="Additional FFmpeg arguments"
    )


# OpenCV codec configurations
OPENCV_MP4V_PARAMETERS = CodecParameters(
    video_file_type="mp4",
    video_fps=30,
    frame_height=1080,
    frame_width=1920,
    pix_fmt="yuv420p"
)

OPENCV_MJPG_PARAMETERS = CodecParameters(
    video_file_type="avi",
    video_fps=30,
    frame_height=1080,
    frame_width=1920,
    pix_fmt="yuv420p"
)

OPENCV_XVID_PARAMETERS = CodecParameters(
    video_file_type="avi",
    video_fps=30,
    frame_height=1080,
    frame_width=1920,
    pix_fmt="yuv420p"
)

OPENCV_DIVX_PARAMETERS = CodecParameters(
    video_file_type="avi",
    video_fps=30,
    frame_height=1080,
    frame_width=1920,
    pix_fmt="yuv420p"
)

OPENCV_H264_PARAMETERS = CodecParameters(
    video_file_type="mp4",
    video_fps=30,
    frame_height=1080,
    frame_width=1920,
    pix_fmt="yuv420p"
)

OPENCV_X264_PARAMETERS = CodecParameters(
    video_file_type="mp4",
    video_fps=30,
    frame_height=1080,
    frame_width=1920,
    pix_fmt="yuv420p"
)

OPENCV_HEVC_PARAMETERS = CodecParameters(
    video_file_type="mp4",
    video_fps=30,
    frame_height=1080,
    frame_width=1920,
    pix_fmt="yuv420p"
)

# FFmpeg codec configurations
MP4V_PARAMETERS = CodecParameters(
    video_file_type="mp4",
    video_fps=15,
    frame_height=256,
    frame_width=256,
    video_crf=18,  # Visually lossless
    video_preset="medium",
    video_profile="high",
    pix_fmt="yuv420p",
    extra_ffmpeg_args="-x265-params keyint=1:tune=stillimage"
)

H265_PARAMETERS = CodecParameters(
    video_file_type="mkv",
    video_fps=30,
    video_crf=28,
    frame_height=256,
    frame_width=256,
    video_preset="slower",
    video_profile="mainstillpicture",
    pix_fmt="yuv420p",
    extra_ffmpeg_args=(
        "-x265-params keyint=1:tune=stillimage:no-scenecut:"
        "strong-intra-smoothing:constrained-intra:rect:amp"
    )
)

H264_PARAMETERS = CodecParameters(
    video_file_type="mkv",
    video_fps=30,
    video_crf=28,
    frame_height=256,
    frame_width=256,
    video_preset="slower",
    video_profile="main",
    pix_fmt="yuv420p",
    extra_ffmpeg_args=(
        "-x265-params keyint=1:tune=stillimage:no-scenecut:"
        "strong-intra-smoothing:constrained-intra:rect:amp"
    )
)

AV1_PARAMETERS = CodecParameters(
    video_file_type="mkv",
    video_crf=28,
    video_fps=60,
    frame_height=720,
    frame_width=720,
    video_preset="slower",
    video_profile="mainstillpicture",
    pix_fmt="yuv420p",
    extra_ffmpeg_args="-x265-params keyint=1:tune=stillimage"
)

# Codec mapping with aliases
CODEC_PARAMETERS: Dict[str, CodecParameters] = {
    # OpenCV codecs
    "mp4v": OPENCV_MP4V_PARAMETERS,
    "mjpg": OPENCV_MJPG_PARAMETERS,
    "xvid": OPENCV_XVID_PARAMETERS,
    "divx": OPENCV_DIVX_PARAMETERS,
    "h264": OPENCV_H264_PARAMETERS,
    "x264": OPENCV_X264_PARAMETERS,
    "hevc": OPENCV_HEVC_PARAMETERS,

    # FFmpeg codecs
    "libx264": H264_PARAMETERS,
    "libx265": H265_PARAMETERS,
    "libaom-av1": AV1_PARAMETERS,
    "h265": H265_PARAMETERS,
    "av1": AV1_PARAMETERS
}

# Default codec parameters
DEFAULT_PARAMETERS = CodecParameters()


def get_codec_parameters(codec: str) -> Tuple[CodecParameters, bool]:
    """
    Get codec parameters for a given codec.

    Args:
        codec: The codec to get parameters for

    Returns:
        Tuple containing the codec parameters and a boolean indicating if the codec is supported
    """
    codec = codec.lower()
    return CODEC_PARAMETERS.get(codec, DEFAULT_PARAMETERS), codec in CODEC_PARAMETERS
