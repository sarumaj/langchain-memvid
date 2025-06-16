"""
Configuration for Memvid.

Original source: https://github.com/Olow304/memvid/blob/main/memvid/config.py
"""

from typing import TypedDict, Optional, Dict, Any


class VideoCodecConfig(TypedDict):
    """Video codec configuration parameters."""
    video_fps: int
    frame_width: int
    frame_height: int
    video_preset: str
    video_crf: int
    pix_fmt: str
    video_file_type: str
    video_profile: str
    extra_ffmpeg_args: Optional[str]


# Common video parameters
COMMON_PARAMS: Dict[str, Any] = {
    "frame_height": 256,
    "frame_width": 256,
    "pix_fmt": "yuv420p",
}

# Common x265 parameters for still image optimization
STILL_IMAGE_PARAMS: str = (
    "keyint=1:tune=stillimage:no-scenecut:strong-intra-smoothing:"
    "constrained-intra:rect:amp"
)

MP4V_PARAMETERS: VideoCodecConfig = {
    "video_file_type": "mp4",
    "video_fps": 15,
    "video_crf": 18,  # Constant Rate Factor (0-51, lower = better quality)
    "video_preset": "medium",  # ultrafast, superfast, veryfast, faster, fast, medium
    "video_profile": "high",  # baseline, main, high (baseline for max compatibility)
    "extra_ffmpeg_args": "-x265-params keyint=1:tune=stillimage",
    **COMMON_PARAMS,
}

H265_PARAMETERS: VideoCodecConfig = {
    "video_file_type": "mkv",  # AKA HEVC
    "video_fps": 30,
    "video_crf": 28,
    "video_preset": "slower",
    "video_profile": "mainstillpicture",
    "extra_ffmpeg_args": f"-x265-params {STILL_IMAGE_PARAMS}",
    **COMMON_PARAMS,
}

H264_PARAMETERS: VideoCodecConfig = {
    "video_file_type": "mkv",  # AKA AVC
    "video_fps": 30,
    "video_crf": 28,
    "video_preset": "slower",
    "video_profile": "main",
    "extra_ffmpeg_args": f"-x265-params {STILL_IMAGE_PARAMS}",
    **COMMON_PARAMS,
}

AV1_PARAMETERS: VideoCodecConfig = {
    "video_file_type": "mkv",
    "video_crf": 28,
    "video_fps": 60,
    "frame_height": 720,
    "frame_width": 720,
    "video_preset": "slower",
    "video_profile": "mainstillpicture",
    "extra_ffmpeg_args": "-x265-params keyint=1:tune=stillimage",
    "pix_fmt": "yuv420p",
}

# Video codec configuration parameters
CODEC_PARAMETERS: Dict[str, VideoCodecConfig] = {
    "mp4v": MP4V_PARAMETERS,
    "h265": H265_PARAMETERS,
    "h264": H264_PARAMETERS,
    "avc": H264_PARAMETERS,
    "av1": AV1_PARAMETERS,
    "hevc": H265_PARAMETERS
}
