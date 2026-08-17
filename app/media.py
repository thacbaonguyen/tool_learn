from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class MediaProbeError(ValueError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    duration_seconds: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None


def _parse_fps(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    try:
        return round(float(Fraction(value)), 3)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: Path) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        payload = json.loads(result.stdout)
    except FileNotFoundError as exc:
        raise MediaProbeError("Không tìm thấy ffprobe. Hãy cài FFmpeg.") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise MediaProbeError("File không phải video hợp lệ hoặc không đọc được metadata.") from exc

    streams = payload.get("streams", [])
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio_stream = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not video_stream:
        raise MediaProbeError("File không có video stream.")

    raw_duration = video_stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise MediaProbeError("Không đọc được thời lượng video.") from exc

    return VideoMetadata(
        duration_seconds=round(duration, 3),
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        fps=_parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        video_codec=str(video_stream.get("codec_name") or "unknown"),
        audio_codec=str(audio_stream.get("codec_name")) if audio_stream else None,
    )

