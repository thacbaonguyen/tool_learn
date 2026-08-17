from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import Settings
from .tts_engine import TTSError, extract_reference_audio, save_json


SUPPORTED_AUDIO_VIDEO = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
}


def media_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TTSError("Không tìm thấy FFprobe.") from exc
    if result.returncode != 0:
        raise TTSError("Không đọc được file giọng mẫu.")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TTSError("Không xác định được thời lượng file giọng mẫu.") from exc


def find_source(voice_dir: Path) -> Path:
    candidates = [
        path
        for path in voice_dir.iterdir()
        if path.is_file()
        and path.name != "reference.wav"
        and path.suffix.lower() in SUPPORTED_AUDIO_VIDEO
    ]
    if not candidates:
        raise TTSError(
            "Không tìm thấy file mẫu trong data/voice. "
            "Hãy chép một file WAV, MP3 hoặc MP4 vào thư mục này."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> int:
    parser = argparse.ArgumentParser(description="Thiết lập giọng mẫu dùng cho mọi video")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    settings = Settings.from_env()
    voice_dir = settings.data_dir / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = find_source(voice_dir)
        total_duration = media_duration(source)
        if args.start < 0 or args.start >= total_duration:
            raise TTSError("Thời điểm bắt đầu nằm ngoài file giọng mẫu.")
        duration = min(args.duration, 30.0, total_duration - args.start)
        if duration < 3:
            raise TTSError("Giọng mẫu phải có ít nhất 3 giây audio.")
        output = voice_dir / "reference.wav"
        extract_reference_audio(
            source,
            output,
            start_seconds=args.start,
            duration_seconds=duration,
        )
        save_json(
            voice_dir / "reference.json",
            {
                "source": source.name,
                "start_seconds": round(args.start, 3),
                "duration_seconds": round(duration, 3),
                "scope": "global",
            },
        )
    except TTSError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    print(f"Giọng mẫu toàn cục đã sẵn sàng: {output}")
    print("Mọi video sẽ tự dùng giọng này khi tạo TTS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
