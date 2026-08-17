from __future__ import annotations

import shutil
import subprocess
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2
CHANNELS = 1
BLOCK_FRAMES = SAMPLE_RATE


class FinalizeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimelineSegment:
    start_ms: int
    audio_path: Path
    cache_key: str


@dataclass(frozen=True)
class FinalJobSpec:
    job_id: str
    project_id: int
    video_id: int
    video_path: Path
    output_dir: Path
    duration_seconds: float
    segments: list[TimelineSegment]


def _canonical_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as source:
            return (
                source.getframerate() == SAMPLE_RATE
                and source.getnchannels() == CHANNELS
                and source.getsampwidth() == SAMPLE_WIDTH
                and source.getcomptype() == "NONE"
            )
    except (OSError, wave.Error):
        return False


def normalize_wav(source: Path, destination: Path) -> Path:
    if _canonical_wav(source):
        return source
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.part{destination.suffix}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        str(CHANNELS),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(temporary),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()
            raise FinalizeError(
                "Không chuẩn hóa được audio TTS: "
                + (detail[-1] if detail else source.name)
            )
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def build_audio_timeline(
    segments: list[TimelineSegment],
    duration_seconds: float,
    output: Path,
    work_dir: Path,
    progress: Callable[[float], None] | None = None,
) -> None:
    if duration_seconds <= 0:
        raise FinalizeError("Thời lượng video không hợp lệ.")
    if not segments:
        raise FinalizeError("Không có audio TTS để tạo timeline.")
    work_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[tuple[int, int, Path]] = []
    unique_paths: dict[str, Path] = {}
    total_segments = len(segments)
    for index, segment in enumerate(segments, start=1):
        if not segment.audio_path.is_file():
            raise FinalizeError(f"Thiếu audio cache: {segment.cache_key}")
        normalized_path = unique_paths.get(segment.cache_key)
        if normalized_path is None:
            normalized_path = normalize_wav(
                segment.audio_path, work_dir / f"{segment.cache_key}.wav"
            )
            unique_paths[segment.cache_key] = normalized_path
        with wave.open(str(normalized_path), "rb") as source:
            frame_count = source.getnframes()
        start_frame = max(0, round(segment.start_ms * SAMPLE_RATE / 1000))
        normalized.append((start_frame, start_frame + frame_count, normalized_path))
        if progress:
            progress(0.25 * index / total_segments)

    normalized.sort(key=lambda item: item[0])
    total_frames = round(duration_seconds * SAMPLE_RATE)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.part{output.suffix}")
    active: list[tuple[int, int, Path]] = []
    next_segment = 0
    try:
        with wave.open(str(temporary), "wb") as destination:
            destination.setparams(
                (CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE, 0, "NONE", "not compressed")
            )
            for block_start in range(0, total_frames, BLOCK_FRAMES):
                block_end = min(total_frames, block_start + BLOCK_FRAMES)
                frame_count = block_end - block_start
                while (
                    next_segment < len(normalized)
                    and normalized[next_segment][0] < block_end
                ):
                    active.append(normalized[next_segment])
                    next_segment += 1
                active = [item for item in active if item[1] > block_start]
                mixed = array("i", [0]) * frame_count
                for segment_start, segment_end, path in active:
                    overlap_start = max(block_start, segment_start)
                    overlap_end = min(block_end, segment_end)
                    if overlap_end <= overlap_start:
                        continue
                    source_offset = overlap_start - segment_start
                    destination_offset = overlap_start - block_start
                    with wave.open(str(path), "rb") as source:
                        source.setpos(source_offset)
                        samples = array("h")
                        samples.frombytes(source.readframes(overlap_end - overlap_start))
                    for sample_index, sample in enumerate(samples):
                        mixed[destination_offset + sample_index] += sample
                clipped = array(
                    "h", (max(-32768, min(32767, value)) for value in mixed)
                )
                destination.writeframes(clipped.tobytes())
                if progress:
                    progress(0.25 + 0.75 * block_end / total_frames)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def mux_vietnamese_audio(
    video: Path,
    timeline: Path,
    output: Path,
    duration_seconds: float,
    progress: Callable[[float], None] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.part{output.suffix}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-i",
        str(timeline),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-t",
        f"{duration_seconds:.3f}",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        "-y",
        str(temporary),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            key, _, value = line.strip().partition("=")
            if key in {"out_time_us", "out_time_ms"} and progress:
                try:
                    elapsed = int(value) / 1_000_000
                    progress(min(1.0, elapsed / duration_seconds))
                except ValueError:
                    pass
        stderr = process.stderr.read() if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            detail = stderr.strip().splitlines()
            raise FinalizeError(
                "Ghép video/audio thất bại: "
                + (detail[-1] if detail else "FFmpeg không có chi tiết.")
            )
        temporary.replace(output)
    except FileNotFoundError as exc:
        raise FinalizeError("Không tìm thấy FFmpeg.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def execute_final_job(
    spec: FinalJobSpec,
    update: Callable[[int, str], None],
) -> None:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = spec.output_dir / f"work-{spec.job_id}"
    timeline = spec.output_dir / "vietnamese_timeline.wav"
    output = spec.output_dir / "final.mp4"
    try:
        update(5, "Đang chuẩn hóa audio từng cue")
        build_audio_timeline(
            spec.segments,
            spec.duration_seconds,
            timeline,
            work_dir,
            lambda value: update(5 + round(value * 65), "Đang tạo audio timeline"),
        )
        update(72, "Đang ghép audio tiếng Việt với video")
        mux_vietnamese_audio(
            spec.video_path,
            timeline,
            output,
            spec.duration_seconds,
            lambda value: update(72 + round(value * 27), "Đang ghép MP4"),
        )
        update(100, "Hoàn thành")
    finally:
        if work_dir.is_dir():
            shutil.rmtree(work_dir)
