from __future__ import annotations

import json
import re
import resource
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import psutil


@dataclass(slots=True)
class Cue:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def padded(self, pixels: int, frame_width: int, frame_height: int) -> "BoundingBox":
        x = max(0, self.x - pixels)
        y = max(0, self.y - pixels)
        right = min(frame_width, self.x + self.width + pixels)
        bottom = min(frame_height, self.y + self.height + pixels)
        return BoundingBox(x, y, right - x, bottom - y)


def run_command(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def probe_video(path: Path) -> dict:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream["r_frame_rate"],
        "duration": float(payload["format"]["duration"]),
    }


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(cues: Iterable[Cue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text.strip()}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


_SRT_BLOCK = re.compile(
    r"(?:^|\n)\s*\d+\s*\n"
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n"
    r"(?P<text>.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def _parse_timestamp(value: str) -> float:
    hours, minutes, rest = value.replace(".", ",").split(":")
    seconds, milliseconds = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def read_srt(path: Path) -> list[Cue]:
    content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    cues = []
    for match in _SRT_BLOCK.finditer(content):
        text = " ".join(line.strip() for line in match.group("text").splitlines()).strip()
        cues.append(Cue(_parse_timestamp(match.group("start")), _parse_timestamp(match.group("end")), text))
    return cues


T = TypeVar("T")


def measure(operation: Callable[[], T]) -> tuple[T, dict[str, float]]:
    process = psutil.Process()
    stop = threading.Event()
    peak_rss = 0

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop.wait(0.05):
            processes = [process]
            try:
                processes.extend(process.children(recursive=True))
            except psutil.Error:
                pass
            total = 0
            for item in processes:
                try:
                    total += item.memory_info().rss
                except psutil.Error:
                    continue
            peak_rss = max(peak_rss, total)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    started = time.perf_counter()
    sampler.start()
    try:
        result = operation()
    finally:
        stop.set()
        sampler.join()
    elapsed = time.perf_counter() - started
    own_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    peak_rss = max(peak_rss, own_peak)
    return result, {"elapsed_seconds": round(elapsed, 3), "peak_rss_mb": round(peak_rss / 1024 / 1024, 2)}


def save_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def box_as_dict(box: BoundingBox) -> dict[str, int]:
    return asdict(box)

