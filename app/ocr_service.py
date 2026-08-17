from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from poc.core import measure, save_json
from poc.extract_subtitles import execute


@dataclass(frozen=True)
class OcrJobSpec:
    job_id: str
    project_id: int
    video_id: int
    video_path: Path
    output_dir: Path
    fps: float = 5.0
    bottom_ratio: float = 0.35


def execute_ocr_job(
    spec: OcrJobSpec,
    update: Callable[[int, str], None],
) -> None:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    output = spec.output_dir / "english.srt"
    temporary_output = spec.output_dir / ".english.part.srt"
    metrics_path = spec.output_dir / "ocr_metrics.json"
    update(5, "Đang trích frame và nhận diện subtitle tiếng Anh")
    args = argparse.Namespace(
        input=spec.video_path,
        output=temporary_output,
        artifacts=spec.output_dir,
        fps=spec.fps,
        bottom_ratio=spec.bottom_ratio,
        metrics=metrics_path,
    )
    try:
        result, performance = measure(lambda: execute(args))
        temporary_output.replace(output)
        result["output_srt"] = str(output)
        result["performance"] = performance
        save_json(result, metrics_path)
        update(100, f"Đã trích xuất {result['subtitle_cues']} subtitle")
    finally:
        temporary_output.unlink(missing_ok=True)
