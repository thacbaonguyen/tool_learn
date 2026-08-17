from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .finalizer import FinalJobSpec, TimelineSegment, execute_final_job
from .rendering import RenderCue, RenderStyle, render_video
from .tts_engine import (
    LocalTTSService,
    generate_tts_cues,
    parse_pronunciation_dictionary,
    save_json,
)


@dataclass(frozen=True)
class AutomaticVideoSpec:
    job_id: str
    project_id: int
    video_id: int
    source_video: Path
    render_video: Path
    ass_path: Path
    final_dir: Path
    tts_dir: Path
    reference_audio: Path
    duration_seconds: float
    width: int
    height: int
    style: RenderStyle
    cues: list[RenderCue]
    dictionary_text: str = ""


class AutomaticVideoPipeline:
    def __init__(self, tts_service: LocalTTSService | None = None) -> None:
        self.tts_service = tts_service or LocalTTSService()

    def execute(
        self,
        spec: AutomaticVideoSpec,
        update: Callable[[int, str], None],
    ) -> None:
        update(2, "Đang tạo giọng đọc tiếng Việt")
        manifest = generate_tts_cues(
            [
                {
                    "sequence": index,
                    "start_ms": cue.start_ms,
                    "end_ms": cue.end_ms,
                    "text": cue.text,
                }
                for index, cue in enumerate(spec.cues, start=1)
            ],
            spec.reference_audio,
            parse_pronunciation_dictionary(spec.dictionary_text),
            spec.tts_dir / "cache",
            self.tts_service,
            progress=lambda done, total: update(
                2 + round(43 * done / total),
                f"Đang tạo TTS {done}/{total}",
            ),
        )
        save_json(spec.tts_dir / "manifest.json", manifest)

        update(48, "Đang che subtitle Anh và render subtitle Việt")
        render_video(
            spec.source_video,
            spec.render_video,
            spec.ass_path,
            width=spec.width,
            height=spec.height,
            style=spec.style,
            cues=spec.cues,
        )

        items = manifest["items"]
        segments = [
            TimelineSegment(
                start_ms=int(item["start_ms"]),
                audio_path=spec.tts_dir / "cache" / f"{item['cache_key']}.wav",
                cache_key=str(item["cache_key"]),
            )
            for item in items
        ]
        update(75, "Đang ghép audio Việt vào video")
        execute_final_job(
            FinalJobSpec(
                job_id=spec.job_id,
                project_id=spec.project_id,
                video_id=spec.video_id,
                video_path=spec.render_video,
                output_dir=spec.final_dir,
                duration_seconds=spec.duration_seconds,
                segments=segments,
            ),
            lambda progress, message: update(
                75 + round(progress * 0.25), message
            ),
        )
