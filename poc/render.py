from __future__ import annotations

import argparse
import json
from pathlib import Path

from poc.core import BoundingBox, Cue, measure, probe_video, read_srt, run_command, save_json


def ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{fraction:02d}"


def escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def create_ass(cues: list[Cue], output: Path, width: int, height: int, box: BoundingBox) -> None:
    margin_v = max(0, height - (box.y + box.height) + 8)
    font_size = max(24, round(box.height * 0.48))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{font_size},&H00FFFFFF,&H00FFFFFF,&H00064D40,&H00064D40,-1,0,0,0,100,100,0,0,1,2,0,2,30,30,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for cue in cues:
        events.append(
            f"Dialogue: 0,{ass_time(cue.start)},{ass_time(cue.end)},Default,,0,0,0,,{escape_ass(cue.text)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def escape_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def execute(args: argparse.Namespace) -> dict:
    metadata = probe_video(args.video.resolve())
    extraction = json.loads(args.ocr_metrics.read_text(encoding="utf-8"))
    raw_box = extraction["cover_box"]
    source_box = BoundingBox(raw_box["x"], raw_box["y"], raw_box["width"], raw_box["height"])
    left = max(0, source_box.x - args.box_padding_x)
    top = max(0, source_box.y - args.box_padding_y)
    right = min(metadata["width"], source_box.x + source_box.width + args.box_padding_x)
    bottom = min(metadata["height"], source_box.y + source_box.height + args.box_padding_y)
    box = BoundingBox(left, top, right - left, bottom - top)
    cues = [cue for cue in read_srt(args.srt) if cue.start < args.duration]
    ass_path = args.artifacts.resolve() / "vietnamese.ass"
    create_ass(cues, ass_path, metadata["width"], metadata["height"], box)
    filter_value = (
        f"drawbox=x={box.x}:y={box.y}:w={box.width}:h={box.height}:"
        f"color=0x064d40:t=fill,ass=filename='{escape_filter_path(ass_path)}'"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(args.video.resolve()),
    ]
    if args.audio:
        command.extend(["-i", str(args.audio.resolve())])
    command.extend(
        [
            "-t",
            str(args.duration),
            "-vf",
            filter_value,
            "-map",
            "0:v:0",
        ]
    )
    if args.audio:
        command.extend(["-map", "1:a:0", "-c:a", "aac", "-b:a", "128k"])
    else:
        command.append("-an")
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            "-y",
            str(args.output.resolve()),
        ]
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    run_command(command)
    return {
        "input": str(args.video.resolve()),
        "output": str(args.output.resolve()),
        "duration": args.duration,
        "source_audio_discarded": True,
        "tts_audio_added": bool(args.audio),
        "cover_box": {"x": box.x, "y": box.y, "width": box.width, "height": box.height},
        "subtitle_cues": len(cues),
        "encoder": "libx264/veryfast",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a short Vietnamese subtitle and TTS POC clip")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--srt", type=Path, required=True)
    parser.add_argument("--ocr-metrics", type=Path, default=Path("artifacts/ocr_metrics.json"))
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--box-padding-x", type=int, default=80)
    parser.add_argument("--box-padding-y", type=int, default=8)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase0_preview.mp4"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/render_metrics.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result, performance = measure(lambda: execute(args))
    result["performance"] = performance
    save_json(result, args.metrics)
    print(f"Rendered {args.output} in {performance['elapsed_seconds']}s; peak RSS: {performance['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
