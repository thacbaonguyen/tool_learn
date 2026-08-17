from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderCue:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class RenderStyle:
    x_ratio: float
    y_ratio: float
    width_ratio: float
    height_ratio: float
    font_family: str
    font_size_ratio: float
    text_color: str
    background_color: str


FONT_MAP = {
    "Arial": "Liberation Sans",
    "Verdana": "DejaVu Sans",
    "Tahoma": "DejaVu Sans",
    "Georgia": "Liberation Serif",
    "Courier New": "Liberation Mono",
}


def ass_time(milliseconds: int) -> str:
    centiseconds = max(0, round(milliseconds / 10))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{fraction:02d}"


def escape_ass(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def ass_color(css_color: str) -> str:
    red = css_color[1:3]
    green = css_color[3:5]
    blue = css_color[5:7]
    return f"&H00{blue}{green}{red}".upper()


def pixel_box(style: RenderStyle, width: int, height: int) -> tuple[int, int, int, int]:
    x = min(width - 1, max(0, round(style.x_ratio * width)))
    y = min(height - 1, max(0, round(style.y_ratio * height)))
    box_width = min(width - x, max(1, round(style.width_ratio * width)))
    box_height = min(height - y, max(1, round(style.height_ratio * height)))
    return x, y, box_width, box_height


def create_ass(
    cues: list[RenderCue],
    output: Path,
    width: int,
    height: int,
    style: RenderStyle,
    *,
    start_ms: int = 0,
    duration_ms: int | None = None,
) -> int:
    x, y, box_width, box_height = pixel_box(style, width, height)
    center_x = x + box_width // 2
    center_y = y + box_height // 2
    font_size = max(8, round(style.font_size_ratio * height))
    font_name = FONT_MAP.get(style.font_family, "DejaVu Sans")
    margin_left = x
    margin_right = max(0, width - x - box_width)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{ass_color(style.text_color)},{ass_color(style.text_color)},&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,{margin_left},{margin_right},0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    segment_end = start_ms + duration_ms if duration_ms is not None else None
    events: list[str] = []
    for cue in cues:
        if cue.end_ms <= start_ms:
            continue
        if segment_end is not None and cue.start_ms >= segment_end:
            continue
        event_start = max(cue.start_ms, start_ms) - start_ms
        event_end = cue.end_ms - start_ms
        if duration_ms is not None:
            event_end = min(event_end, duration_ms)
        position = rf"{{\an5\pos({center_x},{center_y})}}"
        events.append(
            f"Dialogue: 0,{ass_time(event_start)},{ass_time(event_end)},"
            f"Default,,0,0,0,,{position}{escape_ass(cue.text)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return len(events)


def escape_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def render_video(
    source: Path,
    output: Path,
    ass_path: Path,
    *,
    width: int,
    height: int,
    style: RenderStyle,
    cues: list[RenderCue],
    start_seconds: float = 0,
    duration_seconds: float | None = None,
) -> None:
    start_ms = round(start_seconds * 1000)
    duration_ms = round(duration_seconds * 1000) if duration_seconds is not None else None
    create_ass(
        cues,
        ass_path,
        width,
        height,
        style,
        start_ms=start_ms,
        duration_ms=duration_ms,
    )
    x, y, box_width, box_height = pixel_box(style, width, height)
    cover_color = style.background_color.removeprefix("#")
    filters = (
        f"drawbox=x={x}:y={y}:w={box_width}:h={box_height}:"
        f"color=0x{cover_color}:t=fill,ass=filename='{escape_filter_path(ass_path)}'"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.stem}.part{output.suffix}")
    temporary_output.unlink(missing_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start_seconds > 0:
        command.extend(["-ss", f"{start_seconds:.3f}"])
    command.extend(["-i", str(source)])
    if duration_seconds is not None:
        command.extend(["-t", f"{duration_seconds:.3f}"])
    command.extend(
        [
            "-vf",
            filters,
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(temporary_output),
        ]
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            message = result.stderr.strip().splitlines()
            detail = message[-1] if message else "FFmpeg không trả về chi tiết."
            raise RenderError(f"Render thất bại: {detail}")
        temporary_output.replace(output)
    except FileNotFoundError as exc:
        raise RenderError("Không tìm thấy FFmpeg.") from exc
    finally:
        temporary_output.unlink(missing_ok=True)

