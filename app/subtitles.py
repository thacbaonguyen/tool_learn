from __future__ import annotations

import re
from dataclasses import dataclass


TIMING_PATTERN = re.compile(
    r"^(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)


class SrtParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedCue:
    sequence: int
    start_ms: int
    end_ms: int
    text: str


def timestamp_to_ms(value: str) -> int:
    normalized = value.replace(".", ",")
    try:
        clock, milliseconds = normalized.split(",", 1)
        hours, minutes, seconds = (int(part) for part in clock.split(":"))
        millis = int(milliseconds)
    except (ValueError, TypeError) as exc:
        raise SrtParseError(f"Timestamp SRT không hợp lệ: {value}") from exc
    if minutes > 59 or seconds > 59 or millis > 999:
        raise SrtParseError(f"Timestamp SRT không hợp lệ: {value}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def parse_srt(content: str) -> list[ParsedCue]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise SrtParseError("File SRT đang trống.")

    cues: list[ParsedCue] = []
    for position, block in enumerate(re.split(r"\n\s*\n", normalized), start=1):
        lines = [line.rstrip() for line in block.split("\n")]
        if not lines:
            continue
        timing_index = 1 if lines[0].strip().isdigit() else 0
        if timing_index >= len(lines):
            raise SrtParseError(f"Cue {position} thiếu dòng thời gian.")
        match = TIMING_PATTERN.match(lines[timing_index].strip())
        if not match:
            raise SrtParseError(f"Dòng thời gian ở cue {position} không hợp lệ.")
        text = "\n".join(line.strip() for line in lines[timing_index + 1 :]).strip()
        if not text:
            raise SrtParseError(f"Cue {position} không có nội dung.")
        start_ms = timestamp_to_ms(match.group("start"))
        end_ms = timestamp_to_ms(match.group("end"))
        if end_ms <= start_ms:
            raise SrtParseError(f"Cue {position} có thời gian kết thúc không hợp lệ.")
        cues.append(
            ParsedCue(
                sequence=position,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
        )

    if not cues:
        raise SrtParseError("Không tìm thấy cue nào trong file SRT.")
    return cues

