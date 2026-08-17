from pathlib import Path

from poc.core import BoundingBox, Cue
from poc.render import ass_time, create_ass
from poc.tts import atempo_chain


def test_ass_time() -> None:
    assert ass_time(65.239) == "0:01:05.24"


def test_atempo_chain_stays_in_ffmpeg_range() -> None:
    assert atempo_chain(4.5) == "atempo=2.000000,atempo=2.000000,atempo=1.125000"


def test_create_ass_contains_vietnamese(tmp_path: Path) -> None:
    output = tmp_path / "subtitle.ass"
    create_ass([Cue(0, 2, "Xin chào Việt Nam")], output, 1920, 1080, BoundingBox(400, 920, 1100, 70))
    content = output.read_text(encoding="utf-8")
    assert "PlayResX: 1920" in content
    assert "Xin chào Việt Nam" in content

