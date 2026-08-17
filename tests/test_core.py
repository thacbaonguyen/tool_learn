from pathlib import Path

import pytest

from poc.core import BoundingBox, Cue, read_srt, write_srt
from poc.evaluate_ocr import edit_distance, normalize


def test_srt_round_trip(tmp_path: Path) -> None:
    expected = [Cue(0.16, 1.719, "Hello world"), Cue(1.719, 3.28, "Second cue")]
    path = tmp_path / "sample.srt"
    write_srt(expected, path)
    actual = read_srt(path)
    assert [cue.text for cue in actual] == [cue.text for cue in expected]
    assert [cue.start for cue in actual] == pytest.approx([cue.start for cue in expected])
    assert [cue.end for cue in actual] == pytest.approx([cue.end for cue in expected])


def test_bounding_box_padding_is_clamped() -> None:
    assert BoundingBox(2, 3, 10, 20).padded(5, 100, 100) == BoundingBox(0, 0, 17, 28)


def test_word_edit_distance() -> None:
    assert edit_distance(normalize("IAM access advisor"), normalize("IAM accesses advisor")) == 1
