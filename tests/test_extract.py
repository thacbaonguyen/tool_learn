from PIL import Image, ImageDraw

from poc.core import Cue
from poc.evaluate_ocr import evaluate
from poc.extract_subtitles import detect_green_box


def test_detect_green_subtitle_box_in_bottom_region() -> None:
    image = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(image).rectangle((50, 160, 350, 190), fill=(6, 77, 64))
    box = detect_green_box(image, bottom_ratio=0.35)
    assert box is not None
    assert abs(box.x - 50) <= 2
    assert abs(box.y - 160) <= 2
    assert abs(box.width - 301) <= 3
    assert abs(box.height - 31) <= 3


def test_transcript_evaluation_is_independent_of_visual_cue_splitting() -> None:
    predicted = [Cue(0, 2, "HELLO LOCAL WORLD")]
    reference = [Cue(0, 1, "Hello local"), Cue(1, 2, "world")]
    result = evaluate(predicted, reference)
    assert result["transcript_word_error_rate"] == 0

