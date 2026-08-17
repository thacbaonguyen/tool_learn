from __future__ import annotations

import argparse
import re
from pathlib import Path

from poc.core import Cue, read_srt, save_json


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_word in enumerate(left, start=1):
        current = [row]
        for column, right_word in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1]


def overlap(left: Cue, right: Cue) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def evaluate(predicted: list[Cue], reference: list[Cue]) -> dict:
    predicted_transcript = normalize(" ".join(cue.text for cue in predicted))
    reference_transcript = normalize(" ".join(cue.text for cue in reference))
    transcript_errors = edit_distance(reference_transcript, predicted_transcript)
    total_words = 0
    total_errors = 0
    details = []
    for cue in predicted:
        matches = [item for item in reference if overlap(cue, item) > 0]
        expected = " ".join(item.text for item in matches)
        expected_words = normalize(expected)
        actual_words = normalize(cue.text)
        errors = edit_distance(expected_words, actual_words)
        total_words += len(expected_words)
        total_errors += errors
        details.append(
            {
                "start": cue.start,
                "end": cue.end,
                "ocr": cue.text,
                "reference": expected,
                "word_errors": errors,
                "reference_words": len(expected_words),
            }
        )
    return {
        "predicted_cues": len(predicted),
        "reference_cues": len(reference),
        "transcript_word_error_rate": round(transcript_errors / len(reference_transcript), 4)
        if reference_transcript
        else None,
        "transcript_word_errors": transcript_errors,
        "transcript_reference_words": len(reference_transcript),
        "timing_aligned_word_error_rate": round(total_errors / total_words, 4) if total_words else None,
        "total_word_errors": total_errors,
        "reference_words_compared": total_words,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR SRT against an optional embedded SRT track")
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ocr_evaluation.json"))
    args = parser.parse_args()
    result = evaluate(read_srt(args.predicted), read_srt(args.reference))
    save_json(result, args.output)
    print(
        f"Transcript WER: {result['transcript_word_error_rate']}; "
        f"visual/embedded cues: {result['predicted_cues']}/{result['reference_cues']}"
    )


if __name__ == "__main__":
    main()
