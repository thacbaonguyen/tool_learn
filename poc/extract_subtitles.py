from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageFilter

from poc.core import BoundingBox, Cue, box_as_dict, measure, probe_video, run_command, save_json, write_srt


@dataclass(slots=True)
class FrameState:
    timestamp: float
    path: Path
    box: BoundingBox | None
    signature: tuple[int, int, bytes] | None


@dataclass(slots=True)
class Cluster:
    frames: list[FrameState] = field(default_factory=list)


def _is_green(pixel: tuple[int, int, int]) -> bool:
    red, green, blue = pixel
    return red < 55 and 45 <= green <= 125 and 30 <= blue <= 110 and green - red >= 25 and blue - red >= 20


def detect_green_box(image: Image.Image, bottom_ratio: float) -> BoundingBox | None:
    image = image.convert("RGB")
    width, height = image.size
    crop_y = int(height * (1 - bottom_ratio))
    bottom = image.crop((0, crop_y, width, height))
    scale = min(1.0, 640 / width)
    sampled = bottom.resize((round(width * scale), round(bottom.height * scale)))

    xs: list[int] = []
    ys: list[int] = []
    for y in range(sampled.height):
        for x in range(sampled.width):
            if _is_green(sampled.getpixel((x, y))):
                xs.append(x)
                ys.append(y)
    if len(xs) < sampled.width * sampled.height * 0.005:
        return None

    inverse = 1 / scale
    left = round(min(xs) * inverse)
    top = crop_y + round(min(ys) * inverse)
    right = round((max(xs) + 1) * inverse)
    bottom_y = crop_y + round((max(ys) + 1) * inverse)
    return BoundingBox(left, top, right - left, bottom_y - top)


def make_signature(image: Image.Image, box: BoundingBox | None) -> tuple[int, int, bytes] | None:
    if box is None:
        return None
    crop = image.crop((box.x, box.y, box.x + box.width, box.y + box.height)).convert("L")
    normalized = crop.resize((128, 20))
    bits = bytes(1 if value > 155 else 0 for value in normalized.getdata())
    return round(box.x / 12), round(box.width / 12), bits


def signatures_match(left: tuple[int, int, bytes] | None, right: tuple[int, int, bytes] | None) -> bool:
    if left is None or right is None:
        return left is right
    if abs(left[0] - right[0]) > 2 or abs(left[1] - right[1]) > 2:
        return False
    difference = sum(a != b for a, b in zip(left[2], right[2])) / len(left[2])
    return difference <= 0.025


def prepare_for_ocr(image: Image.Image, box: BoundingBox) -> Image.Image:
    # Removing the rounded banner edge prevents Tesseract from reading it as | or _.
    horizontal_inset = min(18, max(2, box.width // 20))
    vertical_inset = min(5, max(1, box.height // 10))
    crop = image.crop(
        (
            box.x + horizontal_inset,
            box.y + vertical_inset,
            box.x + box.width - horizontal_inset,
            box.y + box.height - vertical_inset,
        )
    ).convert("RGB")
    # Keep only the bright subtitle glyphs, then invert to black text on white.
    binary = Image.new("L", crop.size, 255)
    binary.putdata(
        [
            0 if red > 145 and green > 145 and blue > 145 else 255
            for red, green, blue in crop.getdata()
        ]
    )
    binary = binary.resize((binary.width * 3, binary.height * 3))
    return binary.filter(ImageFilter.SHARPEN)


def ocr_image(image: Image.Image, box: BoundingBox, debug_path: Path) -> str:
    prepared = prepare_for_ocr(image, box)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.save(debug_path)
    result = subprocess.run(
        [
            "tesseract",
            str(debug_path),
            "stdout",
            "-l",
            "eng",
            "--oem",
            "1",
            "--psm",
            "7",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    text = re.sub(r"\s+", " ", result.stdout).strip()
    text = re.sub(r"^[|_]+\s*|\s*[|_]+$", "", text).strip()
    return re.sub(r"(?<!\w)\|(?!\w)", "I", text)


def extract_frames(video: Path, directory: Path, fps: float) -> list[Path]:
    pattern = directory / "frame_%06d.png"
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={fps}",
            "-start_number",
            "0",
            str(pattern),
        ]
    )
    return sorted(directory.glob("frame_*.png"))


def cluster_frames(paths: list[Path], fps: float, bottom_ratio: float) -> list[Cluster]:
    clusters: list[Cluster] = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            box = detect_green_box(image, bottom_ratio)
            state = FrameState(index / fps, path, box, make_signature(image, box))
        if not clusters or not signatures_match(clusters[-1].frames[-1].signature, state.signature):
            clusters.append(Cluster([state]))
        else:
            clusters[-1].frames.append(state)
    return clusters


def clusters_to_cues(
    clusters: list[Cluster], duration: float, fps: float, debug_dir: Path
) -> tuple[list[Cue], list[BoundingBox], int]:
    cues: list[Cue] = []
    boxes: list[BoundingBox] = []
    failures = 0
    for cluster in clusters:
        states = cluster.frames
        available = [state for state in states if state.box is not None]
        if not available:
            continue
        representative = available[len(available) // 2]
        with Image.open(representative.path) as image:
            text = ocr_image(image, representative.box, debug_dir / f"cue_{len(cues) + 1:03d}.png")
        if not text:
            text = "[OCR_FAILED]"
            failures += 1
        start = states[0].timestamp
        end = min(duration, states[-1].timestamp + 1 / fps)
        if end - start < (0.9 / fps):
            continue
        if cues and cues[-1].text == text and start - cues[-1].end <= 1 / fps + 0.01:
            cues[-1].end = end
        else:
            cues.append(Cue(start, end, text))
        boxes.extend(state.box for state in available if state.box is not None)
    return cues, boxes, failures


def median_box(boxes: list[BoundingBox]) -> BoundingBox:
    if not boxes:
        raise RuntimeError("No green subtitle background was detected")
    return BoundingBox(
        round(statistics.median(box.x for box in boxes)),
        round(statistics.median(box.y for box in boxes)),
        round(statistics.median(box.width for box in boxes)),
        round(statistics.median(box.height for box in boxes)),
    )


def union_box(boxes: list[BoundingBox]) -> BoundingBox:
    if not boxes:
        raise RuntimeError("No green subtitle background was detected")
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return BoundingBox(left, top, right - left, bottom - top)


def extract_ground_truth(video: Path, output: Path) -> bool:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-map",
            "0:s:0",
            "-c:s",
            "srt",
            "-y",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0 and output.exists()


def execute(args: argparse.Namespace) -> dict:
    video = args.input.resolve()
    output = args.output.resolve()
    debug_dir = args.artifacts.resolve() / "ocr_crops"
    args.artifacts.resolve().mkdir(parents=True, exist_ok=True)
    if debug_dir.exists():
        for old_crop in debug_dir.glob("cue_*.png"):
            old_crop.unlink()
    metadata = probe_video(video)
    with tempfile.TemporaryDirectory(
        prefix="work-ocr-", dir=args.artifacts.resolve()
    ) as temporary:
        frame_paths = extract_frames(video, Path(temporary), args.fps)
        clusters = cluster_frames(frame_paths, args.fps, args.bottom_ratio)
        cues, boxes, failures = clusters_to_cues(
            clusters, metadata["duration"], args.fps, debug_dir
        )
    write_srt(cues, output)
    ground_truth = args.artifacts.resolve() / "ground_truth.srt"
    has_ground_truth = extract_ground_truth(video, ground_truth)
    representative = median_box(boxes)
    return {
        "input": str(video),
        "output_srt": str(output),
        "video": metadata,
        "sampling_fps": args.fps,
        "sampled_frames": round(metadata["duration"] * args.fps),
        "detected_clusters": len(clusters),
        "subtitle_cues": len(cues),
        "ocr_failures": failures,
        "representative_box": box_as_dict(representative),
        "cover_box": box_as_dict(union_box(boxes)),
        "ground_truth_srt": str(ground_truth) if has_ground_truth else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract burned-in green-background subtitles into SRT")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/english_ocr.srt"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--bottom-ratio", type=float, default=0.35)
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/ocr_metrics.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result, performance = measure(lambda: execute(args))
    result["performance"] = performance
    save_json(result, args.metrics)
    print(f"Created {result['subtitle_cues']} cues at {args.output}")
    print(f"Elapsed: {performance['elapsed_seconds']}s; peak RSS: {performance['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
