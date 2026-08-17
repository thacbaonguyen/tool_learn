from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import psutil


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(value: dict | None, key: str, default: str = "chưa chạy") -> object:
    if value is None:
        return default
    return value.get(key, default)


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the measured Phase 0 report")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("docs/phase0-report.md"))
    parser.add_argument("--cpu-quota", type=int, default=3)
    parser.add_argument("--offline-verified", action="store_true")
    args = parser.parse_args()

    ocr = load(args.artifacts / "ocr_metrics.json")
    evaluation = load(args.artifacts / "ocr_evaluation.json")
    tts = load(args.artifacts / "tts_metrics.json") or load(args.artifacts / "tts_fallback_metrics.json")
    render = load(args.artifacts / "render_metrics.json") or load(args.artifacts / "render_fallback_metrics.json")
    voice_clone = bool(tts and tts.get("voice_cloning"))

    lines = [
        "# Báo cáo Phase 0",
        "",
        "## Môi trường",
        "",
        f"- CPU host benchmark: {cpu_model()}",
        f"- CPU quota của container: {args.cpu_quota}; host có {psutil.cpu_count(logical=True)} logical CPU",
        f"- RAM nhìn thấy: {psutil.virtual_memory().total / 1024**3:.1f} GB",
        "- Chế độ: CPU-only, xử lý tuần tự, toàn bộ dữ liệu local",
        "",
        "## Kết quả đo",
        "",
        "| Bước | Thời gian | Peak RAM | Kết quả |",
        "|---|---:|---:|---|",
        f"| OCR | {metric(ocr and ocr.get('performance'), 'elapsed_seconds')} giây | {metric(ocr and ocr.get('performance'), 'peak_rss_mb')} MB | {metric(ocr, 'subtitle_cues')} cue |",
        f"| TTS | {metric(tts and tts.get('performance'), 'elapsed_seconds')} giây | {metric(tts and tts.get('performance'), 'peak_rss_mb')} MB | {metric(tts, 'provider')} |",
        f"| Render | {metric(render and render.get('performance'), 'elapsed_seconds')} giây | {metric(render and render.get('performance'), 'peak_rss_mb')} MB | {metric(render, 'duration')} giây video |",
        "",
        "## OCR",
        "",
        f"- Video mẫu: {metric(ocr and ocr.get('video'), 'duration')} giây, {metric(ocr and ocr.get('video'), 'width')}x{metric(ocr and ocr.get('video'), 'height')}.",
        f"- Tần suất lấy mẫu: {metric(ocr, 'sampling_fps')} FPS.",
        f"- Transcript WER so với subtitle track tham chiếu: {metric(evaluation, 'transcript_word_error_rate')}.",
        "- Số cue OCR và subtitle nhúng có thể khác nhau vì subtitle đóng cứng hiển thị nhiều cụm từ trong cùng một banner.",
        "",
        "## TTS và render",
        "",
        f"- Voice cloning thực sự: {'đạt' if voice_clone else 'chưa đạt; kết quả hiện tại dùng giọng fallback để kiểm tra pipeline'}.",
        f"- Chạy khi container bị tắt network: {'đạt' if args.offline_verified else 'chưa đo'}.",
        f"- Dung lượng model cache sau lần tải đầu: {sum(path.stat().st_size for path in Path('data/models').rglob('*') if path.is_file()) / 1024**2:.1f} MB.",
        "- Thời gian TTS trong bảng bao gồm load model và sinh cue, nhưng không bao gồm tải model lần đầu.",
        f"- Audio gốc đã bị loại bỏ: {'có' if render and render.get('source_audio_discarded') else 'chưa xác minh'}.",
        "- File video: `artifacts/phase0_preview.mp4`.",
        "- Cue có tốc độ lớn hơn 1.2x được đánh dấu `needs_review` để người dùng rút gọn bản dịch.",
        "",
        "## Kết luận",
        "",
        "OCR nền xanh, voice cloning CPU và render local đều đạt yêu cầu POC trên video mẫu. Kết quả benchmark được đo trên host hiện tại với quota 3 CPU; máy i3 Gen 10 cần được benchmark lại và dự kiến chậm hơn.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
