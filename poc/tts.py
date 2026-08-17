from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Protocol

from poc.core import Cue, measure, probe_video, read_srt, run_command, save_json


class TTSProvider(Protocol):
    name: str
    voice_cloning: bool

    def synthesize(self, text: str, reference_audio: Path, output: Path) -> None: ...


class EspeakProvider:
    name = "espeak-ng-fallback"
    voice_cloning = False

    def synthesize(self, text: str, reference_audio: Path, output: Path) -> None:
        del reference_audio
        if shutil.which("espeak-ng") is None:
            raise RuntimeError("espeak-ng is not installed")
        run_command(["espeak-ng", "-v", "vi", "-s", "155", "-w", str(output), text])


class VTTSProvider:
    name = "v-tts-zero-shot"
    voice_cloning = True

    def __init__(self) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        try:
            from huggingface_hub import hf_hub_download
            from v_tts import ZeroShotTTS
        except ImportError as exc:
            raise RuntimeError(
                "v-tts is not installed. Install the local CPU model before using --provider vtts."
            ) from exc
        repo_id = os.environ.get("V_TTS_HF_REPO", "letrggghieu/v-zeroshot-voice-cloning")
        revision = os.environ.get("V_TTS_HF_REVISION", "83deb9f5ab591ca79840c9042dbef37dd2e5780e")
        cache_base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        package_cache = cache_base / "v_tts" / "models"
        model_root = package_cache / "zeroshot-vietnamese"
        checkpoint = model_root / "pretrained" / "zeroshot" / "G_175000.pth"
        config = model_root / "pretrained" / "zeroshot" / "config.json"
        speaker_weights = model_root / "pretrained" / "hasp" / "pytorch_model.bin"
        for relative, destination in (
            ("pretrained/zeroshot/G_175000.pth", checkpoint),
            ("pretrained/zeroshot/config.json", config),
            ("pretrained/hasp/pytorch_model.bin", speaker_weights),
        ):
            if destination.exists():
                continue
            if os.environ.get("TTS_LOCAL_ONLY") == "1":
                raise RuntimeError(
                    f"Local TTS model is missing: {destination}. "
                    "Download the pinned model before enabling offline mode."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=relative,
                    local_dir=str(model_root),
                    repo_type="space",
                    revision=revision,
                )
            )
            if downloaded != destination:
                shutil.copy2(downloaded, destination)
        previous_directory = Path.cwd()
        try:
            # SpeakerEncoder resolves its weights relative to cwd.
            os.chdir(model_root)
            self.model = ZeroShotTTS(
                checkpoint_path=str(checkpoint),
                config_path=str(config),
                device="cpu",
            )
        finally:
            os.chdir(previous_directory)

    def synthesize(self, text: str, reference_audio: Path, output: Path) -> None:
        self.model.clone_voice(
            text=text,
            reference_audio=str(reference_audio),
            output_path=str(output),
        )


def create_provider(name: str) -> TTSProvider:
    if name == "vtts":
        return VTTSProvider()
    if name == "espeak":
        return EspeakProvider()
    raise ValueError(f"Unknown provider: {name}")


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def atempo_chain(tempo: float) -> str:
    factors: list[float] = []
    while tempo > 2.0:
        factors.append(2.0)
        tempo /= 2.0
    while tempo < 0.5:
        factors.append(0.5)
        tempo /= 0.5
    factors.append(tempo)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def fit_audio(source: Path, destination: Path, slot_duration: float) -> dict:
    original_duration = audio_duration(source)
    tempo = max(1.0, original_duration / slot_duration)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-af",
            f"{atempo_chain(tempo)},apad",
            "-t",
            f"{slot_duration:.6f}",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
        ]
    )
    return {
        "original_duration": round(original_duration, 3),
        "slot_duration": round(slot_duration, 3),
        "tempo": round(tempo, 3),
        "needs_review": tempo > 1.2,
    }


def write_timeline(cues: list[Cue], fitted_files: list[Path], output: Path, total_duration: float) -> None:
    rate = 24000
    channels = 1
    sample_width = 2
    output.parent.mkdir(parents=True, exist_ok=True)
    cursor = 0
    with wave.open(str(output), "wb") as destination:
        destination.setparams((channels, sample_width, rate, 0, "NONE", "not compressed"))
        for cue, path in zip(cues, fitted_files):
            target_start = round(cue.start * rate)
            if target_start > cursor:
                destination.writeframes(b"\x00" * (target_start - cursor) * sample_width)
                cursor = target_start
            with wave.open(str(path), "rb") as source:
                frames = source.readframes(source.getnframes())
            if target_start < cursor:
                skipped = (cursor - target_start) * sample_width
                frames = frames[skipped:]
            destination.writeframes(frames)
            cursor += len(frames) // sample_width
        total_frames = round(total_duration * rate)
        if total_frames > cursor:
            destination.writeframes(b"\x00" * (total_frames - cursor) * sample_width)


def extract_reference(video: Path, output: Path, start: float, duration: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            str(video),
            "-vn",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output),
        ]
    )


def execute(args: argparse.Namespace) -> dict:
    cues = read_srt(args.srt)[: args.max_cues]
    if not cues:
        raise RuntimeError("The Vietnamese SRT contains no cues")
    output_dir = args.artifacts.resolve() / "tts"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = output_dir / "reference.wav"
    extract_reference(args.video.resolve(), reference, args.reference_start, args.reference_duration)
    provider = create_provider(args.provider)
    segments: list[dict] = []
    fitted_files: list[Path] = []
    for index, cue in enumerate(cues, start=1):
        raw = output_dir / f"cue_{index:03d}_raw.wav"
        fitted = output_dir / f"cue_{index:03d}_fitted.wav"
        provider.synthesize(cue.text, reference, raw)
        timing = fit_audio(raw, fitted, cue.end - cue.start)
        segments.append({"index": index, "text": cue.text, **timing})
        fitted_files.append(fitted)
    total_duration = min(probe_video(args.video.resolve())["duration"], args.timeline_duration)
    write_timeline(cues, fitted_files, args.output.resolve(), total_duration)
    return {
        "provider": provider.name,
        "voice_cloning": provider.voice_cloning,
        "reference_audio": str(reference),
        "output_audio": str(args.output.resolve()),
        "cue_count": len(cues),
        "timeline_duration": total_duration,
        "segments": segments,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a local Vietnamese TTS timeline")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--srt", type=Path, required=True)
    parser.add_argument("--provider", choices=("vtts", "espeak"), default="vtts")
    parser.add_argument("--reference-start", type=float, default=20.0)
    parser.add_argument("--reference-duration", type=float, default=8.0)
    parser.add_argument("--max-cues", type=int, default=4)
    parser.add_argument("--timeline-duration", type=float, default=15.0)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tts_timeline.wav"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/tts_metrics.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result, performance = measure(lambda: execute(args))
    result["performance"] = performance
    save_json(result, args.metrics)
    print(
        f"Created {result['cue_count']} TTS cues using {result['provider']} in "
        f"{performance['elapsed_seconds']}s; peak RSS: {performance['peak_rss_mb']} MB"
    )


if __name__ == "__main__":
    main()
