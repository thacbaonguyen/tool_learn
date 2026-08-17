from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class TTSError(RuntimeError):
    pass


class VoiceCloneProvider(Protocol):
    name: str

    def synthesize(self, text: str, reference_audio: Path, output: Path) -> None: ...


def create_local_vtts_provider() -> VoiceCloneProvider:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TTS_LOCAL_ONLY", "1")
    try:
        from poc.tts import VTTSProvider

        return VTTSProvider()
    except (ImportError, RuntimeError) as exc:
        raise TTSError(str(exc)) from exc


class LocalTTSService:
    def __init__(
        self,
        provider_factory: Callable[[], VoiceCloneProvider] = create_local_vtts_provider,
    ) -> None:
        self.provider_factory = provider_factory
        self._provider: VoiceCloneProvider | None = None
        self._lock = threading.RLock()

    def provider(self) -> VoiceCloneProvider:
        with self._lock:
            if self._provider is None:
                self._provider = self.provider_factory()
            return self._provider

    def synthesize(self, text: str, reference_audio: Path, output: Path) -> None:
        with self._lock:
            provider = self.provider()
            provider.synthesize(text, reference_audio, output)


def extract_reference_audio(
    video: Path,
    output: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.part{output.suffix}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
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
        str(temporary),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()
            raise TTSError(
                "Không trích được audio mẫu: "
                + (detail[-1] if detail else "video không có audio phù hợp.")
            )
        temporary.replace(output)
    except FileNotFoundError as exc:
        raise TTSError("Không tìm thấy FFmpeg.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (wave.Error, OSError) as exc:
        raise TTSError(f"Không đọc được WAV: {path.name}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pronunciation_dictionary(content: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise TTSError(f"Dòng từ điển {line_number} phải có dạng từ=cách đọc.")
        term, pronunciation = (part.strip() for part in line.split("=", 1))
        if not term or not pronunciation:
            raise TTSError(f"Dòng từ điển {line_number} đang thiếu từ hoặc cách đọc.")
        if len(term) > 100 or len(pronunciation) > 200:
            raise TTSError(f"Dòng từ điển {line_number} quá dài.")
        entries[term] = pronunciation
    if len(entries) > 500:
        raise TTSError("Từ điển chỉ hỗ trợ tối đa 500 mục.")
    return entries


def apply_pronunciation_dictionary(text: str, entries: dict[str, str]) -> str:
    result = text
    for term in sorted(entries, key=len, reverse=True):
        result = re.sub(re.escape(term), lambda _: entries[term], result, flags=re.IGNORECASE)
    return result


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def cue_cache_key(
    processed_text: str,
    reference_hash: str,
    provider_name: str,
) -> str:
    payload = f"phase5-v1\0{provider_name}\0{reference_hash}\0{processed_text}".encode()
    return hashlib.sha256(payload).hexdigest()


def generate_tts_cues(
    cues: list[dict],
    reference_audio: Path,
    dictionary: dict[str, str],
    cache_dir: Path,
    service: LocalTTSService,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    if not reference_audio.is_file():
        raise TTSError("Hãy chọn và lưu đoạn audio làm giọng mẫu trước.")
    provider = service.provider()
    reference_hash = file_sha256(reference_audio)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    cache_hits = 0
    for cue in cues:
        original_text = str(cue["text"])
        processed_text = apply_pronunciation_dictionary(original_text, dictionary)
        cache_key = cue_cache_key(processed_text, reference_hash, provider.name)
        output = cache_dir / f"{cache_key}.wav"
        if output.is_file():
            cache_hits += 1
        else:
            temporary = cache_dir / f".{cache_key}.part.wav"
            temporary.unlink(missing_ok=True)
            try:
                service.synthesize(processed_text, reference_audio, temporary)
                if not temporary.is_file():
                    raise TTSError("Model không tạo file audio đầu ra.")
                temporary.replace(output)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                if isinstance(exc, TTSError):
                    raise
                raise TTSError(f"TTS thất bại ở cue {cue['sequence']}: {exc}") from exc
        generated_duration = audio_duration(output)
        slot_duration = (int(cue["end_ms"]) - int(cue["start_ms"])) / 1000
        overflow = max(0.0, generated_duration - slot_duration)
        results.append(
            {
                "sequence": int(cue["sequence"]),
                "start_ms": int(cue["start_ms"]),
                "end_ms": int(cue["end_ms"]),
                "original_text": original_text,
                "processed_text": processed_text,
                "cache_key": cache_key,
                "audio_duration": round(generated_duration, 3),
                "slot_duration": round(slot_duration, 3),
                "overflow_seconds": round(overflow, 3),
                "needs_review": overflow > 0.01,
            }
        )
        if progress:
            progress(len(results), len(cues))
    return {
        "provider": provider.name,
        "reference_hash": reference_hash,
        "cue_count": len(results),
        "cache_hits": cache_hits,
        "items": results,
    }
