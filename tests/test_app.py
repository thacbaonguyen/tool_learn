from __future__ import annotations

import subprocess
import wave
from array import array
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import Settings
from app.database import Database
from app.finalizer import (
    FinalJobSpec,
    TimelineSegment,
    build_audio_timeline,
)
from app.job_queue import JobBusyError, JobRepository
from app.main import create_app
from app.models import Project, SubtitleTrack, Video
from app.pipeline import AutomaticVideoPipeline
from app.rendering import RenderCue, RenderStyle, ass_color, create_ass, pixel_box
from app.subtitles import SrtParseError, parse_srt
from app.tts_engine import (
    LocalTTSService,
    TTSError,
    apply_pronunciation_dictionary,
    load_json,
    parse_pronunciation_dictionary,
)
from app.worker import WorkerService, cleanup_temporary_files
from poc.core import save_json


@pytest.fixture()
def client(tmp_path: Path):
    application = create_app(Settings(data_dir=tmp_path, max_upload_bytes=5 * 1024 * 1024))
    with TestClient(application) as test_client:
        yield test_client, application, tmp_path


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-shortest",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-y",
            str(path),
        ],
        check=True,
        timeout=30,
    )
    return path


@pytest.fixture()
def voice_video(tmp_path: Path) -> Path:
    path = tmp_path / "voice-sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:d=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-shortest",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-y",
            str(path),
        ],
        check=True,
        timeout=30,
    )
    return path


class FakeVoiceCloneProvider:
    name = "fake-local-clone-v1"

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, text: str, reference_audio: Path, output: Path) -> None:
        assert reference_audio.is_file()
        assert text
        self.calls += 1
        rate = 24_000
        with wave.open(str(output), "wb") as audio:
            audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            audio.writeframes((array("h", [1200]) * round(rate * 0.8)).tobytes())


def write_constant_wav(path: Path, value: int, duration: float) -> None:
    rate = 24_000
    samples = array("h", [value]) * round(rate * duration)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())


def create_project(client: TestClient, name: str = "IAM lesson") -> int:
    response = client.post("/projects", data={"name": name}, follow_redirects=False)
    assert response.status_code == 303
    return int(response.headers["location"].rsplit("/", 1)[1])


def upload_sample_video(client: TestClient, project_id: int, path: Path) -> int:
    with path.open("rb") as source:
        response = client.post(
            f"/projects/{project_id}/videos",
            files={"video_file": ("lesson.mp4", source, "video/mp4")},
            follow_redirects=False,
        )
    assert response.status_code == 303
    return project_id


def test_project_crud(client) -> None:
    test_client, application, _ = client
    project_id = create_project(test_client)

    detail = test_client.get(f"/projects/{project_id}")
    assert detail.status_code == 200
    assert "IAM lesson" in detail.text

    renamed = test_client.post(
        f"/projects/{project_id}/rename",
        data={"name": "AWS IAM"},
        follow_redirects=False,
    )
    assert renamed.status_code == 303
    assert "AWS IAM" in test_client.get(f"/projects/{project_id}").text

    deleted = test_client.post(
        f"/projects/{project_id}/delete", follow_redirects=False
    )
    assert deleted.status_code == 303
    with application.state.database.session_factory() as session:
        assert session.get(Project, project_id) is None


def test_upload_probe_stream_and_delete(client, sample_video: Path) -> None:
    test_client, application, data_dir = client
    project_id = create_project(test_client)

    with sample_video.open("rb") as source:
        response = test_client.post(
            f"/projects/{project_id}/videos",
            files={"video_file": ("lesson.mp4", source, "video/mp4")},
            follow_redirects=False,
        )
    assert response.status_code == 303

    with application.state.database.session_factory() as session:
        video = session.scalar(select(Video).where(Video.project_id == project_id))
        assert video is not None
        video_id = video.id
        storage_path = video.storage_path
        assert video.width == 320
        assert video.height == 180
        assert video.duration_seconds == pytest.approx(1, abs=0.1)
        assert video.video_codec == "mpeg4"

    stored_file = data_dir / storage_path
    assert stored_file.is_file()
    detail = test_client.get(f"/projects/{project_id}")
    assert "320 × 180" in detail.text
    assert "lesson.mp4" in detail.text

    streamed = test_client.get(
        f"/projects/{project_id}/videos/{video_id}/stream",
        headers={"range": "bytes=0-99"},
    )
    assert streamed.status_code in {200, 206}
    assert streamed.content

    deleted = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/delete",
        follow_redirects=False,
    )
    assert deleted.status_code == 303
    assert not stored_file.exists()


def test_invalid_upload_is_rejected_and_cleaned(client) -> None:
    test_client, application, data_dir = client
    project_id = create_project(test_client)

    response = test_client.post(
        f"/projects/{project_id}/videos",
        files={"video_file": ("broken.mp4", b"not a video", "video/mp4")},
    )
    assert response.status_code == 400
    assert "không đọc được metadata" in response.text

    with application.state.database.session_factory() as session:
        assert session.scalar(select(Video)) is None
    assert not list((data_dir / "projects").rglob("*.mp4"))


def test_parse_multiline_srt() -> None:
    cues = parse_srt(
        "1\n00:00:00,100 --> 00:00:01,200\nXin chào\nAWS!\n\n"
        "2\n00:00:01.300 --> 00:00:02.000\nBài học IAM\n"
    )
    assert len(cues) == 2
    assert cues[0].start_ms == 100
    assert cues[0].end_ms == 1200
    assert cues[0].text == "Xin chào\nAWS!"
    assert cues[1].start_ms == 1300


def test_parse_srt_rejects_invalid_range() -> None:
    with pytest.raises(SrtParseError):
        parse_srt("1\n00:00:02,000 --> 00:00:01,000\nSai thời gian")


def test_import_subtitle_and_update_normalized_style(client, sample_video: Path) -> None:
    test_client, application, _ = client
    project_id = create_project(test_client)
    upload_sample_video(test_client, project_id, sample_video)
    with application.state.database.session_factory() as session:
        video_id = session.scalar(select(Video.id))
    assert video_id is not None

    srt = (
        "1\n00:00:00,000 --> 00:00:00,700\nXin chào AWS\n\n"
        "2\n00:00:00,700 --> 00:00:01,000\nHọc IAM\n"
    )
    imported = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/subtitles/import",
        files={"srt_file": ("vi.srt", srt.encode("utf-8"), "application/x-subrip")},
        follow_redirects=False,
    )
    assert imported.status_code == 303

    editor = test_client.get(
        f"/projects/{project_id}/videos/{video_id}/subtitles"
    )
    assert editor.status_code == 200
    assert 'id="subtitle-cues"' in editor.text
    assert "Xin ch\\u00e0o AWS" in editor.text
    assert "2 cue" in editor.text

    style = {
        "x_ratio": 0.12,
        "y_ratio": 0.72,
        "width_ratio": 0.76,
        "height_ratio": 0.18,
        "font_family": "Verdana",
        "font_size_ratio": 0.052,
        "text_color": "#ffff00",
        "background_color": "#123456",
    }
    saved = test_client.put(
        f"/projects/{project_id}/videos/{video_id}/subtitles/style", json=style
    )
    assert saved.status_code == 200
    assert saved.json() == {"status": "saved"}

    with application.state.database.session_factory() as session:
        track = session.scalar(select(SubtitleTrack))
        assert track is not None
        assert track.x_ratio == pytest.approx(0.12)
        assert track.width_ratio == pytest.approx(0.76)
        assert track.font_family == "Verdana"
        assert track.font_size_ratio == pytest.approx(0.052)
        assert track.text_color == "#ffff00"
        assert len(track.cues) == 2

    invalid_style = {**style, "x_ratio": 0.5, "width_ratio": 0.8}
    rejected = test_client.put(
        f"/projects/{project_id}/videos/{video_id}/subtitles/style",
        json=invalid_style,
    )
    assert rejected.status_code == 422


def test_invalid_srt_does_not_create_track(client, sample_video: Path) -> None:
    test_client, application, _ = client
    project_id = create_project(test_client)
    upload_sample_video(test_client, project_id, sample_video)
    with application.state.database.session_factory() as session:
        video_id = session.scalar(select(Video.id))
    assert video_id is not None

    response = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/subtitles/import",
        files={"srt_file": ("bad.srt", b"not an srt", "application/x-subrip")},
    )
    assert response.status_code == 400
    assert "không hợp lệ" in response.text
    with application.state.database.session_factory() as session:
        assert session.scalar(select(SubtitleTrack)) is None


def test_create_ass_uses_normalized_box_and_shifted_timeline(tmp_path: Path) -> None:
    style = RenderStyle(
        x_ratio=0.1,
        y_ratio=0.75,
        width_ratio=0.8,
        height_ratio=0.2,
        font_family="Arial",
        font_size_ratio=0.05,
        text_color="#12abef",
        background_color="#075e54",
    )
    assert pixel_box(style, 1920, 1080) == (192, 810, 1536, 216)
    assert ass_color("#12abef") == "&H00EFAB12"

    ass_path = tmp_path / "preview.ass"
    count = create_ass(
        [
            RenderCue(9_000, 11_000, "Cue trước"),
            RenderCue(12_000, 14_000, "Xin {chào}\nAWS"),
            RenderCue(30_000, 31_000, "Cue sau"),
        ],
        ass_path,
        1920,
        1080,
        style,
        start_ms=10_000,
        duration_ms=10_000,
    )
    content = ass_path.read_text(encoding="utf-8")
    assert count == 2
    assert "Dialogue: 0,0:00:00.00,0:00:01.00" in content
    assert "Dialogue: 0,0:00:02.00,0:00:04.00" in content
    assert r"Xin \{chào\}\NAWS" in content
    assert r"\pos(960,918)" in content


def test_render_short_preview_without_original_audio(client, sample_video: Path) -> None:
    test_client, application, data_dir = client
    project_id = create_project(test_client)
    upload_sample_video(test_client, project_id, sample_video)
    with application.state.database.session_factory() as session:
        video_id = session.scalar(select(Video.id))
    assert video_id is not None

    srt = "1\n00:00:00,000 --> 00:00:01,000\nPhụ đề Việt\n"
    imported = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/subtitles/import",
        files={"srt_file": ("vi.srt", srt.encode("utf-8"), "application/x-subrip")},
        follow_redirects=False,
    )
    assert imported.status_code == 303

    rendered = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/renders/preview",
        data={"start_seconds": "0", "duration_seconds": "1"},
        follow_redirects=False,
    )
    assert rendered.status_code == 303
    output = data_dir / "projects" / str(project_id) / "renders" / str(video_id) / "preview.mp4"
    ass_file = output.with_suffix(".ass")
    assert output.is_file()
    assert ass_file.is_file()
    assert "Phụ đề Việt" in ass_file.read_text(encoding="utf-8")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip().splitlines() == ["video"]

    streamed = test_client.get(
        f"/projects/{project_id}/videos/{video_id}/renders/preview"
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"] == "video/mp4"

    editor = test_client.get(
        f"/projects/{project_id}/videos/{video_id}/subtitles"
    )
    assert "Preview" in editor.text
    assert "Tải MP4" in editor.text


def test_extract_english_srt_job_and_download(client, sample_video: Path) -> None:
    test_client, application, data_dir = client
    project_id = create_project(test_client)
    upload_sample_video(test_client, project_id, sample_video)
    with application.state.database.session_factory() as session:
        video_id = session.scalar(select(Video.id))
    assert video_id is not None

    project_page = test_client.get(f"/projects/{project_id}")
    assert "Trích xuất SRT tiếng Anh" in project_page.text
    started = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/extract-subtitles",
        follow_redirects=False,
    )
    assert started.status_code == 303
    job = application.state.job_repository.latest_for_video(
        project_id, video_id, "extract_subtitles"
    )
    assert job is not None
    assert job["status"] == "queued"

    def fake_ocr(spec, update) -> None:
        update(50, "Đang OCR")
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        (spec.output_dir / "english.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nEnglish subtitle\n",
            encoding="utf-8",
        )
        save_json(
            {"subtitle_cues": 1, "ocr_failures": 0},
            spec.output_dir / "ocr_metrics.json",
        )

    worker = WorkerService(
        application.state.job_repository,
        data_dir,
        ocr_executor=fake_ocr,
    )
    assert worker.process_one() is True
    completed = test_client.get(f"/jobs/{job['job_id']}").json()
    assert completed["status"] == "completed"

    downloaded = test_client.get(
        f"/projects/{project_id}/videos/{video_id}/english-subtitles"
    )
    assert downloaded.status_code == 200
    assert "English subtitle" in downloaded.text
    refreshed = test_client.get(f"/projects/{project_id}")
    assert "Tải SRT tiếng Anh" in refreshed.text
    assert "1 cue" in refreshed.text


def test_import_srt_automatically_builds_playable_video_with_audio(
    client, sample_video: Path
) -> None:
    test_client, application, data_dir = client
    provider = FakeVoiceCloneProvider()
    application.state.tts_service = LocalTTSService(lambda: provider)
    voice_dir = data_dir / "voice"
    voice_dir.mkdir(parents=True)
    write_constant_wav(voice_dir / "reference.wav", 1000, 3.0)
    save_json(
        {"source": "sample.wav", "duration_seconds": 3, "scope": "global"},
        voice_dir / "reference.json",
    )
    project_id = create_project(test_client)
    upload_sample_video(test_client, project_id, sample_video)
    with application.state.database.session_factory() as session:
        video_id = session.scalar(select(Video.id))
    assert video_id is not None

    imported = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/subtitles/import",
        files={
            "srt_file": (
                "vi.srt",
                b"1\n00:00:00,000 --> 00:00:01,000\nXin chao\n",
                "application/x-subrip",
            )
        },
        follow_redirects=False,
    )
    assert imported.status_code == 303
    assert "job=" in imported.headers["location"]
    queued = application.state.job_repository.latest_for_video(
        project_id, video_id, "build_video"
    )
    assert queued is not None

    worker = WorkerService(
        application.state.job_repository,
        data_dir,
        pipeline_executor=AutomaticVideoPipeline(
            application.state.tts_service
        ).execute,
    )
    assert worker.process_one() is True
    assert application.state.job_repository.get(queued["job_id"])["status"] == "completed"

    final_path = data_dir / "projects" / str(project_id) / "final" / str(video_id) / "final.mp4"
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
            "-of", "csv=p=0", str(final_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines() == ["video", "audio"]
    decoded_audio = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(final_path), "-map", "0:a:0",
            "-f", "s16le", "-ac", "1", "-ar", "24000", "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    samples = array("h")
    samples.frombytes(decoded_audio)
    assert samples and max(abs(sample) for sample in samples) > 0
    streamed = test_client.get(
        f"/projects/{project_id}/videos/{video_id}/final"
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"] == "video/mp4"


def test_pronunciation_dictionary() -> None:
    entries = parse_pronunciation_dictionary(
        "# technical words\nAWS IAM=ây ét ét ai am\nAWS=ây ét ét\n"
    )
    assert apply_pronunciation_dictionary("AWS IAM và AWS", entries) == (
        "ây ét ét ai am và ây ét ét"
    )
    with pytest.raises(TTSError):
        parse_pronunciation_dictionary("AWS without equals")


def test_reference_voice_tts_cache_and_overflow_warning(client, voice_video: Path) -> None:
    test_client, application, data_dir = client
    fake_provider = FakeVoiceCloneProvider()
    application.state.tts_service = LocalTTSService(lambda: fake_provider)
    project_id = create_project(test_client)
    upload_sample_video(test_client, project_id, voice_video)
    with application.state.database.session_factory() as session:
        video_id = session.scalar(select(Video.id))
    assert video_id is not None

    srt = (
        "1\n00:00:00,000 --> 00:00:00,500\nHọc AWS\n\n"
        "2\n00:00:00,500 --> 00:00:02,500\nThực hành IAM\n"
    )
    imported = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/subtitles/import",
        files={"srt_file": ("vi.srt", srt.encode(), "application/x-subrip")},
        follow_redirects=False,
    )
    assert imported.status_code == 303

    reference = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/tts/reference",
        data={"start_seconds": "0", "duration_seconds": "3"},
        follow_redirects=False,
    )
    assert reference.status_code == 303
    reference_audio = (
        data_dir / "projects" / str(project_id) / "tts" / str(video_id) / "reference.wav"
    )
    assert reference_audio.is_file()

    dictionary = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/tts/dictionary",
        data={"dictionary_text": "AWS=ây ét ét\nIAM=ai am\n"},
        follow_redirects=False,
    )
    assert dictionary.status_code == 303

    generated = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/tts/generate",
        follow_redirects=False,
    )
    assert generated.status_code == 303
    assert fake_provider.calls == 2

    tts_dir = reference_audio.parent
    manifest = load_json(tts_dir / "manifest.json")
    assert manifest["provider"] == "fake-local-clone-v1"
    assert manifest["cache_hits"] == 0
    assert manifest["items"][0]["processed_text"] == "Học ây ét ét"
    assert manifest["items"][0]["needs_review"] is True
    assert manifest["items"][0]["overflow_seconds"] == pytest.approx(0.3)
    assert manifest["items"][1]["needs_review"] is False

    generated_again = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/tts/generate",
        follow_redirects=False,
    )
    assert generated_again.status_code == 303
    assert fake_provider.calls == 2
    cached_manifest = load_json(tts_dir / "manifest.json")
    assert cached_manifest["cache_hits"] == 2

    first_key = cached_manifest["items"][0]["cache_key"]
    audio = test_client.get(
        f"/projects/{project_id}/videos/{video_id}/tts/audio/{first_key}"
    )
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"

    editor = test_client.get(
        f"/projects/{project_id}/videos/{video_id}/subtitles"
    )
    assert "Dài hơn 0.30s" in editor.text
    assert "2/2 cache hit" in editor.text

    subtitle_video = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/renders/full",
        follow_redirects=False,
    )
    assert subtitle_video.status_code == 303
    started = test_client.post(
        f"/projects/{project_id}/videos/{video_id}/finalize",
        follow_redirects=False,
    )
    assert started.status_code == 303
    job_id = started.headers["location"].split("job=", 1)[1].split("#", 1)[0]
    worker = WorkerService(
        application.state.job_repository,
        data_dir,
    )
    assert worker.process_one() is True
    job = test_client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    assert job["progress"] == 100

    final_path = data_dir / "projects" / str(project_id) / "final" / str(video_id) / "final.mp4"
    timeline_path = final_path.with_name("vietnamese_timeline.wav")
    assert final_path.is_file()
    assert timeline_path.is_file()
    final_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(final_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert final_probe.stdout.strip().splitlines() == ["video", "audio"]
    downloaded = test_client.get(
        f"/projects/{project_id}/videos/{video_id}/final?download=true"
    )
    assert downloaded.status_code == 200
    assert "attachment" in downloaded.headers["content-disposition"]


def test_streaming_timeline_mixes_overlaps_and_clamps(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    output = tmp_path / "timeline.wav"
    write_constant_wav(first, 20_000, 1.0)
    write_constant_wav(second, 20_000, 1.0)

    reported_progress: list[float] = []
    build_audio_timeline(
        [
            TimelineSegment(0, first, "a" * 64),
            TimelineSegment(500, second, "b" * 64),
        ],
        2.0,
        output,
        tmp_path / "work",
        reported_progress.append,
    )
    with wave.open(str(output), "rb") as audio:
        samples = array("h")
        samples.frombytes(audio.readframes(audio.getnframes()))
        assert audio.getnframes() == 48_000
    assert samples[6_000] == 20_000
    assert samples[18_000] == 32_767
    assert samples[30_000] == 20_000
    assert samples[42_000] == 0
    assert reported_progress[-1] == pytest.approx(1.0)


def test_sqlite_job_queue_retries_and_rejects_second_job(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'app.db'}")
    database.create_schema()
    repository = JobRepository(database.session_factory)
    first = FinalJobSpec(
        job_id="a" * 32,
        project_id=1,
        video_id=1,
        video_path=tmp_path / "video.mp4",
        output_dir=tmp_path / "output",
        duration_seconds=1,
        segments=[],
    )
    second = FinalJobSpec(
        job_id="b" * 32,
        project_id=2,
        video_id=2,
        video_path=tmp_path / "video-2.mp4",
        output_dir=tmp_path / "output-2",
        duration_seconds=1,
        segments=[],
    )
    repository.enqueue_final(first)
    with pytest.raises(JobBusyError):
        repository.enqueue_final(second)

    for expected_attempt in range(1, 4):
        claimed = repository.claim_next()
        assert claimed is not None
        assert claimed.attempts == expected_attempt
        repository.fail_or_retry(first.job_id, "test failure")
    failed = repository.get(first.job_id)
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["attempts"] == 3

    retried = repository.retry(first.job_id)
    assert retried["status"] == "queued"
    assert retried["attempts"] == 0
    database.close()


def test_cleanup_only_removes_generated_temporary_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "projects" / "1" / "final" / "2"
    work_dir = output_dir / "work-job"
    work_dir.mkdir(parents=True)
    (work_dir / "chunk.wav").write_bytes(b"temporary")
    part_file = output_dir / ".final.part.mp4"
    part_file.write_bytes(b"temporary")
    keep_file = output_dir / "final.mp4"
    keep_file.write_bytes(b"keep")

    assert cleanup_temporary_files(tmp_path) == 2
    assert not work_dir.exists()
    assert not part_file.exists()
    assert keep_file.read_bytes() == b"keep"
