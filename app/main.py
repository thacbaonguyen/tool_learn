from __future__ import annotations

import shutil
import re
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .config import Settings
from .database import Database
from .finalizer import (
    FinalJobSpec,
    TimelineSegment,
)
from .job_queue import JobBusyError, JobRepository, new_job_id
from .media import MediaProbeError, probe_video
from .models import Project, SubtitleCue, SubtitleTrack, Video
from .ocr_service import OcrJobSpec
from .pipeline import AutomaticVideoSpec
from .rendering import RenderCue, RenderError, RenderStyle, render_video
from .storage import UploadError, clean_original_name, resolve_stored_path, save_upload
from .subtitles import SrtParseError, parse_srt
from .tts_engine import (
    LocalTTSService,
    TTSError,
    extract_reference_audio,
    file_sha256,
    generate_tts_cues,
    load_json,
    parse_pronunciation_dictionary,
    save_json,
)


BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


templates.env.filters["filesize"] = format_bytes
templates.env.filters["duration"] = format_duration

FONT_FAMILIES = {"Arial", "Verdana", "Tahoma", "Georgia", "Courier New"}
MAX_SRT_BYTES = 2 * 1024 * 1024


class SubtitleStyleUpdate(BaseModel):
    x_ratio: float = Field(ge=0, le=1)
    y_ratio: float = Field(ge=0, le=1)
    width_ratio: float = Field(ge=0.1, le=1)
    height_ratio: float = Field(ge=0.05, le=1)
    font_family: str
    font_size_ratio: float = Field(ge=0.015, le=0.15)
    text_color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    background_color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")

    @field_validator("font_family")
    @classmethod
    def supported_font(cls, value: str) -> str:
        if value not in FONT_FAMILIES:
            raise ValueError("Font không được hỗ trợ.")
        return value

    @model_validator(mode="after")
    def box_inside_video(self) -> "SubtitleStyleUpdate":
        if self.x_ratio + self.width_ratio > 1.0001:
            raise ValueError("Khung subtitle vượt quá chiều rộng video.")
        if self.y_ratio + self.height_ratio > 1.0001:
            raise ValueError("Khung subtitle vượt quá chiều cao video.")
        return self


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings.from_env()
    app_settings.data_dir.mkdir(parents=True, exist_ok=True)
    app_settings.projects_dir.mkdir(parents=True, exist_ok=True)
    database = Database(app_settings.database_url)
    job_repository = JobRepository(database.session_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.create_schema()
        yield
        database.close()

    app = FastAPI(title="Local Video Lessons", lifespan=lifespan)
    app.state.settings = app_settings
    app.state.database = database
    app.state.tts_service = LocalTTSService()
    app.state.job_repository = job_repository
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    def get_session() -> Iterator[Session]:
        yield from database.sessions()

    def find_project(session: Session, project_id: int) -> Project:
        project = session.scalar(
            select(Project)
            .where(Project.id == project_id)
            .options(selectinload(Project.videos))
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy project.")
        return project

    def find_video(session: Session, project_id: int, video_id: int) -> Video:
        video = session.scalar(
            select(Video)
            .where(Video.id == video_id, Video.project_id == project_id)
            .options(
                selectinload(Video.subtitle_track).selectinload(SubtitleTrack.cues)
            )
        )
        if video is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy video.")
        return video

    def automatic_video_spec(
        video: Video,
        track: SubtitleTrack,
        job_id: str,
    ) -> AutomaticVideoSpec:
        project_id = video.project_id
        tts_dir = app_settings.projects_dir / str(project_id) / "tts" / str(video.id)
        local_reference = tts_dir / "reference.wav"
        global_reference = app_settings.data_dir / "voice" / "reference.wav"
        reference = local_reference if local_reference.is_file() else global_reference
        if not reference.is_file():
            raise TTSError(
                "Chưa có giọng mẫu. Chép file vào data/voice rồi chạy sudo make voice."
            )
        dictionary_path = tts_dir / "pronunciation.txt"
        render_dir = (
            app_settings.projects_dir / str(project_id) / "renders" / str(video.id)
        )
        return AutomaticVideoSpec(
            job_id=job_id,
            project_id=project_id,
            video_id=video.id,
            source_video=resolve_stored_path(video.storage_path, app_settings),
            render_video=render_dir / "full.mp4",
            ass_path=render_dir / "full.ass",
            final_dir=(
                app_settings.projects_dir / str(project_id) / "final" / str(video.id)
            ),
            tts_dir=tts_dir,
            reference_audio=reference,
            duration_seconds=video.duration_seconds,
            width=video.width,
            height=video.height,
            style=RenderStyle(
                x_ratio=track.x_ratio,
                y_ratio=track.y_ratio,
                width_ratio=track.width_ratio,
                height_ratio=track.height_ratio,
                font_family=track.font_family,
                font_size_ratio=track.font_size_ratio,
                text_color=track.text_color,
                background_color=track.background_color,
            ),
            cues=[
                RenderCue(
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    text=cue.text,
                )
                for cue in track.cues
            ],
            dictionary_text=(
                dictionary_path.read_text(encoding="utf-8")
                if dictionary_path.is_file()
                else ""
            ),
        )

    def render_project(
        request: Request,
        project: Project,
        *,
        error: str | None = None,
        response_status: int = 200,
    ) -> HTMLResponse:
        ocr_outputs = {}
        for video in project.videos:
            output_dir = (
                app_settings.projects_dir
                / str(project.id)
                / "subtitles"
                / str(video.id)
            )
            srt_path = output_dir / "english.srt"
            ocr_outputs[video.id] = {
                "exists": srt_path.is_file(),
                "size": srt_path.stat().st_size if srt_path.is_file() else 0,
                "metrics": load_json(output_dir / "ocr_metrics.json"),
                "job": job_repository.latest_for_video(
                    project.id, video.id, "extract_subtitles"
                ),
            }
        return templates.TemplateResponse(
            request=request,
            name="project_detail.html",
            context={
                "project": project,
                "ocr_outputs": ocr_outputs,
                "job_queue_busy": job_repository.busy(),
                "error": error,
            },
            status_code=response_status,
        )

    def render_subtitle_editor(
        request: Request,
        project: Project,
        video: Video,
        *,
        error: str | None = None,
        response_status: int = 200,
    ) -> HTMLResponse:
        track = video.subtitle_track
        cues = [
            {
                "sequence": cue.sequence,
                "start_ms": cue.start_ms,
                "end_ms": cue.end_ms,
                "text": cue.text,
            }
            for cue in (track.cues if track else [])
        ]
        render_dir = app_settings.projects_dir / str(project.id) / "renders" / str(video.id)
        render_outputs = {}
        for render_kind in ("preview", "full"):
            render_path = render_dir / f"{render_kind}.mp4"
            render_outputs[render_kind] = {
                "exists": render_path.is_file(),
                "version": render_path.stat().st_mtime_ns if render_path.is_file() else 0,
                "size": render_path.stat().st_size if render_path.is_file() else 0,
            }
        tts_dir = app_settings.projects_dir / str(project.id) / "tts" / str(video.id)
        local_reference_path = tts_dir / "reference.wav"
        global_reference_path = app_settings.data_dir / "voice" / "reference.wav"
        reference_path = (
            local_reference_path
            if local_reference_path.is_file()
            else global_reference_path
        )
        reference_metadata_path = (
            tts_dir / "reference.json"
            if local_reference_path.is_file()
            else app_settings.data_dir / "voice" / "reference.json"
        )
        reference_metadata = load_json(reference_metadata_path)
        if not reference_path.is_file():
            reference_metadata = {}
        dictionary_path = tts_dir / "pronunciation.txt"
        pronunciation_dictionary = (
            dictionary_path.read_text(encoding="utf-8")
            if dictionary_path.is_file()
            else "# Mỗi dòng: từ=cách đọc\nAWS=ây ét ét\n"
        )
        tts_manifest = load_json(tts_dir / "manifest.json")
        manifest_matches_voice = False
        if reference_path.is_file():
            manifest_matches_voice = (
                tts_manifest.get("reference_hash") == file_sha256(reference_path)
            )
        cue_texts = {cue.sequence: cue.text for cue in (track.cues if track else [])}
        tts_items = {}
        for item in tts_manifest.get("items", []):
            sequence = item.get("sequence")
            cache_key = item.get("cache_key", "")
            cached_audio = tts_dir / "cache" / f"{cache_key}.wav"
            if (
                manifest_matches_voice
                and
                cue_texts.get(sequence) == item.get("original_text")
                and cached_audio.is_file()
            ):
                tts_items[sequence] = item
        final_dir = app_settings.projects_dir / str(project.id) / "final" / str(video.id)
        final_path = final_dir / "final.mp4"
        requested_job_id = request.query_params.get("job")
        final_job = (
            job_repository.get(requested_job_id)
            if requested_job_id and re.fullmatch(r"[0-9a-f]{32}", requested_job_id)
            else job_repository.latest_for_video(
                project.id, video.id, ["build_video", "finalize"]
            )
        )
        return templates.TemplateResponse(
            request=request,
            name="subtitle_editor.html",
            context={
                "project": project,
                "video": video,
                "track": track,
                "cues": cues,
                "font_families": sorted(FONT_FAMILIES),
                "render_outputs": render_outputs,
                "tts_reference": reference_metadata,
                "tts_reference_scope": (
                    "video" if local_reference_path.is_file() else "global"
                ) if reference_path.is_file() else None,
                "tts_has_reference": reference_path.is_file(),
                "tts_dictionary": pronunciation_dictionary,
                "tts_manifest": tts_manifest,
                "tts_items": tts_items,
                "tts_ready": bool(cues) and len(tts_items) == len(cues),
                "final_job": final_job,
                "final_runner_busy": job_repository.busy(),
                "final_output": {
                    "exists": final_path.is_file(),
                    "size": final_path.stat().st_size if final_path.is_file() else 0,
                    "version": final_path.stat().st_mtime_ns if final_path.is_file() else 0,
                },
                "error": error,
            },
            status_code=response_status,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/projects", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @app.get("/projects", response_class=HTMLResponse)
    def list_projects(
        request: Request,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        projects = session.scalars(
            select(Project)
            .options(selectinload(Project.videos))
            .order_by(Project.updated_at.desc())
        ).all()
        return templates.TemplateResponse(
            request=request,
            name="projects.html",
            context={"projects": projects, "error": None},
        )

    @app.post("/projects")
    def create_project(
        request: Request,
        name: str = Form(...),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        clean_name = name.strip()
        if not clean_name or len(clean_name) > 120:
            projects = session.scalars(
                select(Project)
                .options(selectinload(Project.videos))
                .order_by(Project.updated_at.desc())
            ).all()
            return templates.TemplateResponse(
                request=request,
                name="projects.html",
                context={
                    "projects": projects,
                    "error": "Tên project phải có từ 1 đến 120 ký tự.",
                },
                status_code=400,
            )
        project = Project(name=clean_name)
        session.add(project)
        session.commit()
        return RedirectResponse(
            f"/projects/{project.id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @app.get("/projects/{project_id}", response_class=HTMLResponse)
    def project_detail(
        request: Request,
        project_id: int,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        return render_project(request, find_project(session, project_id))

    @app.post("/projects/{project_id}/rename")
    def rename_project(
        request: Request,
        project_id: int,
        name: str = Form(...),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        clean_name = name.strip()
        if not clean_name or len(clean_name) > 120:
            return render_project(
                request,
                project,
                error="Tên project phải có từ 1 đến 120 ký tự.",
                response_status=400,
            )
        project.name = clean_name
        session.commit()
        return RedirectResponse(
            f"/projects/{project.id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @app.post("/projects/{project_id}/delete")
    def delete_project(
        project_id: int,
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        project = find_project(session, project_id)
        if job_repository.active_for(project_id):
            raise HTTPException(
                status_code=409,
                detail="Không thể xóa project khi worker đang xử lý video của project.",
            )
        project_dir = app_settings.projects_dir / str(project.id)
        session.delete(project)
        session.commit()
        if project_dir.is_dir() and project_dir.resolve().is_relative_to(
            app_settings.projects_dir.resolve()
        ):
            shutil.rmtree(project_dir)
        return RedirectResponse("/projects", status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/projects/{project_id}/videos")
    async def upload_video(
        request: Request,
        project_id: int,
        video_file: UploadFile,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        saved_path: Path | None = None
        try:
            saved_path, relative_path, original_name, size = await save_upload(
                video_file, project.id, app_settings
            )
            metadata = probe_video(saved_path)
            video = Video(
                project_id=project.id,
                original_filename=original_name,
                storage_path=relative_path,
                size_bytes=size,
                duration_seconds=metadata.duration_seconds,
                width=metadata.width,
                height=metadata.height,
                fps=metadata.fps,
                video_codec=metadata.video_codec,
                audio_codec=metadata.audio_codec,
            )
            session.add(video)
            session.commit()
        except (UploadError, MediaProbeError) as exc:
            if saved_path:
                saved_path.unlink(missing_ok=True)
            session.rollback()
            session.expire_all()
            project = find_project(session, project_id)
            return render_project(
                request, project, error=str(exc), response_status=400
            )
        except Exception:
            if saved_path:
                saved_path.unlink(missing_ok=True)
            session.rollback()
            raise
        return RedirectResponse(
            f"/projects/{project.id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @app.get("/projects/{project_id}/videos/{video_id}/stream")
    def stream_video(
        project_id: int,
        video_id: int,
        session: Session = Depends(get_session),
    ) -> FileResponse:
        video = session.scalar(
            select(Video).where(
                Video.id == video_id,
                Video.project_id == project_id,
            )
        )
        if video is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy video.")
        path = resolve_stored_path(video.storage_path, app_settings)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File video không còn tồn tại.")
        return FileResponse(path)

    @app.post("/projects/{project_id}/videos/{video_id}/extract-subtitles")
    def extract_english_subtitles(
        request: Request,
        project_id: int,
        video_id: int,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        video = find_video(session, project_id, video_id)
        spec = OcrJobSpec(
            job_id=new_job_id(),
            project_id=project_id,
            video_id=video_id,
            video_path=resolve_stored_path(video.storage_path, app_settings),
            output_dir=(
                app_settings.projects_dir
                / str(project_id)
                / "subtitles"
                / str(video_id)
            ),
        )
        try:
            job_repository.enqueue_ocr(spec)
        except JobBusyError as exc:
            return render_project(
                request, project, error=str(exc), response_status=409
            )
        return RedirectResponse(
            f"/projects/{project_id}#video-{video_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/projects/{project_id}/videos/{video_id}/english-subtitles")
    def download_english_subtitles(
        project_id: int,
        video_id: int,
        session: Session = Depends(get_session),
    ) -> FileResponse:
        video = find_video(session, project_id, video_id)
        path = (
            app_settings.projects_dir
            / str(project_id)
            / "subtitles"
            / str(video_id)
            / "english.srt"
        )
        if not path.is_file():
            raise HTTPException(status_code=404, detail="SRT tiếng Anh chưa được tạo.")
        return FileResponse(
            path,
            media_type="application/x-subrip",
            filename=f"{Path(video.original_filename).stem}-en.srt",
        )

    @app.get(
        "/projects/{project_id}/videos/{video_id}/subtitles",
        response_class=HTMLResponse,
    )
    def subtitle_editor(
        request: Request,
        project_id: int,
        video_id: int,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        video = find_video(session, project_id, video_id)
        return render_subtitle_editor(request, project, video)

    @app.post("/projects/{project_id}/videos/{video_id}/subtitles/import")
    async def import_subtitles(
        request: Request,
        project_id: int,
        video_id: int,
        srt_file: UploadFile,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        video = find_video(session, project_id, video_id)
        try:
            filename = clean_original_name(srt_file.filename)
            if Path(filename).suffix.lower() != ".srt":
                raise SrtParseError("Vui lòng chọn file có đuôi .srt.")
            raw_content = await srt_file.read(MAX_SRT_BYTES + 1)
            if len(raw_content) > MAX_SRT_BYTES:
                raise SrtParseError("File SRT vượt quá giới hạn 2 MB.")
            try:
                content = raw_content.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise SrtParseError("File SRT phải dùng mã hóa UTF-8.") from exc
            parsed_cues = parse_srt(content)
        except SrtParseError as exc:
            return render_subtitle_editor(
                request, project, video, error=str(exc), response_status=400
            )
        finally:
            await srt_file.close()

        track = video.subtitle_track
        if track is None:
            track = SubtitleTrack(video=video, source_filename=filename)
            session.add(track)
        else:
            track.source_filename = filename
            track.cues.clear()
        track.cues.extend(
            SubtitleCue(
                sequence=cue.sequence,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=cue.text,
            )
            for cue in parsed_cues
        )
        session.commit()
        job_id = new_job_id()
        try:
            spec = automatic_video_spec(video, track, job_id)
            job_repository.enqueue_build(spec)
        except TTSError:
            return RedirectResponse(
                f"/projects/{project_id}/videos/{video_id}/subtitles#final-output",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except JobBusyError as exc:
            return render_subtitle_editor(
                request, project, video, error=str(exc), response_status=409
            )
        return RedirectResponse(
            f"/projects/{project_id}/videos/{video_id}/subtitles"
            f"?job={job_id}#final-output",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/projects/{project_id}/videos/{video_id}/build")
    def build_video_automatically(
        request: Request,
        project_id: int,
        video_id: int,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        video = find_video(session, project_id, video_id)
        track = video.subtitle_track
        if track is None or not track.cues:
            return render_subtitle_editor(
                request, project, video, error="Hãy import SRT trước.", response_status=400
            )
        job_id = new_job_id()
        try:
            job_repository.enqueue_build(automatic_video_spec(video, track, job_id))
        except (TTSError, JobBusyError) as exc:
            return render_subtitle_editor(
                request, project, video, error=str(exc), response_status=409
            )
        return RedirectResponse(
            f"/projects/{project_id}/videos/{video_id}/subtitles"
            f"?job={job_id}#final-output",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.put("/projects/{project_id}/videos/{video_id}/subtitles/style")
    def update_subtitle_style(
        project_id: int,
        video_id: int,
        style: SubtitleStyleUpdate,
        session: Session = Depends(get_session),
    ) -> dict[str, str]:
        video = find_video(session, project_id, video_id)
        track = video.subtitle_track
        if track is None:
            raise HTTPException(status_code=409, detail="Hãy import SRT trước.")
        for field_name, value in style.model_dump().items():
            setattr(track, field_name, value)
        session.commit()
        return {"status": "saved"}

    def video_tts_dir(project_id: int, video_id: int) -> Path:
        return app_settings.projects_dir / str(project_id) / "tts" / str(video_id)

    @app.post("/projects/{project_id}/videos/{video_id}/tts/reference")
    def save_tts_reference(
        request: Request,
        project_id: int,
        video_id: int,
        start_seconds: float = Form(...),
        duration_seconds: float = Form(...),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        video = find_video(session, project_id, video_id)
        if not video.audio_codec:
            return render_subtitle_editor(
                request, project, video, error="Video không có audio gốc.", response_status=400
            )
        if start_seconds < 0 or start_seconds >= video.duration_seconds:
            return render_subtitle_editor(
                request,
                project,
                video,
                error="Thời điểm bắt đầu audio mẫu nằm ngoài video.",
                response_status=400,
            )
        if duration_seconds < 3 or duration_seconds > 30:
            return render_subtitle_editor(
                request,
                project,
                video,
                error="Đoạn giọng mẫu phải dài từ 3 đến 30 giây.",
                response_status=400,
            )
        actual_duration = min(duration_seconds, video.duration_seconds - start_seconds)
        if actual_duration < 3:
            return render_subtitle_editor(
                request,
                project,
                video,
                error="Đoạn audio còn lại quá ngắn để làm giọng mẫu.",
                response_status=400,
            )
        tts_dir = video_tts_dir(project_id, video_id)
        try:
            extract_reference_audio(
                resolve_stored_path(video.storage_path, app_settings),
                tts_dir / "reference.wav",
                start_seconds=start_seconds,
                duration_seconds=actual_duration,
            )
        except TTSError as exc:
            return render_subtitle_editor(
                request, project, video, error=str(exc), response_status=500
            )
        save_json(
            tts_dir / "reference.json",
            {
                "start_seconds": round(start_seconds, 3),
                "duration_seconds": round(actual_duration, 3),
                "scope": "video",
            },
        )
        (tts_dir / "manifest.json").unlink(missing_ok=True)
        return RedirectResponse(
            f"/projects/{project_id}/videos/{video_id}/subtitles#tts-section",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/projects/{project_id}/videos/{video_id}/tts/reference")
    def stream_tts_reference(
        project_id: int,
        video_id: int,
        session: Session = Depends(get_session),
    ) -> FileResponse:
        find_video(session, project_id, video_id)
        local_path = video_tts_dir(project_id, video_id) / "reference.wav"
        global_path = app_settings.data_dir / "voice" / "reference.wav"
        path = local_path if local_path.is_file() else global_path
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Chưa có audio mẫu.")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/projects/{project_id}/videos/{video_id}/tts/dictionary")
    def save_tts_dictionary(
        request: Request,
        project_id: int,
        video_id: int,
        dictionary_text: str = Form(""),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        video = find_video(session, project_id, video_id)
        try:
            parse_pronunciation_dictionary(dictionary_text)
        except TTSError as exc:
            return render_subtitle_editor(
                request, project, video, error=str(exc), response_status=400
            )
        tts_dir = video_tts_dir(project_id, video_id)
        tts_dir.mkdir(parents=True, exist_ok=True)
        dictionary_path = tts_dir / "pronunciation.txt"
        previous = dictionary_path.read_text(encoding="utf-8") if dictionary_path.is_file() else None
        dictionary_path.write_text(dictionary_text, encoding="utf-8")
        if previous != dictionary_text:
            (tts_dir / "manifest.json").unlink(missing_ok=True)
        return RedirectResponse(
            f"/projects/{project_id}/videos/{video_id}/subtitles#tts-section",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/projects/{project_id}/videos/{video_id}/tts/generate")
    def generate_video_tts(
        request: Request,
        project_id: int,
        video_id: int,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        video = find_video(session, project_id, video_id)
        track = video.subtitle_track
        if track is None or not track.cues:
            return render_subtitle_editor(
                request, project, video, error="Hãy import SRT trước.", response_status=400
            )
        tts_dir = video_tts_dir(project_id, video_id)
        dictionary_path = tts_dir / "pronunciation.txt"
        dictionary_text = (
            dictionary_path.read_text(encoding="utf-8")
            if dictionary_path.is_file()
            else ""
        )
        try:
            dictionary = parse_pronunciation_dictionary(dictionary_text)
            manifest = generate_tts_cues(
                [
                    {
                        "sequence": cue.sequence,
                        "start_ms": cue.start_ms,
                        "end_ms": cue.end_ms,
                        "text": cue.text,
                    }
                    for cue in track.cues
                ],
                (
                    tts_dir / "reference.wav"
                    if (tts_dir / "reference.wav").is_file()
                    else app_settings.data_dir / "voice" / "reference.wav"
                ),
                dictionary,
                tts_dir / "cache",
                app.state.tts_service,
            )
            save_json(tts_dir / "manifest.json", manifest)
        except TTSError as exc:
            return render_subtitle_editor(
                request, project, video, error=str(exc), response_status=500
            )
        return RedirectResponse(
            f"/projects/{project_id}/videos/{video_id}/subtitles#tts-section",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/projects/{project_id}/videos/{video_id}/tts/audio/{cache_key}")
    def stream_tts_cue(
        project_id: int,
        video_id: int,
        cache_key: str,
        session: Session = Depends(get_session),
    ) -> FileResponse:
        find_video(session, project_id, video_id)
        if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
            raise HTTPException(status_code=404, detail="Cache key không hợp lệ.")
        path = video_tts_dir(project_id, video_id) / "cache" / f"{cache_key}.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Audio cache không tồn tại.")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/projects/{project_id}/videos/{video_id}/finalize")
    def start_final_job(
        request: Request,
        project_id: int,
        video_id: int,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        video = find_video(session, project_id, video_id)
        track = video.subtitle_track
        if track is None or not track.cues:
            return render_subtitle_editor(
                request, project, video, error="Hãy import SRT trước.", response_status=400
            )
        render_video_path = (
            app_settings.projects_dir
            / str(project_id)
            / "renders"
            / str(video_id)
            / "full.mp4"
        )
        if not render_video_path.is_file():
            return render_subtitle_editor(
                request,
                project,
                video,
                error="Hãy render toàn bộ video có subtitle Việt ở Phase 4 trước.",
                response_status=400,
            )
        tts_dir = video_tts_dir(project_id, video_id)
        manifest = load_json(tts_dir / "manifest.json")
        manifest_items = {
            item.get("sequence"): item for item in manifest.get("items", [])
        }
        segments: list[TimelineSegment] = []
        for cue in track.cues:
            item = manifest_items.get(cue.sequence)
            if not item or item.get("original_text") != cue.text:
                return render_subtitle_editor(
                    request,
                    project,
                    video,
                    error=f"Cue {cue.sequence} chưa có audio TTS hợp lệ.",
                    response_status=400,
                )
            cache_key = str(item.get("cache_key", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
                return render_subtitle_editor(
                    request,
                    project,
                    video,
                    error=f"Cache audio cue {cue.sequence} không hợp lệ.",
                    response_status=400,
                )
            audio_path = tts_dir / "cache" / f"{cache_key}.wav"
            if not audio_path.is_file():
                return render_subtitle_editor(
                    request,
                    project,
                    video,
                    error=f"Thiếu file audio cho cue {cue.sequence}.",
                    response_status=400,
                )
            segments.append(
                TimelineSegment(
                    start_ms=cue.start_ms,
                    audio_path=audio_path,
                    cache_key=cache_key,
                )
            )
        job_id = new_job_id()
        spec = FinalJobSpec(
            job_id=job_id,
            project_id=project_id,
            video_id=video_id,
            video_path=render_video_path,
            output_dir=(
                app_settings.projects_dir / str(project_id) / "final" / str(video_id)
            ),
            duration_seconds=video.duration_seconds,
            segments=segments,
        )
        try:
            job_repository.enqueue_final(spec)
        except JobBusyError as exc:
            return render_subtitle_editor(
                request, project, video, error=str(exc), response_status=409
            )
        return RedirectResponse(
            f"/projects/{project_id}/videos/{video_id}/subtitles?job={job_id}#final-output",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/jobs/{job_id}")
    def final_job_status(job_id: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(status_code=404, detail="Job không hợp lệ.")
        state = job_repository.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy job.")
        return state

    @app.post("/jobs/{job_id}/retry")
    def retry_final_job(job_id: str) -> RedirectResponse:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(status_code=404, detail="Job không hợp lệ.")
        try:
            state = job_repository.retry(job_id)
        except JobBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if state["kind"] == "extract_subtitles":
            location = f"/projects/{state['project_id']}#video-{state['video_id']}"
        else:
            location = (
                f"/projects/{state['project_id']}/videos/{state['video_id']}"
                f"/subtitles?job={job_id}#final-output"
            )
        return RedirectResponse(location, status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/projects/{project_id}/videos/{video_id}/final")
    def stream_final_video(
        project_id: int,
        video_id: int,
        download: bool = False,
        session: Session = Depends(get_session),
    ) -> FileResponse:
        video = find_video(session, project_id, video_id)
        path = (
            app_settings.projects_dir / str(project_id) / "final" / str(video_id) / "final.mp4"
        )
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Video cuối chưa được tạo.")
        filename = f"{Path(video.original_filename).stem}-vi-final.mp4"
        return FileResponse(path, filename=filename if download else None)

    def execute_render(
        project_id: int,
        video: Video,
        render_kind: str,
        *,
        start_seconds: float = 0,
        duration_seconds: float | None = None,
    ) -> None:
        track = video.subtitle_track
        if track is None or not track.cues:
            raise RenderError("Hãy import SRT trước khi render.")
        source = resolve_stored_path(video.storage_path, app_settings)
        if not source.is_file():
            raise RenderError("File video gốc không còn tồn tại.")
        render_dir = app_settings.projects_dir / str(project_id) / "renders" / str(video.id)
        style = RenderStyle(
            x_ratio=track.x_ratio,
            y_ratio=track.y_ratio,
            width_ratio=track.width_ratio,
            height_ratio=track.height_ratio,
            font_family=track.font_family,
            font_size_ratio=track.font_size_ratio,
            text_color=track.text_color,
            background_color=track.background_color,
        )
        cues = [
            RenderCue(start_ms=cue.start_ms, end_ms=cue.end_ms, text=cue.text)
            for cue in track.cues
        ]
        render_video(
            source,
            render_dir / f"{render_kind}.mp4",
            render_dir / f"{render_kind}.ass",
            width=video.width,
            height=video.height,
            style=style,
            cues=cues,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )

    @app.post("/projects/{project_id}/videos/{video_id}/renders/preview")
    def render_preview(
        request: Request,
        project_id: int,
        video_id: int,
        start_seconds: float = Form(0),
        duration_seconds: float = Form(15),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        video = find_video(session, project_id, video_id)
        if start_seconds < 0 or start_seconds >= video.duration_seconds:
            return render_subtitle_editor(
                request,
                project,
                video,
                error="Thời điểm bắt đầu preview nằm ngoài video.",
                response_status=400,
            )
        if duration_seconds < 1 or duration_seconds > 60:
            return render_subtitle_editor(
                request,
                project,
                video,
                error="Preview phải dài từ 1 đến 60 giây.",
                response_status=400,
            )
        actual_duration = min(duration_seconds, video.duration_seconds - start_seconds)
        try:
            execute_render(
                project_id,
                video,
                "preview",
                start_seconds=start_seconds,
                duration_seconds=actual_duration,
            )
        except RenderError as exc:
            return render_subtitle_editor(
                request, project, video, error=str(exc), response_status=500
            )
        return RedirectResponse(
            f"/projects/{project_id}/videos/{video_id}/subtitles#render-output",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/projects/{project_id}/videos/{video_id}/renders/full")
    def render_full_video(
        request: Request,
        project_id: int,
        video_id: int,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        project = find_project(session, project_id)
        video = find_video(session, project_id, video_id)
        try:
            execute_render(project_id, video, "full")
        except RenderError as exc:
            return render_subtitle_editor(
                request, project, video, error=str(exc), response_status=500
            )
        return RedirectResponse(
            f"/projects/{project_id}/videos/{video_id}/subtitles#render-output",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/projects/{project_id}/videos/{video_id}/renders/{render_kind}")
    def stream_render(
        project_id: int,
        video_id: int,
        render_kind: str,
        download: bool = False,
        session: Session = Depends(get_session),
    ) -> FileResponse:
        video = find_video(session, project_id, video_id)
        if render_kind not in {"preview", "full"}:
            raise HTTPException(status_code=404, detail="Output không hợp lệ.")
        path = (
            app_settings.projects_dir
            / str(project_id)
            / "renders"
            / str(video.id)
            / f"{render_kind}.mp4"
        )
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Output chưa được render.")
        filename = f"{Path(video.original_filename).stem}-{render_kind}-vi.mp4"
        return FileResponse(path, filename=filename if download else None)

    @app.post("/projects/{project_id}/videos/{video_id}/delete")
    def delete_video(
        project_id: int,
        video_id: int,
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        video = session.scalar(
            select(Video).where(
                Video.id == video_id,
                Video.project_id == project_id,
            )
        )
        if video is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy video.")
        if job_repository.active_for(project_id, video_id):
            raise HTTPException(
                status_code=409,
                detail="Không thể xóa video khi worker đang xử lý video này.",
            )
        path = resolve_stored_path(video.storage_path, app_settings)
        render_dir = (
            app_settings.projects_dir / str(project_id) / "renders" / str(video.id)
        )
        tts_dir = video_tts_dir(project_id, video.id)
        final_dir = (
            app_settings.projects_dir / str(project_id) / "final" / str(video.id)
        )
        english_subtitle_dir = (
            app_settings.projects_dir
            / str(project_id)
            / "subtitles"
            / str(video.id)
        )
        session.delete(video)
        session.commit()
        path.unlink(missing_ok=True)
        if render_dir.is_dir() and render_dir.resolve().is_relative_to(
            app_settings.projects_dir.resolve()
        ):
            shutil.rmtree(render_dir)
        for generated_dir in (tts_dir, final_dir, english_subtitle_dir):
            if generated_dir.is_dir() and generated_dir.resolve().is_relative_to(
                app_settings.projects_dir.resolve()
            ):
                shutil.rmtree(generated_dir)
        return RedirectResponse(
            f"/projects/{project_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    return app


app = create_app()
