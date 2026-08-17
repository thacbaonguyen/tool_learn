from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from .finalizer import FinalJobSpec, TimelineSegment
from .models import ProcessingJob
from .ocr_service import OcrJobSpec
from .pipeline import AutomaticVideoSpec
from .rendering import RenderCue, RenderStyle


class JobBusyError(RuntimeError):
    pass


def job_to_dict(job: ProcessingJob) -> dict:
    return {
        "job_id": job.id,
        "kind": job.kind,
        "project_id": job.project_id,
        "video_id": job.video_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
    }


def spec_to_payload(spec: FinalJobSpec) -> str:
    return json.dumps(
        {
            "video_path": str(spec.video_path),
            "output_dir": str(spec.output_dir),
            "duration_seconds": spec.duration_seconds,
            "segments": [
                {
                    "start_ms": segment.start_ms,
                    "audio_path": str(segment.audio_path),
                    "cache_key": segment.cache_key,
                }
                for segment in spec.segments
            ],
        }
    )


def payload_to_spec(job: ProcessingJob) -> FinalJobSpec:
    try:
        payload = json.loads(job.payload)
        return FinalJobSpec(
            job_id=job.id,
            project_id=job.project_id,
            video_id=job.video_id,
            video_path=Path(payload["video_path"]),
            output_dir=Path(payload["output_dir"]),
            duration_seconds=float(payload["duration_seconds"]),
            segments=[
                TimelineSegment(
                    start_ms=int(item["start_ms"]),
                    audio_path=Path(item["audio_path"]),
                    cache_key=str(item["cache_key"]),
                )
                for item in payload["segments"]
            ],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Payload job không hợp lệ.") from exc


def ocr_payload_to_spec(job: ProcessingJob) -> OcrJobSpec:
    try:
        payload = json.loads(job.payload)
        return OcrJobSpec(
            job_id=job.id,
            project_id=job.project_id,
            video_id=job.video_id,
            video_path=Path(payload["video_path"]),
            output_dir=Path(payload["output_dir"]),
            fps=float(payload.get("fps", 5.0)),
            bottom_ratio=float(payload.get("bottom_ratio", 0.35)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Payload OCR không hợp lệ.") from exc


def build_payload_to_spec(job: ProcessingJob) -> AutomaticVideoSpec:
    try:
        payload = json.loads(job.payload)
        return AutomaticVideoSpec(
            job_id=job.id,
            project_id=job.project_id,
            video_id=job.video_id,
            source_video=Path(payload["source_video"]),
            render_video=Path(payload["render_video"]),
            ass_path=Path(payload["ass_path"]),
            final_dir=Path(payload["final_dir"]),
            tts_dir=Path(payload["tts_dir"]),
            reference_audio=Path(payload["reference_audio"]),
            duration_seconds=float(payload["duration_seconds"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            style=RenderStyle(**payload["style"]),
            cues=[RenderCue(**cue) for cue in payload["cues"]],
            dictionary_text=str(payload.get("dictionary_text", "")),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Payload tạo video tự động không hợp lệ.") from exc


class JobRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def enqueue_final(self, spec: FinalJobSpec) -> dict:
        with self.session_factory() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            active = session.scalar(
                select(ProcessingJob.id).where(
                    ProcessingJob.status.in_(["queued", "running"])
                )
            )
            if active:
                session.rollback()
                raise JobBusyError("Một job khác đang chạy. Hãy chờ job đó hoàn thành.")
            job = ProcessingJob(
                id=spec.job_id,
                kind="finalize",
                project_id=spec.project_id,
                video_id=spec.video_id,
                status="queued",
                progress=0,
                message="Đang chờ worker",
                payload=spec_to_payload(spec),
            )
            session.add(job)
            session.commit()
            return job_to_dict(job)

    def enqueue_ocr(self, spec: OcrJobSpec) -> dict:
        with self.session_factory() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            active = session.scalar(
                select(ProcessingJob.id).where(
                    ProcessingJob.status.in_(["queued", "running"])
                )
            )
            if active:
                session.rollback()
                raise JobBusyError("Một job khác đang chạy. Hãy chờ job đó hoàn thành.")
            job = ProcessingJob(
                id=spec.job_id,
                kind="extract_subtitles",
                project_id=spec.project_id,
                video_id=spec.video_id,
                status="queued",
                progress=0,
                message="Đang chờ worker OCR",
                payload=json.dumps(
                    {
                        "video_path": str(spec.video_path),
                        "output_dir": str(spec.output_dir),
                        "fps": spec.fps,
                        "bottom_ratio": spec.bottom_ratio,
                    }
                ),
            )
            session.add(job)
            session.commit()
            return job_to_dict(job)

    def enqueue_build(self, spec: AutomaticVideoSpec) -> dict:
        with self.session_factory() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            active = session.scalar(
                select(ProcessingJob.id).where(
                    ProcessingJob.status.in_(["queued", "running"])
                )
            )
            if active:
                session.rollback()
                raise JobBusyError("Một job khác đang chạy. Hãy chờ job đó hoàn thành.")
            job = ProcessingJob(
                id=spec.job_id,
                kind="build_video",
                project_id=spec.project_id,
                video_id=spec.video_id,
                status="queued",
                progress=0,
                message="Đang chờ tạo video tiếng Việt",
                payload=json.dumps(
                    {
                        "source_video": str(spec.source_video),
                        "render_video": str(spec.render_video),
                        "ass_path": str(spec.ass_path),
                        "final_dir": str(spec.final_dir),
                        "tts_dir": str(spec.tts_dir),
                        "reference_audio": str(spec.reference_audio),
                        "duration_seconds": spec.duration_seconds,
                        "width": spec.width,
                        "height": spec.height,
                        "style": {
                            field: getattr(spec.style, field)
                            for field in spec.style.__dataclass_fields__
                        },
                        "cues": [
                            {
                                "start_ms": cue.start_ms,
                                "end_ms": cue.end_ms,
                                "text": cue.text,
                            }
                            for cue in spec.cues
                        ],
                        "dictionary_text": spec.dictionary_text,
                    },
                    ensure_ascii=False,
                ),
            )
            session.add(job)
            session.commit()
            return job_to_dict(job)

    def claim_next(self) -> ProcessingJob | None:
        with self.session_factory() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            running = session.scalar(
                select(ProcessingJob.id).where(ProcessingJob.status == "running")
            )
            if running:
                session.rollback()
                return None
            job = session.scalar(
                select(ProcessingJob)
                .where(ProcessingJob.status == "queued")
                .order_by(ProcessingJob.created_at.asc())
                .limit(1)
            )
            if job is None:
                session.rollback()
                return None
            job.status = "running"
            job.progress = max(1, job.progress)
            job.message = "Worker đã nhận job"
            job.error = None
            job.attempts += 1
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def update_progress(self, job_id: str, progress: int, message: str) -> None:
        with self.session_factory() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None or job.status != "running":
                return
            job.progress = max(job.progress, min(100, int(progress)))
            job.message = message[:255]
            session.commit()

    def complete(self, job_id: str) -> None:
        with self.session_factory() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                return
            job.status = "completed"
            job.progress = 100
            job.message = "Hoàn thành"
            job.error = None
            session.commit()

    def fail_or_retry(self, job_id: str, error: str) -> None:
        with self.session_factory() as session:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                return
            job.error = error
            if job.attempts < job.max_attempts:
                job.status = "queued"
                job.progress = 0
                job.message = f"Lỗi lần {job.attempts}; đang chờ retry"
            else:
                job.status = "failed"
                job.message = f"Thất bại sau {job.attempts} lần"
            session.commit()

    def retry(self, job_id: str) -> dict:
        with self.session_factory() as session:
            session.execute(text("BEGIN IMMEDIATE"))
            active = session.scalar(
                select(ProcessingJob.id).where(
                    ProcessingJob.status.in_(["queued", "running"])
                )
            )
            if active:
                session.rollback()
                raise JobBusyError("Một job khác đang chạy. Chưa thể retry.")
            job = session.get(ProcessingJob, job_id)
            if job is None or job.status != "failed":
                session.rollback()
                raise ValueError("Chỉ có thể retry job đã thất bại.")
            job.status = "queued"
            job.progress = 0
            job.message = "Đang chờ retry thủ công"
            job.error = None
            job.attempts = 0
            session.commit()
            return job_to_dict(job)

    def recover_interrupted(self) -> int:
        with self.session_factory() as session:
            jobs = session.scalars(
                select(ProcessingJob).where(ProcessingJob.status == "running")
            ).all()
            for job in jobs:
                job.status = "queued" if job.attempts < job.max_attempts else "failed"
                job.progress = 0
                job.message = "Worker trước bị gián đoạn; đang chờ retry"
                job.error = "Worker đã dừng trước khi job hoàn thành."
            session.commit()
            return len(jobs)

    def get(self, job_id: str) -> dict | None:
        with self.session_factory() as session:
            job = session.get(ProcessingJob, job_id)
            return job_to_dict(job) if job else None

    def latest_for_video(
        self,
        project_id: int,
        video_id: int,
        kind: str | list[str] | None = None,
    ) -> dict | None:
        with self.session_factory() as session:
            query = select(ProcessingJob).where(
                ProcessingJob.project_id == project_id,
                ProcessingJob.video_id == video_id,
            )
            if isinstance(kind, str):
                query = query.where(ProcessingJob.kind == kind)
            elif kind:
                query = query.where(ProcessingJob.kind.in_(kind))
            job = session.scalar(query.order_by(ProcessingJob.created_at.desc()).limit(1))
            return job_to_dict(job) if job else None

    def busy(self) -> bool:
        with self.session_factory() as session:
            return bool(
                session.scalar(
                    select(ProcessingJob.id).where(
                        ProcessingJob.status.in_(["queued", "running"])
                    )
                )
            )

    def active_for(self, project_id: int, video_id: int | None = None) -> bool:
        with self.session_factory() as session:
            query = select(ProcessingJob.id).where(
                ProcessingJob.project_id == project_id,
                ProcessingJob.status.in_(["queued", "running"]),
            )
            if video_id is not None:
                query = query.where(ProcessingJob.video_id == video_id)
            return bool(session.scalar(query.limit(1)))


def new_job_id() -> str:
    return uuid4().hex
