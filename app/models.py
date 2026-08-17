from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)
    videos: Mapped[list["Video"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    original_filename: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(String(500), unique=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    duration_seconds: Mapped[float] = mapped_column(Float)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    fps: Mapped[float] = mapped_column(Float)
    video_codec: Mapped[str] = mapped_column(String(50))
    audio_codec: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    project: Mapped[Project] = relationship(back_populates="videos")
    subtitle_track: Mapped["SubtitleTrack | None"] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


class SubtitleTrack(Base):
    __tablename__ = "subtitle_tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), unique=True, index=True
    )
    source_filename: Mapped[str] = mapped_column(String(255))
    x_ratio: Mapped[float] = mapped_column(Float, default=0.1)
    y_ratio: Mapped[float] = mapped_column(Float, default=0.78)
    width_ratio: Mapped[float] = mapped_column(Float, default=0.8)
    height_ratio: Mapped[float] = mapped_column(Float, default=0.15)
    font_family: Mapped[str] = mapped_column(String(50), default="Arial")
    font_size_ratio: Mapped[float] = mapped_column(Float, default=0.045)
    text_color: Mapped[str] = mapped_column(String(7), default="#ffffff")
    background_color: Mapped[str] = mapped_column(String(7), default="#075e54")
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)
    video: Mapped[Video] = relationship(back_populates="subtitle_track")
    cues: Mapped[list["SubtitleCue"]] = relationship(
        back_populates="track",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SubtitleCue.sequence",
    )


class SubtitleCue(Base):
    __tablename__ = "subtitle_cues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    track_id: Mapped[int] = mapped_column(
        ForeignKey("subtitle_tracks.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    track: Mapped[SubtitleTrack] = relationship(back_populates="cues")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(30), index=True)
    project_id: Mapped[int] = mapped_column(Integer, index=True)
    video_id: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(String(255), default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    payload: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

