from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from .config import Settings


ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
CHUNK_SIZE = 1024 * 1024


class UploadError(ValueError):
    pass


def clean_original_name(filename: str | None) -> str:
    name = Path(filename or "video").name.strip()
    name = re.sub(r"[\x00-\x1f]", "", name)
    return name[:255] or "video"


async def save_upload(
    upload: UploadFile,
    project_id: int,
    settings: Settings,
) -> tuple[Path, str, str, int]:
    original_name = clean_original_name(upload.filename)
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UploadError(f"Định dạng không hỗ trợ. Chấp nhận: {allowed}.")

    project_dir = settings.projects_dir / str(project_id) / "videos"
    project_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}{suffix}"
    final_path = project_dir / stored_name
    partial_path = project_dir / f"{stored_name}.part"
    size = 0
    try:
        with partial_path.open("wb") as target:
            while chunk := await upload.read(CHUNK_SIZE):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise UploadError("Video vượt quá dung lượng upload cho phép.")
                target.write(chunk)
        if size == 0:
            raise UploadError("File upload đang trống.")
        partial_path.replace(final_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    relative_path = final_path.relative_to(settings.data_dir).as_posix()
    return final_path, relative_path, original_name, size


def resolve_stored_path(relative_path: str, settings: Settings) -> Path:
    root = settings.data_dir.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Đường dẫn video không hợp lệ.")
    return candidate

