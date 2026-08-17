from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_upload_bytes: int = 10 * 1024 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("APP_DATA_DIR", "data")).resolve(),
            max_upload_bytes=int(
                os.getenv("APP_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024 * 1024))
            ),
        )

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'app.db'}"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

