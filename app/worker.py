from __future__ import annotations

import argparse
import fcntl
import json
import logging
import shutil
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread

from .config import Settings
from .database import Database
from .finalizer import FinalJobSpec, execute_final_job
from .job_queue import (
    JobRepository,
    build_payload_to_spec,
    ocr_payload_to_spec,
    payload_to_spec,
)
from .ocr_service import OcrJobSpec, execute_ocr_job
from .pipeline import AutomaticVideoPipeline, AutomaticVideoSpec


LOGGER = logging.getLogger("local-video-worker")


def cleanup_temporary_files(data_dir: Path) -> int:
    """Only remove temporary names created by this application."""
    removed = 0
    projects_dir = data_dir / "projects"
    if projects_dir.is_dir():
        for project_dir in projects_dir.iterdir():
            if not project_dir.is_dir() or not project_dir.name.isdigit():
                continue
            for category in ("renders", "tts", "final", "subtitles"):
                root = project_dir / category
                if not root.is_dir():
                    continue
                for path in sorted(root.rglob("*"), reverse=True):
                    if path.is_dir() and path.name.startswith("work-"):
                        shutil.rmtree(path, ignore_errors=True)
                        removed += 1
                    elif path.is_file() and (
                        path.name.endswith(".part") or ".part." in path.name
                    ):
                        path.unlink(missing_ok=True)
                        removed += 1
    return removed


class WorkerService:
    def __init__(
        self,
        repository: JobRepository,
        data_dir: Path,
        executor: Callable[
            [FinalJobSpec, Callable[[int, str], None]], None
        ] = execute_final_job,
        ocr_executor: Callable[
            [OcrJobSpec, Callable[[int, str], None]], None
        ] = execute_ocr_job,
        pipeline_executor: Callable[
            [AutomaticVideoSpec, Callable[[int, str], None]], None
        ] | None = None,
    ) -> None:
        self.repository = repository
        self.data_dir = data_dir
        self.executor = executor
        self.ocr_executor = ocr_executor
        self.pipeline_executor = (
            pipeline_executor or AutomaticVideoPipeline().execute
        )
        self.stop_event = Event()
        self.current_job_id: str | None = None

    def write_heartbeat(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.data_dir / "worker-heartbeat.json"
        temporary = path.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(
                {"timestamp": time.time(), "job_id": self.current_job_id},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    def process_one(self) -> bool:
        job = self.repository.claim_next()
        if job is None:
            return False
        self.current_job_id = job.id
        self.write_heartbeat()
        heartbeat_stop = Event()

        def keep_heartbeat_alive() -> None:
            while not heartbeat_stop.wait(10):
                self.write_heartbeat()

        heartbeat_thread = Thread(target=keep_heartbeat_alive, daemon=True)
        heartbeat_thread.start()
        try:
            def update(progress: int, message: str) -> None:
                self.repository.update_progress(job.id, progress, message)

            if job.kind == "finalize":
                self.executor(payload_to_spec(job), update)
            elif job.kind == "extract_subtitles":
                self.ocr_executor(ocr_payload_to_spec(job), update)
            elif job.kind == "build_video":
                self.pipeline_executor(build_payload_to_spec(job), update)
            else:
                raise ValueError(f"Loại job không được hỗ trợ: {job.kind}")
            self.repository.complete(job.id)
            LOGGER.info("Job %s completed", job.id)
        except Exception as exc:
            LOGGER.exception("Job %s failed", job.id)
            self.repository.fail_or_retry(job.id, str(exc)[:4000])
            cleanup_temporary_files(self.data_dir)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2)
            self.current_job_id = None
            self.write_heartbeat()
        return True

    def run(self, poll_seconds: float = 1.0) -> None:
        self.repository.recover_interrupted()
        removed = cleanup_temporary_files(self.data_dir)
        if removed:
            LOGGER.info("Removed %s stale temporary paths", removed)
        while not self.stop_event.is_set():
            self.write_heartbeat()
            if self.process_one():
                self.stop_event.wait(2.0)
            else:
                self.stop_event.wait(poll_seconds)


def heartbeat_is_healthy(data_dir: Path, max_age_seconds: float = 60.0) -> bool:
    try:
        payload = json.loads(
            (data_dir / "worker-heartbeat.json").read_text(encoding="utf-8")
        )
        return time.time() - float(payload["timestamp"]) <= max_age_seconds
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Local video processing worker")
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.healthcheck:
        return 0 if heartbeat_is_healthy(settings.data_dir) else 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = settings.data_dir / "worker.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOGGER.error("Một worker khác đang giữ khóa %s", lock_path)
            return 2

        database = Database(settings.database_url)
        database.create_schema()
        service = WorkerService(
            JobRepository(database.session_factory), settings.data_dir
        )

        def request_stop(_signum: int, _frame: object) -> None:
            service.stop_event.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        try:
            service.run()
        finally:
            database.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
