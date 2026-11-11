"""Storage abstraction.

The pipeline needs real files on local disk (ffmpeg, OpenCV), so each job owns a working
directory. ``Storage`` is the seam for production object storage: an S3 implementation would
materialise inputs into the job directory before processing and publish results afterwards.
Only the local backend is implemented today.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from .config import get_settings
from .security import is_valid_id, resolve_inside


class Storage(Protocol):
    def job_dir(self, job_id: str) -> Path: ...
    def source_path(self, job_id: str, ext: str) -> Path: ...
    def clip_dir(self, job_id: str, clip_id: str) -> Path: ...
    def delete_job(self, job_id: str) -> None: ...


class LocalStorage:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or get_settings().jobs_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        if not is_valid_id(job_id):
            raise PermissionError("invalid job id")
        p = resolve_inside(self.root, job_id)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def source_path(self, job_id: str, ext: str) -> Path:
        return self.job_dir(job_id) / f"source{ext}"

    def analysis_dir(self, job_id: str) -> Path:
        p = self.job_dir(job_id) / "analysis"
        p.mkdir(exist_ok=True)
        return p

    def clip_dir(self, job_id: str, clip_id: str) -> Path:
        if not is_valid_id(clip_id):
            raise PermissionError("invalid clip id")
        p = self.job_dir(job_id) / "clips" / clip_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def clip_file(self, job_id: str, clip_id: str, version: int, kind: str) -> Path:
        ext = {"video": "mp4", "srt": "srt", "thumb": "jpg", "ass": "ass"}[kind]
        return self.clip_dir(job_id, clip_id) / f"v{int(version)}.{ext}"

    def delete_job(self, job_id: str) -> None:
        if not is_valid_id(job_id):
            return
        p = resolve_inside(self.root, job_id)
        shutil.rmtree(p, ignore_errors=True)


def get_storage() -> LocalStorage:
    return LocalStorage()
