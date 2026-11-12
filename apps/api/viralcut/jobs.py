"""Database-backed job queue: claiming, progress, logs, cancellation and retention.

The queue is the ``jobs``/``clips`` tables themselves. Claims are single atomic UPDATEs, so
any number of worker processes can share one database (SQLite in development, PostgreSQL in
production) without an extra broker.
"""

from __future__ import annotations

import datetime as dt
import logging
import socket
import time

from sqlalchemy import select, update

from .config import get_settings
from .db import Clip, Job, JobLog, session_scope, utcnow
from .storage import get_storage

log = logging.getLogger(__name__)

# (key, label, weight in overall progress)
STAGES: list[tuple[str, str, float]] = [
    ("analyzing", "Analyzing video", 0.10),
    ("transcribing", "Transcribing speech", 0.27),
    ("discovering", "Discovering high-impact moments", 0.04),
    ("ranking", "Ranking clips", 0.04),
    ("syncing", "Syncing captions to the audio", 0.05),
    ("reframing", "Smart reframing", 0.10),
    ("subtitles", "Generating subtitles", 0.02),
    ("rendering", "Rendering videos", 0.38),
    ("ready", "Ready to publish", 0.0),
]
STAGE_KEYS = [k for k, _, _ in STAGES]


def initial_stages() -> dict:
    return {k: {"status": "pending", "progress": 0.0} for k in STAGE_KEYS}


def overall_progress(stages: dict) -> float:
    total = 0.0
    for key, _, weight in STAGES:
        st = stages.get(key, {})
        if st.get("status") in ("done", "skipped"):
            total += weight
        elif st.get("status") == "active":
            total += weight * float(st.get("progress", 0.0))
    return round(min(1.0, total), 4)


def add_log(job_id: str, message: str, level: str = "info", stage: str = "") -> None:
    with session_scope() as s:
        s.add(JobLog(job_id=job_id, level=level, stage=stage, message=message[:2000]))


def enqueue(job_id: str) -> None:
    with session_scope() as s:
        s.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(
                status="queued",
                queued_at=utcnow(),
                stages=initial_stages(),
                stage="",
                progress=0.0,
                stage_progress=0.0,
                error="",
                cancel_requested=False,
            )
        )


def claim_job(worker_id: str) -> str | None:
    """Atomically take the oldest queued job."""
    with session_scope() as s:
        for _ in range(3):
            job_id = s.execute(
                select(Job.id).where(Job.status == "queued").order_by(Job.queued_at, Job.created_at).limit(1)
            ).scalar_one_or_none()
            if job_id is None:
                return None
            res = s.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == "queued")
                .values(
                    status="processing",
                    worker_id=worker_id,
                    started_at=utcnow(),
                    heartbeat_at=utcnow(),
                    attempts=Job.attempts + 1,
                )
            )
            s.commit()
            if res.rowcount == 1:
                return job_id
    return None


def claim_clip(worker_id: str) -> tuple[str, str] | None:
    """Atomically take a clip waiting for a re-render. Returns (job_id, clip_id)."""
    with session_scope() as s:
        for _ in range(3):
            row = s.execute(
                select(Clip.id, Clip.job_id).where(Clip.status == "queued").order_by(Clip.updated_at).limit(1)
            ).first()
            if row is None:
                return None
            res = s.execute(
                update(Clip)
                .where(Clip.id == row.id, Clip.status == "queued")
                .values(status="rendering", worker_id=worker_id, progress=0.0, error="")
            )
            s.commit()
            if res.rowcount == 1:
                return row.job_id, row.id
    return None


class JobReporter:
    """Writes stage progress and heartbeats (throttled) and polls the cancel flag."""

    def __init__(self, job_id: str, min_interval: float = 0.4) -> None:
        self.job_id = job_id
        self.min_interval = min_interval
        self._last_write = 0.0
        self._last_cancel_check = 0.0
        self._cancelled = False
        self.stage = ""
        self._stage_t0: dict[str, float] = {}

    def _update(self, mutate) -> None:  # noqa: ANN001
        with session_scope() as s:
            job = s.get(Job, self.job_id)
            if job is None:
                self._cancelled = True
                return
            stages = dict(job.stages or initial_stages())
            mutate(stages)
            job.stages = stages
            job.stage = self.stage
            job.stage_progress = float(stages.get(self.stage, {}).get("progress", 0.0)) if self.stage else 0.0
            job.progress = overall_progress(stages)
            job.heartbeat_at = utcnow()
            self._cancelled = self._cancelled or bool(job.cancel_requested)

    def start(self, stage: str) -> None:
        self.stage = stage
        self._stage_t0[stage] = time.monotonic()

        def m(stages: dict) -> None:
            stages[stage] = {"status": "active", "progress": 0.0}

        self._update(m)
        self._last_write = time.monotonic()

    def progress(self, value: float, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_write < self.min_interval:
            return
        self._last_write = now
        stage = self.stage
        value = max(0.0, min(1.0, float(value)))

        def m(stages: dict) -> None:
            cur = stages.get(stage, {"status": "active", "progress": 0.0})
            stages[stage] = {"status": "active", "progress": max(value, float(cur.get("progress", 0.0)))}

        self._update(m)

    def finish(self, stage: str | None = None, skipped: bool = False) -> None:
        stage = stage or self.stage
        t0 = self._stage_t0.get(stage)
        elapsed = round(time.monotonic() - t0, 2) if t0 is not None and not skipped else None

        def m(stages: dict) -> None:
            stages[stage] = {"status": "skipped" if skipped else "done", "progress": 1.0}
            if elapsed is not None:
                stages[stage]["elapsed_s"] = elapsed  # real time spent, shown in the UI and benchmarks

        self._update(m)

    def sub(self, lo: float, hi: float):
        """Progress callback mapping 0..1 onto a sub-range of the current stage."""
        return lambda p: self.progress(lo + (hi - lo) * max(0.0, min(1.0, p)))

    def cancelled(self) -> bool:
        now = time.monotonic()
        if not self._cancelled and now - self._last_cancel_check > 1.0:
            self._last_cancel_check = now
            with session_scope() as s:
                flag = s.execute(
                    select(Job.cancel_requested).where(Job.id == self.job_id)
                ).scalar_one_or_none()
                self._cancelled = flag is None or bool(flag)
        return self._cancelled

    def log(self, message: str, level: str = "info") -> None:
        log.log(
            logging.WARNING if level == "warn" else logging.ERROR if level == "error" else logging.INFO,
            "[%s] %s",
            self.job_id[:8],
            message,
        )
        add_log(self.job_id, message, level=level, stage=self.stage)


def requeue_stale() -> int:
    """Recover jobs whose worker died (no heartbeat): retry while attempts remain, else fail."""
    s_ = get_settings()
    cutoff = utcnow() - dt.timedelta(minutes=s_.stale_job_minutes)
    n = 0
    with session_scope() as s:
        jobs = (
            s.execute(select(Job).where(Job.status == "processing", Job.heartbeat_at < cutoff))
            .scalars()
            .all()
        )
        for job in jobs:
            n += 1
            if job.attempts < s_.max_attempts:
                job.status = "queued"
                job.queued_at = utcnow()
                s.add(
                    JobLog(job_id=job.id, level="warn", message="Worker stopped responding; job re-queued.")
                )
            else:
                job.status = "failed"
                job.error = "Processing stopped unexpectedly. Please try again."
                job.finished_at = utcnow()
        clips = (
            s.execute(select(Clip).where(Clip.status == "rendering", Clip.updated_at < cutoff))
            .scalars()
            .all()
        )
        for clip in clips:
            if clip.job.status in ("completed",):
                clip.status = "failed"
                clip.error = "Rendering stopped unexpectedly. Please try again."
    return n


def _process_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:  # pragma: no cover - psutil missing: assume alive, the stale timeout still applies
        return True


def recover_orphans() -> int:
    """At worker start-up: free work claimed by a process on this machine that no longer exists.

    Closing the API window mid-job left the video "processing" (it could not be canceled or
    deleted) until the 30-minute stale timeout re-queued it. A worker id is host:pid:thread, so a
    claim from this host whose process is gone is certainly dead and is recovered right away.
    """
    host = socket.gethostname()
    n = 0
    with session_scope() as s:
        for job in s.execute(select(Job).where(Job.status == "processing")).scalars().all():
            h, _, rest = (job.worker_id or "").partition(":")
            pid = rest.partition(":")[0]
            if h != host or not pid.isdigit() or _process_alive(int(pid)):
                continue
            n += 1
            if job.cancel_requested:
                job.status = "canceled"
                job.finished_at = utcnow()
            else:
                job.status = "queued"
                job.queued_at = utcnow()
                s.add(
                    JobLog(
                        job_id=job.id,
                        level="warn",
                        message="ViralCut was closed during processing; job re-queued.",
                    )
                )
        for clip in s.execute(select(Clip).where(Clip.status == "rendering")).scalars().all():
            h, _, rest = (clip.worker_id or "").partition(":")
            pid = rest.partition(":")[0]
            if h == host and pid.isdigit() and not _process_alive(int(pid)):
                n += 1
                clip.status = "queued"
    return n


def purge_expired() -> int:
    """Retention: delete jobs (database rows and every file) past their expiry time."""
    storage = get_storage()
    with session_scope() as s:
        ids = list(
            s.execute(
                select(Job.id).where(Job.expires_at < utcnow(), Job.status.notin_(("processing", "queued")))
            )
            .scalars()
            .all()
        )
        # uploads that never finished (client vanished, server restarted mid-transfer)
        abandoned = utcnow() - dt.timedelta(hours=6)
        ids += list(
            s.execute(select(Job.id).where(Job.status == "uploading", Job.created_at < abandoned))
            .scalars()
            .all()
        )
    for job_id in ids:
        storage.delete_job(job_id)
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job is not None:
                s.delete(job)
    return len(ids)
