"""Worker process: claims queued jobs / clip re-renders and runs the pipeline."""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time

from .. import cache
from ..config import get_settings
from ..db import get_engine
from ..jobs import claim_clip, claim_job, purge_expired, requeue_stale
from ..pipeline.ffmpeg import require_binaries
from ..pipeline.orchestrator import rerender_clip, run_job
from ..resources import active_plan

log = logging.getLogger("viralcut.worker")

# what the in-process worker is doing, reported by GET /api/health so the site can say why a
# video is still waiting ("off" = no embedded worker here: a separate worker process may run)
STATUS: dict[str, str] = {"state": "off", "error": "", "job": ""}


class Worker:
    def __init__(self) -> None:
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}"
        self.stop_event = threading.Event()
        self._last_maintenance = 0.0

    def stop(self, *_: object) -> None:
        self.stop_event.set()

    def maintenance(self) -> None:
        now = time.monotonic()
        if now - self._last_maintenance < 60:
            return
        self._last_maintenance = now
        try:
            stale = requeue_stale()
            purged = purge_expired()
            cached = cache.purge()  # cached transcripts expire with the uploads they came from
            if stale or purged or cached:
                log.info(
                    "maintenance: %d stale job(s) recovered, %d expired job(s) purged, %d cache file(s) removed",
                    stale, purged, cached,
                )  # fmt: skip
        except Exception:
            log.exception("maintenance failed")

    def run_once(self) -> bool:
        """Process one unit of work. Returns False when the queue is empty."""
        # studio re-renders are short and interactive: serve them first
        claimed = claim_clip(self.worker_id)
        if claimed is not None:
            job_id, clip_id = claimed
            log.info("re-rendering clip %s", clip_id[:8])
            STATUS.update(state="busy", job=job_id)
            try:
                rerender_clip(job_id, clip_id)
            finally:
                STATUS.update(state="idle", job="")
            return True
        job_id = claim_job(self.worker_id)
        if job_id is not None:
            log.info("processing job %s", job_id[:8])
            STATUS.update(state="busy", job=job_id)
            try:
                run_job(job_id)
            finally:
                STATUS.update(state="idle", job="")
            return True
        return False

    def _wait_until_ready(self) -> bool:
        """Retry start-up (database, ffmpeg) instead of dying: a dead worker thread leaves every
        video "Waiting for a free worker" forever while the API keeps answering normally."""
        STATUS.update(state="starting", error="")
        while not self.stop_event.is_set():
            try:
                get_engine()
                require_binaries()
                STATUS.update(state="idle", error="")
                return True
            except Exception as e:
                STATUS.update(state="error", error=str(e))
                log.error("worker cannot start: %s (retrying in 15 s)", e)
                self.stop_event.wait(15)
        return False

    def run_forever(self) -> None:
        if not self._wait_until_ready():
            return
        interval = get_settings().worker_poll_interval_s
        log.info("worker %s started", self.worker_id)
        while not self.stop_event.is_set():
            self.maintenance()
            try:
                worked = self.run_once()
            except Exception:
                log.exception("worker iteration failed")
                worked = False
            if not worked:
                self.stop_event.wait(interval)
        log.info("worker stopped")


def start_embedded() -> Worker:
    """Run a worker thread inside the API process (single-process development mode)."""
    worker = Worker()
    threading.Thread(target=worker.run_forever, name="viralcut-worker", daemon=True).start()
    return worker


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    plan = active_plan()
    if plan.nice > 0 and hasattr(os, "nice"):
        try:  # a standalone worker yields the CPU to the user's own applications (ECO / BALANCED)
            os.nice(plan.nice)
        except OSError:
            pass
    worker = Worker()
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run_forever()


if __name__ == "__main__":
    main()
