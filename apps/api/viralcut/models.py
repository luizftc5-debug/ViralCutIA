"""Downloaded AI models: where they live, and a progress report while one is being fetched.

The first video downloads the speech model (about 1.6 GB) and the caption-sync model. The
libraries that fetch them report nothing, so the stage used to sit at 0 % for many minutes and
looked frozen. ``DownloadWatch`` measures the model folder while it fills and reports
"Downloading …: 640 MB of 1.6 GB" through a callback (the job page shows it under the stage).
"""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from .config import get_settings

log = logging.getLogger(__name__)

StatusFn = Callable[[str], None]

_sink: StatusFn | None = None  # where reports go when the caller passes no callback (the running job)


def set_status_sink(fn: StatusFn | None) -> None:
    """Route download reports of the code that has no callback of its own to ``fn`` (``None``: nowhere)."""
    global _sink
    _sink = fn


def report(text: str, on_status: StatusFn | None = None) -> None:
    """Send one status line; a failing callback never breaks the work being reported."""
    fn = on_status or _sink
    if fn is None:
        return
    try:
        fn(text)
    except Exception:  # pragma: no cover - reporting is best effort
        log.debug("status callback failed", exc_info=True)


def hub_cache_dir() -> Path:
    s = get_settings()
    if s.models_dir:
        return Path(s.models_dir)
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def repo_dir(repo_id: str) -> Path:
    """Folder of one Hugging Face repository inside the model cache."""
    return hub_cache_dir() / ("models--" + repo_id.replace("/", "--"))


def dir_bytes(path: Path) -> int:
    """Bytes on disk under ``path``; links are not followed, so a linked file counts once."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass  # a temporary file renamed or removed while we were looking
    return total


def _repo_bytes(repo_id: str, patterns: Sequence[str] | None) -> int | None:
    """Total size of the files that will be downloaded (``None`` when the hub cannot be asked)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True, timeout=10)
        sizes = [
            int(f.size or 0)
            for f in info.siblings or []
            if patterns is None or any(fnmatch.fnmatch(f.rfilename, p) for p in patterns)
        ]
        return sum(sizes) or None
    except Exception:
        return None


def human_size(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


class DownloadWatch:
    """Context manager around a blocking download: reports how much of it is on disk."""

    def __init__(
        self,
        what: str,
        repo_id: str,
        on_status: StatusFn | None = None,
        patterns: Sequence[str] | None = None,
        interval: float = 1.0,
    ) -> None:
        self.what = what
        self.repo_id = repo_id
        self.on_status = on_status
        self.patterns = patterns
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> DownloadWatch:
        log.info("%s (%s)", self.what, self.repo_id)
        report(f"{self.what}…", self.on_status)
        self._thread = threading.Thread(target=self._run, name="viralcut-download-watch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        folder = repo_dir(self.repo_id)
        total = _repo_bytes(self.repo_id, self.patterns)
        last = ""
        while not self._stop.wait(self.interval):
            done = dir_bytes(folder)
            if total:
                text = f"{self.what}: {human_size(min(done, total))} of {human_size(total)}"
            else:
                text = f"{self.what}: {human_size(done)} so far"
            if text != last:
                last = text
                report(text, self.on_status)
