"""Thin, safe wrappers around the ffmpeg/ffprobe binaries."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from ..resources import active_plan, lower_priority
from ..security import FAMILY_DEMUXER

ProgressFn = Callable[[float], None]
CancelFn = Callable[[], bool]


class FFmpegError(RuntimeError):
    pass


class Cancelled(Exception):
    """Raised when the job owner asked to stop processing."""


def require_binaries() -> None:
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise FFmpegError(f"{name} not found on PATH")


def input_args(
    path: Path | str,
    family: str | None,
    *,
    seek: float | None = None,
    duration: float | None = None,
    threads: int | None = None,
) -> list[str]:
    """Arguments that open an untrusted upload.

    The protocol whitelist stops crafted files from making ffmpeg read other local files or
    reach the network, and forcing the demuxer stops playlist/script formats from being probed.
    The decoder gets the active profile's thread budget instead of one thread per core.
    """
    args = ["-protocol_whitelist", "file,pipe", "-threads", str(threads or active_plan().threads)]
    if family in FAMILY_DEMUXER:
        args += ["-f", FAMILY_DEMUXER[family]]
    if seek is not None:
        args += ["-ss", f"{max(0.0, seek):.6f}"]
    if duration is not None:
        args += ["-t", f"{max(0.0, duration):.6f}"]
    args += ["-i", str(path)]
    return args


def _drain(stream, sink: list[bytes], limit: int = 64_000) -> None:  # noqa: ANN001
    size = 0
    for chunk in iter(lambda: stream.read(4096), b""):
        if size < limit:
            sink.append(chunk)
            size += len(chunk)


def run_ffmpeg(
    args: Sequence[str],
    *,
    total_duration: float | None = None,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> None:
    """Run ffmpeg to completion, reporting real progress parsed from ``-progress``."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-nostats",
        "-progress",
        "pipe:1",
        *args,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    lower_priority(proc.pid)
    err: list[bytes] = []
    t = threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)
    t.start()
    started = time.monotonic()
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("out_time_us=") and total_duration and on_progress:
                try:
                    done = int(line.split("=", 1)[1]) / 1e6
                    on_progress(max(0.0, min(1.0, done / total_duration)))
                except ValueError:
                    pass
            if should_cancel and should_cancel():
                raise Cancelled()
            if timeout and time.monotonic() - started > timeout:
                raise FFmpegError("ffmpeg timed out")
        code = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        t.join(timeout=2)
    if code != 0:
        tail = b"".join(err).decode("utf-8", "replace").strip().splitlines()[-12:]
        raise FFmpegError("ffmpeg failed: " + " | ".join(tail))
    if on_progress:
        on_progress(1.0)


def read_pipe(
    args: Sequence[str],
    *,
    chunk_size: int,
    exact: bool = True,
    should_cancel: CancelFn | None = None,
    cwd: Path | None = None,
) -> Iterator[bytes]:
    """Yield chunks from an ffmpeg process writing raw data to stdout.

    With ``exact=True`` every chunk has exactly ``chunk_size`` bytes (one video frame) and a
    trailing partial chunk is discarded; with ``exact=False`` chunks are yielded as read.
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *args, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    lower_priority(proc.pid)
    err: list[bytes] = []
    t = threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)
    t.start()
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(chunk_size)
            if not buf:
                break
            if exact and len(buf) < chunk_size:
                break
            yield buf
            if should_cancel and should_cancel():
                raise Cancelled()
        code = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        t.join(timeout=2)
    if code != 0:
        tail = b"".join(err).decode("utf-8", "replace").strip().splitlines()[-8:]
        raise FFmpegError("ffmpeg failed: " + " | ".join(tail))


def run_capture(
    cmd: Sequence[str], timeout: float = 120, cwd: Path | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run to completion and capture output; the child is killed on timeout (never left running)."""
    proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    lower_priority(proc.pid)
    try:
        out, err = proc.communicate(timeout=timeout)
    except BaseException:
        proc.kill()
        proc.communicate()
        raise
    return subprocess.CompletedProcess(list(cmd), proc.returncode, out, err)
