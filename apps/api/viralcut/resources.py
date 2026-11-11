"""Hardware detection and the ECO / BALANCED / TURBO resource profiles.

Everything here describes the machine that *executes the work* — the worker process — never
the browser. The API reports it at ``GET /api/system`` so the web app can show what the server
has and which profile it recommends, without ever guessing from the user's own computer.

A profile is a small set of concrete limits that every heavy stage reads:

* ``threads``          — ffmpeg filter/decoder/encoder threads for a render or analysis decode
* ``asr_threads``      — CTranslate2 (Whisper) intra-op threads
* ``onnx_threads``     — ONNX Runtime threads for the forced aligner
* ``render_workers``   — clips rendered at the same time (each with ``threads`` threads)
* ``x264_preset``      — encoder effort (quality is held constant by CRF; see docs/PERFORMANCE.md)
* ``nice``             — scheduling priority of child processes (higher = yields to the user's apps)
* ``keep_asr_model``   — keep Whisper loaded between jobs (fast) or release it after use (lean)

The numbers are not guesses: they come from the measurements in docs/PERFORMANCE.md.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .config import get_settings

log = logging.getLogger(__name__)

PROFILES = ("eco", "balanced", "turbo")


@dataclass(frozen=True)
class Hardware:
    logical_cpus: int  # CPUs this process may run on (affinity and cgroup quota applied)
    physical_cpus: int
    ram_gb: float | None
    ram_available_gb: float | None
    gpu: dict = field(default_factory=dict)  # {"name", "vram_gb", "vram_free_gb"} when a CUDA GPU exists
    platform: str = sys.platform

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResourcePlan:
    profile: str
    recommended: str
    reason: str
    threads: int
    asr_threads: int
    onnx_threads: int
    render_workers: int
    x264_preset: str
    nice: int
    keep_asr_model: bool
    hardware: Hardware

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hardware"] = self.hardware.to_dict()
        return d


def _cgroup_cpus() -> float | None:
    """CPU quota of a container (cgroup v2 ``cpu.max`` or v1 cfs files), in CPUs."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()[:2]
            if quota != "max":
                return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        with (
            open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq,
            open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp,
        ):
            quota, period = int(fq.read()), int(fp.read())
            if quota > 0 and period > 0:
                return quota / period
    except (OSError, ValueError):
        pass
    return None


def _gpu() -> dict:
    """First CUDA GPU with its memory, via nvidia-smi (no CUDA libraries are imported here)."""
    if get_settings().whisper_device == "cpu" or shutil.which("nvidia-smi") is None:
        return {}
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )  # fmt: skip
        line = res.stdout.strip().splitlines()[0] if res.returncode == 0 and res.stdout.strip() else ""
        name, total, free = (p.strip() for p in line.split(","))
        return {
            "name": name,
            "vram_gb": round(int(total) / 1024, 1),
            "vram_free_gb": round(int(free) / 1024, 1),
        }
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


@lru_cache(maxsize=1)
def detect_hardware() -> Hardware:
    try:
        logical = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:  # macOS / Windows
        logical = os.cpu_count() or 2
    quota = _cgroup_cpus()
    if quota:
        logical = max(1, min(logical, int(quota + 0.5)))
    physical = logical
    ram = avail = None
    try:
        import psutil

        physical = min(logical, psutil.cpu_count(logical=False) or logical)
        vm = psutil.virtual_memory()
        ram, avail = round(vm.total / 2**30, 1), round(vm.available / 2**30, 1)
    except Exception:  # psutil missing or unsupported platform: sysconf is enough on Linux/macOS
        try:
            ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        except (ValueError, OSError, AttributeError):
            pass
    return Hardware(logical, max(1, physical), ram, avail, _gpu())


def recommend(hw: Hardware) -> tuple[str, str]:
    ram = hw.ram_gb or 8.0
    if hw.logical_cpus <= 2 or ram < 6:
        return (
            "eco",
            f"{hw.logical_cpus} CPU core(s) and {ram:.0f} GB RAM: a light profile keeps the computer usable",
        )
    if (hw.logical_cpus >= 8 and ram >= 16) or hw.gpu:
        why = f"{hw.logical_cpus} cores and {ram:.0f} GB RAM" + (f", GPU {hw.gpu['name']}" if hw.gpu else "")
        return "turbo", f"{why}: enough headroom to run heavier work in parallel"
    return (
        "balanced",
        f"{hw.logical_cpus} cores and {ram:.0f} GB RAM: fast processing that leaves room for other apps",
    )


def plan_for(profile: str | None = None, hw: Hardware | None = None) -> ResourcePlan:
    """Concrete limits for ``profile`` (``auto``/empty = the recommended one) on this machine."""
    s = get_settings()
    hw = hw or detect_hardware()
    rec, why = recommend(hw)
    name = (profile or s.performance_profile or "auto").lower()
    if name not in PROFILES:
        name = rec
    n = hw.logical_cpus
    phys = hw.physical_cpus
    ram = hw.ram_gb or 8.0
    if name == "eco":
        # half the machine, low priority, nothing kept in memory between jobs
        threads = max(1, n // 2)
        plan = dict(threads=threads, asr_threads=max(1, min(phys, n // 2)), onnx_threads=max(1, n // 2),
                    render_workers=1, x264_preset="medium", nice=10, keep_asr_model=False)  # fmt: skip
    elif name == "turbo":
        # two clips at once where cores and memory allow it: measured -21 % render time on 4 cores
        # for +0.6 GB peak RAM (docs/PERFORMANCE.md); ffmpeg's filter graph is partly serial, so
        # two half-width renders keep the cores busier than one full-width render
        workers = 2 if (n >= 4 and ram >= 8) else 1
        plan = dict(threads=max(1, n // workers), asr_threads=max(1, min(phys, 16)), onnx_threads=max(1, min(8, n)),
                    render_workers=workers, x264_preset="medium", nice=0, keep_asr_model=True)  # fmt: skip
    else:  # balanced: one core always left for the system and the web UI
        threads = max(1, n - 1) if n > 2 else n
        plan = dict(threads=threads, asr_threads=max(1, min(phys, threads, 8)), onnx_threads=max(1, min(4, threads)),
                    render_workers=1, x264_preset="medium", nice=5, keep_asr_model=True)  # fmt: skip
    # explicit operator settings always win over the profile
    if s.ffmpeg_threads > 0:
        plan["threads"] = s.ffmpeg_threads
    if s.x264_preset:
        plan["x264_preset"] = s.x264_preset
    if s.render_workers > 0:
        plan["render_workers"] = s.render_workers
    return ResourcePlan(profile=name, recommended=rec, reason=why, hardware=hw, **plan)


_override: str | None = None


@lru_cache(maxsize=8)
def _plan_cached(name: str) -> ResourcePlan:
    plan = plan_for(None if name == "auto" else name)
    log.info(
        "resource profile %s (recommended %s): %d threads, ASR %d, aligner %d, %d render worker(s), nice %+d",
        plan.profile, plan.recommended, plan.threads, plan.asr_threads, plan.onnx_threads,
        plan.render_workers, plan.nice,
    )  # fmt: skip
    return plan


def active_plan() -> ResourcePlan:
    """The plan in force: the job's chosen profile while a job runs, else the server default."""
    return _plan_cached(_override or "auto")


@contextmanager
def use_profile(name: str | None):  # noqa: ANN201
    """Run a job under the profile its owner chose (``auto``/None = the server default).

    A worker runs one job at a time, so a process-wide setting is enough and it also reaches
    the render threads. Thread pools inside ONNX Runtime / CTranslate2 are sized when their
    model loads, so a per-job change of those two only applies to models loaded afterwards.
    """
    global _override
    previous = _override
    _override = name if name in PROFILES else None
    try:
        yield active_plan()
    finally:
        _override = previous


def lower_priority(pid: int) -> None:
    """Lower a child process's scheduling priority per the active profile (never raises).

    Done after the process starts rather than in ``preexec_fn``, which is unsafe in a
    multi-threaded process. Only processes this application started are touched.
    """
    nice = active_plan().nice
    if nice <= 0:
        return
    try:
        if os.name == "nt":  # pragma: no cover - Windows
            import psutil

            psutil.Process(pid).nice(
                psutil.BELOW_NORMAL_PRIORITY_CLASS if nice < 10 else psutil.IDLE_PRIORITY_CLASS
            )
        else:
            # absolute level: a child of an already-niced worker is not lowered twice
            if os.getpriority(os.PRIO_PROCESS, pid) < nice:
                os.setpriority(os.PRIO_PROCESS, pid, min(19, nice))
    except Exception:  # the process may already have exited, or the platform refuses it
        pass
