#!/usr/bin/env python3
"""Measure what one full ViralCut job really costs: wall time per stage, CPU, peak RAM, disk.

Each run starts a fresh process (so model loading is counted, as it is for a worker that has
just started) in an empty data directory, and samples the whole process tree (Python + every
ffmpeg child) every 0.25 s with psutil:

* ``peak_rss_mb``       — largest sum of resident memory across the tree at any sample
* ``cpu_s``             — user+system CPU seconds of the tree (from ``getrusage``: exact, not sampled)
* ``avg_cores`` / ``peak_cores`` — CPU seconds per wall second (1.0 = one core fully busy)
* ``peak_disk_mb``      — largest size of the data directory (source copy, analysis, temp files)
* ``stages``            — seconds spent in each pipeline stage (read from the job's own record)

With ``--rerender`` a second process then re-renders clip 1 after a caption-style change, which
is what the studio does when a user tweaks a clip.

    python scripts/benchmark_pipeline.py video.mp4 --clips 3 --label baseline --out report.json

All knobs (profile, transcription mode, model...) come from the same ``VIRALCUT_*`` environment
variables the application reads, so a configuration measured here is the one that ships.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

API_DIR = Path(__file__).resolve().parents[1] / "apps" / "api"

RERENDER_SNIPPET = r"""
import sys, json
from viralcut.db import Clip, session_scope
from viralcut.pipeline.orchestrator import rerender_clip
job_id, style = sys.argv[1], sys.argv[2]
with session_scope() as s:
    clips = sorted(s.query(Clip).filter(Clip.job_id == job_id, Clip.status == "ready").all(), key=lambda c: c.rank)
    if not clips:
        print(json.dumps({"error": "no ready clip"})); sys.exit(1)
    c = clips[0]
    c.captions = {**(c.captions or {}), "style": style}
    c.status = "rendering"
    clip_id = c.id
rerender_clip(job_id, clip_id)
with session_scope() as s:
    c = s.get(Clip, clip_id)
    print(json.dumps({"status": c.status, "version": c.version, "qa": (c.qa or {}).get("passed")}))
"""


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


class Sampler:
    def __init__(self, proc: subprocess.Popen, data_dir: Path) -> None:
        self.root = psutil.Process(proc.pid)
        self.data_dir = data_dir
        self.peak_rss = 0
        self.peak_disk = 0
        self.peak_cores = 0.0
        self.stage_times: dict[str, list[float]] = {}  # stage -> [first seen, last seen] (or [0, exact])
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._last_cpu: tuple[float, float] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _tree(self) -> list[psutil.Process]:
        try:
            return [self.root, *self.root.children(recursive=True)]
        except psutil.Error:
            return []

    def _poll_stages(self) -> None:
        db = self.data_dir / "viralcut.db"
        if not db.exists():
            return
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.2)
            rows = con.execute("SELECT stages FROM jobs ORDER BY created_at DESC LIMIT 1").fetchall()
            con.close()
        except sqlite3.Error:
            return
        now = time.monotonic() - self._t0
        for (raw,) in rows:
            for key, st in (json.loads(raw or "{}") or {}).items():
                status = st.get("status")
                if status == "active" and key not in self.stage_times:
                    self.stage_times[key] = [now, now]
                elif status in ("done", "skipped") and st.get("elapsed_s") is not None:
                    self.stage_times[key] = [0.0, float(st["elapsed_s"])]  # exact, recorded by the worker
                elif status in ("done", "skipped") and key in self.stage_times:
                    if self.stage_times[key][1] == self.stage_times[key][0]:
                        self.stage_times[key][1] = now

    def _run(self) -> None:
        tick = 0
        while not self._stop.is_set():
            rss, cpu = 0, 0.0
            for p in self._tree():
                try:
                    rss += p.memory_info().rss
                    t = p.cpu_times()
                    cpu += t.user + t.system + t.children_user + t.children_system
                except psutil.Error:
                    pass
            now = time.monotonic()
            if self._last_cpu is not None and cpu >= self._last_cpu[1]:
                dt = now - self._last_cpu[0]
                if dt > 0:
                    self.peak_cores = max(self.peak_cores, (cpu - self._last_cpu[1]) / dt)
            self._last_cpu = (now, cpu)
            self.peak_rss = max(self.peak_rss, rss)
            if tick % 4 == 0:
                self.peak_disk = max(self.peak_disk, _dir_size(self.data_dir))
                self._poll_stages()
            tick += 1
            self._stop.wait(0.25)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._poll_stages()
        self._stop.set()
        self.thread.join()


def run_measured(cmd: list[str], env: dict, data_dir: Path) -> tuple[dict, str]:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, env=env, cwd=API_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = Sampler(proc, data_dir)
    sampler.start()
    out, err = proc.communicate()
    wall = time.monotonic() - t0
    sampler.stop()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    stages = {k: round(v[1] - v[0], 2) for k, v in sampler.stage_times.items()}
    if proc.returncode != 0:
        sys.stderr.write(err[-4000:])
    return (
        {
            "returncode": proc.returncode,
            "wall_s": round(wall, 2),
            "cpu_s": round(cpu, 2),
            "avg_cores": round(cpu / wall, 2) if wall else 0.0,
            "peak_cores": round(sampler.peak_cores, 2),
            "peak_rss_mb": round(sampler.peak_rss / 2**20, 1),
            "peak_disk_mb": round(sampler.peak_disk / 2**20, 1),
            "stages": stages,
        },
        out,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--clips", default="3")
    ap.add_argument("--duration", default="auto")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--rerender", action="store_true", help="also time a studio re-render after a caption-style change")
    ap.add_argument("--rerender-style", default="clean_editorial")
    ap.add_argument("--repeat-job", action="store_true", help="run the same job twice in the same data dir (cache reuse)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra viralcut-cli arguments")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep", type=Path, default=None, help="copy rendered clips here for quality checks")
    args = ap.parse_args()

    data_dir = Path(tempfile.mkdtemp(prefix="viralcut-bench-"))
    env = dict(os.environ, VIRALCUT_DATA_DIR=str(data_dir), PYTHONUNBUFFERED="1")
    py = sys.executable
    cli = [py, "-m", "viralcut.cli", str(args.video.resolve()), "--clips", args.clips, "--duration", args.duration, "--min-score", "0"]
    if args.no_captions:
        cli.append("--no-captions")
    cli += args.extra
    if args.keep:
        cli += ["--out", str(args.keep.resolve())]
    report: dict = {"label": args.label, "video": args.video.name, "env": {k: v for k, v in os.environ.items() if k.startswith("VIRALCUT_")}}
    report["cpu"] = {"logical": psutil.cpu_count(), "physical": psutil.cpu_count(logical=False)}
    job, out = run_measured(cli, env, data_dir)
    try:
        result = json.loads(out[out.index("{"):])
    except ValueError:
        result = {}
    job["clips_ready"] = sum(1 for c in result.get("clips", []) if c.get("status") == "ready")
    job["asr"] = (result.get("analysis") or {}).get("asr", {}).get("model")
    report["job"] = job
    if args.repeat_job:
        job2, _ = run_measured(cli, env, data_dir)
        report["job_repeat"] = job2
    if args.rerender and result.get("job_id"):
        cmd = [py, "-c", RERENDER_SNIPPET, result["job_id"], args.rerender_style]
        rr, out2 = run_measured(cmd, env, data_dir)
        rr["result"] = out2.strip().splitlines()[-1] if out2.strip() else ""
        report["rerender"] = rr
    shutil.rmtree(data_dir, ignore_errors=True)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0 if job["returncode"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
