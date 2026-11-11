"""Command-line entry point: process a local video without the web app.

viralcut-cli path/to/video.mp4 --clips 3 --style high_retention
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from .config import get_settings
from .db import Job, session_scope
from .jobs import enqueue
from .options import CaptionOptions, JobOptions
from .pipeline.orchestrator import run_job
from .pipeline.probe import probe
from .security import sanitize_filename, sniff_container, validate_extension_and_magic
from .storage import get_storage


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ViralCut AI — generate short-form clips from a video file.")
    p.add_argument("video", type=Path)
    p.add_argument("--clips", default="auto")
    p.add_argument("--duration", default="auto", choices=["15-30", "30-60", "60-90", "auto"])
    p.add_argument("--no-captions", action="store_true")
    p.add_argument(
        "--style",
        default="high_retention",
        choices=["high_retention", "clean_editorial", "dynamic_creator", "gaming"],
    )
    p.add_argument(
        "--edit-style", default="auto", choices=["dynamic", "editorial", "podcast", "gaming", "auto"]
    )
    p.add_argument("--platform", default="universal", choices=["shorts", "reels", "tiktok", "universal"])
    p.add_argument("--language", default="auto")
    p.add_argument("--framing", default="auto", choices=["auto", "face", "split", "blur", "center"])
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--out", type=Path, default=None, help="copy rendered clips to this directory")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    storage = get_storage()
    with args.video.open("rb") as fh:
        head = fh.read(64)
    ext, family = validate_extension_and_magic(args.video.name, head, settings.allowed_extensions)
    options = JobOptions(
        captions=CaptionOptions(enabled=not args.no_captions, style=args.style),
        clip_count="auto" if args.clips == "auto" else int(args.clips),
        duration=args.duration,
        platform=args.platform,
        language=args.language,
        style=args.edit_style,
        framing=args.framing,
        min_score=args.min_score,
    )
    with session_scope() as s:
        job = Job(
            original_filename=sanitize_filename(args.video.name),
            source_ext=ext,
            options=options.model_dump(),
            size_bytes=args.video.stat().st_size,
        )
        s.add(job)
        s.flush()
        job_id = job.id
    shutil.copyfile(args.video, storage.source_path(job_id, ext))
    media = probe(storage.source_path(job_id, ext), sniff_container(head))
    with session_scope() as s:
        s.get(Job, job_id).media = media.to_dict()
    enqueue(job_id)
    with session_scope() as s:
        j = s.get(Job, job_id)
        j.status, j.attempts = "processing", settings.max_attempts  # run inline, no retry loop
    run_job(job_id)

    with session_scope() as s:
        job = s.get(Job, job_id)
        result = {
            "job_id": job_id,
            "status": job.status,
            "error": job.error,
            "analysis": job.analysis,
            "clips": [
                {
                    "rank": c.rank,
                    "status": c.status,
                    "score": c.score,
                    "start": c.start,
                    "end": c.end,
                    "duration": c.duration,
                    "explanation": c.explanation,
                    "hook": c.hook,
                    "titles": c.titles,
                    "layouts": (c.framing or {}).get("layouts"),
                    "qa_passed": (c.qa or {}).get("passed"),
                    "file": str(storage.clip_file(job_id, c.id, c.version, "video")),
                }
                for c in job.clips
            ],
        }
        clips = [(c.rank, c.id, c.version) for c in job.clips if c.status == "ready"]
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for rank, cid, version in clips:
            shutil.copyfile(
                storage.clip_file(job_id, cid, version, "video"), args.out / f"clip-{rank:02d}.mp4"
            )
            srt = storage.clip_file(job_id, cid, version, "srt")
            if srt.exists():
                shutil.copyfile(srt, args.out / f"clip-{rank:02d}.srt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
