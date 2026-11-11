"""Command-line entry point: process a local video without the web app.

viralcut-cli path/to/video.mp4 --clips 3 --style high_retention
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from .config import get_settings
from .db import Job, session_scope
from .jobs import enqueue
from .options import JobOptions
from .pipeline.orchestrator import run_job
from .pipeline.probe import probe
from .security import sanitize_filename, sniff_container, validate_extension_and_magic
from .storage import get_storage


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ViralCut AI — generate short-form clips from a video file.")
    p.add_argument("video", type=Path)
    p.add_argument("--clips", default="auto")
    p.add_argument("--duration", default="auto", choices=["15-30", "30-60", "60-90", "auto", "custom"])
    p.add_argument("--min-len", type=float, default=None, help="custom length: minimum seconds")
    p.add_argument("--max-len", type=float, default=None, help="custom length: maximum seconds")
    p.add_argument("--no-captions", action="store_true")
    p.add_argument(
        "--style",
        default="high_retention",
        choices=[
            "auto",
            "viral_dynamic",
            "energy",
            "podcast",
            "kinetic",
            "clean_editorial",
            "gaming",
            "cinematic",
            "high_retention",
            "dynamic_creator",
        ],
        help="caption style (auto = the caption design of the video style)",
    )
    p.add_argument(
        "--edit-style",
        "--video-style",
        dest="edit_style",
        default="auto",
        choices=[
            "auto",
            "viral",
            "podcast",
            "gaming",
            "cinematic",
            "educational",
            "reaction",
            "minimal",
            "storytime",
            "news",
            "dynamic",
            "editorial",
        ],
    )
    p.add_argument("--intensity", default="balanced", choices=["subtle", "balanced", "high"])
    p.add_argument("--variations", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--quality", default="high", choices=["high", "max"])
    p.add_argument("--performance", default="auto", choices=["auto", "eco", "balanced", "turbo"])
    p.add_argument(
        "--preset", default="", help="Creator Studio preset (gaming_highlights, podcast, rankings, ...)"
    )
    p.add_argument(
        "--content-type", default="shorts", choices=["shorts", "tiktok", "reels", "youtube", "compilation"]
    )
    p.add_argument("--aspect", default="9:16", choices=["9:16", "16:9"])
    p.add_argument(
        "--editors", default="{}", help='editor settings as JSON, e.g. \'{"lettering":{"enabled":true}}\''
    )
    p.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help="add a music/image/reference file (repeatable); refer to it in --editors as @1, @2, ...",
    )
    p.add_argument("--platform", default="universal", choices=["shorts", "reels", "tiktok", "universal"])
    p.add_argument("--language", default="auto")
    p.add_argument(
        "--framing", default="auto", choices=["auto", "face", "speaker", "split", "blur", "center"]
    )
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--no-zoom", action="store_true", help="disable the eased punch-ins on stressed words")
    p.add_argument("--glossary", default="", help="comma-separated names/terms the recogniser should prefer")
    p.add_argument(
        "--mode",
        default=None,
        choices=["auto", "fast", "balanced", "accurate"],
        help="speech-recognition mode (default: VIRALCUT_TRANSCRIPTION_MODE)",
    )
    p.add_argument("--out", type=Path, default=None, help="copy rendered clips to this directory")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.mode:
        os.environ["VIRALCUT_TRANSCRIPTION_MODE"] = args.mode
        get_settings.cache_clear()
    settings = get_settings()
    storage = get_storage()
    with args.video.open("rb") as fh:
        head = fh.read(64)
    ext, family = validate_extension_and_magic(args.video.name, head, settings.allowed_extensions)
    base = dict(
        captions={"enabled": not args.no_captions, "style": args.style},
        clip_count="auto" if args.clips == "auto" else int(args.clips),
        duration=args.duration,
        platform=args.platform,
        language=args.language,
        style=args.edit_style,
        framing=args.framing,
        min_score=args.min_score,
        smart_zoom=not args.no_zoom,
        glossary=args.glossary,
        intensity=args.intensity,
        variations=args.variations,
        quality=args.quality,
        performance=args.performance,
        content_type=args.content_type,
        aspect=args.aspect,
        editors=json.loads(args.editors),
    )
    if args.min_len is not None:
        base["custom_min"] = args.min_len
    if args.max_len is not None:
        base["custom_max"] = args.max_len
    if args.preset:
        from .pipeline.editors import apply_preset

        base = apply_preset(base, args.preset)
    options_raw = base
    with session_scope() as s:
        job = Job(
            original_filename=sanitize_filename(args.video.name),
            source_ext=ext,
            options={},
            size_bytes=args.video.stat().st_size,
        )
        s.add(job)
        s.flush()
        job_id = job.id
    shutil.copyfile(args.video, storage.source_path(job_id, ext))
    # user assets for the editors (@1, @2... in --editors refer to them in order)
    from .pipeline.studio import assets_dir, list_assets, save_asset_index

    index, raw = list_assets(storage.job_dir(job_id)), json.dumps(options_raw)
    for n, spec in enumerate(args.asset, 1):
        kind, _, path = spec.partition("=")
        src = Path(path)
        aid = f"{n:02d}" + job_id[:30]
        dst = assets_dir(storage.job_dir(job_id)) / f"{aid}{src.suffix.lower()}"
        shutil.copyfile(src, dst)
        index[aid] = {"kind": kind, "name": src.name, "file": dst.name}
        raw = raw.replace(f'"@{n}"', f'"{aid}"')
    save_asset_index(storage.job_dir(job_id), index)
    options = JobOptions.model_validate(json.loads(raw))
    with session_scope() as s:
        s.get(Job, job_id).options = options.model_dump(by_alias=True)
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
                    "variant": (c.edit or {}).get("variant"),
                    "style": ((c.framing or {}).get("edit") or {}).get("style"),
                    "recommendation": (c.edit or {}).get("recommendation"),
                    "notes": ((c.framing or {}).get("edit") or {}).get("notes"),
                    "audio": ((c.framing or {}).get("edit") or {}).get("audio"),
                    "kind": (c.edit or {}).get("kind", "clip"),
                    "growth": (c.edit or {}).get("growth"),
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
                storage.clip_file(job_id, cid, version, "video"), args.out / f"clip-{rank:02d}-{cid[:6]}.mp4"
            )
            srt = storage.clip_file(job_id, cid, version, "srt")
            if srt.exists():
                shutil.copyfile(srt, args.out / f"clip-{rank:02d}-{cid[:6]}.srt")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
