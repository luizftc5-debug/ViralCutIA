"""HTTP API. Job ids are unguessable capability tokens: knowing one is what grants access."""

from __future__ import annotations

import datetime as dt
import json
import logging
import tempfile
import time
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from starlette.background import BackgroundTask

from .. import __version__
from ..config import get_settings
from ..db import Clip, ClipMetric, Job, JobLog, session_scope, utcnow
from ..jobs import enqueue
from ..options import SAFE_AREAS, CaptionOptions, JobOptions
from ..pipeline import captions as cap
from ..pipeline.orchestrator import Artifacts, effective_words
from ..pipeline.probe import MediaInfo, ProbeError, probe
from ..pipeline.transcribe import plan_asr
from ..security import UploadRejected, is_valid_id, safe_stem, sanitize_filename, validate_extension_and_magic
from ..storage import get_storage
from .serializers import clip_out, job_out

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_upload_hits: dict[str, deque[float]] = defaultdict(deque)


def _rate_limit(ip: str) -> None:
    limit = get_settings().uploads_per_hour_per_ip
    if limit <= 0:
        return
    now = time.time()
    hits = _upload_hits[ip]
    while hits and now - hits[0] > 3600:
        hits.popleft()
    if len(hits) >= limit:
        raise HTTPException(429, "Too many uploads from this address. Please try again later.")
    hits.append(now)


def _get_job(s, job_id: str) -> Job:  # noqa: ANN001
    if not is_valid_id(job_id):
        raise HTTPException(404, "Job not found.")
    job = s.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


def _get_clip(s, job_id: str, clip_id: str) -> Clip:  # noqa: ANN001
    _get_job(s, job_id)
    if not is_valid_id(clip_id):
        raise HTTPException(404, "Clip not found.")
    clip = s.get(Clip, clip_id)
    if clip is None or clip.job_id != job_id:
        raise HTTPException(404, "Clip not found.")
    return clip


# ----------------------------------------------------------------------------- meta


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


@router.get("/config")
def config() -> dict[str, Any]:
    s = get_settings()
    return {
        "version": __version__,
        "limits": {
            "max_upload_mb": s.max_upload_mb,
            "max_duration_s": s.max_duration_s,
            "min_duration_s": s.min_duration_s,
            "formats": [e.lstrip(".").upper() for e in s.allowed_extensions],
            "retention_hours": s.retention_hours,
        },
        "caption_styles": [
            {
                "key": k,
                "label": v.label,
                "description": v.description,
                "accent": v.accent,
                "font": v.font,
                "uppercase": v.upper,
            }
            for k, v in cap.STYLES.items()
        ],
        "caption_fonts": [
            {"key": "auto", "label": "Style default"},
            {"key": "inter", "label": "Inter"},
            {"key": "poppins", "label": "Poppins"},
            {"key": "anton", "label": "Anton"},
            {"key": "dm_serif", "label": "DM Serif Display"},
        ],
        "platforms": list(SAFE_AREAS),
        "engine": {
            "transcription": f"faster-whisper:{plan_asr().model}",
            "transcription_mode": plan_asr().mode,
            "forced_alignment": s.align_mode != "off",
            "narrative_analysis": f"llm:{s.llm_model}" if s.llm_active else "heuristic",
            "llm_enabled": s.llm_active,
            "min_score": s.min_score,
        },
    }


# ----------------------------------------------------------------------------- upload


@router.post("/jobs", status_code=201)
async def upload(request: Request, filename: str = "video.mp4") -> dict[str, Any]:
    """Upload a video as the raw request body (streamed to disk, size-limited, content-validated)."""
    s = get_settings()
    storage = get_storage()
    _rate_limit(request.client.host if request.client else "unknown")
    max_bytes = s.max_upload_mb * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise HTTPException(413, f"The file is larger than the {s.max_upload_mb} MB limit.")

    display_name = sanitize_filename(filename)
    with session_scope() as db:
        job = Job(original_filename=display_name, status="uploading")
        db.add(job)
        db.flush()
        job_id = job.id

    tmp = storage.job_dir(job_id) / "upload.part"
    size = 0
    ext = family = None
    try:
        with tmp.open("wb") as fh:
            head = b""
            async for chunk in request.stream():
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"The file is larger than the {s.max_upload_mb} MB limit.")
                if ext is None:
                    head += chunk
                    if len(head) >= 64:
                        # reject wrong content before accepting the rest of the body
                        ext, family = validate_extension_and_magic(
                            display_name, head[:64], s.allowed_extensions
                        )
                fh.write(chunk)
            if ext is None:
                ext, family = validate_extension_and_magic(display_name, head[:64], s.allowed_extensions)
        if size == 0:
            raise UploadRejected("The upload was empty.")
        source = storage.source_path(job_id, ext)
        tmp.replace(source)
        media: MediaInfo = probe(source, family)
        if media.duration > s.max_duration_s:
            raise UploadRejected(f"The video is longer than the {s.max_duration_s // 60} minute limit.")
        if media.duration < s.min_duration_s:
            raise UploadRejected("The video is too short to cut clips from.")
    except (UploadRejected, ProbeError) as exc:
        _discard(job_id)
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        _discard(job_id)
        raise
    except Exception:
        _discard(job_id)
        log.exception("upload failed")
        raise HTTPException(500, "The upload could not be processed.") from None

    with session_scope() as db:
        job = db.get(Job, job_id)
        job.status = "uploaded"
        job.source_ext = ext
        job.size_bytes = size
        job.media = media.to_dict()
        job.expires_at = utcnow() + dt.timedelta(hours=s.retention_hours)
        return job_out(job)


def _discard(job_id: str) -> None:
    get_storage().delete_job(job_id)
    with session_scope() as db:
        job = db.get(Job, job_id)
        if job is not None:
            db.delete(job)


# ----------------------------------------------------------------------------- jobs


@router.post("/jobs/{job_id}/start")
def start(job_id: str, options: JobOptions) -> dict[str, Any]:
    with session_scope() as s:
        job = _get_job(s, job_id)
        if job.status not in ("uploaded", "failed", "canceled", "completed", "review"):
            raise HTTPException(409, "This job is already being processed.")
        for clip in list(job.clips):
            s.delete(clip)
        job.options = options.model_dump()
        job.attempts = 0
        job.analysis = {}
    import shutil

    shutil.rmtree(get_storage().job_dir(job_id) / "clips", ignore_errors=True)
    enqueue(job_id)
    with session_scope() as s:
        return job_out(_get_job(s, job_id))


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with session_scope() as s:
        return job_out(_get_job(s, job_id))


@router.get("/jobs/{job_id}/logs")
def get_logs(job_id: str, after: int = 0) -> dict[str, Any]:
    with session_scope() as s:
        _get_job(s, job_id)
        rows = (
            s.execute(
                select(JobLog)
                .where(JobLog.job_id == job_id, JobLog.id > after)
                .order_by(JobLog.id)
                .limit(500)
            )
            .scalars()
            .all()
        )
        return {
            "logs": [
                {
                    "id": r.id,
                    "ts": r.ts.isoformat() + "Z",
                    "level": r.level,
                    "stage": r.stage,
                    "message": r.message,
                }
                for r in rows
            ]
        }


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict[str, Any]:
    with session_scope() as s:
        job = _get_job(s, job_id)
        if job.status in ("queued", "review"):
            job.status = "canceled"
            job.finished_at = utcnow()
        elif job.status == "processing":
            job.cancel_requested = True
        return job_out(job)


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    with session_scope() as s:
        job = _get_job(s, job_id)
        if job.status == "processing":
            raise HTTPException(409, "Cancel the job before deleting it.")
        s.delete(job)
    get_storage().delete_job(job_id)
    return Response(status_code=204)


# ----------------------------------------------------------------------------- transcript review


class WordEdit(BaseModel):
    index: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=80)


class ReplaceRule(BaseModel):
    find: str = Field(min_length=1, max_length=120)
    replace: str = Field(max_length=120)
    case_sensitive: bool = False


class TranscriptUpdate(BaseModel):
    words: list[WordEdit] = Field(default_factory=list, max_length=5000)
    replace: list[ReplaceRule] = Field(default_factory=list, max_length=50)


def _transcript_path(job_id: str) -> Path:
    return get_storage().analysis_dir(job_id) / "transcript.json"


@router.get("/jobs/{job_id}/transcript")
def get_transcript(job_id: str) -> dict[str, Any]:
    """The full recognised transcript with word timings, plus what the recogniser discarded."""
    with session_scope() as s:
        job = _get_job(s, job_id)
        if job.status not in ("review", "completed"):
            raise HTTPException(409, "The transcript is available once the video has been analysed.")
    path = _transcript_path(job_id)
    if not path.exists():
        raise HTTPException(404, "No transcript for this video.")
    data = json.loads(path.read_text(encoding="utf-8"))
    original = path.with_name("transcript.original.json")
    return {
        "language": data.get("language", ""),
        "engine": data.get("engine", ""),
        "words": [
            {
                "index": i,
                "text": w["text"],
                "start": w["start"],
                "end": w["end"],
                "confidence": w.get("prob", 1.0),
            }
            for i, w in enumerate(data.get("words", []))
        ],
        "quality": data.get("quality", {}),
        "edited": original.exists(),
    }


@router.put("/jobs/{job_id}/transcript")
def update_transcript(job_id: str, body: TranscriptUpdate) -> dict[str, Any]:
    """Correct words of the whole transcript (by index or find-and-replace).

    Only the text changes; every word keeps the time it was spoken at. Before rendering the
    corrections simply apply to the upcoming clips; on a finished job the clips that contain
    corrected words are queued for re-rendering.
    """
    from ..pipeline.transcribe import Transcript
    from ..pipeline.transcript_edit import Replacement, apply_replacements, apply_word_edits

    with session_scope() as s:
        job = _get_job(s, job_id)
        if job.status not in ("review", "completed"):
            raise HTTPException(409, "The transcript can be corrected once the video has been analysed.")
    path = _transcript_path(job_id)
    if not path.exists():
        raise HTTPException(404, "No transcript for this video.")
    transcript = Transcript.from_dict(json.loads(path.read_text(encoding="utf-8")))
    words = transcript.words
    words, by_index = apply_word_edits(words, {e.index: e.text for e in body.words})
    words, by_rule = apply_replacements(
        words, [Replacement(r.find, r.replace, r.case_sensitive) for r in body.replace]
    )
    if not by_index and not by_rule and len(words) == len(transcript.words):
        return {"changed": 0, "affected_clips": [], "words": len(words)}
    original = path.with_name("transcript.original.json")
    if not original.exists():  # keep what the recogniser actually produced
        original.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    changed_times = [(words[i].start, words[i].end) for i in by_index if i < len(words)]
    changed_times += [(words[i].start, words[i].end) for i in by_rule if i < len(words)]
    transcript.words = words
    path.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False), encoding="utf-8")

    affected: list[str] = []
    with session_scope() as s:
        job = _get_job(s, job_id)
        for clip in job.clips:
            hit = any(a < clip.end and b > clip.start for a, b in changed_times)
            if not hit:
                continue
            affected.append(clip.id)
            if job.status == "completed" and clip.status == "ready":
                clip.text_edits = {}  # a global correction supersedes older per-clip text, which was keyed to the old words
                clip.status, clip.progress, clip.error, clip.updated_at = "queued", 0.0, "", utcnow()
    return {"changed": len(set(by_index) | set(by_rule)), "affected_clips": affected, "words": len(words)}


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict[str, Any]:
    """Continue a job that was paused for transcript review."""
    with session_scope() as s:
        job = _get_job(s, job_id)
        if job.status != "review":
            raise HTTPException(409, "This job is not waiting for a transcript review.")
        job.status = "queued"
        job.queued_at = utcnow()
        job.cancel_requested = False
        job.analysis = {**(job.analysis or {}), "review": "resume"}
        return job_out(job)


# ----------------------------------------------------------------------------- clips


class PageEdit(BaseModel):
    index: int = Field(ge=0)
    text: str = Field(max_length=400)


class ClipUpdate(BaseModel):
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, gt=0)
    captions: CaptionOptions | None = None
    framing_mode: Literal["auto", "face", "split", "blur", "center"] | None = None
    smart_trim: bool | None = None
    page_edits: list[PageEdit] | None = Field(default=None, max_length=500)


@router.get("/jobs/{job_id}/clips/{clip_id}")
def get_clip(job_id: str, clip_id: str) -> dict[str, Any]:
    with session_scope() as s:
        clip = _get_clip(s, job_id, clip_id)
        out = clip_out(clip, detail=True)
        out["source_duration"] = (clip.job.media or {}).get("duration")
        return out


@router.patch("/jobs/{job_id}/clips/{clip_id}")
def update_clip(job_id: str, clip_id: str, body: ClipUpdate) -> dict[str, Any]:
    """Apply studio edits and queue a re-render of this clip."""
    storage = get_storage()
    with session_scope() as s:
        clip = _get_clip(s, job_id, clip_id)
        job = clip.job
        if job.status != "completed":
            raise HTTPException(409, "The job is still processing.")
        if clip.status in ("queued", "rendering"):
            raise HTTPException(409, "This clip is already being rendered.")
        media_duration = float((job.media or {}).get("duration") or 0)

        if body.page_edits:
            # corrections are applied to the words of the version currently on screen
            media = MediaInfo.from_dict(job.media)
            art = Artifacts.load(storage.analysis_dir(job_id), media)
            words = effective_words(clip.start, clip.end, clip.text_edits or {}, art.transcript)
            pages = clip.pages or []
            for edit in sorted(body.page_edits, key=lambda e: e.index, reverse=True):
                if edit.index >= len(pages):
                    raise HTTPException(422, "Caption index out of range.")
                page = pages[edit.index]
                words = cap.apply_page_edit(words, int(page["k0"]), int(page["k1"]), edit.text.strip())
            clip.text_edits = {
                "cov": [clip.start, clip.end],
                "words": [{"text": w.text, "start": w.start, "end": w.end, "prob": w.prob} for w in words],
            }

        start = clip.start if body.start is None else body.start
        end = clip.end if body.end is None else body.end
        end = min(end, media_duration) if media_duration else end
        if end - start < 3.0:
            raise HTTPException(422, "A clip must be at least 3 seconds long.")
        if end - start > 180.0:
            raise HTTPException(422, "A clip can be at most 180 seconds long.")
        if (start, end) != (clip.start, clip.end):
            dims = dict(clip.dimensions or {})
            dims["stale"] = True  # the score describes the originally selected boundaries
            clip.dimensions = dims
        clip.start, clip.end = round(start, 3), round(end, 3)
        if body.captions is not None:
            clip.captions = body.captions.model_dump()
        if body.framing_mode is not None:
            clip.framing_mode = body.framing_mode
        if body.smart_trim is not None:
            clip.smart_trim = body.smart_trim
        clip.status = "queued"
        clip.progress = 0.0
        clip.error = ""
        clip.updated_at = utcnow()
        s.flush()
        return clip_out(clip, detail=True)


def _clip_file(job_id: str, clip_id: str, kind: str) -> tuple[Path, Clip, str]:
    with session_scope() as s:
        clip = _get_clip(s, job_id, clip_id)
        path = get_storage().clip_file(job_id, clip_id, clip.version, kind)
        stem = safe_stem(clip.job.original_filename, "video")
        rank = clip.rank
    if not path.exists():
        raise HTTPException(404, "File not available.")
    return path, clip, f"{stem}-clip-{rank:02d}"


@router.get("/jobs/{job_id}/clips/{clip_id}/video")
def clip_video(job_id: str, clip_id: str) -> FileResponse:
    path, _, _ = _clip_file(job_id, clip_id, "video")
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/jobs/{job_id}/clips/{clip_id}/download")
def clip_download(job_id: str, clip_id: str) -> FileResponse:
    path, _, name = _clip_file(job_id, clip_id, "video")
    return FileResponse(path, media_type="video/mp4", filename=f"{name}.mp4")


@router.get("/jobs/{job_id}/clips/{clip_id}/srt")
def clip_srt(job_id: str, clip_id: str) -> FileResponse:
    path, _, name = _clip_file(job_id, clip_id, "srt")
    return FileResponse(path, media_type="application/x-subrip", filename=f"{name}.srt")


@router.get("/jobs/{job_id}/clips/{clip_id}/thumbnail")
def clip_thumbnail(job_id: str, clip_id: str) -> FileResponse:
    path, _, _ = _clip_file(job_id, clip_id, "thumb")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/jobs/{job_id}/export.zip")
def export_zip(job_id: str) -> FileResponse:
    """All ready clips (MP4 + SRT) plus their titles/captions/scores in one archive."""
    storage = get_storage()
    with session_scope() as s:
        job = _get_job(s, job_id)
        stem = safe_stem(job.original_filename, "video")
        entries = []
        manifest = []
        for clip in job.clips:
            if clip.status != "ready":
                continue
            name = f"{stem}-clip-{clip.rank:02d}"
            video = storage.clip_file(job_id, clip.id, clip.version, "video")
            if not video.exists():
                continue
            srt = storage.clip_file(job_id, clip.id, clip.version, "srt")
            entries.append((video, f"{name}.mp4"))
            if srt.exists():
                entries.append((srt, f"{name}.srt"))
            manifest.append(
                {
                    "file": f"{name}.mp4",
                    "rank": clip.rank,
                    "score": clip.score,
                    "source_start_s": clip.start,
                    "source_end_s": clip.end,
                    "duration_s": clip.duration,
                    "hook": (clip.hook or {}).get("text", ""),
                    "why": clip.explanation,
                    "titles": clip.titles,
                    "caption": clip.caption,
                    "copy_source": clip.copy_source,
                    "language": clip.language,
                }
            )
    if not entries:
        raise HTTPException(404, "No clips are ready to export.")
    tmp = tempfile.NamedTemporaryFile(prefix="viralcut-export-", suffix=".zip", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_STORED) as zf:
        for path, arcname in entries:
            zf.write(path, arcname)
        zf.writestr("clips.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        lines = []
        for m in manifest:
            lines += [f"{m['file']}  —  score {m['score']:.0f}/100", f"Why: {m['why']}"]
            lines += [f"Title option: {t}" for t in m["titles"]]
            lines += [f"Caption: {m['caption']}", ""]
        zf.writestr("publishing-notes.txt", "\n".join(lines))
    return FileResponse(
        tmp.name,
        media_type="application/zip",
        filename=f"{stem}-clips.zip",
        background=BackgroundTask(lambda: Path(tmp.name).unlink(missing_ok=True)),
    )


# ----------------------------------------------------------------------------- performance feedback


class MetricsIn(BaseModel):
    """Observed performance of a published clip, reported by its owner."""

    platform: Literal["shorts", "reels", "tiktok", "other"]
    views: int | None = Field(default=None, ge=0)
    likes: int | None = Field(default=None, ge=0)
    comments: int | None = Field(default=None, ge=0)
    shares: int | None = Field(default=None, ge=0)
    saves: int | None = Field(default=None, ge=0)
    avg_watch_time_s: float | None = Field(default=None, ge=0)
    avg_percent_watched: float | None = Field(default=None, ge=0, le=1000)
    retention_3s: float | None = Field(default=None, ge=0, le=100)
    hours_since_publish: float | None = Field(default=None, ge=0)


@router.post("/jobs/{job_id}/clips/{clip_id}/metrics", status_code=201)
def add_metrics(job_id: str, clip_id: str, body: MetricsIn) -> dict[str, Any]:
    with session_scope() as s:
        clip = _get_clip(s, job_id, clip_id)
        dims = clip.dimensions or {}
        row = ClipMetric(
            clip_id=clip.id,
            platform=body.platform,
            metrics=body.model_dump(exclude={"platform"}, exclude_none=True),
            features={
                "score": clip.score,
                "dimensions": dims.get("values", {}),
                "engine": dims.get("engine"),
                "duration": clip.duration,
                "hook_types": [t["type"] for t in (clip.hook or {}).get("types", [])],
                "caption_style": (clip.captions or {}).get("style")
                if (clip.captions or {}).get("enabled")
                else None,
                "layouts": (clip.framing or {}).get("layouts", []),
            },
        )
        s.add(row)
        s.flush()
        return {"id": row.id}


@router.get("/jobs/{job_id}/clips/{clip_id}/metrics")
def list_metrics(job_id: str, clip_id: str) -> dict[str, Any]:
    with session_scope() as s:
        clip = _get_clip(s, job_id, clip_id)
        return {
            "metrics": [
                {
                    "id": m.id,
                    "platform": m.platform,
                    "reported_at": m.reported_at.isoformat() + "Z",
                    **m.metrics,
                }
                for m in clip.metrics
            ]
        }


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=status)
