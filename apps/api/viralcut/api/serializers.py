"""Database rows -> API JSON."""

from __future__ import annotations

import datetime as dt
from typing import Any

from ..db import Clip, Job
from ..jobs import STAGES, initial_stages
from ..pipeline.scoring import DIMENSION_LABELS, DIMENSIONS


def _iso(value: dt.datetime | None) -> str | None:
    return value.replace(tzinfo=dt.UTC).isoformat().replace("+00:00", "Z") if value else None


def decision_notes(clip: Clip) -> list[str]:
    """Plain-language list of what the engine decided for this clip, and why."""
    framing = clip.framing or {}
    edit = framing.get("edit") or {}
    notes = list(framing.get("notes", []))
    style = edit.get("style")
    if style:
        notes.append(
            f"Video style: {style['label']} at {style['intensity']} intensity ({style['reason']}); "
            f"pauses kept up to {style['max_pause'] if style['max_pause'] is not None else 'any length'} s, "
            f"cuts: {style['transition']}, audio: {style['audio']}."
        )
    profile = edit.get("profile")
    if profile:
        why = profile["evidence"][0] if profile.get("evidence") else "no single signal dominated"
        notes.append(f"Content type: {profile['name']} ({why}).")
    notes += edit.get("notes", [])
    for z in edit.get("zooms", []):
        ev = z.get("evidence", {})
        notes.append(
            f"Punch-in {z['t0']:.1f}-{z['t1']:.1f}s, {z['amp'] * 100:.0f}%: {z['reason']}"
            + (
                f" on \u201c{ev['word']}\u201d (+{ev.get('lift_db', 0):.0f} dB over the speaker's baseline)"
                if ev
                else ""
            )
            + "."
        )
    align = framing.get("align") or {}
    if align.get("method") == "ctc":
        notes.append(
            f"Captions aligned to the audio: {align['aligned']} of {align['words']} words, median correction "
            f"{align['median_shift_s'] * 1000:+.0f} ms."
        )
    for line in (edit.get("audio") or {}).get("applied") or []:
        notes.append(f"Audio: {line}.")
    return notes


def clip_out(clip: Clip, detail: bool = False) -> dict[str, Any]:
    dims = clip.dimensions or {}
    values = dims.get("values", {})
    weights = dims.get("weights", {})
    base = f"/api/jobs/{clip.job_id}/clips/{clip.id}"
    v = f"?v={clip.version}"
    has_file = clip.version > 0 and clip.qa.get("passed", False) if clip.qa else False
    out: dict[str, Any] = {
        "id": clip.id,
        "rank": clip.rank,
        "status": clip.status,
        "progress": clip.progress,
        "start": clip.start,
        "end": clip.end,
        "duration": clip.duration,
        "score": clip.score,
        "score_stale": bool(dims.get("stale", False)),
        "engine": dims.get("engine", "heuristic"),
        "dimensions": [
            {
                "key": k,
                "label": DIMENSION_LABELS[k],
                "value": round(100 * float(values.get(k, 0.0))),
                "weight": round(float(weights.get(k, 0.0)), 3),
            }
            for k in DIMENSIONS
        ],
        "explanation": clip.explanation,
        "hook": clip.hook or {},
        "titles": clip.titles or [],
        "caption": clip.caption,
        "copy_source": clip.copy_source,
        "language": clip.language,
        "captions": clip.captions or {},
        "framing_mode": clip.framing_mode,
        "smart_trim": clip.smart_trim,
        "style": (clip.edit or {}).get("style", "auto"),
        "intensity": (clip.edit or {}).get("intensity", "balanced"),
        "quality": (clip.edit or {}).get("quality", "high"),
        "applied_style": ((clip.framing or {}).get("edit") or {}).get("style"),
        "variant": (clip.edit or {}).get("variant"),
        "kind": (clip.edit or {}).get("kind", "clip"),
        "aspect": (clip.edit or {}).get("aspect", "9:16"),
        "growth": (clip.edit or {}).get("growth"),
        "hook_first": (clip.edit or {}).get("hook_first"),
        "editors": (clip.edit or {}).get("editors") or {},
        "recommendation": (clip.edit or {}).get("recommendation"),
        "layouts": (clip.framing or {}).get("layouts", []),
        "framing_notes": decision_notes(clip),
        "has_captions_burned": bool((clip.qa or {}).get("captions_burned", False)),
        "has_srt": bool(clip.pages),
        "qa": {
            k: (clip.qa or {}).get(k)
            for k in (
                "passed",
                "checks",
                "width",
                "height",
                "fps",
                "size_bytes",
                "bit_rate",
                "mean_volume_db",
                "integrated_lufs",
                "true_peak_db",
            )
        }
        if clip.qa
        else None,
        "version": clip.version,
        "error": clip.error,
        "urls": {
            "video": f"{base}/video{v}",
            "download": f"{base}/download{v}",
            "srt": f"{base}/srt{v}",
            "thumbnail": f"{base}/thumbnail{v}",
        }
        if has_file
        else None,
    }
    if detail:
        out["transcript"] = clip.transcript
        out["pages"] = clip.pages or []
        out["details"] = dims.get("details", {})
        out["alignment"] = (clip.framing or {}).get("align")
        out["edit"] = (clip.framing or {}).get("edit")
    return out


def job_out(job: Job, include_clips: bool = True) -> dict[str, Any]:
    stages = job.stages or initial_stages()
    media = job.media or {}
    out: dict[str, Any] = {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "stages": [
            {
                "key": key,
                "label": label,
                "status": stages.get(key, {}).get("status", "pending"),
                "progress": stages.get(key, {}).get("progress", 0.0),
                "elapsed_s": stages.get(key, {}).get("elapsed_s"),
            }
            for key, label, _ in STAGES
        ],
        "error": job.error,
        "filename": job.original_filename,
        "size_bytes": job.size_bytes,
        "media": {
            k: media.get(k)
            for k in ("duration", "width", "height", "fps", "video_codec", "audio_codec", "has_audio")
        },
        "options": job.options or {},
        "analysis": job.analysis or {},
        "attempts": job.attempts,
        "cancel_requested": job.cancel_requested,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
        "expires_at": _iso(job.expires_at),
    }
    if include_clips:
        # variations sit right after the version they belong to
        ordered = sorted(
            job.clips, key=lambda c: (c.rank, int(((c.edit or {}).get("variant") or {}).get("index", 1)))
        )
        out["clips"] = [clip_out(c) for c in ordered]
    return out
