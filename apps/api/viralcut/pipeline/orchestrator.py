"""Runs a job through every pipeline stage and persists real progress and results."""

from __future__ import annotations

import json
import logging
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import cache
from ..config import get_settings
from ..db import Clip, Job, session_scope, utcnow
from ..jobs import JobReporter, add_log
from ..options import CaptionOptions, JobOptions
from ..resources import active_plan, use_profile
from ..security import FAMILY_DEMUXER, sniff_container
from ..storage import LocalStorage, get_storage
from . import captions as cap
from . import llm
from .align import align_words, refine_with_energy
from .audio import AudioSignals, analyze_audio, decode_audio, silent_signals
from .candidates import signal_candidates, speech_candidates
from .copywriting import extractive_copy
from .editing import ContentProfile, Zoom, classify_content, plan_zooms, tighten_opening
from .ffmpeg import Cancelled, FFmpegError
from .probe import MediaInfo, ProbeError, probe
from .qa import QAError, verify_clip
from .reframe import FramingPlan, Shot, plan_framing
from .render import (
    add_punch_transitions,
    assign_zooms,
    build_segments,
    jump_cuts,
    make_thumbnail,
    output_size,
    plan_audio_polish,
    render_clip,
    segment_boundaries,
    segments_time_map,
)
from .scoring import AnalysisContext, score_signal, score_speech
from .selection import auto_count, select
from .styles import STYLES as VIDEO_STYLES
from .styles import EditParams, edit_params
from .styles import recommend as recommend_style
from .styles import resolve as resolve_style
from .text import build_sentences, norm
from .transcribe import Transcript, TranscriptionUnavailable, Word, release_models, transcribe
from .transcribe import cache_key as transcribe_cache_key
from .trimming import plan_keeps
from .variations import plan_variations
from .visual import VisualSignals, analyze_visual

log = logging.getLogger(__name__)

MIN_SPEECH_WORDS = 12


class UserFacingError(RuntimeError):
    """An error whose message can be shown to the user as-is."""


@dataclass
class Artifacts:
    """Analysis results persisted per job so clips can be re-rendered without re-analysing."""

    media: MediaInfo
    family: str | None
    transcript: Transcript
    audio: AudioSignals
    visual: VisualSignals
    keywords: list[str]

    def save(self, directory: Path) -> None:
        (directory / "transcript.json").write_text(json.dumps(self.transcript.to_dict(), ensure_ascii=False))
        (directory / "meta.json").write_text(json.dumps({"family": self.family, "keywords": self.keywords}))
        a, v = self.audio.to_arrays(), self.visual.to_arrays()
        np.savez_compressed(
            directory / "signals.npz",
            rms_db=a["rms_db"],
            audio_meta=a["meta"],
            motion=v["motion"],
            cuts=v["cuts"],
        )

    @classmethod
    def load(cls, directory: Path, media: MediaInfo) -> Artifacts:
        meta = json.loads((directory / "meta.json").read_text())
        transcript = Transcript.from_dict(json.loads((directory / "transcript.json").read_text()))
        with np.load(directory / "signals.npz") as z:
            audio = AudioSignals.from_arrays(z["rms_db"], z["audio_meta"])
            visual = VisualSignals.from_arrays(z["motion"], z["cuts"])
        return cls(media, meta.get("family"), transcript, audio, visual, meta.get("keywords", []))


def _family(path: Path) -> str | None:
    with path.open("rb") as fh:
        return sniff_container(fh.read(64))


def _mid(w: Word) -> float:
    return (w.start + w.end) / 2


def effective_words(start: float, end: float, text_edits: dict, transcript: Transcript) -> list[Word]:
    """Words spoken inside [start, end) on the source timeline, with the user's corrections applied.

    Leading disfluencies ("uh", "hum") are dropped here, in the one place every code path reads
    words from, so caption indices used by the studio always refer to the same list.
    """
    base = [w for w in transcript.words if start <= _mid(w) < end]
    if not text_edits or text_edits.get("words") is None:
        return tighten_opening(base)[0]
    c0, c1 = text_edits.get("cov", [0.0, 0.0])
    edited = [Word(w["text"], w["start"], w["end"], w.get("prob", 1.0)) for w in text_edits["words"]]
    edited = [w for w in edited if start <= _mid(w) < end]
    outside = [w for w in base if not (c0 <= _mid(w) < c1)]
    return tighten_opening(sorted(outside + edited, key=lambda w: w.start))[0]


# --------------------------------------------------------------------------- per-clip steps


class ClipCache:
    """Per-job memo of a clip's expensive, deterministic steps (alignment, face tracking, loudness).

    Keys hash every input of the step, so a studio edit that does not touch those inputs (caption
    style, colours, font, burn-in on/off...) reuses them, and any edit that does (new in/out
    points, corrected words, another framing mode) recomputes them. Files live in the job's
    analysis directory and are deleted with the job.
    """

    def __init__(self, directory: Path | None) -> None:
        self.dir = directory
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def get(self, *parts) -> dict | None:  # noqa: ANN002
        if self.dir is None:
            return None
        try:
            return json.loads((self.dir / f"{cache.make_key(*parts)}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, value: dict, *parts) -> None:  # noqa: ANN002
        if self.dir is None:
            return
        try:
            (self.dir / f"{cache.make_key(*parts)}.json").write_text(json.dumps(value), encoding="utf-8")
        except OSError:
            pass


NO_CACHE = ClipCache(None)


def _words_sig(words: list[Word]) -> list:
    return [(w.text, round(w.start, 3), round(w.end, 3)) for w in words]


@dataclass
class ClipPlan:
    clip_id: str
    start: float
    end: float
    words: list[Word]  # source timeline
    plan: FramingPlan | None = None
    segments: list | None = None
    out_words: list[Word] | None = None  # output timeline
    align: dict | None = None  # how the word timestamps were obtained (CTC alignment report)
    profile: ContentProfile | None = None
    zooms: list[Zoom] = field(default_factory=list)
    edit_notes: list[str] = field(default_factory=list)
    audio_polish: object | None = None
    duration: float = 0.0
    ass_path: Path | None = None
    version: int = 1
    style: str = "auto"  # video style requested for this clip (pipeline/styles.py)
    intensity: str = "balanced"
    params: EditParams | None = None  # concrete editing parameters applied
    quality: str = "high"
    cap_opts: CaptionOptions | None = None  # this clip's caption options (variations may differ)


def step_align(
    cp: ClipPlan,
    art: Artifacts,
    source: Path,
    should_cancel=None,
    memo: ClipCache = NO_CACHE,  # noqa: ANN001
) -> dict:
    """Re-time the clip's words to the waveform (CTC forced alignment, then loudness refinement).

    Whisper's native timestamps lead the audio by ~0.2 s on average; captions make that visible.
    Failure here must never block rendering: the words then keep Whisper's timing.
    """
    key = (
        "align-v1",
        cp.start,
        cp.end,
        _words_sig(cp.words),
        art.transcript.language,
        get_settings().align_mode,
    )
    hit = memo.get(*key)
    if hit is not None:
        cp.words = [Word(w[0], w[1], w[2], w[3]) for w in hit["words"]]
        cp.align, cp.start = hit["info"], hit["start"]
        cp.edit_notes.extend(hit["notes"])
        return cp.align
    notes_before = len(cp.edit_notes)
    info = _align(cp, art, source, should_cancel)
    memo.put(
        {"words": [(w.text, w.start, w.end, w.prob) for w in cp.words], "info": info, "start": cp.start,
         "notes": cp.edit_notes[notes_before:]},
        *key,
    )  # fmt: skip
    return info


def _align(cp: ClipPlan, art: Artifacts, source: Path, should_cancel=None) -> dict:  # noqa: ANN001
    info: dict = {"method": "whisper", "note": "no words"}
    if not cp.words:
        cp.align = info
        return info
    lo = max(0.0, cp.start - 1.0)
    hi = min(art.media.duration, cp.end + 1.0)
    try:
        if art.media.has_audio:
            samples = decode_audio(source, art.family, hi - lo, None, should_cancel, seek=lo, limit=hi - lo)
            cp.words, stats = align_words(cp.words, samples, art.transcript.language, origin=lo)
            info = stats.to_dict()
    except Cancelled:
        raise
    except Exception as exc:  # alignment is an improvement, never a hard dependency
        log.warning("forced alignment failed for clip %s: %s", cp.clip_id, exc)
        info = {"method": "whisper", "note": f"alignment failed: {type(exc).__name__}"}
    cp.words, rs = refine_with_energy(cp.words, art.audio)
    info["energy_refined"] = {"starts": rs.moved_starts, "ends": rs.moved_ends}
    cp.align = info
    # tight in-point: the first frame is the first real word (a dropped disfluency or a long silent
    # lead-in is cut), with a short breath before it
    lead = cp.words[0].start - cp.start if cp.words else 0.0
    if lead > 0.35 and cp.end - (cp.words[0].start - 0.2) >= 3.0:
        cp.edit_notes.append(
            f"opening tightened: {lead - 0.2:.1f}s of lead-in removed so the clip starts on the first word"
        )
        cp.start = cp.words[0].start - 0.2
    return info


def _recommendation(cp: ClipPlan, art: Artifacts):  # noqa: ANN202
    """Style recommendation for a framed clip, from its measured signals (applied to a Clip row)."""
    dur = max(1.0, cp.end - cp.start)
    wpm = 60.0 * len(cp.words) / dur
    motion_pct = art.visual.motion_percentile(art.visual.motion_level(cp.start, cp.end))
    face = cp.plan.face_ratio if cp.plan else 0.0
    name = cp.profile.name if cp.profile else "general"

    def apply(clip: Clip) -> dict:
        dims = (clip.dimensions or {}).get("values", {})
        return recommend_style(dims, name, wpm, motion_pct, face)

    return apply


def framing_json(cp: ClipPlan) -> dict:
    out = cp.plan.to_dict()
    if cp.align is not None:
        out["align"] = cp.align
    out["edit"] = {
        "style": cp.params.to_dict() if cp.params else None,
        "profile": cp.profile.to_dict() if cp.profile else None,
        "zooms": [z.to_dict() for z in cp.zooms],
        "notes": cp.edit_notes,
        "audio": getattr(cp.audio_polish, "report", None),
    }
    return out


def step_framing(
    cp: ClipPlan,
    art: Artifacts,
    source: Path,
    framing_mode: str,
    max_pause: float | None,
    smart_trim: bool,
    on_progress=None,
    should_cancel=None,
    *,
    user_style: str = "auto",
    smart_zoom: bool = True,
    memo: ClipCache = NO_CACHE,
) -> None:  # noqa: ANN001
    """Decide the clip's reframing, pacing and punch-ins from what is actually in it."""
    from .render import output_fps

    fkey = ("frame-v1", round(cp.start, 4), round(cp.end, 4), framing_mode)
    hit = memo.get(*fkey)
    if hit is not None:
        cp.plan = FramingPlan.from_dict(hit)
        if on_progress:
            on_progress(1.0)
    else:
        cp.plan = plan_framing(
            source, art.family, art.media, cp.start, cp.end, framing_mode, on_progress, should_cancel
        )
        memo.put(cp.plan.to_dict(), *fkey)
    split = any(s.layout["type"] == "split" for s in cp.plan.shots)
    style = resolve_style(cp.style if cp.style != "auto" else user_style)
    forced = VIDEO_STYLES[style].profile if style != "auto" else None
    cp.profile = classify_content(
        cp.words,
        art.transcript.language,
        art.audio,
        art.visual,
        cp.start,
        cp.end,
        face_ratio=cp.plan.face_ratio,
        split_layout=split,
        user_style=forced or "auto",
    )
    # the video style (chosen, or picked from the detected content) sets pacing, moves and audio
    cp.params = edit_params(
        style, cp.intensity, cp.profile.name, cp.profile.evidence, (cp.profile.max_pause, cp.profile.zoom)
    )
    cp.profile.max_pause, cp.profile.zoom = cp.params.max_pause, dict(cp.params.zoom)
    pause = cp.params.max_pause
    keeps = plan_keeps(cp.words, cp.start, cp.end, art.audio, pause if smart_trim else None)
    fps = output_fps(art.media)
    cp.segments = build_segments(keeps, cp.plan.shots, cp.start, fps)
    if not cp.segments:
        cp.segments = build_segments(
            [(cp.start, cp.end)], [Shot(0.0, cp.end - cp.start, cp.plan.shots[0].layout)], cp.start, fps
        )
    tmap = segments_time_map(cp.segments, fps)
    cp.duration = tmap.duration
    cp.out_words = tmap.map_words(cp.words, cp.start)
    keywords = {norm(k) for k in art.keywords[:10]}
    zooms = plan_zooms(
        [w for w in cp.words],
        lambda t: tmap.to_output(t - cp.start),
        art.audio,
        keywords,
        cp.profile,
        cp.duration,
        segment_boundaries(cp.segments, fps),
        enabled=smart_zoom,
    )
    cp.zooms = assign_zooms(cp.segments, fps, zooms)
    if cp.params.transition == "punch" and smart_zoom:
        punches = add_punch_transitions(cp.segments, fps, amp=max(0.05, cp.params.zoom.get("amp", 0.06)))
        if punches:
            cp.edit_notes.append(f"{punches} jump cut(s) covered with a quick punch-in")
    cp.audio_polish = plan_audio_polish(art.audio, cp.start, cp.end, cp.params.audio)
    if cp.params.transition == "smooth":
        cp.audio_polish.join_fade = 0.025  # softer joins: no audible edge where a pause was removed
        cuts = len(jump_cuts(cp.segments))
        if cuts:
            cp.edit_notes.append(f"{cuts} join(s) softened (25 ms audio crossfade, no punch effects)")
    removed = (cp.end - cp.start) - cp.duration
    if removed > 0.05:
        cp.edit_notes.append(
            f"removed {removed:.1f}s of silence (kept pauses up to {pause if pause is not None else 'n/a'} s)"
        )
    if cp.profile.zoom["allowed"] and smart_zoom and not cp.zooms:
        cp.edit_notes.append("no punch-in: no word was stressed enough to justify one")


def step_captions(
    cp: ClipPlan, art: Artifacts, storage: LocalStorage, job_id: str, opts: CaptionOptions, platform: str
) -> list[dict]:
    """Write the SRT sidecar (whenever there is speech) and the ASS file (only when burn-in is on)."""
    if not cp.out_words:
        return []
    if opts.style == "auto":  # the caption design that belongs to the clip's video style
        opts = opts.model_copy(update={"style": cp.params.caption if cp.params else "high_retention"})
    ow, oh = output_size(art.media)
    if (
        opts.position == "bottom"
        and cp.plan is not None
        and any(s.layout["type"] == "split" for s in cp.plan.shots)
    ):
        # in a split screen the lower third is someone's face: captions sit on the seam instead
        opts = opts.model_copy(update={"position": "middle"})
        cp.plan.notes.append("captions moved to the middle so they do not cover the lower speaker")
    pages = cap.paginate(cp.out_words, opts, cp.duration, set(art.keywords[:10]))
    if not pages:
        return []
    center_y = None
    if cp.plan is not None and cp.plan.shots and all(s.layout["type"] == "blur" for s in cp.plan.shots):
        # letterboxed picture: put captions in the blurred band, clear of the gameplay / screen content
        center_y = cap.blur_band_anchor(
            opts.position, platform, art.media.width, art.media.height, ow, oh, opts.style, opts.size, opts
        )
        if center_y is not None:
            cp.plan.notes.append("captions placed in the blurred band so they do not cover the picture")
    ass = cap.build_ass(
        pages, opts, platform, ow, oh, center_y
    )  # also computes the line layout used by the SRT
    storage.clip_file(job_id, cp.clip_id, cp.version, "srt").write_text(
        cap.build_srt(pages), encoding="utf-8"
    )
    if opts.enabled:
        cp.ass_path = storage.clip_file(job_id, cp.clip_id, cp.version, "ass")
        cp.ass_path.write_text(ass, encoding="utf-8")
    return [p.to_dict() for p in pages]


def step_render(
    cp: ClipPlan,
    art: Artifacts,
    source: Path,
    storage: LocalStorage,
    job_id: str,
    on_progress=None,
    should_cancel=None,
    memo: ClipCache = NO_CACHE,
) -> dict:  # noqa: ANN001
    out = storage.clip_file(job_id, cp.clip_id, cp.version, "video")
    params = render_clip(
        source, art.family, art.media, cp.start, cp.segments, out, cp.ass_path, on_progress, should_cancel,
        cp.audio_polish, quality=cp.quality, loudness_memo=memo,
    )  # fmt: skip
    report = verify_clip(
        out,
        expected_width=params["width"],
        expected_height=params["height"],
        expected_duration=params["duration"],
        expected_fps=params["fps"],
        source_has_audio=art.media.has_audio,
    )
    report["render"] = params
    report["captions_burned"] = cp.ass_path is not None
    if not report["passed"]:
        failed = [k for k, ok in report["checks"].items() if not ok]
        out.unlink(missing_ok=True)
        raise QAError("quality checks failed: " + ", ".join(failed))
    try:
        make_thumbnail(
            out, storage.clip_file(job_id, cp.clip_id, cp.version, "thumb"), at=min(1.0, cp.duration / 3)
        )
    except FFmpegError:
        pass  # a missing poster frame is cosmetic
    if cp.ass_path is not None:
        cp.ass_path.unlink(missing_ok=True)
    return report


# --------------------------------------------------------------------------- full job


def run_job(job_id: str) -> None:
    """Entry point used by the worker. Never raises: outcomes are written to the database."""
    settings = get_settings()
    storage = get_storage()
    rep = JobReporter(job_id)
    with session_scope() as s:
        job = s.get(Job, job_id)
        profile = ((job.options or {}).get("performance") if job else None) or "auto"
    try:
        with use_profile(profile) as plan:
            rep.log(
                f"Resource profile: {plan.profile.upper()} ({plan.threads} threads per task, "
                f"{plan.render_workers} render at a time; recommended for this machine: {plan.recommended.upper()})."
            )
            _run(job_id, rep, storage)
    except Cancelled:
        _finish(job_id, "canceled", "")
        rep.log("Job canceled.", "warn")
        storage_cleanup_partial(storage, job_id)
    except (UserFacingError, ProbeError) as exc:
        _finish(job_id, "failed", str(exc))
        rep.log(f"Failed: {exc}", "error")
    except Exception as exc:  # unexpected: retry while attempts remain
        log.error("job %s crashed: %s\n%s", job_id, exc, traceback.format_exc())
        add_log(
            job_id, f"Internal error: {type(exc).__name__}: {str(exc)[:500]}", level="error", stage=rep.stage
        )
        with session_scope() as s:
            job = s.get(Job, job_id)
            attempts = job.attempts if job else settings.max_attempts
        if attempts < settings.max_attempts:
            with session_scope() as s:
                job = s.get(Job, job_id)
                if job is not None:
                    for clip in list(job.clips):
                        s.delete(clip)
                    job.analysis = {k: v for k, v in (job.analysis or {}).items() if k != "review"}
                    job.status = "queued"
                    job.queued_at = utcnow()
            add_log(job_id, f"Retrying (attempt {attempts + 1} of {settings.max_attempts}).", level="warn")
            storage_cleanup_partial(storage, job_id)
        else:
            _finish(job_id, "failed", "Something went wrong while processing this video. Please try again.")


def storage_cleanup_partial(storage: LocalStorage, job_id: str) -> None:
    """Remove temporary/partial artifacts of an interrupted run (the uploaded source is kept)."""
    import shutil

    shutil.rmtree(storage.job_dir(job_id) / "clips", ignore_errors=True)


def _finish(job_id: str, status: str, error: str) -> None:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        job.status = status
        job.error = error
        job.finished_at = utcnow()
        if status == "completed":
            job.progress = 1.0


def _run(job_id: str, rep: JobReporter, storage: LocalStorage) -> None:
    settings = get_settings()
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        options = JobOptions.model_validate(job.options or {})
        source = storage.source_path(job_id, job.source_ext)
        stored_media = dict(job.media or {})
        resume = (job.analysis or {}).get("review") == "resume"
        if not resume:
            # a retry starts clean
            for clip in list(job.clips):
                s.delete(clip)
    if not source.exists():
        raise UserFacingError("The uploaded file is no longer available. Please upload it again.")
    cancelled = rep.cancelled
    if resume:
        _resume(job_id, rep, storage, options, source)
        return

    # ---- 1. analyzing -----------------------------------------------------------------
    rep.start("analyzing")
    family = _family(source)
    if family not in FAMILY_DEMUXER:
        raise UserFacingError("The file is not a supported video container.")
    try:  # probed once at upload; probing again would only repeat the same ffprobe call
        media = MediaInfo.from_dict(stored_media) if stored_media.get("duration") else probe(source, family)
    except (TypeError, KeyError, ValueError):
        media = probe(source, family)
    rep.log(
        f"Source: {media.width}x{media.height} @ {media.fps_float:.2f} fps, {media.duration:.1f}s, "
        f"video={media.video_codec}, audio={media.audio_codec or 'none'}"
    )
    warnings: list[str] = []
    if media.is_hdr:
        warnings.append("HDR source: converted to SDR with a basic tone-mapping.")
    # Expensive results are cached by the *content* of the file: the same video processed again
    # (a retry, other clip lengths or styles, a duplicate upload) skips decoding and recognition.
    digest = cache.file_digest(source)
    lang = None if options.language == "auto" else options.language
    sig_key = cache.make_key("signals-v1", digest)
    cached_sig = cache.get_npz("signals", sig_key)
    asr_key = transcribe_cache_key(digest, lang, options.glossary) if media.has_audio else ""
    cached_tr = cache.get_json("transcripts", asr_key) if asr_key else None
    samples = None
    if not media.has_audio:
        warnings.append("The video has no audio track: clips were selected from visual signals only.")
    if cached_sig is not None:
        audio = AudioSignals.from_arrays(cached_sig["rms_db"], cached_sig["audio_meta"])
        visual = VisualSignals.from_arrays(cached_sig["motion"], cached_sig["cuts"])
        rep.log("Reused the audio and visual analysis of this exact file (content hash match).")
    else:
        if media.has_audio:
            samples = decode_audio(source, family, media.duration, rep.sub(0.0, 0.35), cancelled)
            audio = analyze_audio(samples)
        else:
            audio = silent_signals(media.duration)
        visual = analyze_visual(source, family, media.duration, rep.sub(0.35, 1.0), cancelled)
        a, v = audio.to_arrays(), visual.to_arrays()
        cache.put_npz(
            "signals", sig_key, rms_db=a["rms_db"], audio_meta=a["meta"], motion=v["motion"], cuts=v["cuts"]
        )
    rep.log(
        f"Audio: speech level {audio.speech_db:.0f} dBFS, noise floor {audio.noise_floor_db:.0f} dBFS. Visual: {len(visual.cuts)} shot changes."
    )
    rep.finish()

    # ---- 2. transcribing --------------------------------------------------------------
    rep.start("transcribing")
    transcript = Transcript()
    if cached_tr is not None:
        transcript = Transcript.from_dict(cached_tr)
        rep.log(
            f"Reused the transcript of this exact file ({len(transcript.words)} words, {transcript.engine}): "
            "same audio and recognition settings, so recognition was not repeated."
        )
        rep.finish()
    elif media.has_audio:
        if samples is None:
            samples = decode_audio(source, family, media.duration, None, cancelled)
        if len(samples):
            try:
                transcript = transcribe(samples, lang, rep.progress, cancelled, audio, options.glossary)
                q = transcript.quality
                rep.log(
                    f"Transcribed {len(transcript.words)} words, language={transcript.language or '?'} "
                    f"({transcript.engine}, {q.get('mode', '?')} mode)."
                )
                if q.get("dropped_segments"):
                    rep.log(
                        f"Discarded {len(q['dropped_segments'])} segment(s) with no evidence of speech "
                        "(music, noise or silence), so no words were invented."
                    )
                if q.get("enhancement", {}).get("applied"):
                    rep.log("Audio prepared for recognition: " + "; ".join(q["enhancement"]["applied"]) + ".")
                if q.get("requested_model") and q.get("model") != q.get("requested_model"):
                    warnings.append(
                        f"Speech model '{q['requested_model']}' was unavailable; a smaller model ({transcript.engine}) was used, so accuracy is lower."
                    )
                elif transcript.words:
                    # only results of the requested model are reused (a fallback is never cached)
                    cache.put_json("transcripts", asr_key, transcript.to_dict())
            except TranscriptionUnavailable as exc:
                rep.log(f"Transcription unavailable: {exc}", "warn")
                warnings.append(
                    "Speech recognition was unavailable: clips were selected from audio/visual signals only, without captions."
                )
            finally:
                if not active_plan().keep_asr_model:
                    release_models()  # ECO: give the model's memory back before rendering
        rep.finish()
    else:
        rep.finish(skipped=True)
    del samples

    # ---- 3. discovering ---------------------------------------------------------------
    rep.start("discovering")
    min_d, max_d = options.duration_range
    sentences = build_sentences(transcript.words, transcript.language)
    ctx = AnalysisContext.build(transcript.words, sentences, transcript.language, audio, visual, media)
    has_speech = len(transcript.words) >= MIN_SPEECH_WORDS and bool(sentences)
    if has_speech:
        cands = speech_candidates(sentences, transcript.words, media.duration, min_d, max_d)
        mode = "speech"
        if transcript.language[:2] not in ("en", "pt"):
            warnings.append(
                f"Text heuristics are tuned for English and Portuguese; for '{transcript.language}' the ranking relies on language-agnostic signals"
                + (" and the LLM." if settings.llm_active else ".")
            )
    else:
        cands = signal_candidates(audio, visual, media.duration, min_d, max_d)
        mode = "signals"
        if media.has_audio and "Speech recognition was unavailable" not in " ".join(warnings):
            warnings.append(
                "No usable speech was detected: clips were selected from audio/visual signals only (lower confidence)."
            )
    total = max(1, len(cands))
    for i, c in enumerate(cands):
        (score_speech if c.kind == "speech" else score_signal)(c, ctx)
        if i % 200 == 0:
            rep.progress(i / total)
            if cancelled():
                raise Cancelled()
    rep.log(f"Generated and scored {len(cands)} candidates ({mode} mode).")
    art = Artifacts(media, family, transcript, audio, visual, ctx.tfidf.keywords)
    art.save(storage.analysis_dir(job_id))
    rep.finish()

    # ---- 4. ranking -------------------------------------------------------------------
    rep.start("ranking")
    want = auto_count(media.duration) if options.clip_count == "auto" else int(options.clip_count)
    want = min(want, settings.max_clips)
    min_score = options.min_score if options.min_score is not None else settings.min_score
    if mode == "signals" and options.min_score is None:
        min_score = min(min_score, 30.0)  # signal-only scores are capped by the neutral text dimensions
    llm_info: dict = {"used": False}
    if settings.llm_active and mode == "speech":
        shortlist = select(
            cands,
            ctx,
            max_clips=settings.llm_max_candidates,
            min_score=0.0,
            max_overlap=0.5,
            max_similarity=0.8,
        )
        rep.log(f"Asking {settings.llm_model} to rate {len(shortlist)} shortlisted candidates.")
        llm_info = llm.enhance(shortlist, ctx)
        if llm_info.get("used"):
            pool = shortlist
            rep.log(f"LLM rated {llm_info['rated']} candidates.")
        else:
            pool = cands
            warnings.append(
                f"LLM analysis was not applied ({llm_info.get('reason', 'unknown reason')}); scores are heuristic."
            )
    else:
        pool = cands
    chosen = select(pool, ctx, max_clips=want, min_score=min_score)
    rep.progress(0.8, force=True)
    engine = "signals-only" if mode == "signals" else ("llm+signals" if llm_info.get("used") else "heuristic")
    best = max((c.score for c in cands), default=0.0)
    rep.log(
        f"Selected {len(chosen)} of {want} requested clips (min score {min_score:.0f}, best candidate {best:.0f})."
    )

    plans: list[ClipPlan] = []
    with session_scope() as s:
        job = s.get(Job, job_id)
        for rank, c in enumerate(chosen, 1):
            llm_copy = (c.details.get("llm") or {}) if c.source == "llm" else {}
            if llm_copy.get("titles"):
                titles, caption, copy_source = llm_copy["titles"], llm_copy.get("caption", ""), "llm"
            elif c.kind == "speech":
                titles, caption = extractive_copy(c, ctx)
                copy_source = "extractive"
            else:
                titles, caption, copy_source = [], "", "none"
            common = dict(
                job_id=job_id,
                rank=rank,
                status="pending",
                score=c.score,
                dimensions={"values": c.dims, "details": c.details, "engine": engine, "weights": ctx.weights},
                explanation=c.explanation,
                hook=c.hook,
                titles=titles,
                caption=caption,
                copy_source=copy_source,
                language=transcript.language,
                transcript=c.text,
                framing_mode=options.default_framing,
                smart_trim=options.trim_max_pause is not None,
                version=1,
            )
            clip = Clip(
                **common,
                start=round(c.start, 3),
                end=round(c.end, 3),
                captions=options.captions.model_dump(),
                edit={"style": options.style, "intensity": options.intensity, "quality": options.quality},
            )
            s.add(clip)
            s.flush()
            words = effective_words(c.start, c.end, {}, transcript)
            plans.append(
                ClipPlan(clip.id, c.start, c.end, words, style=options.style, intensity=options.intensity,
                         quality=options.quality, cap_opts=options.captions)
            )  # fmt: skip
            # alternative versions of the same moment: no new analysis, only another edit
            for var in plan_variations(c, ctx, options.variations, options.style, options.intensity, min_d):
                vcap = options.captions
                if var.caption_style and vcap.style != "auto":
                    vcap = vcap.model_copy(update={"style": var.caption_style})
                vclip = Clip(
                    **common,
                    start=round(var.start, 3),
                    end=round(var.end, 3),
                    captions=vcap.model_dump(),
                    edit={"style": var.style, "intensity": var.intensity, "quality": options.quality,
                          "variant": {**var.to_dict(), "of": clip.id}},
                )  # fmt: skip
                s.add(vclip)
                s.flush()
                plans.append(
                    ClipPlan(vclip.id, var.start, var.end, effective_words(var.start, var.end, {}, transcript),
                             style=var.style, intensity=var.intensity, quality=options.quality, cap_opts=vcap)
                )  # fmt: skip
        job.analysis = {
            "engine": engine,
            "mode": mode,
            "language": transcript.language,
            "language_probability": transcript.language_probability,
            "transcription": transcript.engine,
            "asr": {
                **transcript.quality,
                "dropped_segments": transcript.quality.get("dropped_segments", [])[:20],
            },
            "words": len(transcript.words),
            "candidates": len(cands),
            "requested": want,
            "selected": len(chosen),
            "variations": options.variations,
            "min_score": min_score,
            "best_candidate_score": best,
            "llm": llm_info,
            "warnings": warnings,
        }
        job.media = media.to_dict()
    rep.finish()

    if plans and options.review_transcript:
        with session_scope() as s:
            job = s.get(Job, job_id)
            job.status = "review"
            job.analysis = {**(job.analysis or {}), "review": "waiting"}
        rep.log("Paused so the transcript can be reviewed. Correct any wrong words, then continue to render.")
        return
    _produce(job_id, rep, storage, options, art, plans, source, has_speech)


def _variant_index(clip: Clip) -> int:
    return int(((clip.edit or {}).get("variant") or {}).get("index", 1))


def plan_from_clip(clip: Clip, art: Artifacts, options: JobOptions, version: int | None = None) -> ClipPlan:
    """A ClipPlan from a stored clip: its own edits (style, intensity, captions) over the job's options."""
    edit = dict(clip.edit or {})
    return ClipPlan(
        clip.id,
        clip.start,
        clip.end,
        effective_words(clip.start, clip.end, dict(clip.text_edits or {}), art.transcript),
        version=clip.version if version is None else version,
        style=edit.get("style", options.style),
        intensity=edit.get("intensity", options.intensity),
        quality=edit.get("quality", options.quality),
        cap_opts=CaptionOptions.model_validate(clip.captions or {}) if clip.captions else options.captions,
    )


def _resume(job_id: str, rep: JobReporter, storage: LocalStorage, options: JobOptions, source: Path) -> None:
    """Continue a job that was paused for transcript review: render with the corrected words."""
    with session_scope() as s:
        job = s.get(Job, job_id)
        media = MediaInfo.from_dict(job.media)
        art = Artifacts.load(storage.analysis_dir(job_id), media)
        plans = [
            plan_from_clip(c, art, options)
            for c in sorted(job.clips, key=lambda c: (c.rank, _variant_index(c)))
        ]
        job.analysis = {**(job.analysis or {}), "review": "done"}
    rep.log("Resuming with the reviewed transcript.")
    _produce(job_id, rep, storage, options, art, plans, source, len(art.transcript.words) >= MIN_SPEECH_WORDS)


def _produce(
    job_id: str,
    rep: JobReporter,
    storage: LocalStorage,
    options: JobOptions,
    art: Artifacts,
    plans: list[ClipPlan],
    source: Path,
    has_speech: bool,
) -> None:
    """Stages 5-8: sync, reframe, subtitles, render. Runs after clip selection (or after review)."""
    cancelled = rep.cancelled
    if not plans:
        rep.log("No segment met the quality bar; nothing was rendered.", "warn")
        for key in ("syncing", "reframing", "subtitles", "rendering"):
            rep.finish(key, skipped=True)
        rep.start("ready")
        rep.finish()
        _finish(job_id, "completed", "")
        return

    # ---- 5. syncing: word timestamps re-timed against the waveform ----------------------
    n = len(plans)
    memo = ClipCache(storage.analysis_dir(job_id) / "clips")
    rep.start("syncing")
    for i, cp in enumerate(plans):
        info = step_align(cp, art, source, cancelled, memo)
        rep.progress((i + 1) / n, force=True)
        if info.get("method") == "ctc":
            rep.log(
                f"Clip {i + 1}: {info['aligned']}/{info['words']} words forced-aligned to the audio "
                f"(median correction {info['median_shift_s'] * 1000:+.0f} ms)."
            )
    rep.finish()

    # ---- 6. smart reframing -----------------------------------------------------------
    rep.start("reframing")
    for i, cp in enumerate(plans):
        step_framing(
            cp,
            art,
            source,
            options.default_framing,
            options.trim_max_pause,
            True,
            rep.sub(i / n, (i + 1) / n),
            cancelled,
            user_style=options.style,
            smart_zoom=options.smart_zoom,
            memo=memo,
        )
        rec = _recommendation(cp, art)
        with session_scope() as s:
            clip = s.get(Clip, cp.clip_id)
            clip.framing = framing_json(cp)
            clip.duration = round(cp.duration, 3)
            clip.edit = {**(clip.edit or {}), "recommendation": rec(clip)}
        removed = (cp.end - cp.start) - cp.duration
        rep.log(
            f"Clip {i + 1}: layout {'/'.join(sorted({s_.layout['type'] for s_ in cp.plan.shots}))}, "
            f"faces in {cp.plan.face_ratio:.0%} of samples ({cp.plan.detector})"
            + (f", trimmed {removed:.1f}s of silence" if removed > 0.05 else "")
        )
        if cancelled():
            raise Cancelled()
    rep.finish()

    # ---- 7. subtitles -----------------------------------------------------------------
    if has_speech:
        rep.start("subtitles")
        for i, cp in enumerate(plans):
            pages = step_captions(cp, art, storage, job_id, cp.cap_opts or options.captions, options.platform)
            with session_scope() as s:
                clip = s.get(Clip, cp.clip_id)
                clip.pages = pages
                clip.framing = framing_json(cp)
            rep.progress((i + 1) / n, force=True)
        rep.log(
            "Captions burned in and SRT written."
            if options.captions.enabled
            else "Burn-in disabled: SRT sidecar files written only."
        )
        rep.finish()
    else:
        rep.finish("subtitles", skipped=True)

    # ---- 8. rendering -----------------------------------------------------------------
    rep.start("rendering")
    # Clips are rendered one at a time unless the profile allows more (TURBO on a machine with
    # the cores and memory for it): each render already uses the profile's thread budget, so
    # parallel renders split the budget instead of oversubscribing the CPU.
    workers = max(1, min(active_plan().render_workers, n))
    lock = threading.Lock()
    done_frac = [0.0] * n

    def report_progress(i: int):  # noqa: ANN202
        def f(p: float) -> None:
            with lock:
                done_frac[i] = max(done_frac[i], 0.97 * p)
                rep.progress(sum(done_frac) / n)

        return f

    def render_one(i: int, cp: ClipPlan) -> bool:
        with session_scope() as s:
            s.get(Clip, cp.clip_id).status = "rendering"
        try:
            report = step_render(cp, art, source, storage, job_id, report_progress(i), cancelled, memo)
        except (FFmpegError, QAError) as exc:
            with lock:
                rep.log(f"Clip {i + 1} failed: {exc}", "error")
            with session_scope() as s:
                clip = s.get(Clip, cp.clip_id)
                clip.status = "failed"
                clip.error = "This clip could not be rendered."
            return False
        with session_scope() as s:
            clip = s.get(Clip, cp.clip_id)
            clip.status = "ready"
            clip.progress = 1.0
            clip.qa = report
            clip.duration = report["duration"]
        with lock:
            done_frac[i] = 1.0
            rep.log(
                f"Clip {i + 1} rendered and verified: {report['width']}x{report['height']}, {report['duration']:.1f}s, {report['size_bytes'] / 1e6:.1f} MB."
            )
            rep.progress(sum(done_frac) / n, force=True)
        return True

    if workers == 1:
        ready = sum(render_one(i, cp) for i, cp in enumerate(plans))
    else:
        from concurrent.futures import ThreadPoolExecutor

        rep.log(f"Rendering {workers} clips at a time ({active_plan().profile} profile).")
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="render") as pool:
            futures = [pool.submit(render_one, i, cp) for i, cp in enumerate(plans)]
            try:
                ready = sum(f.result() for f in futures)
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    rep.finish()
    if ready == 0:
        raise UserFacingError("None of the selected clips could be rendered.")

    rep.start("ready")
    rep.finish()
    _finish(job_id, "completed", "")
    rep.log(f"Done: {ready} clip(s) ready.")


# --------------------------------------------------------------------------- studio re-render


def rerender_clip(job_id: str, clip_id: str) -> None:
    """Re-render one clip after the user edited it in the studio. Never raises."""
    storage = get_storage()
    try:
        with session_scope() as s:
            clip = s.get(Clip, clip_id)
            job = s.get(Job, job_id)
            if clip is None or job is None:
                return
            options = JobOptions.model_validate(job.options or {})
            media = MediaInfo.from_dict(job.media)
            source = storage.source_path(job_id, job.source_ext)
            start, end = clip.start, clip.end
            text_edits = dict(clip.text_edits or {})
            cap_opts = CaptionOptions.model_validate(clip.captions or {})
            framing_mode, smart_trim = clip.framing_mode, clip.smart_trim
            new_version = clip.version + 1
            edit = dict(clip.edit or {})
        art = Artifacts.load(storage.analysis_dir(job_id), media)
        words = effective_words(start, end, text_edits, art.transcript)
        cp = ClipPlan(
            clip_id, start, end, words, version=new_version,
            style=edit.get("style", options.style), intensity=edit.get("intensity", options.intensity),
            quality=edit.get("quality", options.quality), cap_opts=cap_opts,
        )  # fmt: skip

        def prog(lo: float, hi: float):
            def f(p: float) -> None:
                with session_scope() as s:
                    c = s.get(Clip, clip_id)
                    if c is not None:
                        c.progress = round(lo + (hi - lo) * p, 3)

            return _throttle(f)

        memo = ClipCache(storage.analysis_dir(job_id) / "clips")
        step_align(cp, art, source, memo=memo)
        step_framing(
            cp, art, source, framing_mode, options.trim_max_pause, smart_trim, prog(0.0, 0.2),
            user_style=options.style, smart_zoom=options.smart_zoom, memo=memo,
        )  # fmt: skip
        pages = step_captions(cp, art, storage, job_id, cap_opts, options.platform)
        report = step_render(cp, art, source, storage, job_id, prog(0.2, 1.0), memo=memo)
        rec = _recommendation(cp, art)
        with session_scope() as s:
            clip = s.get(Clip, clip_id)
            old_version = clip.version
            clip.version = new_version
            clip.status = "ready"
            clip.progress = 1.0
            clip.error = ""
            clip.pages = pages
            clip.framing = framing_json(cp)
            clip.qa = report
            clip.duration = report["duration"]
            clip.edit = {**(clip.edit or {}), "recommendation": rec(clip)}
            # corrections are now keyed to this exact word list
            clip.text_edits = (
                {
                    "cov": [start, end],
                    "words": [
                        {"text": w.text, "start": w.start, "end": w.end, "prob": w.prob} for w in words
                    ],
                }
                if text_edits.get("words") is not None
                else {}
            )
        for kind in ("video", "srt", "thumb", "ass"):
            storage.clip_file(job_id, clip_id, old_version, kind).unlink(missing_ok=True)
        add_log(job_id, f"Clip re-rendered (v{new_version}): {report['duration']:.1f}s, verified.")
    except Exception as exc:
        log.error("re-render of clip %s failed: %s\n%s", clip_id, exc, traceback.format_exc())
        add_log(job_id, f"Clip re-render failed: {type(exc).__name__}: {str(exc)[:300]}", level="error")
        with session_scope() as s:
            clip = s.get(Clip, clip_id)
            if clip is not None:
                # the previous version (if any) is still on disk and stays downloadable
                has_previous = storage.clip_file(job_id, clip_id, clip.version, "video").exists()
                clip.status = "ready" if has_previous else "failed"
                clip.error = (
                    "The new version could not be rendered; your previous version was kept."
                    if has_previous
                    else "This clip could not be rendered."
                )


def _throttle(fn, interval: float = 0.5):  # noqa: ANN001
    import time

    last = [0.0]

    def wrapped(p: float) -> None:
        now = time.monotonic()
        if p >= 1.0 or now - last[0] >= interval:
            last[0] = now
            fn(p)

    return wrapped
