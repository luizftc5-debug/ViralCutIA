"""Runs a job through every pipeline stage and persists real progress and results."""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..db import Clip, Job, session_scope, utcnow
from ..jobs import JobReporter, add_log
from ..options import CaptionOptions, JobOptions
from ..security import FAMILY_DEMUXER, sniff_container
from ..storage import LocalStorage, get_storage
from . import captions as cap
from . import llm
from .audio import AudioSignals, analyze_audio, decode_audio, silent_signals
from .candidates import signal_candidates, speech_candidates
from .copywriting import extractive_copy
from .ffmpeg import Cancelled, FFmpegError
from .probe import MediaInfo, ProbeError, probe
from .qa import QAError, verify_clip
from .reframe import FramingPlan, Shot, plan_framing
from .render import build_segments, make_thumbnail, output_size, render_clip, segments_time_map
from .scoring import AnalysisContext, score_signal, score_speech
from .selection import auto_count, select
from .text import build_sentences
from .transcribe import Transcript, TranscriptionUnavailable, Word, transcribe
from .trimming import plan_keeps
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
    """Words spoken inside [start, end) on the source timeline, with the user's corrections applied."""
    base = [w for w in transcript.words if start <= _mid(w) < end]
    if not text_edits or text_edits.get("words") is None:
        return base
    c0, c1 = text_edits.get("cov", [0.0, 0.0])
    edited = [Word(w["text"], w["start"], w["end"], w.get("prob", 1.0)) for w in text_edits["words"]]
    edited = [w for w in edited if start <= _mid(w) < end]
    outside = [w for w in base if not (c0 <= _mid(w) < c1)]
    return sorted(outside + edited, key=lambda w: w.start)


# --------------------------------------------------------------------------- per-clip steps


@dataclass
class ClipPlan:
    clip_id: str
    start: float
    end: float
    words: list[Word]  # source timeline
    plan: FramingPlan | None = None
    segments: list | None = None
    out_words: list[Word] | None = None  # output timeline
    duration: float = 0.0
    ass_path: Path | None = None
    version: int = 1


def step_framing(
    cp: ClipPlan,
    art: Artifacts,
    source: Path,
    framing_mode: str,
    max_pause: float | None,
    smart_trim: bool,
    on_progress=None,
    should_cancel=None,
) -> None:  # noqa: ANN001
    from .render import output_fps

    keeps = plan_keeps(cp.words, cp.start, cp.end, art.audio, max_pause if smart_trim else None)
    cp.plan = plan_framing(
        source, art.family, art.media, cp.start, cp.end, framing_mode, on_progress, should_cancel
    )
    fps = output_fps(art.media)
    cp.segments = build_segments(keeps, cp.plan.shots, cp.start, fps)
    if not cp.segments:
        cp.segments = build_segments(
            [(cp.start, cp.end)], [Shot(0.0, cp.end - cp.start, cp.plan.shots[0].layout)], cp.start, fps
        )
    tmap = segments_time_map(cp.segments, fps)
    cp.duration = tmap.duration
    cp.out_words = tmap.map_words(cp.words, cp.start)


def step_captions(
    cp: ClipPlan, art: Artifacts, storage: LocalStorage, job_id: str, opts: CaptionOptions, platform: str
) -> list[dict]:
    """Write the SRT sidecar (whenever there is speech) and the ASS file (only when burn-in is on)."""
    if not cp.out_words:
        return []
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
    ass = cap.build_ass(pages, opts, platform, ow, oh)  # also computes the line layout used by the SRT
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
) -> dict:  # noqa: ANN001
    out = storage.clip_file(job_id, cp.clip_id, cp.version, "video")
    params = render_clip(
        source, art.family, art.media, cp.start, cp.segments, out, cp.ass_path, on_progress, should_cancel
    )
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
    try:
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
        # a retry starts clean
        for clip in list(job.clips):
            s.delete(clip)
    if not source.exists():
        raise UserFacingError("The uploaded file is no longer available. Please upload it again.")
    cancelled = rep.cancelled

    # ---- 1. analyzing -----------------------------------------------------------------
    rep.start("analyzing")
    family = _family(source)
    if family not in FAMILY_DEMUXER:
        raise UserFacingError("The file is not a supported video container.")
    media = probe(source, family)
    rep.log(
        f"Source: {media.width}x{media.height} @ {media.fps_float:.2f} fps, {media.duration:.1f}s, "
        f"video={media.video_codec}, audio={media.audio_codec or 'none'}"
    )
    warnings: list[str] = []
    if media.is_hdr:
        warnings.append("HDR source: converted to SDR with a basic tone-mapping.")
    samples = None
    if media.has_audio:
        samples = decode_audio(source, family, media.duration, rep.sub(0.0, 0.35), cancelled)
        audio = analyze_audio(samples)
    else:
        audio = silent_signals(media.duration)
        warnings.append("The video has no audio track: clips were selected from visual signals only.")
    visual = analyze_visual(source, family, media.duration, rep.sub(0.35, 1.0), cancelled)
    rep.log(
        f"Audio: speech level {audio.speech_db:.0f} dBFS, noise floor {audio.noise_floor_db:.0f} dBFS. Visual: {len(visual.cuts)} shot changes."
    )
    rep.finish()

    # ---- 2. transcribing --------------------------------------------------------------
    rep.start("transcribing")
    transcript = Transcript()
    if samples is not None and len(samples):
        try:
            lang = None if options.language == "auto" else options.language
            transcript = transcribe(samples, lang, rep.progress, cancelled)
            rep.log(
                f"Transcribed {len(transcript.words)} words, language={transcript.language or '?'} ({transcript.engine})."
            )
            if (
                transcript.engine != f"faster-whisper:{settings.whisper_model}"
                and transcript.engine != "none"
            ):
                warnings.append(
                    f"Speech model '{settings.whisper_model}' was unavailable; a smaller model ({transcript.engine}) was used, so accuracy is lower."
                )
        except TranscriptionUnavailable as exc:
            rep.log(f"Transcription unavailable: {exc}", "warn")
            warnings.append(
                "Speech recognition was unavailable: clips were selected from audio/visual signals only, without captions."
            )
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
            clip = Clip(
                job_id=job_id,
                rank=rank,
                status="pending",
                start=round(c.start, 3),
                end=round(c.end, 3),
                score=c.score,
                dimensions={"values": c.dims, "details": c.details, "engine": engine, "weights": ctx.weights},
                explanation=c.explanation,
                hook=c.hook,
                titles=titles,
                caption=caption,
                copy_source=copy_source,
                language=transcript.language,
                transcript=c.text,
                captions=options.captions.model_dump(),
                framing_mode=options.default_framing,
                smart_trim=options.trim_max_pause is not None,
                version=1,
            )
            s.add(clip)
            s.flush()
            words = effective_words(c.start, c.end, {}, transcript)
            plans.append(ClipPlan(clip.id, c.start, c.end, words))
        job.analysis = {
            "engine": engine,
            "mode": mode,
            "language": transcript.language,
            "language_probability": transcript.language_probability,
            "transcription": transcript.engine,
            "words": len(transcript.words),
            "candidates": len(cands),
            "requested": want,
            "selected": len(chosen),
            "min_score": min_score,
            "best_candidate_score": best,
            "llm": llm_info,
            "warnings": warnings,
        }
        job.media = media.to_dict()
    rep.finish()

    if not plans:
        rep.log("No segment met the quality bar; nothing was rendered.", "warn")
        for key in ("reframing", "subtitles", "rendering"):
            rep.finish(key, skipped=True)
        rep.start("ready")
        rep.finish()
        _finish(job_id, "completed", "")
        return

    # ---- 5. smart reframing -----------------------------------------------------------
    rep.start("reframing")
    n = len(plans)
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
        )
        with session_scope() as s:
            clip = s.get(Clip, cp.clip_id)
            clip.framing = cp.plan.to_dict()
            clip.duration = round(cp.duration, 3)
        removed = (cp.end - cp.start) - cp.duration
        rep.log(
            f"Clip {i + 1}: layout {'/'.join(sorted({s_.layout['type'] for s_ in cp.plan.shots}))}, "
            f"faces in {cp.plan.face_ratio:.0%} of samples ({cp.plan.detector})"
            + (f", trimmed {removed:.1f}s of silence" if removed > 0.05 else "")
        )
        if cancelled():
            raise Cancelled()
    rep.finish()

    # ---- 6. subtitles -----------------------------------------------------------------
    if has_speech:
        rep.start("subtitles")
        for i, cp in enumerate(plans):
            pages = step_captions(cp, art, storage, job_id, options.captions, options.platform)
            with session_scope() as s:
                clip = s.get(Clip, cp.clip_id)
                clip.pages = pages
                clip.framing = cp.plan.to_dict()
            rep.progress((i + 1) / n, force=True)
        rep.log(
            "Captions burned in and SRT written."
            if options.captions.enabled
            else "Burn-in disabled: SRT sidecar files written only."
        )
        rep.finish()
    else:
        rep.finish("subtitles", skipped=True)

    # ---- 7. rendering -----------------------------------------------------------------
    rep.start("rendering")
    ready = 0
    for i, cp in enumerate(plans):
        with session_scope() as s:
            s.get(Clip, cp.clip_id).status = "rendering"
        try:
            report = step_render(cp, art, source, storage, job_id, rep.sub(i / n, (i + 0.97) / n), cancelled)
        except Cancelled:
            raise
        except (FFmpegError, QAError) as exc:
            rep.log(f"Clip {i + 1} failed: {exc}", "error")
            with session_scope() as s:
                clip = s.get(Clip, cp.clip_id)
                clip.status = "failed"
                clip.error = "This clip could not be rendered."
            continue
        with session_scope() as s:
            clip = s.get(Clip, cp.clip_id)
            clip.status = "ready"
            clip.progress = 1.0
            clip.qa = report
            clip.duration = report["duration"]
        ready += 1
        rep.log(
            f"Clip {i + 1} rendered and verified: {report['width']}x{report['height']}, {report['duration']:.1f}s, {report['size_bytes'] / 1e6:.1f} MB."
        )
        rep.progress((i + 1) / n, force=True)
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
        art = Artifacts.load(storage.analysis_dir(job_id), media)
        words = effective_words(start, end, text_edits, art.transcript)
        cp = ClipPlan(clip_id, start, end, words, version=new_version)

        def prog(lo: float, hi: float):
            def f(p: float) -> None:
                with session_scope() as s:
                    c = s.get(Clip, clip_id)
                    if c is not None:
                        c.progress = round(lo + (hi - lo) * p, 3)

            return _throttle(f)

        step_framing(cp, art, source, framing_mode, options.trim_max_pause or 0.7, smart_trim, prog(0.0, 0.2))
        pages = step_captions(cp, art, storage, job_id, cap_opts, options.platform)
        report = step_render(cp, art, source, storage, job_id, prog(0.2, 1.0))
        with session_scope() as s:
            clip = s.get(Clip, clip_id)
            old_version = clip.version
            clip.version = new_version
            clip.status = "ready"
            clip.progress = 1.0
            clip.error = ""
            clip.pages = pages
            clip.framing = cp.plan.to_dict()
            clip.qa = report
            clip.duration = report["duration"]
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
