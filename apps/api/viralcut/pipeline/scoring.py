"""Stage 7 — Viral Potential Score.

Six dimensions, each 0..1, combined with configurable weights into a 0..100 score. The weights
are product hypotheses, not a validated virality formula; ``docs/SCORING.md`` describes how
they are meant to be calibrated with observed performance data.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

from .audio import AudioSignals
from .candidates import Candidate
from .hooks import HOOK_LABELS, HookResult, analyze_hook, analyze_signal_hook, opener_kind
from .probe import MediaInfo
from .text import (
    BACKREF_PATTERNS,
    CONTRAST_PATTERNS,
    DEPENDENT_OPENERS,
    FILLERS,
    HOOK_PATTERNS,
    MILD_OPENERS,
    PAYOFF_PATTERNS,
    STRUCTURE_PATTERNS,
    TRAILING_WORDS,
    Sentence,
    TfIdf,
    lang_key,
    norm,
    topic_boundaries,
)
from .transcribe import Word
from .visual import VisualSignals

DIMENSIONS = ("hook", "narrative", "value", "impact", "self_contained", "technical")

DEFAULT_WEIGHTS: dict[str, float] = {
    "hook": 0.25,
    "narrative": 0.20,
    "value": 0.20,
    "impact": 0.15,
    "self_contained": 0.10,
    "technical": 0.10,
}

DIMENSION_LABELS = {
    "hook": "Hook strength",
    "narrative": "Narrative retention",
    "value": "Content value",
    "impact": "Emotional / visual impact",
    "self_contained": "Standalone clarity",
    "technical": "Technical quality",
}


def load_weights() -> dict[str, float]:
    """Weights can be overridden with VIRALCUT_SCORE_WEIGHTS='{"hook":0.3,...}' (renormalised)."""
    weights = dict(DEFAULT_WEIGHTS)
    raw = os.environ.get("VIRALCUT_SCORE_WEIGHTS", "").strip()
    if raw:
        try:
            for k, v in json.loads(raw).items():
                if k in weights and float(v) >= 0:
                    weights[k] = float(v)
        except (ValueError, TypeError, AttributeError):
            pass
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sat(x: float, k: float) -> float:
    return 1.0 - math.exp(-max(0.0, x) / k)


def _band(x: float, zero_lo: float, full_lo: float, full_hi: float, zero_hi: float) -> float:
    if x <= zero_lo or x >= zero_hi:
        return 0.0
    if x < full_lo:
        return (x - zero_lo) / (full_lo - zero_lo)
    if x > full_hi:
        return (zero_hi - x) / (zero_hi - full_hi)
    return 1.0


_CAP_RE = re.compile(r"\b[A-ZÀ-Ý][a-zà-ÿ]{2,}")


@dataclass
class AnalysisContext:
    """Everything the scorer needs about the source video."""

    words: list[Word]
    sentences: list[Sentence]
    language: str
    audio: AudioSignals
    visual: VisualSignals
    media: MediaInfo
    tfidf: TfIdf
    boundaries: set[int]
    weights: dict[str, float]
    breaks: list[float] = field(
        default_factory=list
    )  # strength (0..1) of the topic break before each sentence
    info: list[float] = field(default_factory=list)  # informativeness (0..1) of each sentence

    @classmethod
    def build(cls, words, sentences, language, audio, visual, media) -> AnalysisContext:  # noqa: ANN001
        tfidf = TfIdf([s.content for s in sentences])
        boundaries = topic_boundaries(sentences, tfidf)
        ctx = cls(
            words=words,
            sentences=sentences,
            language=language,
            audio=audio,
            visual=visual,
            media=media,
            tfidf=tfidf,
            boundaries=boundaries,
            weights=load_weights(),
        )
        ctx.breaks = [_break_strength(s, boundaries, visual) for s in sentences]
        ctx.info = [_sentence_info(s, ctx) for s in sentences]
        return ctx


def _break_strength(s: Sentence, boundaries: set[int], visual: VisualSignals) -> float:
    """How strongly the speaker moves on before this sentence: lexical shift, long pause, shot change."""
    if s.idx == 0:
        return 1.0
    strength = 0.35 if s.idx in boundaries else 0.0
    strength += 0.55 * _clamp((s.gap_before - 0.5) / 0.6)
    if s.gap_before >= 0.4 and visual.cuts_between(s.start - s.gap_before - 0.4, s.start + 0.4):
        strength += 0.25  # a shot change during a pause usually marks a new section
    return _clamp(strength)


def _sentence_info(s: Sentence, ctx: AnalysisContext) -> float:
    """Informativeness of one sentence: content share, topical keywords, specifics, actionable wording."""
    lk = lang_key(ctx.language)
    if not s.tokens:
        return 0.0
    content_ratio = len(s.content) / len(s.tokens)
    kw_hits = len(set(s.content) & set(ctx.tfidf.keywords[:20]))
    numbers = sum(1 for t in s.tokens if any(ch.isdigit() for ch in t))
    proper = len(_CAP_RE.findall(s.text.split(" ", 1)[1])) if " " in s.text else 0
    text_norm = norm(s.text)
    marker = bool(
        lk
        and (
            HOOK_PATTERNS["value"][lk].search(text_norm)
            or HOOK_PATTERNS["stakes"][lk].search(text_norm)
            or STRUCTURE_PATTERNS[lk].search(text_norm)
        )
    )
    fillers = sum(1 for t in s.tokens if lk and t in FILLERS[lk]) / len(s.tokens)
    first = s.tokens[0]
    drifting = bool(
        lk and (first in DEPENDENT_OPENERS[lk] or first in MILD_OPENERS[lk]) and kw_hits == 0 and not marker
    )
    info = (
        0.40 * _clamp(content_ratio / 0.5)
        + 0.25 * _clamp(kw_hits / 2)
        + 0.15 * _sat(numbers + 0.5 * proper, 1.2)
        + 0.20 * (1.0 if marker else 0.0)
        - 0.4 * _clamp(fillers / 0.12)
        - (0.15 if drifting else 0.0)
    )
    return _clamp(info)


def _technical(ctx: AnalysisContext, start: float, end: float, words: list[Word]) -> tuple[float, dict]:
    a = ctx.audio
    level = a.level(start, end)
    active = a.peak(start, end)
    snr = max(0.0, a.speech_db - a.noise_floor_db)
    level_ok = _band(max(level, active - 12), -48.0, -32.0, -6.0, -1.0) if ctx.media.has_audio else 0.5
    snr_ok = _clamp((snr - 8.0) / 17.0) if ctx.media.has_audio else 0.5
    short_side = min(ctx.media.width, ctx.media.height)
    res = 1.0 if short_side >= 1080 else 0.85 if short_side >= 720 else 0.6 if short_side >= 480 else 0.4
    if words:
        conf = sum(w.prob for w in words) / len(words)
        asr = _clamp((conf - 0.55) / 0.35)
        tech = 0.30 * asr + 0.20 * level_ok + 0.25 * snr_ok + 0.25 * res
    else:
        asr = 0.0
        tech = 0.35 * level_ok + 0.30 * snr_ok + 0.35 * res
    clip_pen = _clamp(a.clip_ratio / 0.01) * 0.25
    return _clamp(tech - clip_pen), {
        "asr_confidence": round(asr, 3),
        "snr_db": round(snr, 1),
        "resolution": res,
    }


def score_speech(c: Candidate, ctx: AnalysisContext) -> HookResult:
    lk = lang_key(ctx.language)
    sents = ctx.sentences[c.s0 : c.s1 + 1]
    words = ctx.words[sents[0].w0 : sents[-1].w1]
    n_s, n_w = len(sents), len(words)
    dur = max(0.1, c.duration)
    first, last = sents[0], sents[-1]
    text_norm = norm(c.text)
    middle_norm = norm(" ".join(s.text for s in sents[1:])) if n_s > 1 else ""
    tail_norm = norm(" ".join(s.text for s in sents[-2:]))

    # ---- hook
    hook = analyze_hook(
        c.start, ctx.sentences, ctx.words, c.s0, c.s1, ctx.language, ctx.audio, ctx.visual, c.end
    )

    # ---- narrative retention: hook -> development -> payoff
    payoff = 0.25 if last.terminal else 0.0
    payoff_marker = bool(lk and PAYOFF_PATTERNS[lk].search(tail_norm))
    if payoff_marker:
        payoff += 0.45
    opens_with_question = any(s.is_question for s in sents[:2])
    answered = opens_with_question and sum(1 for s in sents[1:] if not s.is_question) >= 2
    if answered:
        payoff += 0.25
    next_break = ctx.breaks[c.s1 + 1] if c.s1 + 1 < len(ctx.breaks) else 1.0
    ends_on_break = next_break >= 0.45
    if ends_on_break:
        payoff += 0.2
    if last.n_words <= 12 and last.terminal:
        payoff += 0.1
    if last.is_question:
        payoff -= 0.2  # ends on an open question: no resolution inside the clip
    payoff = _clamp(payoff)

    development = {1: 0.1, 2: 0.3}.get(n_s, 0.5)
    contrast = bool(lk and CONTRAST_PATTERNS[lk].search(middle_norm))
    if contrast:
        development += 0.25
    if lk and (
        HOOK_PATTERNS["value"][lk].search(middle_norm)
        or HOOK_PATTERNS["tension"][lk].search(middle_norm)
        or STRUCTURE_PATTERNS[lk].search(middle_norm)
    ):
        development += 0.25
    development = _clamp(development)

    speech_time = max(0.1, sum(w.end - w.start for w in words))
    span = max(0.1, words[-1].end - words[0].start)
    wps = n_w / span
    pacing = _band(wps, 1.2, 2.0, 3.9, 5.2)
    long_pauses = sum(max(0.0, words[k + 1].start - words[k].end - 0.8) for k in range(n_w - 1))
    pacing *= 1.0 - _clamp(long_pauses / span / 0.25) * 0.4
    dur_fit = _band(dur, 6.0, 15.0, 45.0, 120.0)
    # a clip is one narrative unit: every topic break inside it weakens the arc
    inside_breaks = [ctx.breaks[k] for k in range(c.s0 + 1, c.s1 + 1)]
    unity = 1.0
    for b in inside_breaks:
        unity *= 1.0 - 0.45 * b
    narrative = _clamp(
        (0.40 * payoff + 0.30 * development + 0.20 * pacing + 0.10 * dur_fit) * (0.35 + 0.65 * unity)
    )

    # ---- content value: duration-weighted informativeness, so filler dilutes it and length does not inflate it
    content = [t for s in sents for t in s.content]
    uniq = len(set(content))
    density = _clamp((uniq / dur) / 0.9)
    weights_d = [max(0.3, s.duration) for s in sents]
    mean_info = sum(ctx.info[s.idx] * w for s, w in zip(sents, weights_d, strict=True)) / sum(weights_d)
    weakest = min(ctx.info[s.idx] for s in sents)
    salience = mean_info
    n_markers = (
        len(HOOK_PATTERNS["value"][lk].findall(text_norm)) + len(STRUCTURE_PATTERNS[lk].findall(text_norm))
        if lk
        else 0
    )
    instruct = _sat(n_markers / max(1.0, dur / 20.0), 2.0) if lk else 0.4
    all_tokens = [t for s in sents for t in s.tokens]
    fillers = sum(1 for t in all_tokens if lk and t in FILLERS[lk]) / max(1, len(all_tokens))
    value = _clamp(
        0.55 * mean_info + 0.20 * density + 0.15 * instruct + 0.10 * weakest - _clamp(fillers / 0.12) * 0.2
    )
    value = _clamp(value / 0.8)  # a clip made only of strong sentences reaches the top of the scale

    # ---- emotional / visual impact
    emo_hits = (
        len(HOOK_PATTERNS["emotion"][lk].findall(text_norm))
        + len(HOOK_PATTERNS["stakes"][lk].findall(text_norm)) * 0.6
        if lk
        else 0.0
    )
    emo = _sat(emo_hits / max(1, n_s), 0.5)
    punct = _sat(sum(s.text.count("!") + (1 if s.is_question else 0) for s in sents) / max(1, n_s), 0.4)
    dyn = ctx.audio.dynamics(c.start, c.end)
    audio_energy = 0.6 * _clamp((dyn - 2.0) / 6.0) + 0.4 * ctx.audio.percentile_of(
        ctx.audio.level(c.start, c.end)
    )
    motion = ctx.visual.motion_percentile(ctx.visual.motion_level(c.start, c.end))
    impact = _clamp(0.35 * emo + 0.15 * punct + 0.30 * audio_energy + 0.20 * motion)

    # ---- standalone clarity
    opener = opener_kind(first, lk)
    start_clean = {"clean": 1.0, "mild": 0.8, "dependent": 0.35}[opener]
    head_norm = norm(" ".join(s.text for s in sents[:2]))
    if lk and BACKREF_PATTERNS[lk].search(head_norm):
        start_clean *= 0.6
    last_tok = last.tokens[-1] if last.tokens else ""
    if lk and last_tok in TRAILING_WORDS[lk]:
        end_complete = 0.3
    else:
        end_complete = 1.0 if last.terminal else 0.5
    inside = sum(1 for b in inside_breaks if b >= 0.45)
    cohesion = unity
    fresh = 0.55 + 0.45 * ctx.breaks[c.s0]
    self_contained = _clamp(0.35 * start_clean + 0.20 * end_complete + 0.20 * cohesion + 0.25 * fresh)

    # ---- technical quality
    technical, tech_details = _technical(ctx, c.start, c.end, words)

    c.dims = {
        "hook": hook.score,
        "narrative": narrative,
        "value": value,
        "impact": impact,
        "self_contained": self_contained,
        "technical": technical,
    }
    c.details = {
        "payoff": round(payoff, 3),
        "payoff_marker": payoff_marker,
        "question_answered": answered,
        "development": round(development, 3),
        "contrast": contrast,
        "words_per_second": round(wps, 2),
        "speech_ratio": round(speech_time / dur, 2),
        "density": round(density, 3),
        "salience": round(salience, 3),
        "unity": round(unity, 3),
        "instructional": round(instruct, 3),
        "emotion": round(emo, 3),
        "audio_dynamics_db": round(dyn, 1),
        "motion_percentile": round(motion, 2),
        "opener": opener,
        "ends_complete": end_complete >= 1.0,
        "topic_breaks_inside": inside,
        **tech_details,
    }
    c.hook = hook.to_dict()
    finalize(c, ctx.weights)
    return hook


def score_signal(c: Candidate, ctx: AnalysisContext) -> HookResult:
    """Score footage without usable speech from measured audio/visual signals only.

    Narrative, value and standalone clarity cannot be judged without language, so they are set
    to conservative neutral values and the clip is flagged as a low-confidence estimate.
    """
    hook = analyze_signal_hook(c.start, c.end, ctx.audio, ctx.visual)
    dyn = ctx.audio.dynamics(c.start, c.end)
    audio_energy = 0.5 * _clamp((dyn - 2.0) / 6.0) + 0.5 * ctx.audio.percentile_of(
        ctx.audio.level(c.start, c.end)
    )
    motion = ctx.visual.motion_percentile(ctx.visual.motion_level(c.start, c.end))
    cuts = len(ctx.visual.cuts_between(c.start, c.end)) / max(1.0, c.duration / 10)
    impact = _clamp(0.45 * motion + 0.35 * audio_energy + 0.20 * _sat(cuts, 1.5))
    technical, tech_details = _technical(ctx, c.start, c.end, [])
    starts_on_cut = bool(ctx.visual.cuts_between(max(0, c.start - 0.3), c.start + 0.3)) or c.start < 0.3
    ends_on_cut = bool(ctx.visual.cuts_between(c.end - 0.3, c.end + 0.3)) or c.end >= ctx.media.duration - 0.3
    c.dims = {
        "hook": hook.score,
        "narrative": 0.35,
        "value": 0.30,
        "impact": impact,
        "self_contained": 0.4 + (0.2 if starts_on_cut else 0.0) + (0.2 if ends_on_cut else 0.0),
        "technical": technical,
    }
    c.details = {
        "mode": "signals-only",
        "audio_dynamics_db": round(dyn, 1),
        "motion_percentile": round(motion, 2),
        **tech_details,
    }
    c.hook = hook.to_dict()
    finalize(c, ctx.weights)
    return hook


def finalize(c: Candidate, weights: dict[str, float]) -> None:
    c.dims = {k: round(_clamp(float(c.dims.get(k, 0.0))), 4) for k in DIMENSIONS}
    c.score = round(100.0 * sum(weights[k] * c.dims[k] for k in DIMENSIONS), 1)
    c.explanation = explain(c)


def explain(c: Candidate) -> str:
    """One objective sentence describing why the clip scored the way it did."""
    d = c.dims
    parts: list[str] = []
    types = c.hook.get("types") or []
    if d["hook"] >= 0.7:
        parts.append(f"Strong opening ({types[0]['label']})" if types else "Strong opening")
    elif d["hook"] >= 0.45:
        parts.append(f"Solid opening ({types[0]['label']})" if types else "Solid opening")
    else:
        parts.append("Soft opening")
    if c.kind == "signal":
        if d["impact"] >= 0.6:
            parts.append("high audio/visual energy")
        parts.append("no speech detected, so narrative could not be assessed")
        return ", ".join(parts).capitalize() + "."
    if d["narrative"] >= 0.65:
        parts.append(
            "clear narrative payoff" if c.details.get("payoff", 0) >= 0.6 else "well-paced development"
        )
    if d["value"] >= 0.6:
        parts.append(
            "actionable, information-dense content"
            if c.details.get("instructional", 0) >= 0.5
            else "information-dense content"
        )
    if d["impact"] >= 0.6:
        parts.append(
            "high emotional contrast" if c.details.get("emotion", 0) >= 0.5 else "energetic delivery"
        )
    if d["self_contained"] >= 0.8:
        parts.append("works without extra context")
    weak = [k for k in ("narrative", "value", "impact", "self_contained", "technical") if d[k] < 0.4]
    text = ", ".join(parts)
    text = text[0].upper() + text[1:] + "."
    if weak:
        names = {
            "narrative": "narrative payoff",
            "value": "content density",
            "impact": "emotional impact",
            "self_contained": "standalone clarity",
            "technical": "audio/technical quality",
        }
        text += " Weaker on " + " and ".join(names[k] for k in weak[:2]) + "."
    return text


def hook_summary(hook: dict) -> str:
    types = hook.get("types") or []
    if not types:
        return ""
    return ", ".join(HOOK_LABELS.get(t["type"], t["type"]) for t in types)
