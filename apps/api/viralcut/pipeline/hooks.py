"""Stage 6 — Hook Detection Engine.

Looks at the first ~3 seconds of a candidate and measures the presence of authentic hook
signals in the source material. Nothing is invented: every hook type is backed by words that
were actually spoken or by measured audio/visual events at the opening.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .audio import AudioSignals
from .text import (
    DEPENDENT_OPENERS,
    FILLERS,
    HOOK_PATTERNS,
    MILD_OPENERS,
    SECOND_PERSON,
    Sentence,
    has_number,
    lang_key,
    norm,
    tokenize,
)
from .transcribe import Word
from .visual import VisualSignals

HOOK_WINDOW_S = 3.0

HOOK_LABELS = {
    "curiosity": "curiosity gap",
    "interrupt": "pattern interruption",
    "stakes": "high stakes",
    "emotion": "emotional impact",
    "value": "immediate value",
    "tension": "narrative tension",
    "visual": "strong visual opening",
}


@dataclass
class HookResult:
    score: float  # 0..1
    text: str  # words spoken in the hook window
    types: list[dict] = field(default_factory=list)  # [{type, label, strength}], strongest first
    opener: str = "clean"  # clean | mild | dependent
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"score": round(self.score, 3), "text": self.text, "types": self.types, "opener": self.opener}


def opener_kind(sentence: Sentence, lk: str | None) -> str:
    if lk is None or not sentence.tokens:
        return "clean"
    first = sentence.tokens[0]
    if first in DEPENDENT_OPENERS[lk]:
        return "dependent"
    if first in MILD_OPENERS[lk] or first in FILLERS[lk]:
        return "mild"
    return "clean"


def analyze_hook(
    start: float,
    sentences: list[Sentence],
    words: list[Word],
    s0: int,
    s1: int,
    language: str,
    audio: AudioSignals,
    visual: VisualSignals,
    clip_end: float,
) -> HookResult:
    lk = lang_key(language)
    first = sentences[s0]
    # words spoken within the hook window (at least 4, at most 16)
    opening: list[Word] = []
    for w in words[first.w0 : sentences[s1].w1]:
        if w.start - start <= HOOK_WINDOW_S or len(opening) < 4:
            opening.append(w)
        else:
            break
        if len(opening) >= 16:
            break
    opening_text = " ".join(w.text for w in opening)
    opening_norm = norm(opening_text)
    first_norm = norm(first.text)
    # the opening idea may extend slightly past 3 s: also consider the sentences that begin within 6 s
    early_norm = norm(" ".join(s.text for s in sentences[s0 : s1 + 1] if s.start - start <= 6.0))

    strengths: dict[str, float] = {}

    def lex(cat: str) -> float:
        if lk is None:
            return 0.0
        rx = HOOK_PATTERNS[cat][lk]
        if rx.search(opening_norm):
            return 1.0
        if rx.search(first_norm):
            return 0.75
        if rx.search(early_norm):
            return 0.5
        return 0.0

    question_open = first.is_question or "?" in opening_text
    early_question = any(s.is_question for s in sentences[s0 : s1 + 1] if s.start - start <= 6.0)

    # --- curiosity gap
    cur = 0.75 * lex("curiosity")
    if question_open:
        cur = max(cur, 0.85) + (0.15 if cur > 0 else 0.0)
    elif early_question:
        cur = max(cur, 0.55)
    strengths["curiosity"] = min(1.0, cur)

    # --- pattern interruption: contrarian wording, abrupt audio onset, shot change at the opening
    open_level = audio.level(start, min(clip_end, start + 1.5))
    clip_level = audio.level(start, clip_end)
    loud_open = audio.percentile_of(open_level) >= 0.75 and open_level >= clip_level + 1.0
    cut_open = bool(visual.cuts_between(max(0.0, start - 0.3), start + 1.5))
    intr = 0.7 * lex("interrupt") + (0.2 if loud_open else 0.0) + (0.15 if cut_open else 0.0)
    strengths["interrupt"] = min(1.0, intr)

    # --- high stakes
    stk = 0.8 * lex("stakes")
    if stk > 0 and has_number(opening_text):
        stk += 0.15
    strengths["stakes"] = min(1.0, stk)

    # --- emotional impact
    emo = 0.7 * lex("emotion")
    if "!" in opening_text:
        emo += 0.15
    if emo > 0 and audio.dynamics(start, min(clip_end, start + 4.0)) >= 6.0:
        emo += 0.15
    strengths["emotion"] = min(1.0, emo)

    # --- immediate value
    val = 0.75 * lex("value")
    second_person = bool(lk and SECOND_PERSON[lk].search(opening_norm))
    if val > 0 and second_person:
        val += 0.15
    if val > 0 and has_number(opening_text):
        val += 0.15
    strengths["value"] = min(1.0, val)

    # --- narrative tension
    strengths["tension"] = min(1.0, 0.75 * lex("tension"))

    # --- strong visual opening (measured motion in the first two seconds vs. the whole video)
    m_open = visual.motion_level(start, start + 2.0)
    pct = visual.motion_percentile(m_open)
    vis = max(0.0, (pct - 0.6) / 0.4) * 0.7 + (0.2 if cut_open else 0.0)
    strengths["visual"] = min(1.0, vis if m_open > 0.004 else 0.0)

    ranked = sorted(strengths.items(), key=lambda kv: kv[1], reverse=True)
    top = [v for _, v in ranked[:3]] + [0.0, 0.0, 0.0]

    # delivery: short, punchy first sentences hold attention better than long wind-ups
    n_first = first.n_words
    punch = 1.0 if n_first <= 12 else max(0.5, 1.0 - (n_first - 12) * 0.03)
    raw = 0.62 * top[0] + 0.20 * top[1] + 0.08 * top[2] + 0.18 * punch
    if second_person:
        raw += 0.04

    opener = opener_kind(first, lk)
    penalty = {"clean": 1.0, "mild": 0.88, "dependent": 0.55}[opener]
    first_tokens = tokenize(first.text)
    if lk and first_tokens and first_tokens[0] in FILLERS[lk]:
        penalty *= 0.9
    score = max(0.0, min(1.0, raw * penalty))

    types = [{"type": k, "label": HOOK_LABELS[k], "strength": round(v, 2)} for k, v in ranked if v >= 0.45][
        :3
    ]
    # what the UI quotes as "the hook": the whole first sentence when short, otherwise the first 3 seconds
    quote = first.text if first.n_words <= 20 else opening_text.rstrip(",;: ") + "…"
    return HookResult(
        score=score,
        text=quote,
        types=types,
        opener=opener,
        signals={k: round(v, 3) for k, v in strengths.items()},
    )


def analyze_signal_hook(start: float, end: float, audio: AudioSignals, visual: VisualSignals) -> HookResult:
    """Hook estimate for footage without speech: only measured audio/visual events are available."""
    open_level = audio.level(start, min(end, start + 1.5))
    loud = audio.percentile_of(open_level)
    cut_open = bool(visual.cuts_between(max(0.0, start - 0.3), start + 1.5))
    m_open = visual.motion_level(start, start + 2.0)
    pct = visual.motion_percentile(m_open) if m_open > 0.004 else 0.0
    vis = min(1.0, max(0.0, (pct - 0.5) / 0.5) * 0.8 + (0.2 if cut_open else 0.0))
    intr = min(1.0, max(0.0, (loud - 0.6) / 0.4) * 0.7 + (0.2 if cut_open else 0.0))
    score = min(1.0, 0.55 * max(vis, intr) + 0.25 * min(vis, intr) + 0.1)
    types = []
    if vis >= 0.45:
        types.append({"type": "visual", "label": HOOK_LABELS["visual"], "strength": round(vis, 2)})
    if intr >= 0.45:
        types.append({"type": "interrupt", "label": HOOK_LABELS["interrupt"], "strength": round(intr, 2)})
    return HookResult(
        score=score, text="", types=types, signals={"visual": round(vis, 3), "interrupt": round(intr, 3)}
    )
