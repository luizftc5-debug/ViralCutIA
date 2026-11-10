"""Stage 5 — clip candidate generation.

Candidates are never fixed-size blocks: with speech, every candidate starts at a sentence
boundary and ends at a sentence boundary; without speech, windows are anchored on shot changes
and audio onsets. Scoring and selection happen in later stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .audio import AudioSignals
from .text import Sentence
from .transcribe import Word
from .visual import VisualSignals

START_PAD = 0.15
END_PAD = 0.40


@dataclass
class Candidate:
    start: float
    end: float
    kind: str = "speech"  # speech | signal
    s0: int = -1  # first sentence index (speech candidates)
    s1: int = -1  # last sentence index, inclusive
    text: str = ""
    dims: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    hook: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    source: str = "heuristic"  # heuristic | llm (who proposed the boundaries)

    @property
    def duration(self) -> float:
        return self.end - self.start


def clip_bounds(
    sentences: list[Sentence], words: list[Word], s0: int, s1: int, media_duration: float
) -> tuple[float, float]:
    """Pad sentence boundaries without ever cutting into a neighbouring word."""
    first, last = sentences[s0], sentences[s1]
    prev_end = words[first.w0 - 1].end if first.w0 > 0 else 0.0
    next_start = words[last.w1].start if last.w1 < len(words) else media_duration
    start = first.start - min(START_PAD, max(0.0, first.start - prev_end) / 2)
    end = last.end + min(END_PAD, max(0.0, next_start - last.end) * 0.6)
    return max(0.0, start), min(media_duration, end)


def speech_candidates(
    sentences: list[Sentence],
    words: list[Word],
    media_duration: float,
    min_d: float,
    max_d: float,
) -> list[Candidate]:
    out: list[Candidate] = []
    n = len(sentences)
    if n == 0:
        return out
    total = sentences[-1].end - sentences[0].start
    if total < min_d:
        # the whole talk is shorter than the requested range: offer it as a single clip
        if total >= 5.0:
            st, en = clip_bounds(sentences, words, 0, n - 1, media_duration)
            out.append(Candidate(st, en, s0=0, s1=n - 1, text=" ".join(s.text for s in sentences)))
        return out
    for i in range(n):
        for j in range(i, n):
            dur = sentences[j].end - sentences[i].start
            if dur > max_d:
                break
            if dur >= min_d:
                st, en = clip_bounds(sentences, words, i, j, media_duration)
                out.append(Candidate(st, en, s0=i, s1=j))
    for c in out:
        c.text = " ".join(s.text for s in sentences[c.s0 : c.s1 + 1])
    return out


def signal_candidates(
    audio: AudioSignals,
    visual: VisualSignals,
    media_duration: float,
    min_d: float,
    max_d: float,
) -> list[Candidate]:
    """Windows for footage without usable speech, anchored on shot changes and audio onsets."""
    if media_duration < 5.0:
        return []
    target = min(max_d, max(min_d, (min_d + max_d) / 2))
    if media_duration <= max(min_d, 8.0):
        return [Candidate(0.0, media_duration, kind="signal")]
    target = min(target, media_duration)
    anchors: set[float] = {0.0}
    anchors.update(round(c, 2) for c in visual.cuts)
    # audio onsets: envelope rising well above the preceding second
    hop = audio.hop
    env = audio.rms_db
    win = max(1, int(1.0 / hop))
    for k in range(win, len(env) - win, max(1, win // 2)):
        before = float(env[k - win : k].mean())
        after = float(env[k : k + win].mean())
        if after - before > 8.0 and after > audio.silence_threshold_db:
            anchors.add(round(k * hop, 2))
    step = max(2.0, target / 4)
    t = 0.0
    while t + min_d <= media_duration:
        anchors.add(round(t, 2))
        t += step
    out: list[Candidate] = []
    for a in sorted(anchors):
        end = min(media_duration, a + target)
        # prefer ending on a shot change near the target end
        near = [
            c for c in visual.cuts if a + min_d <= c <= min(media_duration, a + max_d) and abs(c - end) < 3.0
        ]
        if near:
            end = min(near, key=lambda c: abs(c - end))
        if end - a >= min(min_d, media_duration):
            out.append(Candidate(a, end, kind="signal"))
    return out
