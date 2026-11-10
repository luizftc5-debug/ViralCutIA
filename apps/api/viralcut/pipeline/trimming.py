"""Smart trimming — remove dead air inside a clip without making speech sound unnatural.

A pause is shortened only when (a) it is longer than the style's limit and (b) the audio
envelope confirms it is actually silent, so laughter, music or reactions between words are
never cut. Pauses after a question keep a longer beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import AudioSignals
from .transcribe import Word

KEEP_BEFORE = 0.18  # silence kept after the last word before the cut
KEEP_AFTER = 0.14  # silence kept before the next word
QUESTION_BEAT = 0.9


def _longest_silent_run(audio: AudioSignals, t0: float, t1: float) -> tuple[float, float] | None:
    a, b = int(np.ceil(t0 / audio.hop)), int(np.floor(t1 / audio.hop))
    if b - a < 2:
        return None
    seg = audio.rms_db[a:b] < audio.silence_threshold_db
    best, cur_start = None, None
    for i, quiet in enumerate(list(seg) + [False]):
        if quiet and cur_start is None:
            cur_start = i
        elif not quiet and cur_start is not None:
            if best is None or i - cur_start > best[1] - best[0]:
                best = (cur_start, i)
            cur_start = None
    if best is None:
        return None
    return (a + best[0]) * audio.hop, (a + best[1]) * audio.hop


def plan_keeps(
    words: list[Word],
    start: float,
    end: float,
    audio: AudioSignals,
    max_pause: float | None,
) -> list[tuple[float, float]]:
    """Return the source intervals to keep, in order. ``max_pause=None`` disables trimming."""
    if max_pause is None or not words:
        return [(start, end)]
    cuts: list[tuple[float, float]] = []
    spans: list[tuple[float, float, float]] = []  # (gap start, gap end, allowed pause)
    if words[0].start - start > max_pause:
        spans.append((start, words[0].start, 0.0))
    for a, b in zip(words, words[1:], strict=False):
        allowed = max(max_pause, QUESTION_BEAT) if a.text.rstrip().endswith(("?", "…", "...")) else max_pause
        if b.start - a.end > allowed:
            spans.append((a.end, b.start, allowed))
    if end - words[-1].end > max(max_pause, 0.8):
        spans.append((words[-1].end, end, max_pause))
    for g0, g1, allowed in spans:
        run = _longest_silent_run(audio, g0, g1)
        if run is None:
            continue
        r0, r1 = run
        lead = g0 == start
        tail = g1 == end
        keep_total = (
            min(allowed, KEEP_BEFORE + KEEP_AFTER) if not (lead or tail) else (KEEP_AFTER if lead else 0.35)
        )
        if r1 - r0 <= max(allowed, keep_total + 0.15):
            continue
        if lead:
            c0, c1 = r0, r1 - KEEP_AFTER
        elif tail:
            c0, c1 = r0 + 0.35, r1
        else:
            extra = max(0.0, allowed - (KEEP_BEFORE + KEEP_AFTER)) if allowed > max_pause else 0.0
            c0, c1 = r0 + KEEP_BEFORE + extra / 2, r1 - KEEP_AFTER - extra / 2
        c0, c1 = max(start, c0), min(end, c1)
        if c1 - c0 >= 0.15:
            cuts.append((c0, c1))
    keeps: list[tuple[float, float]] = []
    cursor = start
    for c0, c1 in cuts:
        if c0 - cursor >= 0.25:
            keeps.append((cursor, c0))
        elif keeps:
            pass  # sliver between two cuts: drop it with the silence around it
        cursor = c1
    if end - cursor >= 0.25 or not keeps:
        keeps.append((cursor, end))
    return keeps


@dataclass
class TimeMap:
    """Maps source-relative time (seconds from clip start) to the trimmed output timeline."""

    segments: list[tuple[float, float]]  # kept [a, b) in clip-relative seconds, frame-snapped

    @property
    def duration(self) -> float:
        return sum(b - a for a, b in self.segments)

    def to_output(self, t: float) -> float | None:
        offset = 0.0
        for a, b in self.segments:
            if a <= t < b:
                return offset + (t - a)
            offset += b - a
        return None

    def clamp_to_output(self, t: float) -> float:
        """Like ``to_output`` but snaps times inside removed pauses to the nearest kept instant."""
        offset = 0.0
        for a, b in self.segments:
            if t < a:
                return offset
            if t < b:
                return offset + (t - a)
            offset += b - a
        return offset

    def map_words(self, words: list[Word], clip_start: float) -> list[Word]:
        out: list[Word] = []
        for w in words:
            mid = (w.start + w.end) / 2 - clip_start
            if self.to_output(mid) is None:
                # the midpoint fell in a removed pause (should not happen for real words): skip
                if (
                    self.to_output(w.start - clip_start) is None
                    and self.to_output(w.end - clip_start - 1e-3) is None
                ):
                    continue
            s = self.clamp_to_output(w.start - clip_start)
            e = self.clamp_to_output(w.end - clip_start)
            if out and s < out[-1].end:
                s = out[-1].end
            out.append(Word(w.text, round(s, 3), round(max(e, s + 0.04), 3), w.prob))
        return out
