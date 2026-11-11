"""Stage 4b — whole-video visual analysis: motion energy timeline and shot changes.

One low-resolution decode pass produces both signals. Deblocking and non-reference frames are
skipped because the analysis only needs coarse luminance changes, which keeps long videos fast.
Frame-accurate shot boundaries for the selected clips are refined later in ``reframe``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg import CancelFn, ProgressFn, input_args, read_pipe

ANALYSIS_FPS = 4.0
_W, _H = 96, 54


@dataclass
class VisualSignals:
    motion: np.ndarray  # mean absolute luminance change between consecutive samples (0..1)
    cuts: list[float]  # approximate shot-change times (s)
    fps: float = ANALYSIS_FPS

    def _slice(self, t0: float, t1: float) -> np.ndarray:
        a = max(0, int(t0 * self.fps))
        b = max(a + 1, min(len(self.motion), int(np.ceil(t1 * self.fps))))
        return self.motion[a:b]

    def motion_level(self, t0: float, t1: float) -> float:
        s = self._slice(t0, t1)
        return float(np.mean(s)) if len(s) else 0.0

    def motion_percentile(self, value: float) -> float:
        return float(np.mean(self.motion <= value)) if len(self.motion) else 0.5

    def cuts_between(self, t0: float, t1: float) -> list[float]:
        return [c for c in self.cuts if t0 <= c < t1]

    def to_arrays(self) -> dict:
        return {"motion": self.motion.astype(np.float32), "cuts": np.array(self.cuts, dtype=np.float64)}

    @classmethod
    def from_arrays(cls, motion: np.ndarray, cuts: np.ndarray) -> VisualSignals:
        return cls(motion=motion, cuts=[float(c) for c in cuts])


def detect_cuts(
    diffs: np.ndarray, fps: float, *, floor: float = 0.08, ratio: float = 5.0, min_gap_s: float = 0.75
) -> list[float]:
    """A shot change is a luminance jump that dwarfs the local motion level."""
    cuts: list[float] = []
    n = len(diffs)
    win = max(4, int(fps * 4))
    for i in range(1, n):
        d = diffs[i]
        if d < floor:
            continue
        lo, hi = max(0, i - win), min(n, i + win + 1)
        neighbours = np.concatenate([diffs[lo:i], diffs[i + 1 : hi]])
        local = float(np.median(neighbours)) if len(neighbours) else 0.0
        if d >= max(floor, ratio * local):
            t = i / fps
            if not cuts or t - cuts[-1] >= min_gap_s:
                cuts.append(t)
    return cuts


def analyze_visual(
    path: Path,
    family: str | None,
    duration: float,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
) -> VisualSignals:
    args = [
        "-skip_loop_filter", "all", "-skip_frame", "noref",
        *input_args(path, family),
        "-an", "-sn", "-map", "0:v:0",
        "-vf", f"fps={ANALYSIS_FPS},scale={_W}:{_H}:flags=fast_bilinear,format=gray",
        "-f", "rawvideo",
    ]  # fmt: skip
    expected = max(1.0, duration * ANALYSIS_FPS)
    prev: np.ndarray | None = None
    diffs: list[float] = []
    for i, buf in enumerate(read_pipe(args, chunk_size=_W * _H, should_cancel=should_cancel)):
        frame = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
        diffs.append(0.0 if prev is None else float(np.mean(np.abs(frame - prev))) / 255.0)
        prev = frame
        if on_progress and i % 20 == 0:
            on_progress(min(1.0, i / expected))
    motion = np.array(diffs or [0.0], dtype=np.float32)
    cuts = detect_cuts(motion, ANALYSIS_FPS)
    # shot changes are not "motion": neutralise their spikes so they do not inflate activity scores
    smooth = motion.copy()
    for c in cuts:
        idx = int(round(c * ANALYSIS_FPS))
        if 0 < idx < len(smooth):
            smooth[idx] = smooth[idx - 1]
    return VisualSignals(motion=smooth, cuts=cuts)
