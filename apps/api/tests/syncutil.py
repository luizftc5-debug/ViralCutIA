"""Measure caption timing against the audio of a rendered MP4, independently of the recogniser.

Method: render the same clip twice (captions on / off). The frame difference is exactly where a
caption is visible. Speech activity comes from a voice-activity detector run on the *output*
audio, so the comparison covers the whole chain: recognition, alignment, trimming, timeline
mapping, caption layout and burn-in.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

W, H = 270, 480


def gray_frames(path: Path, fps: float = 30.0) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W)


def caption_mask(with_captions: Path, without: Path, fps: float = 30.0, thr: float = 0.35) -> np.ndarray:
    a, b = gray_frames(with_captions, fps), gray_frames(without, fps)
    n = min(len(a), len(b))
    diff = np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16)).mean(axis=(1, 2))
    return diff > thr


def runs(mask: np.ndarray, fps: float, min_frames: int = 3) -> list[tuple[float, float]]:
    out, start = [], None
    for i, on in enumerate(list(mask) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_frames:
                out.append((start / fps, i / fps))
            start = None
    return out


def speech_intervals(path: Path) -> list[tuple[float, float]]:
    """Voice-activity detection (Silero) on the clip's own audio track."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    x = np.frombuffer(raw, np.float32)
    ts = get_speech_timestamps(
        x, VadOptions(threshold=0.5, min_silence_duration_ms=250, speech_pad_ms=0, min_speech_duration_ms=150)
    )
    return [(t["start"] / 16000, t["end"] / 16000) for t in ts]


def to_mask(intervals: list[tuple[float, float]], n: int, fps: float, dilate: float = 0.0) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in intervals:
        m[max(0, int((a - dilate) * fps)) : min(n, int((b + dilate) * fps) + 1)] = True
    return m


def sync_metrics(with_captions: Path, without: Path, fps: float = 30.0) -> dict:
    cap = caption_mask(with_captions, without, fps)
    speech = speech_intervals(without)
    n = len(cap)
    sp = to_mask(speech, n, fps)
    sp_tol = to_mask(speech, n, fps, dilate=0.40)
    visible = max(1, int(cap.sum()))
    last_speech = max((b for _, b in speech), default=0.0)
    cap_runs = runs(cap, fps)
    return {
        "speech_covered": float((cap & sp).sum() / max(1, sp.sum())),  # speech that has a caption on screen
        "stray_visible": float((cap & ~sp_tol).sum() / visible),  # caption on screen with no speech near it
        "last_caption_off": cap_runs[-1][1] if cap_runs else 0.0,
        "last_speech_end": last_speech,
        "caption_runs": len(cap_runs),
        "speech_runs": len(speech),
        "frames": n,
    }
