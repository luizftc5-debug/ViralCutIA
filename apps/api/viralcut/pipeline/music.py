"""/trilha: tempo and beat analysis of a user-supplied music file (numpy only, first 90 s)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, decode_audio

HOP = 512  # 32 ms at 16 kHz


def onset_envelope(x: np.ndarray) -> np.ndarray:
    """Spectral flux: how much new energy appears in each 32 ms frame."""
    n = 1024
    if len(x) < n * 2:
        return np.zeros(1)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::HOP] * np.hanning(n)
    mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
    flux -= np.convolve(flux, np.ones(16) / 16, mode="same")  # remove slow loudness changes
    return np.maximum(flux, 0.0)


def analyze_music(path: Path, family: str | None = None) -> dict:
    """Duration, tempo (BPM) and the time of the first strong beat."""
    from .ffmpeg import run_capture

    res = run_capture(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_entries", "format=duration",
                       "-of", "csv=p=0", str(path)], timeout=30)  # fmt: skip
    try:
        duration = float(res.stdout.decode().strip().splitlines()[0])
    except (ValueError, IndexError):
        raise ValueError("the music file could not be read") from None
    x = decode_audio(path, None, min(90.0, duration), limit=min(90.0, duration))
    env = onset_envelope(x)
    fr = SAMPLE_RATE / HOP
    tempo, first = 0.0, 0.0
    if len(env) > fr * 4 and env.std() > 0:
        e = (env - env.mean()) / env.std()
        ac = np.correlate(e, e, mode="full")[len(e) - 1 :]
        lo, hi = int(fr * 60 / 180), int(fr * 60 / 60)  # 60..180 BPM
        lag = lo + int(np.argmax(ac[lo:hi]))
        tempo = 60.0 * fr / lag
        strong = np.where(e > 1.5)[0]
        if len(strong):
            # the phase of the beat grid that collects the most onset energy, then its first strong beat
            phase = int(np.argmax([e[p::lag].sum() for p in range(lag)]))
            beats = np.arange(phase, len(e), lag)
            hits = [b for b in beats if e[max(0, b - 2) : b + 3].max() > 1.0]
            first = float((hits[0] if hits else strong[0]) / fr)
    return {"duration": round(duration, 2), "tempo": round(tempo, 1), "first_beat": round(first, 3)}
