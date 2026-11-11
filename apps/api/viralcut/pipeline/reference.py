"""/referencia — measure a reference video's editing and adapt this edit to it.

What is measured (first 3 minutes of the reference; nothing is copied from it):

* shot rhythm: shot changes per minute and average shot length (scene-change detector)
* pauses: how long the silences are that the editor left in (loudness envelope)
* audio dynamics: how compressed the voice is (spread of the active speech level)
* motion: average picture activity
* text band: where on screen burned-in text most likely sits (edge density of the top /
  middle / bottom thirds across sampled frames — a heuristic, reported with its confidence)

How it is applied: the pause limit, the punch-in spacing and strength, the jump-cut handling,
the speech treatment and (when the text band is clear) the caption position follow the
reference. The content, words and meaning of the user's video are untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import analyze_audio, decode_audio
from .ffmpeg import input_args, read_pipe
from .probe import probe
from .visual import analyze_visual

MAX_S = 180.0


def _pauses(rms_db: np.ndarray, threshold: float, hop: float) -> list[float]:
    quiet = rms_db < threshold
    runs, n = [], 0
    for q in quiet:
        if q:
            n += 1
        elif n:
            runs.append(n * hop)
            n = 0
    return [r for r in runs if 0.12 <= r <= 3.0]


def _text_band(path: Path, family: str | None, duration: float) -> tuple[str | None, float]:
    import cv2

    w, h = 180, 320
    n = 12
    fps = n / max(1.0, duration)
    args = [*input_args(path, family, duration=duration, threads=1), "-an", "-vf",
            f"fps={fps:.4f},scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h},format=gray", "-f", "rawvideo"]  # fmt: skip
    dens = []
    for buf in read_pipe(args, chunk_size=w * h):
        f = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
        e = cv2.Canny(f, 120, 240) > 0
        dens.append([e[: h // 3].mean(), e[h // 3 : 2 * h // 3].mean(), e[2 * h // 3 :].mean()])
    if not dens:
        return None, 0.0
    d = np.median(np.array(dens), axis=0)
    i = int(np.argmax(d))
    conf = float((d[i] - np.median(d)) / (d[i] + 1e-6))
    return ("top", "middle", "bottom")[i], round(conf, 2)


def analyze_reference(path: Path, family: str | None) -> dict:
    media = probe(path, family)
    dur = min(MAX_S, media.duration)
    measured: dict = {
        "duration_s": round(dur, 1),
        "aspect": "vertical" if media.height > media.width else "horizontal",
    }
    notes: list[str] = []
    visual = analyze_visual(path, family, dur, limit=dur)
    cuts = [c for c in visual.cuts if c < dur]
    measured["shots_per_min"] = round(len(cuts) / dur * 60, 1)
    measured["avg_shot_s"] = round(dur / (len(cuts) + 1), 2)
    measured["motion"] = round(float(np.mean(visual.motion[: int(dur * 4) or None])), 4)
    if media.has_audio:
        x = decode_audio(path, family, dur, limit=dur)
        a = analyze_audio(x)
        pauses = _pauses(a.rms_db, a.silence_threshold_db, a.hop)
        measured["pause_p75_s"] = round(float(np.percentile(pauses, 75)), 2) if pauses else None
        measured["pauses_per_min"] = round(len(pauses) / dur * 60, 1)
        measured["voice_spread_db"] = round(float(a.dynamics(0, dur)), 1)
    band, conf = _text_band(path, family, dur)
    measured["text_band"], measured["text_band_confidence"] = band, conf

    # ---- adapt
    spm = measured["shots_per_min"]
    applied: dict = {}
    applied["intensity"] = "high" if spm >= 18 else "subtle" if spm < 6 else "balanced"
    p75 = measured.get("pause_p75_s")
    if p75:
        applied["max_pause"] = round(float(np.clip(p75 * 1.1, 0.3, 1.2)), 2)
        notes.append(f"pauses kept up to {applied['max_pause']} s (the reference leaves pauses of ~{p75} s)")
    applied["transition"] = "punch" if spm >= 15 else "smooth" if spm < 5 else "cut"
    gap = float(np.clip(measured["avg_shot_s"] * 1.5, 3.0, 12.0))
    amp = 0.09 if spm >= 18 else 0.05 if spm < 6 else 0.07
    applied["zoom"] = {"min_gap": round(gap, 1), "amp": amp}
    notes.append(
        f"{spm} shot changes/min (avg {measured['avg_shot_s']} s) → punch-ins every ≥{gap:.0f} s, {amp:.0%}, cuts: {applied['transition']}"
    )
    spread = measured.get("voice_spread_db")
    if spread is not None:
        applied["audio"] = "punchy" if spread < 12 else "voice" if spread < 18 else "natural"
        notes.append(f"voice level spread {spread} dB → {applied['audio']} speech treatment")
    if band and conf >= 0.35:
        applied["caption_position"] = band
        notes.append(f"burned-in text mostly in the {band} third (confidence {conf:.0%}) → captions {band}")
    else:
        notes.append("no clear text band in the reference: caption position unchanged")
    return {"measured": measured, "applied": applied, "notes": notes}
