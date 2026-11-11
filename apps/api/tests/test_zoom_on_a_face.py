"""The whole editing chain on a real face: detection -> framing -> content profile -> zoom plan -> render.

The punch-in is verified with the face detector itself: the face must be measurably larger at the
peak of the zoom than before it, by the planned amount, and back to normal afterwards.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from viralcut.pipeline import render as R
from viralcut.pipeline.audio import AudioSignals
from viralcut.pipeline.orchestrator import Artifacts, ClipPlan, step_framing
from viralcut.pipeline.probe import probe
from viralcut.pipeline.transcribe import Transcript, Word
from viralcut.pipeline.visual import VisualSignals

FACE = Path(__file__).parent / "fixtures" / "face" / "astronaut.jpg"
DURATION = 20.0


@pytest.fixture(scope="module")
def face_video(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("face") / "face.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=0x20303c:s=1280x720:r=30:d={DURATION}",
         "-loop", "1", "-i", str(FACE),
         "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=48000:duration={DURATION}",
         "-filter_complex", "[1:v]scale=-2:560[f];[0:v][f]overlay=x=640:y=80:shortest=1,format=yuv420p[v]",
         "-map", "[v]", "-map", "2:a", "-t", str(DURATION), "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out)],
        check=True,
    )  # fmt: skip
    return out


def _artifacts(media) -> tuple[Artifacts, list[Word]]:
    names = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()
    words = [Word(names[i % len(names)], 1.0 + i * 0.5, 1.0 + i * 0.5 + 0.4, 0.95) for i in range(34)]
    hop = 0.05
    env = np.full(int((DURATION + 1) / hop), -30.0, dtype=np.float32)
    for i in (12, 24):  # the speaker leans on two words: +14 dB over their own baseline
        a, b = int(words[i].start / hop), int(words[i].end / hop)
        env[a:b] = -16.0
    audio = AudioSignals(env, noise_floor_db=-60.0, speech_db=-25.0, clip_ratio=0.0)
    visual = VisualSignals(np.full(int(DURATION * 4) + 1, 0.01, dtype=np.float32), [])
    art = Artifacts(media, None, Transcript(language="en", words=words), audio, visual, [])
    return art, words


def _frame(video: Path, t: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(1920, 1080)


def _scale_between(a: np.ndarray, b: np.ndarray) -> float:
    """Image registration (ORB features + a similarity transform): how much larger is b than a?

    Measures the picture itself, not a detector's bounding box (which snaps to anchor sizes)."""
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    m, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    assert m is not None and int(inliers.sum()) > 40, "frames could not be registered"
    return float(np.hypot(m[0, 0], m[1, 0]))


def test_a_stressed_word_gets_a_measured_punch_in_on_a_tracked_face(face_video, tmp_path):
    media = probe(face_video)
    art, words = _artifacts(media)
    cp = ClipPlan("c1", 0.0, 18.0, words)
    step_framing(cp, art, face_video, "auto", 0.7, True, user_style="auto", smart_zoom=True)

    assert {s.layout["type"] for s in cp.plan.shots} == {"crop"}, "the face is found and followed"
    assert cp.plan.face_ratio > 0.9
    assert cp.profile is not None and cp.profile.name in {
        "general",
        "podcast",
        "education",
        "storytelling",
        "entertainment",
    }
    assert cp.zooms, "two words are stressed far above the speaker's baseline"
    z = cp.zooms[0]
    assert z.evidence["lift_db"] >= 10 and 0.04 < z.amp < 0.09 and z.t0 > 5.0 and z.reason == "vocal emphasis"
    assert all(
        b.t0 - a.t0 >= cp.profile.zoom["min_gap"] for a, b in zip(cp.zooms, cp.zooms[1:], strict=False)
    )

    out = tmp_path / "zoomed.mp4"
    R.render_clip(face_video, None, media, cp.start, cp.segments, out, polish=cp.audio_polish)
    peak_t = (z.t0 + z.ramp + z.t1 - z.ramp) / 2
    base = _frame(out, max(0.5, z.t0 - 1.0))
    at_peak = _scale_between(base, _frame(out, peak_t))
    half_way = _scale_between(base, _frame(out, z.t0 + z.ramp / 2))
    after = _scale_between(base, _frame(out, min(cp.duration - 0.3, z.t1 + 1.2)))
    assert at_peak == pytest.approx(1 + z.amp, abs=0.012), (
        f"picture scale {at_peak:.3f} at the peak, planned {1 + z.amp:.3f}"
    )
    assert 1.0 < half_way < at_peak, "eased in: halfway through the ramp it has not reached the peak"
    assert after == pytest.approx(1.0, abs=0.01), "framing returns to normal after the move"
    assert _scale_between(base, _frame(out, max(0.5, z.t0 - 0.5))) == pytest.approx(1.0, abs=0.01), (
        "no creep before the move"
    )


def test_zoom_can_be_switched_off(face_video):
    media = probe(face_video)
    art, words = _artifacts(media)
    cp = ClipPlan("c2", 0.0, 18.0, words)
    step_framing(cp, art, face_video, "auto", 0.7, True, user_style="auto", smart_zoom=False)
    assert cp.zooms == []
