"""Rendering features measured on real ffmpeg output: punch-in zoom, loudness, signalling."""

import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from viralcut.pipeline import render as R
from viralcut.pipeline.audio import AudioSignals, analyze_audio
from viralcut.pipeline.probe import probe
from viralcut.pipeline.qa import verify_clip


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def box_video(tmp_path_factory) -> Path:
    """1920x1080: a 200 px white square at the exact centre, and a quiet tone (-35 dBFS) as audio."""
    out = tmp_path_factory.mktemp("box") / "box.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=30:d=6",
        "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=6",
        "-vf", "drawbox=x=860:y=440:w=200:h=200:color=white:t=fill,format=yuv420p",
        "-af", "volume=-35dB", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(out),
    )  # fmt: skip
    return out


def _crop_segment(frames: int, zooms=()):
    seg = R.Segment(0, frames, {"type": "crop", "w": 608, "h": 1080, "y": 0, "keys": [(0.0, 656)]})
    seg.zooms = list(zooms)
    return seg


def _white_width(video: Path, t: float) -> int:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    img = np.frombuffer(raw, np.uint8).reshape(1920, 1080)
    return int(np.sum(img[960] > 128))


def test_punch_in_zooms_smoothly_by_the_planned_amount(box_video, tmp_path):
    media = probe(box_video)
    fps = Fraction(30)
    seg = _crop_segment(150, [(1.5, 3.5, 0.08, 0.5)])
    out = tmp_path / "zoom.mp4"
    R.render_clip(box_video, None, media, 0.0, [seg], out)
    base = _white_width(out, 0.8)
    assert abs(base - 200 * 1080 / 608) < 6, "unzoomed, the frame is a plain crop-and-scale"
    peak = _white_width(out, 2.5)
    assert peak / base == pytest.approx(1.08, abs=0.015)
    assert _white_width(out, 1.75) / base == pytest.approx(1.04, abs=0.015)  # halfway up the smoothstep
    assert _white_width(out, 4.5) / base == pytest.approx(1.0, abs=0.01), "returns to the original framing"
    assert output_ok(out, media, fps)


def output_ok(out: Path, media, fps) -> bool:
    info = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)],
            capture_output=True,
            text=True,
        ).stdout
    )
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    return (v["width"], v["height"]) == (1080, 1920) and v["codec_name"] == "h264"


def test_zoom_is_ignored_for_layouts_where_it_would_hide_content():
    blur = R.Segment(0, 90, {"type": "blur"})
    split = R.Segment(0, 90, {"type": "split", "top": {}, "bottom": {}})
    from viralcut.pipeline.editing import Zoom

    applied = R.assign_zooms([blur, split], Fraction(30), [Zoom(1.0, 2.5, 0.07, 0.4, "x")])
    assert applied == [] and blur.zooms == [] and split.zooms == []
    crop = _crop_segment(90)
    applied = R.assign_zooms(
        [crop], Fraction(30), [Zoom(1.0, 2.5, 0.07, 0.4, "x"), Zoom(2.9, 3.4, 0.07, 0.2, "outside")]
    )
    assert len(applied) == 1 and crop.zooms == [(1.0, 2.5, 0.07, 0.4)], "a zoom must fit inside one segment"


def _loudness(path: Path) -> tuple[float, float]:
    err = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True,
        text=True,
    ).stderr
    summary = err[err.rfind("Summary:") :]
    return (
        float(re.search(r"I:\s*(-?[\d.]+) LUFS", summary).group(1)),
        float(re.search(r"Peak:\s*(-?[\d.]+) dBFS", summary).group(1)),
    )


def test_two_pass_loudness_hits_the_target_without_clipping(box_video, tmp_path):
    media = probe(box_video)
    assert _loudness(box_video)[0] < -30, "the source really is quiet"
    audio = analyze_audio(np.random.default_rng(0).normal(0, 0.05, 16000 * 6).astype(np.float32))
    polish = R.plan_audio_polish(audio, 0.0, 6.0)
    out = tmp_path / "loud.mp4"
    params = R.render_clip(box_video, None, media, 0.0, [_crop_segment(180)], out, polish=polish)
    lufs, peak = _loudness(out)
    assert lufs == pytest.approx(-14.0, abs=1.0), f"integrated loudness {lufs}"
    assert peak <= -1.0, f"true peak {peak}: no clipping"
    assert params["audio"]["loudness_pass"] == "two-pass linear"
    assert params["audio"]["measured_lufs"] < -30
    report = verify_clip(
        out,
        expected_width=1080,
        expected_height=1920,
        expected_duration=params["duration"],
        expected_fps=params["fps"],
        source_has_audio=True,
    )
    assert report["passed"], report["checks"]
    assert report["integrated_lufs"] == pytest.approx(lufs, abs=0.5) and report["audio_sample_rate"] == 48000


def test_signals_bt709_and_caps_bitrate_spikes(box_video, tmp_path):
    media = probe(box_video)
    out = tmp_path / "tags.mp4"
    R.render_clip(box_video, None, media, 0.0, [_crop_segment(90)], out)
    v = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-select_streams", "v", "-of", "json", str(out)],
            capture_output=True,
            text=True,
        ).stdout
    )["streams"][0]
    assert (v["color_space"], v["color_primaries"], v["color_transfer"], v["color_range"]) == (
        "bt709",
        "bt709",
        "bt709",
        "tv",
    )
    assert v["profile"] == "High" and v["pix_fmt"] == "yuv420p" and v["sample_aspect_ratio"] == "1:1"
    assert abs(Fraction(v["avg_frame_rate"]) - 30) < 0.01, "source frame rate preserved"


def test_polish_is_decided_from_measurements():
    rng = np.random.default_rng(5)

    def sig(noise_db: float, speech_db: float, spread: float = 6.0):
        n = 400
        env = np.concatenate(
            [np.full(n // 2, speech_db) + rng.normal(0, spread / 3, n // 2), np.full(n // 2, noise_db)]
        ).astype(np.float32)
        return AudioSignals(env, noise_floor_db=noise_db, speech_db=speech_db, clip_ratio=0.0)

    clean = R.plan_audio_polish(sig(-70, -20), 0, 20)
    assert clean.denoise_nr is None and not clean.compress
    noisy = R.plan_audio_polish(sig(-38, -30), 0, 20)
    assert (
        noisy.denoise_nr
        and 4 <= noisy.denoise_nr <= 9
        and "noise reduction" in " ".join(noisy.report["applied"])
    )
    swingy = R.plan_audio_polish(sig(-70, -30, spread=40), 0, 20)
    assert swingy.compress
    assert R._polish_chain(clean) == "highpass=f=70", "clean speech only loses rumble"


def test_noise_reduction_and_levelling_chain_renders_cleanly(box_video, tmp_path):
    """The conditional treatments (only chosen for noisy / widely swinging audio) must be valid ffmpeg."""
    media = probe(box_video)
    polish = R.AudioPolish(denoise_nr=8.0, noise_floor_db=-45.0, compress=True)
    out = tmp_path / "treated.mp4"
    params = R.render_clip(box_video, None, media, 0.0, [_crop_segment(120)], out, polish=polish)
    assert params["audio"]["loudness_pass"] == "two-pass linear"
    chain = R._polish_chain(polish)
    assert chain.startswith("highpass=f=70,afftdn=nr=8.0:nf=-45") and "acompressor" in chain
    report = verify_clip(
        out,
        expected_width=1080,
        expected_height=1920,
        expected_duration=params["duration"],
        expected_fps=params["fps"],
        source_has_audio=True,
    )
    assert report["passed"], report["checks"]
    assert report["integrated_lufs"] == pytest.approx(-14.0, abs=1.5)
