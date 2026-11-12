"""Trimming, reframing, rendering and QA."""

from fractions import Fraction

import numpy as np
import pytest

from viralcut.pipeline import reframe, render, trimming
from viralcut.pipeline.audio import AudioSignals, analyze_audio, decode_audio, silent_signals
from viralcut.pipeline.probe import MediaInfo, ProbeError, probe
from viralcut.pipeline.qa import verify_clip
from viralcut.pipeline.reframe import Shot
from viralcut.pipeline.transcribe import Word
from viralcut.pipeline.visual import analyze_visual, detect_cuts
from viralcut.security import sniff_container

MEDIA = MediaInfo(duration=60, width=1920, height=1080, fps="30/1", video_codec="h264", has_audio=True)


def _audio_with_silence(duration: float, silences: list[tuple[float, float]]) -> AudioSignals:
    a = silent_signals(duration)
    a.rms_db[:] = -18.0
    a.speech_db, a.noise_floor_db = -18.0, -60.0
    for s, e in silences:
        a.rms_db[int(s / a.hop) : int(e / a.hop)] = -60.0
    return a


# ---------------------------------------------------------------- trimming


def test_long_silent_pause_is_shortened_but_not_removed():
    words = [Word("one", 0.2, 0.6), Word("two", 0.7, 1.1), Word("three", 4.1, 4.5), Word("four", 4.6, 5.0)]
    audio = _audio_with_silence(6, [(1.15, 4.05)])
    keeps = trimming.plan_keeps(words, 0.0, 5.4, audio, max_pause=0.7)
    assert len(keeps) == 2
    removed = 5.4 - sum(b - a for a, b in keeps)
    assert 2.0 < removed < 2.8
    assert keeps[0][1] >= 1.1 + 0.1 and keeps[1][0] <= 4.1 - 0.1, "never cuts into speech"


def test_pause_with_sound_is_preserved():
    words = [Word("one", 0.2, 0.6), Word("two", 3.6, 4.0)]
    audio = _audio_with_silence(5, [])  # laughter/music fills the gap
    assert trimming.plan_keeps(words, 0.0, 4.5, audio, max_pause=0.7) == [(0.0, 4.5)]


def test_trimming_disabled_and_question_beat():
    words = [Word("why?", 0.2, 0.6), Word("because", 1.4, 1.9)]
    audio = _audio_with_silence(3, [(0.65, 1.35)])
    assert trimming.plan_keeps(words, 0.0, 2.2, audio, max_pause=None) == [(0.0, 2.2)]
    # 0.8 s after a question is a natural beat even in the most aggressive style
    assert trimming.plan_keeps(words, 0.0, 2.2, audio, max_pause=0.45) == [(0.0, 2.2)]


def test_time_map_remaps_words_after_cuts():
    tm = trimming.TimeMap([(0.0, 1.0), (3.0, 5.0)])
    assert tm.duration == 3.0
    assert tm.to_output(0.5) == 0.5 and tm.to_output(2.0) is None and tm.to_output(3.5) == 1.5
    words = tm.map_words([Word("a", 10.2, 10.6), Word("b", 13.2, 13.8)], clip_start=10.0)
    assert [(w.start, w.end) for w in words] == [(0.2, 0.6), (1.2, 1.8)]


# ---------------------------------------------------------------- reframing


def _face(cx: float, cy: float = 400, size: float = 220):
    return (cx - size / 2, cy - size / 2, size, size, 0.95)


def test_crop_size_and_vertical_detection():
    assert reframe.crop_size(1920, 1080) == (608, 1080)
    assert reframe.crop_size(1080, 1920) == (1080, 1920)
    assert reframe.crop_size(1080, 1080) == (608, 1080)
    assert reframe.is_vertical(MediaInfo(1, 1080, 1920, "30/1", "h264", True))
    assert not reframe.is_vertical(MEDIA)


def test_static_face_gets_static_crop_centred_on_it():
    samples = [(i / 5, [_face(1400 + (i % 2) * 6)]) for i in range(50)]
    layout = reframe.layout_for_shot(samples, 0.0, 10.0, MEDIA, "auto")
    assert layout["type"] == "crop" and len(layout["keys"]) == 1
    x = layout["keys"][0][1]
    assert abs((x + 304) - 1403) < 20


def test_moving_face_gets_smooth_bounded_path():
    samples = [(i / 5, [_face(300 + i * 24)]) for i in range(50)]  # pans 300 -> 1476 px
    layout = reframe.layout_for_shot(samples, 0.0, 10.0, MEDIA, "auto")
    keys = layout["keys"]
    assert layout["type"] == "crop" and len(keys) >= 2
    assert all(0 <= x <= 1920 - 608 for _, x in keys)
    assert keys[-1][1] - keys[0][1] > 600
    speeds = [abs(x1 - x0) / (t1 - t0) for (t0, x0), (t1, x1) in zip(keys, keys[1:], strict=False)]
    assert max(speeds) < 400, "no abrupt reframing"


def test_two_distant_faces_use_split_layout_and_close_faces_share_a_crop():
    far = [(i / 5, [_face(350), _face(1550)]) for i in range(40)]
    layout = reframe.layout_for_shot(far, 0.0, 8.0, MEDIA, "auto")
    assert layout["type"] == "split"
    assert layout["top"]["x"] < layout["bottom"]["x"]
    for panel in (layout["top"], layout["bottom"]):
        assert panel["x"] >= 0 and panel["x"] + panel["w"] <= 1920 and panel["y"] + panel["h"] <= 1080
        assert abs(panel["w"] / panel["h"] - 9 / 8) < 0.02
    near = [(i / 5, [_face(850, size=140), _face(1050, size=140)]) for i in range(40)]
    layout = reframe.layout_for_shot(near, 0.0, 8.0, MEDIA, "auto")
    assert layout["type"] == "crop" and abs(layout["keys"][0][1] + 304 - 950) < 30
    assert reframe.layout_for_shot(far, 0.0, 8.0, MEDIA, "face")["type"] == "crop"


def test_no_face_falls_back_to_blur_and_modes_are_respected():
    empty = [(i / 5, []) for i in range(40)]
    assert reframe.layout_for_shot(empty, 0.0, 8.0, MEDIA, "auto")["type"] == "blur"
    assert reframe.layout_for_shot(empty, 0.0, 8.0, MEDIA, "face")["type"] == "crop"
    assert reframe.layout_for_shot(empty, 0.0, 8.0, MEDIA, "blur")["type"] == "blur"
    centre = reframe.layout_for_shot(empty, 0.0, 8.0, MEDIA, "center")
    assert centre["keys"][0][1] == (1920 - 608) // 2


def test_path_simplification_keeps_endpoints():
    t = np.linspace(0, 10, 51)
    keys = reframe.simplify(t, np.where(t < 5, 100.0, 100 + (t - 5) * 80), tolerance=5)
    assert keys[0] == (0.0, 100.0) and keys[-1][0] == 10.0 and 2 <= len(keys) <= 4


def test_cut_detection_ignores_gradual_motion():
    diffs = np.full(200, 0.01, dtype=np.float32)
    diffs[80] = 0.4
    diffs[120:140] = 0.06  # sustained motion, not a cut
    assert detect_cuts(diffs, 4.0) == [20.0]


# ---------------------------------------------------------------- render graph


def test_segments_snap_to_frames_and_follow_shots():
    crop_a = {"type": "crop", "w": 608, "h": 1080, "y": 0, "keys": [(0.0, 100)]}
    blur = {"type": "blur"}
    shots = [Shot(0.0, 4.0, crop_a), Shot(4.0, 10.0, blur)]
    segs = render.build_segments([(100.0, 103.0), (105.0, 110.0)], shots, 100.0, Fraction(30))
    assert [(s.f0, s.f1, s.layout["type"]) for s in segs] == [(0, 90, "crop"), (150, 300, "blur")]
    tm = render.segments_time_map(segs, Fraction(30))
    assert tm.duration == pytest.approx(8.0)


def test_filtergraph_structure():
    pan = {"type": "crop", "w": 608, "h": 1080, "y": 0, "keys": [(0.0, 100), (2.0, 500), (4.0, 500)]}
    segs = [render.Segment(0, 120, pan), render.Segment(150, 300, {"type": "blur"})]
    g = render.build_filtergraph(segs, MEDIA, Fraction(30), 1080, 1920, "v1.ass", True)
    assert "trim=start_frame=0:end_frame=120" in g and "trim=start_frame=150:end_frame=300" in g
    assert "concat=n=2:v=1:a=0" in g and "concat=n=2:v=0:a=1" in g
    assert "ass=filename=v1.ass:fontsdir=fonts" in g and "loudnorm" in g
    assert "clip(100+200.0000*clip(t+0.0000-0.0000,0,2.0000),0,1312)" in g
    no_caps = render.build_filtergraph(segs[:1], MEDIA, Fraction(30), 1080, 1920, None, False)
    assert "ass=" not in no_caps and "concat" not in no_caps and "loudnorm" not in no_caps


def test_output_profile_follows_source_quality():
    assert render.output_size(MEDIA) == (1080, 1920)
    assert render.output_size(MediaInfo(1, 640, 360, "30/1", "h264", True)) == (720, 1280)
    assert render.output_fps(MediaInfo(1, 1920, 1080, "60/1", "h264", True)) == 60
    assert render.output_fps(MediaInfo(1, 1920, 1080, "120/1", "h264", True)) == 60
    assert render.output_fps(MediaInfo(1, 1920, 1080, "24000/1001", "h264", True)) == Fraction(24000, 1001)


# ---------------------------------------------------------------- real media


def test_probe_reports_real_properties(silent_video, no_audio_video, vertical_video):
    info = probe(silent_video, "isobmff")
    assert (info.width, info.height, info.has_audio, info.video_codec) == (640, 360, True, "h264")
    assert 13.5 < info.duration < 14.5 and info.fps_float == pytest.approx(30)
    assert probe(no_audio_video, "matroska").has_audio is False
    assert reframe.is_vertical(probe(vertical_video, "isobmff"))


def test_probe_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"not really a video" * 50)
    with pytest.raises(ProbeError):
        probe(bad, "isobmff")


def test_audio_and_visual_analysis_on_real_file(silent_video):
    info = probe(silent_video, "isobmff")
    samples = decode_audio(silent_video, "isobmff", info.duration)
    assert abs(len(samples) / 16000 - info.duration) < 0.3
    audio = analyze_audio(samples)
    assert -30 < audio.speech_db < 0  # the sine tone
    visual = analyze_visual(silent_video, "isobmff", info.duration)
    assert len(visual.cuts) == 2
    assert abs(visual.cuts[0] - 5.0) < 0.6 and abs(visual.cuts[1] - 9.0) < 0.6


def test_render_and_qa_end_to_end(silent_video, tmp_path):
    """A real render: two kept segments, pan + blur layouts, burned captions, verified output."""
    from viralcut.options import CaptionOptions
    from viralcut.pipeline import captions as cap

    family = sniff_container(silent_video.read_bytes()[:64])
    media = probe(silent_video, family)
    fps = render.output_fps(media)
    cw, ch = reframe.crop_size(media.width, media.height)
    pan = {"type": "crop", "w": cw, "h": ch, "y": 0, "keys": [(0.0, 0), (3.0, media.width - cw)]}
    shots = [Shot(0.0, 3.0, pan), Shot(3.0, 8.0, {"type": "blur"})]
    segs = render.build_segments([(2.0, 4.5), (6.0, 9.0)], shots, 2.0, fps)
    words = [Word("Olá", 0.3, 0.8), Word("coração", 0.9, 1.6), Word("valente", 3.0, 3.8)]
    opts = CaptionOptions(style="dynamic_creator")
    ow, oh = render.output_size(media)
    pages = cap.paginate(words, opts, 5.5)
    ass = tmp_path / "v1.ass"
    ass.write_text(cap.build_ass(pages, opts, "universal", ow, oh), encoding="utf-8")
    out = tmp_path / "v1.mp4"
    params = render.render_clip(silent_video, family, media, 2.0, segs, out, ass)
    assert params["duration"] == pytest.approx(5.5, abs=0.05) and params["segments"] == 2
    report = verify_clip(
        out,
        expected_width=ow,
        expected_height=oh,
        expected_duration=params["duration"],
        expected_fps=params["fps"],
        source_has_audio=True,
    )
    assert report["passed"], report["checks"]
    assert (report["width"], report["height"]) == (720, 1280)
    assert not out.with_suffix(".tmp.mp4").exists()


def test_qa_catches_wrong_output(silent_video):
    report = verify_clip(
        silent_video,
        expected_width=1080,
        expected_height=1920,
        expected_duration=14.0,
        expected_fps="30/1",
        source_has_audio=True,
    )
    assert not report["passed"]
    assert report["checks"]["resolution"] is False and report["checks"]["vertical_9_16"] is False


def test_source_without_audio_gets_silent_track(no_audio_video, tmp_path):
    family = sniff_container(no_audio_video.read_bytes()[:64])
    media = probe(no_audio_video, family)
    fps = render.output_fps(media)
    segs = render.build_segments([(1.0, 5.0)], [Shot(0.0, 4.0, {"type": "blur"})], 1.0, fps)
    out = tmp_path / "v1.mp4"
    params = render.render_clip(no_audio_video, family, media, 1.0, segs, out)
    report = verify_clip(
        out,
        expected_width=params["width"],
        expected_height=params["height"],
        expected_duration=params["duration"],
        expected_fps=params["fps"],
        source_has_audio=False,
    )
    assert report["passed"], report["checks"]


def test_split_layout_moves_bottom_captions_to_the_seam(tmp_path):
    from viralcut.options import CaptionOptions
    from viralcut.pipeline.orchestrator import Artifacts, ClipPlan, step_captions
    from viralcut.pipeline.reframe import FramingPlan
    from viralcut.pipeline.transcribe import Transcript
    from viralcut.pipeline.visual import VisualSignals
    from viralcut.storage import LocalStorage

    panel = {"x": 0, "y": 0, "w": 900, "h": 800}
    words = [Word("hello", 0.2, 0.6), Word("there", 0.7, 1.1)]
    art = Artifacts(
        MEDIA,
        "isobmff",
        Transcript(),
        silent_signals(10),
        VisualSignals(np.zeros(40, dtype=np.float32), []),
        [],
    )
    storage, job_id, clip_id = LocalStorage(tmp_path), "a" * 32, "b" * 32

    def ys(layout):
        cp = ClipPlan(
            clip_id,
            0.0,
            5.0,
            words,
            plan=FramingPlan([Shot(0.0, 5.0, layout)]),
            out_words=words,
            duration=5.0,
        )
        step_captions(cp, art, storage, job_id, CaptionOptions(position="bottom"), "universal")
        text = cp.ass_path.read_text()
        import re

        return [float(m.group(1)) for m in re.finditer(r"\\pos\([0-9.]+,([0-9.]+)\)", text)], cp

    split_y, cp = ys({"type": "split", "top": panel, "bottom": panel})
    crop_y, _ = ys({"type": "blur"})
    assert max(split_y) < 1920 * 0.6 < min(crop_y)
    assert any("middle" in n for n in cp.plan.notes)


@pytest.mark.parametrize("mode", ["auto", "face", "split"])
def test_framing_a_clip_without_shot_change(tmp_path, mode):
    """A clip with no shot change at all (a long gameplay take) must frame without error.

    The cut-detection output used to drop every frame that was not a cut, so in such a clip it
    received no frame; ffmpeg 7+ then aborts the whole command ("Could not open encoder before
    EOF"). The test suite runs ffmpeg with -abort_on empty_output_stream to fail the same way.
    """
    import subprocess

    clip = tmp_path / "one_take.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=s=1280x720:r=30:d=8", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )  # fmt: skip
    media = probe(clip)
    plan = reframe.plan_framing(clip, None, media, 1.0, 7.0, mode)
    assert plan.shots and plan.shots[0].start == 0.0
    assert abs(plan.shots[-1].end - 6.0) < 0.05
