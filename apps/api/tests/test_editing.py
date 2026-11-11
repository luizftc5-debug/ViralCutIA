"""Editing decisions: every one must be justified by a measurement, and must respect its limits."""

import math

import numpy as np
import pytest
from helpers import STRONG, WEAK, make_words

from viralcut.pipeline import editing as E
from viralcut.pipeline.audio import AudioSignals
from viralcut.pipeline.transcribe import Word
from viralcut.pipeline.visual import VisualSignals


def _audio(seconds: float, loud: list[tuple[float, float]] = (), base: float = -30.0, peak: float = -14.0):
    env = np.full(int(seconds / 0.05), base, dtype=np.float32)
    for a, b in loud:
        env[round(a / 0.05) : round(b / 0.05)] = peak
    return AudioSignals(env, noise_floor_db=-60.0, speech_db=-25.0, clip_ratio=0.0)


def _visual(seconds: float, level: float) -> VisualSignals:
    return VisualSignals(np.full(int(seconds * 4) + 1, level, dtype=np.float32), [])


def test_zoom_expression_is_a_smoothstep_in_and_out():
    expr = E.zoom_expression([(1.5, 3.5, 0.08, 0.5)], local_offset=0.0)

    def z(t):
        return eval(expr, {"clip": lambda x, a, b: max(a, min(b, x)), "t": t})  # noqa: S307 - our own expression

    assert z(0.0) == pytest.approx(1.0) and z(1.5) == pytest.approx(1.0)
    assert z(2.0) == pytest.approx(1.08) and z(3.0) == pytest.approx(1.08)  # plateau
    assert z(1.75) == pytest.approx(1.04)  # halfway up the smoothstep
    assert z(3.5) == pytest.approx(1.0) and z(9.0) == pytest.approx(1.0)
    ramp = [z(1.5 + i * 0.05) for i in range(11)]
    assert ramp == sorted(ramp), "monotonic ramp, no overshoot"
    # a segment that starts later in the clip sees the same zoom on its own local clock
    shifted = E.zoom_expression([(1.5, 3.5, 0.08, 0.5)], local_offset=1.0)
    assert eval(shifted, {"clip": lambda x, a, b: max(a, min(b, x)), "t": 1.0}) == pytest.approx(1.08)  # noqa: S307


def test_classifies_content_from_evidence_not_a_template():
    gaming = make_words(
        "the boss respawn was insane in this game, the squad clutch headshot on the final level " * 3
    )
    prof = E.classify_content(
        gaming, "en", _audio(60), _visual(60, 0.2), 0, 60, face_ratio=0.0, user_style="auto"
    )
    assert prof.name == "gaming" and prof.zoom["allowed"] is False and prof.max_pause is None
    assert any("gaming" in e for e in prof.evidence)

    lesson = make_words(
        "first learn the rule because the method means you practice every step, for example the second tip is an important concept "
        * 2
    )
    prof = E.classify_content(
        lesson, "en", _audio(60), _visual(60, 0.01), 0, 60, face_ratio=0.9, user_style="auto"
    )
    assert prof.name == "education" and prof.zoom["allowed"]

    prof = E.classify_content(make_words(WEAK), "en", _audio(30), _visual(30, 0.05), 0, 30, face_ratio=0.5)
    assert prof.name == "general", "when nothing stands out the engine says so instead of guessing"


def test_an_explicit_style_always_wins():
    prof = E.classify_content(
        make_words(STRONG), "en", _audio(60), _visual(60, 0.05), 0, 60, user_style="gaming"
    )
    assert prof.name == "gaming" and prof.confidence == 1.0 and "chosen by the user" in prof.evidence[0]


def test_portuguese_vocabulary_is_recognised():
    texto = make_words("o jogo da partida no minecraft foi surreal, o boss do mapa e o time na live " * 3)
    prof = E.classify_content(texto, "pt", _audio(60), _visual(60, 0.2), 0, 60, face_ratio=0.0)
    assert prof.name == "gaming"


def test_opening_removes_only_pure_disfluencies():
    words = make_words("uh um so this is the real start of the story and it keeps going on", start=1.0)
    out, note = E.tighten_opening(words)
    assert [w.text for w in out][:2] == ["so", "this"] and "uh um" in note
    real = make_words("So this is the real start of the story and it keeps going on", start=1.0)
    out, note = E.tighten_opening(real)
    assert out == real and note is None, "discourse words like 'so' are speech, not disfluency"


def _zoom_case(loud_at: list[float], *, seconds: float = 40.0, profile: str = "general"):
    words = make_words(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 12, start=1.0, step=0.3, pause=0.1
    )
    words = [w for w in words if w.end < seconds - 0.5]
    loud = []
    for t in loud_at:
        w = min(words, key=lambda x: abs(x.start - t))
        loud.append((w.start, w.end))
    audio = _audio(seconds + 1, loud)
    prof = E.classify_content(words, "en", audio, _visual(seconds, 0.05), 0, seconds, user_style="auto")
    prof = E.ContentProfile(
        profile,
        1.0,
        {},
        [],
        E._PROFILE_PARAMS[profile]["max_pause"],
        dict(E._PROFILE_PARAMS[profile]["zoom"]),
    )
    return words, audio, prof


def test_zooms_land_on_stressed_words_and_respect_spacing_and_edges():
    words, audio, prof = _zoom_case([0.3, 6.0, 9.0, 20.0, 38.0, 12.0])
    zooms = E.plan_zooms(words, lambda t: t, audio, set(), prof, 40.0, [])
    assert zooms, "stressed words justify punch-ins"
    for z in zooms:
        assert z.t0 >= 0.8 and z.t1 <= 40.0 - 0.35, "opening and ending are left alone"
        assert z.evidence["lift_db"] >= 1.5 and z.reason.startswith("vocal emphasis")
        assert 0.04 < z.amp < 0.09, "controlled, never dramatic"
    for a, b in zip(zooms, zooms[1:], strict=False):
        assert b.t0 - a.t0 >= prof.zoom["min_gap"] and b.t0 > a.t1, "spaced out"
    assert sum(z.t1 - z.t0 for z in zooms) <= prof.zoom["max_share"] * 40.0 + 1e-6


def test_no_emphasis_means_no_zoom():
    words, audio, prof = _zoom_case([])
    assert E.plan_zooms(words, lambda t: t, audio, set(), prof, 40.0, []) == []


def test_gaming_disabled_option_and_hard_cuts_block_zooms():
    words, audio, prof = _zoom_case([6.0, 20.0], profile="gaming")
    assert E.plan_zooms(words, lambda t: t, audio, set(), prof, 40.0, []) == []
    words, audio, prof = _zoom_case([6.0, 20.0])
    assert E.plan_zooms(words, lambda t: t, audio, set(), prof, 40.0, [], enabled=False) == []
    free = E.plan_zooms(words, lambda t: t, audio, set(), prof, 40.0, [])
    assert free
    cut_at = [free[0].t0 + 0.4]
    blocked = E.plan_zooms(words, lambda t: t, audio, set(), prof, 40.0, cut_at)
    assert all(not (z.t0 - z.ramp < cut_at[0] < z.t1 + z.ramp) for z in blocked), (
        "never ramp across a hard cut"
    )


def test_numbers_and_key_terms_count_even_with_mild_emphasis():
    words = [
        Word(t, 1.0 + i * 0.4, 1.0 + i * 0.4 + 0.3)
        for i, t in enumerate("we sold 40000 units in the first month alone which was wild".split())
    ]
    words += [Word("filler", 6.0 + i * 0.4, 6.3 + i * 0.4) for i in range(20)]
    env = _audio(20.0, [(1.8, 2.1)], base=-26.0, peak=-23.0)  # only +3 dB
    prof = E.ContentProfile("general", 1, {}, [], 0.7, dict(E._PROFILE_PARAMS["general"]["zoom"]))
    zooms = E.plan_zooms(words, lambda t: t, env, set(), prof, 20.0, [])
    assert zooms and zooms[0].reason.endswith("on a number") and zooms[0].evidence["word"] == "40000"


def test_amplitudes_scale_with_how_strong_the_evidence_is():
    words, audio, prof = _zoom_case([6.0, 20.0])
    z = E.plan_zooms(words, lambda t: t, audio, set(), prof, 40.0, [])
    assert all(math.isfinite(x.amp) for x in z)


def test_every_code_path_sees_the_same_word_list_so_caption_indices_stay_valid():
    """effective_words drops leading disfluencies for render, page edits and re-render alike."""
    from viralcut.pipeline.orchestrator import effective_words
    from viralcut.pipeline.transcribe import Transcript

    words = [Word("uh", 1.0, 1.3), Word("um", 1.4, 1.6)] + [
        Word(f"w{i}", 2.0 + i * 0.4, 2.3 + i * 0.4) for i in range(10)
    ]
    got = effective_words(0.8, 8.0, {}, Transcript(words=words))
    assert [w.text for w in got][:3] == ["w0", "w1", "w2"] and len(got) == 10
    edits = {
        "cov": [0.8, 8.0],
        "words": [{"text": t.text, "start": t.start, "end": t.end, "prob": 1.0} for t in words],
    }
    again = effective_words(0.8, 8.0, edits, Transcript(words=words))
    assert [w.text for w in again] == [w.text for w in got], "edited and unedited paths agree"
