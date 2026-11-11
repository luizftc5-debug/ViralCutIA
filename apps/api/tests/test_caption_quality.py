"""Caption timing, completeness and on-screen geometry, verified on real libass renders."""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from helpers import make_words

from viralcut.config import get_settings
from viralcut.options import SAFE_AREAS, CaptionOptions
from viralcut.pipeline import captions as cap
from viralcut.pipeline.transcribe import Word

STYLES = list(cap.STYLES)
# the longest realistic words in the languages we support: a layout that survives these survives anything
LONG = "Inconstitucionalissimamente Donaudampfschifffahrtsgesellschaft pneumoultramicroscopicossilicovulcanoconiótico"


def _words(seed: int, n: int = 40) -> list[Word]:
    rng = np.random.default_rng(seed)
    words, t = [], 0.3
    vocab = "você sabia que o coração bate cem mil vezes por dia isso é incrível, não é? então cuide bem dele.".split()
    for _ in range(n):
        tok = vocab[int(rng.integers(len(vocab)))]
        d = float(rng.uniform(0.12, 0.5))
        words.append(Word(tok, round(t, 3), round(t + d, 3)))
        t += d + float(rng.choice([0.0, 0.02, 0.05, 0.2, 0.5, 1.4], p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]))
    return words


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("seed", range(6))
def test_pages_never_outlive_speech_and_never_overlap(style, seed):
    words = _words(seed)
    opts = CaptionOptions(style=style)
    pages = cap.paginate(words, opts, words[-1].end + 3.0)
    for i, p in enumerate(pages):
        last_end = p.words[-1].end
        nxt = pages[i + 1].start if i + 1 < len(pages) else None
        chained = nxt is not None and nxt - last_end < cap.JOIN_GAP
        limit = nxt if chained else last_end + cap.HOLD_AFTER
        assert p.end <= max(limit, p.start + cap.MIN_PAGE) + 1e-6, "caption lingers after the speech ended"
        assert p.end >= p.start + 0.05
        if nxt is not None:
            assert p.end <= nxt + 1e-6, "pages must not overlap"


@pytest.mark.parametrize("style", STYLES)
def test_every_word_is_shown_exactly_once_in_order(style):
    words = _words(11, 60)
    pages = cap.paginate(words, CaptionOptions(style=style), words[-1].end + 3.0)
    shown = [w.k for p in pages for w in p.words]
    assert shown == sorted(set(shown)) == list(range(len(words))), "no word missing, none duplicated"
    for p in pages:
        for w in p.words:
            assert w.start >= p.start - 1e-6 and w.start <= p.end


@pytest.mark.parametrize("style", STYLES)
def test_ass_events_stay_inside_their_page(style):
    words = _words(3, 30)
    opts = CaptionOptions(style=style)
    pages = cap.paginate(words, opts, words[-1].end + 3.0)
    ass = cap.build_ass(pages, opts, "universal", 1080, 1920)

    def secs(ts: str) -> float:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    spans = [
        (secs(m.group(1)), secs(m.group(2))) for m in re.finditer(r"Dialogue: \d+,([\d:.]+),([\d:.]+),", ass)
    ]
    assert spans, "no caption events were produced"
    first, last = min(a for a, _ in spans), max(b for _, b in spans)
    assert first >= pages[0].start - 0.011
    assert last <= pages[-1].end + 0.011, "no event may extend past the end of the last page"
    assert all(b > a for a, b in spans)
    # nothing is visible while nobody is speaking, beyond the hold: every gap between pages is empty
    for a, b in zip(pages, pages[1:], strict=False):
        mid = (a.end + b.start) / 2
        if b.start - a.end > 0.05:
            assert not any(s0 < mid < s1 for s0, s1 in spans), "caption visible in the silence between pages"


def _frame(ass_text: str, t: float, w: int = 1080, h: int = 1920) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "c.ass").write_text(ass_text, encoding="utf-8")
        os.symlink(get_settings().fonts_dir.resolve(), Path(tmp) / "fonts", target_is_directory=True)
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=30:d={t + 1:.2f}",
             "-vf", "ass=c.ass:fontsdir=fonts,format=gray", "-ss", f"{t:.3f}", "-frames:v", "1", "-f", "rawvideo", "-"],
            cwd=tmp, capture_output=True, check=True,
        ).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(h, w)


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("position", ["top", "middle", "bottom"])
def test_captions_never_leave_the_platform_safe_area(style, position):
    """Rasterise the worst case (very long words, active-word pop at full scale) and measure it."""
    opts = CaptionOptions(style=style, position=position)
    words = [Word(w, 0.2 + 0.5 * i, 0.2 + 0.5 * i + 0.4) for i, w in enumerate(LONG.split())]
    pages = cap.paginate(words, opts, 4.0)
    ass = cap.build_ass(pages, opts, "tiktok", 1080, 1920)
    safe = SAFE_AREAS["tiktok"]
    for p in pages:
        t = p.words[-1].start + 0.2  # the last word is active (largest state)
        img = _frame(ass, t)
        ys, xs = np.where(img > 60)
        assert len(xs), f"nothing rendered for {p.raw_text!r}"
        assert xs.min() >= 1080 * safe["side"] * 0.8 and xs.max() <= 1080 * (1 - safe["side"] * 0.8), (
            f"{style}/{position}: text reaches x={xs.min()}..{xs.max()} outside the frame margins"
        )
        assert ys.min() >= 1920 * safe["top"] * 0.85, f"{style}/{position}: text under the platform header"
        assert ys.max() <= 1920 * (1 - safe["bottom"]) + 1920 * 0.03, (
            f"{style}/{position}: text under the buttons"
        )


def test_blur_band_puts_captions_clear_of_the_picture():
    out_w, out_h = 1080, 1920
    shown_h = out_w * 1080 / 1920  # 16:9 gameplay fitted to the width
    top_edge, bottom_edge = (out_h - shown_h) / 2, (out_h + shown_h) / 2
    cy = cap.blur_band_anchor("bottom", "universal", 1920, 1080, out_w, out_h, "gaming", "m")
    assert cy is not None and bottom_edge < cy < out_h * (1 - SAFE_AREAS["universal"]["bottom"])
    top = cap.blur_band_anchor("top", "universal", 1920, 1080, out_w, out_h, "gaming", "m")
    assert top is not None and out_h * SAFE_AREAS["universal"]["top"] < top < top_edge
    # a 4:3 source leaves too little band for large captions: fall back to the safe-area default
    assert cap.blur_band_anchor("bottom", "universal", 1440, 1080, out_w, out_h, "gaming", "l") is None
    assert cap.blur_band_anchor("middle", "universal", 1920, 1080, out_w, out_h, "gaming", "m") is None

    # rasterised: the text lies entirely below the picture
    opts = CaptionOptions(style="gaming", position="bottom")
    words = [Word("GOOD", 0.2, 0.6), Word("GAME", 0.7, 1.1)]
    pages = cap.paginate(words, opts, 3.0)
    ass = cap.build_ass(pages, opts, "universal", out_w, out_h, cy)
    img = _frame(ass, 0.9)
    ys, _ = np.where(img > 60)
    assert ys.min() > bottom_edge, "captions overlap the gameplay"


def test_fast_speech_keeps_every_word_on_screen_for_a_readable_instant():
    words = [Word(f"w{i}", i * 0.13, i * 0.13 + 0.11) for i in range(30)]  # ~7.7 words per second
    opts = CaptionOptions(style="viral_dynamic")
    pages = cap.paginate(words, opts, 6.0)
    assert sum(len(p.words) for p in pages) == 30
    assert all(len(p.words) <= cap.STYLES["viral_dynamic"].max_words + 1 for p in pages)
    assert all(p.end - p.start >= cap.MIN_PAGE - 1e-6 or p is pages[-1] for p in pages)


def test_viral_dynamic_keywords_get_their_own_colour_and_pages_slide_in():
    words = make_words("The best moment was already in your video", start=0.2)
    opts = CaptionOptions(style="viral_dynamic")
    pages = cap.paginate(words, opts, 5.0, {"moment"})
    ass = cap.build_ass(pages, opts, "universal", 1080, 1920)
    spec = cap.STYLES["viral_dynamic"]
    assert "\\move(" in ass, "kinetic pages slide into place"
    assert cap._c(spec.accent2) in ass and cap._c(spec.accent) in ass
    plain = cap.build_ass(
        pages, CaptionOptions(style="viral_dynamic", animation=False), "universal", 1080, 1920
    )
    assert "\\move(" not in plain and "\\t(" not in plain, "animation off means no movement at all"
