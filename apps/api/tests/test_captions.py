import re

from helpers import make_words

from viralcut.options import CaptionOptions
from viralcut.pipeline import captions as cap
from viralcut.pipeline.transcribe import Word

TEXT = "Você sabia que o coração bate 100 mil vezes por dia? Isso é incrível, não é? Então cuide bem dele."


def _pages(style: str, **kw):
    opts = CaptionOptions(style=style, **kw)
    words = make_words(TEXT, start=0.2)
    pages = cap.paginate(words, opts, words[-1].end + 1, {"coracao"})
    return opts, words, pages


def test_pagination_respects_style_limits_and_time_order():
    for style, spec in cap.STYLES.items():
        _, words, pages = _pages(style)
        assert sum(len(p.words) for p in pages) == len(words)
        prev_end = 0.0
        for p in pages:
            assert len(p.words) <= spec.max_words + 1  # +1 only for a merged short orphan
            assert p.start >= prev_end - 1e-6 and p.end > p.start
            prev_end = p.end


def test_pages_break_at_sentence_ends():
    _, _, pages = _pages("clean_editorial")
    for p in pages[:-1]:
        inner = [w.raw for w in p.words[:-1]]
        assert not any(w.endswith(("?", ".")) for w in inner), "a sentence end always closes the page"


def test_ass_contains_positioned_events_inside_safe_area():
    for style in cap.STYLES:
        opts, _, pages = _pages(style)
        ass = cap.build_ass(pages, opts, "universal", 1080, 1920)
        assert "PlayResX: 1080" in ass and "[Events]" in ass
        events = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
        assert len(events) >= len(pages)
        xs = [float(m.group(1)) for m in re.finditer(r"\\pos\(([0-9.]+),", ass)]
        ys = [float(m.group(1)) for m in re.finditer(r"\\pos\([0-9.]+,([0-9.]+)\)", ass)]
        assert min(xs) > 1080 * 0.09 and max(xs) < 1080 * 0.91
        assert max(ys) < 1920 * (1 - 0.22), "captions stay above the platform UI"
        for p in pages:
            assert 0.3 < p.scale <= 1.0


def test_accents_and_uppercase():
    opts, _, pages = _pages("high_retention")
    ass = cap.build_ass(pages, opts, "universal", 1080, 1920)
    assert "CORAÇÃO" in ass and "VOCÊ" in ass
    opts, _, pages = _pages("clean_editorial")
    ass = cap.build_ass(pages, opts, "universal", 1080, 1920)
    assert "coração" in ass and "Você" in ass


def test_position_size_and_colour_options():
    def ys(**kw):
        opts, _, pages = _pages("high_retention", **kw)
        ass = cap.build_ass(pages, opts, "tiktok", 1080, 1920)
        return [float(m.group(1)) for m in re.finditer(r"\\pos\([0-9.]+,([0-9.]+)\)", ass)], ass

    top, _ = ys(position="top")
    mid, _ = ys(position="middle")
    bottom, ass = ys(position="bottom", color="#00FF88")
    assert max(top) < min(mid) < min(bottom)
    assert min(top) > 1920 * 0.10
    assert "&H88FF00&" in ass  # custom accent, BGR order
    small = cap.make_layout(CaptionOptions(size="s"), "universal", 1080, 1920).size
    large = cap.make_layout(CaptionOptions(size="l"), "universal", 1080, 1920).size
    assert small < large


def test_keyword_highlight_toggle():
    opts, words, _ = _pages("high_retention")
    on = cap.paginate(words, opts, 60, {"coracao"})
    off = cap.paginate(
        words, CaptionOptions(style="high_retention", highlight_keywords=False), 60, {"coracao"}
    )
    assert any(w.emph for p in on for w in p.words)
    assert not any(w.emph for p in off for w in p.words)
    assert all(sum(w.emph for w in p.words) <= 1 for p in on)


def test_long_word_is_scaled_down_not_clipped():
    words = [Word("Pneumoultramicroscopicossilicovulcanoconiótico", 0.0, 1.0)]
    opts = CaptionOptions(style="gaming", size="l")
    pages = cap.paginate(words, opts, 2.0)
    cap.build_ass(pages, opts, "universal", 1080, 1920)
    lay = cap.make_layout(opts, "universal", 1080, 1920)
    width = cap.text_width(pages[0].words[0].text, str(lay.font_path), lay.size * pages[0].scale)
    assert pages[0].scale < 1.0 and width <= lay.max_width


def test_srt_format():
    opts, _, pages = _pages("dynamic_creator")
    cap.build_ass(pages, opts, "universal", 1080, 1920)
    srt = cap.build_srt(pages)
    blocks = srt.strip().split("\n\n")
    assert len(blocks) == len(pages)
    assert re.match(r"1\n00:00:00,\d{3} --> 00:00:0\d,\d{3}\n", blocks[0])
    assert "coração" in srt


def test_apply_page_edit_keeps_or_redistributes_timings():
    words = make_words("one two three four five", start=0.0)
    same = cap.apply_page_edit(words, 1, 3, "TWO THREE")
    assert [w.text for w in same] == ["one", "TWO", "THREE", "four", "five"]
    assert (same[1].start, same[2].end) == (words[1].start, words[2].end)
    more = cap.apply_page_edit(words, 1, 3, "a b c d")
    assert len(more) == 7
    assert more[1].start == words[1].start and more[4].end <= words[2].end + 1e-6
    assert all(more[i].start <= more[i + 1].start for i in range(len(more) - 1))
    removed = cap.apply_page_edit(words, 1, 3, "")
    assert [w.text for w in removed] == ["one", "four", "five"]
