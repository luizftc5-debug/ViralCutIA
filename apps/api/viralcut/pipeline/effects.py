"""Editor effects that are pure computation or ASS/ffmpeg text: no models, no extra decoding.

* filler and retake removal (/corte, /tomada) — return source intervals to cut
* ASS overlays: animated titles (/lettering), the on-screen hook (/ganchos), progress bar and
  emphasis flashes (/animacao), the big word placed behind the subject (/texto-atras)
* the 2.5D camera filter (/3d)

Everything here is rendered by libass/ffmpeg in the same pass as the captions, so it is in the
exported MP4 exactly as previewed.
"""

from __future__ import annotations

import re
import unicodedata

from .captions import FONTS, _ass_color, _c, _rounded_rect, _ts, scale_for, text_width
from .text import build_sentences
from .transcribe import Word

# --------------------------------------------------------------------------- /corte, /tomada

FILLERS = {
    "en": {"um", "uh", "erm", "er", "hmm", "hm", "uhm", "umm", "ah", "mhm"},
    "pt": {"hã", "ãh", "ahn", "hum", "humm", "éé", "ééé", "eee", "ãã", "uhm", "hmm", "ah"},
    "es": {"eh", "ehh", "em", "este", "mmm", "hmm"},
}


def _canon(w: str) -> str:
    w = unicodedata.normalize("NFC", w.lower())
    return re.sub(r"[^\w']+", "", w)


def filler_cuts(words: list[Word], language: str) -> list[tuple[float, float]]:
    """Source intervals of pure disfluencies ("um", "hã") to remove, with a tiny margin."""
    lex = FILLERS.get((language or "en")[:2], FILLERS["en"]) | FILLERS["en"]
    return [(w.start - 0.03, w.end + 0.03) for w in words if _canon(w.text) in lex]


def _tokens(text: str) -> list[str]:
    return [t for t in (_canon(x) for x in text.split()) if t]


def retake_cuts(words: list[Word], language: str, similarity: float = 0.6) -> list[tuple[float, float, str]]:
    """Earlier takes of a sentence the speaker immediately said again (keep the last take).

    A sentence is a retake of the next one when their word sets overlap by ``similarity``
    (Jaccard) and they start the same way, or when it is a false start: a short fragment whose
    words are the beginning of the next sentence. Returns (start, end, text) to remove.
    """
    sents = build_sentences(words, language)
    out: list[tuple[float, float, str]] = []
    for a, b in zip(sents, sents[1:], strict=False):
        ta, tb = _tokens(a.text), _tokens(b.text)
        if len(ta) < 2 or len(tb) < 2 or b.start - a.end > 3.0:
            continue
        sa, sb = set(ta), set(tb)
        jac = len(sa & sb) / max(1, len(sa | sb))
        same_open = ta[:2] == tb[:2]
        false_start = len(ta) <= 0.7 * len(tb) and tb[: len(ta)] == ta
        if (jac >= similarity and same_open) or false_start:
            out.append((words[a.w0].start - 0.05, words[a.w1 - 1].end + 0.05, a.text))
    return out


def subtract(
    keeps: list[tuple[float, float]], cuts: list[tuple[float, float]], min_keep: float = 0.12
) -> list[tuple[float, float]]:
    """Remove ``cuts`` from the kept intervals (both on the source timeline)."""
    out = list(keeps)
    for c0, c1 in sorted(cuts):
        nxt: list[tuple[float, float]] = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 - a >= min_keep:
                nxt.append((a, c0))
            if b - c1 >= min_keep:
                nxt.append((c1, b))
        out = nxt
    return out


def drop_words(words: list[Word], cuts: list[tuple[float, float]]) -> list[Word]:
    return [w for w in words if not any(c0 <= (w.start + w.end) / 2 <= c1 for c0, c1 in cuts)]


# --------------------------------------------------------------------------- ASS overlays


def ass_header(width: int, height: int) -> str:
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.709",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: W,Inter ExtraBold,60,{_ass_color('#FFFFFF')},{_ass_color('#FFFFFF')},{_ass_color('#000000')},"
            f"{_ass_color('#000000', 0x70)},0,0,0,0,100,100,0,0,1,4,2,5,0,0,0,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )


def _ev(layer: int, t0: float, t1: float, text: str) -> str:
    return f"Dialogue: {layer},{_ts(t0)},{_ts(t1)},W,,0,0,0,,{text}"


def _esc(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\\", "/").replace("\n", " ").strip()


def _wrap(text: str, font: str, size: float, max_w: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and text_width(trial, font, size) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines[:3]


def title_events(
    text: str,
    width: int,
    height: int,
    *,
    t0: float = 0.0,
    duration: float = 2.8,
    animation: str = "pop",
    position: str = "top",
    accent: str = "#FFD60A",
    size_scale: float = 1.0,
    layer: int = 5,
) -> list[str]:
    """An animated title card (white type on an accent panel)."""
    text = _esc(text).upper()
    if not text:
        return []
    k = scale_for(width, height)
    name, file = FONTS["poppins"]["heavy"]
    from ..config import get_settings

    fp = str(get_settings().fonts_dir / file)
    size = 82 * k * size_scale
    lines = _wrap(text, fp, size, width * 0.78)
    while len(lines) == 3 and size > 40 * k:  # very long titles get smaller type, not a 4th line
        size *= 0.88
        lines = _wrap(text, fp, size, width * 0.78)
    line_h = size * 1.12
    block = line_h * len(lines)
    cy = height * (0.17 if position == "top" else 0.42) + block / 2
    t1 = t0 + duration
    out: list[str] = []
    widest = max(text_width(ln, fp, size) for ln in lines)
    pw, ph = widest + size * 0.8, block + size * 0.5
    px, py = (width - pw) / 2, cy - ph / 2
    fade = "\\fad(120,180)"
    if animation == "slide":
        panel_in = f"\\move({-pw:.0f},{py:.0f},{px:.0f},{py:.0f},0,260)"
    else:
        panel_in = f"\\pos({px:.0f},{py:.0f})"
    grow = (
        "\\fscx60\\fscy60\\t(0,160,\\fscx106\\fscy106)\\t(160,260,\\fscx100\\fscy100)"
        if animation == "pop"
        else ""
    )
    out.append(
        _ev(
            layer,
            t0,
            t1,
            f"{{\\an7{panel_in}\\p1\\bord0\\shad3\\4a&H80&\\1c{_c(accent)}{fade}{grow}}}{_rounded_rect(pw, ph, size * 0.25)}{{\\p0}}",
        )
    )
    base = f"\\fn{name}\\fs{size:.1f}\\bord{3 * k:.1f}\\shad0\\1c&HFFFFFF&\\3c&H000000&"
    for i, ln in enumerate(lines):
        y = cy - block / 2 + line_h * (i + 0.5)
        if animation == "typewriter":
            # characters appear one by one; hidden characters keep the layout stable
            n = len(ln)
            step = min(0.05, 0.6 / max(1, n))
            start = t0 + i * n * step
            for j in range(1, n + 1):
                a = start + (j - 1) * step
                b = t1 if j == n else start + j * step
                shown, hidden = _esc(ln[:j]), _esc(ln[j:])
                out.append(
                    _ev(
                        layer + 1,
                        a,
                        b,
                        f"{{\\an5\\pos({width / 2:.0f},{y:.0f}){base}}}{shown}{{\\alpha&HFF&}}{hidden}",
                    )
                )
        elif animation == "slide":
            out.append(
                _ev(
                    layer + 1,
                    t0,
                    t1,
                    f"{{\\an5\\move({-width / 2:.0f},{y:.0f},{width / 2:.0f},{y:.0f},0,260){base}{fade}}}{ln}",
                )
            )
        else:
            out.append(
                _ev(layer + 1, t0, t1, f"{{\\an5\\pos({width / 2:.0f},{y:.0f}){base}{fade}{grow}}}{ln}")
            )
    return out


def hook_events(
    text: str, width: int, height: int, *, duration: float = 2.6, accent: str = "#FFFFFF"
) -> list[str]:
    """The opening line shown big for the first seconds (the hook made explicit on screen)."""
    words = text.split()
    short = " ".join(words[:9]) + ("…" if len(words) > 9 else "")
    return title_events(
        short,
        width,
        height,
        duration=duration,
        animation="slide",
        position="middle",
        accent="#000000" if accent == "#FFFFFF" else accent,
        size_scale=0.78,
        layer=7,
    )


def progress_bar_events(duration: float, width: int, height: int, color: str, top_frac: float) -> list[str]:
    """A thin bar along the top that fills with the clip's progress (animated clip rectangle)."""
    k = scale_for(width, height)
    h = max(6, int(9 * k))
    y = int(height * top_frac * 0.5)
    ms = int(duration * 1000)
    shape = f"m 0 0 l {width} 0 l {width} {h} l 0 {h}"
    track = _ev(
        8, 0, duration, f"{{\\an7\\pos(0,{y})\\p1\\bord0\\shad0\\1c&HFFFFFF&\\1a&HB0&}}{shape}{{\\p0}}"
    )
    fill = _ev(
        9, 0, duration,
        f"{{\\an7\\pos(0,{y})\\p1\\bord0\\shad0\\1c{_c(color)}\\clip(0,{y},0,{y + h})\\t(0,{ms},\\clip(0,{y},{width},{y + h}))}}{shape}{{\\p0}}",
    )  # fmt: skip
    return [track, fill]


def emphasis_events(times: list[float], width: int, height: int, color: str) -> list[str]:
    """A short accent-coloured frame flash on emphasised moments (impact without covering faces)."""
    k = scale_for(width, height)
    t = int(14 * k)
    shape = f"m 0 0 l {width} 0 l {width} {height} l 0 {height} m {t} {t} l {t} {height - t} l {width - t} {height - t} l {width - t} {t}"
    out = []
    for at in times:
        out.append(
            _ev(
                9,
                at,
                at + 0.35,
                f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\blur{6 * k:.0f}\\1c{_c(color)}\\1a&H30&\\t(0,350,\\1a&HFF&)}}{shape}{{\\p0}}",
            )
        )
    return out


def behind_text_events(
    text: str, duration: float, width: int, height: int, color: str, size: float = 1.0, y_frac: float = 0.30
) -> list[str]:
    """One huge word for the layer *behind* the subject (/texto-atras)."""
    text = _esc(text).upper()
    if not text:
        return []
    name, file = FONTS["anton"]["heavy"]
    from ..config import get_settings

    fp = str(get_settings().fonts_dir / file)
    fs = 300.0 * scale_for(width, height) * size
    tw = text_width(text, fp, fs)
    if tw > width * 0.94:
        fs *= width * 0.94 / tw
    # vertical centre given by the caller (from the matte: just above the person's head, so the
    # word reads clearly and the head overlaps only its lower part)
    y = max(fs * 0.55, height * y_frac)
    return [
        _ev(
            1, 0, duration,
            f"{{\\an5\\pos({width / 2:.0f},{y:.0f})\\fn{name}\\fs{fs:.1f}\\bord0\\shad0\\1c{_c(color)}"
            f"\\fad(300,300)\\fscx92\\fscy92\\t(0,700,\\fscx100\\fscy100)}}{text}",
        )
    ]  # fmt: skip


def merge_ass(base: str | None, events: list[str], width: int, height: int) -> str:
    """Append overlay events to a caption ASS file (or to a fresh header)."""
    head = base.rstrip("\n") if base else ass_header(width, height)
    return head + "\n" + "\n".join(events) + "\n"


# --------------------------------------------------------------------------- /3d

DEPTH = {"subtle": 0.05, "medium": 0.08, "strong": 0.12}


def depth_filter(strength: str, duration: float, width: int, height: int, fps: float) -> str:
    """2.5D camera: an eased dolly push plus a slow perspective sway (no depth estimation).

    The frame is first pushed in by up to ``A``; the sway moves the corners by less than half of
    that margin, so no black edge is ever revealed.
    """
    a = DEPTH.get(strength, 0.05)
    d = max(1.0, duration)
    z = f"(1+{a:.3f}*(0.5-0.5*cos(PI*min(t/{d:.3f},1))))"
    pad_w = int(width * a * 0.5) // 2 * 2 + 2
    pad_h = int(height * a * 0.5) // 2 * 2 + 2
    w_ = f"sin(on/{fps:.3f}*0.55)"
    if strength != "strong":
        # measured: the eased push + drift costs ~0.25 CPU-s per second of 1080x1920 video;
        # the perspective sway below costs ~0.65 more, so only "strong" uses it
        drift = "0.5" if strength == "subtle" else "0.8"
        return (
            f"scale=w='trunc({width}*{z}/2)*2+{pad_w}':h='trunc({height}*{z}/2)*2+{pad_h}':eval=frame:flags=bicubic,"
            f"crop={width}:{height}:x='(iw-{width})/2*(1+{drift}*sin(t*0.35))':y='(ih-{height})/2*(1-{drift}*0.5*sin(t*0.25))',"
            f"setsar=1"
        )
    # perspective works on the pushed-in frame; its source quad stays inside the frame (margin
    # mx/my larger than the sway dx/dy), then the centre is cropped back to the output size
    mx, my, dx, dy = (
        f"W*{a * 0.2:.4f}",
        f"H*{a * 0.2:.4f}",
        f"W*{a * 0.12:.4f}*{w_}",
        f"H*{a * 0.06:.4f}*{w_}",
    )
    return (
        f"scale=w='trunc({width}*{z}/2)*2+{pad_w}':h='trunc({height}*{z}/2)*2+{pad_h}':eval=frame:flags=bicubic,"
        f"perspective=x0='{mx}+{dx}':y0='{my}+{dy}':x1='W-{mx}+{dx}':y1='{my}-{dy}':"
        f"x2='{mx}-{dx}':y2='H-{my}+{dy}':x3='W-{mx}-{dx}':y3='H-{my}-{dy}':eval=frame:interpolation=linear,"
        f"crop={width}:{height},setsar=1"
    )
