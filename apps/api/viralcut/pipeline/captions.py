"""Stage 9 — caption engine: pagination, layout, ASS (burn-in) and SRT (sidecar) generation.

Every word is laid out explicitly: its width is measured with the real font file, so lines are
balanced, never exceed the platform safe area and each word can be styled/animated on its own
without shifting its neighbours.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from ..config import get_settings
from ..options import SAFE_AREAS, CaptionOptions
from .text import norm
from .transcribe import Word

_TERMINAL = (".", "?", "!", "…")


@dataclass(frozen=True)
class StyleSpec:
    label: str
    description: str
    font: str  # default font family key
    weight: str  # heavy | bold | medium
    size: int  # font size at 1920 px output height
    upper: bool
    max_words: int
    max_lines: int
    max_chars: int  # soft limit per line
    outline: float
    shadow: float
    blur: float
    accent: str  # default accent colour (#RRGGBB)
    active: str  # color | reveal | box | pop
    line_spacing: float = 1.16
    strip_punct: bool = True


STYLES: dict[str, StyleSpec] = {
    "high_retention": StyleSpec(
        label="High Retention",
        description="Bold word-by-word captions with a controlled colour highlight on the spoken word.",
        font="poppins", weight="heavy", size=118, upper=True, max_words=3, max_lines=2, max_chars=13,
        outline=8, shadow=3, blur=0.6, accent="#FFD60A", active="color", line_spacing=0.96,
    ),
    "clean_editorial": StyleSpec(
        label="Clean Editorial",
        description="Quiet, elegant lines that reveal as they are spoken.",
        font="inter", weight="medium", size=72, upper=False, max_words=7, max_lines=2, max_chars=24,
        outline=1.6, shadow=1.5, blur=2.2, accent="#F2C879", active="reveal", line_spacing=1.06, strip_punct=False,
    ),
    "dynamic_creator": StyleSpec(
        label="Dynamic Creator",
        description="A highlight block follows the voice from word to word.",
        font="poppins", weight="bold", size=100, upper=False, max_words=4, max_lines=2, max_chars=14,
        outline=5, shadow=2, blur=0.5, accent="#2F6BFF", active="box", line_spacing=1.14,
    ),
    "gaming": StyleSpec(
        label="Gaming / Entertainment",
        description="Oversized, high-contrast words that punch in with the commentary.",
        font="anton", weight="heavy", size=168, upper=True, max_words=2, max_lines=1, max_chars=14,
        outline=10, shadow=5, blur=0.4, accent="#3DFF8F", active="pop",
    ),
}  # fmt: skip

# family key -> weight class -> (ASS font name, file)
FONTS: dict[str, dict[str, tuple[str, str]]] = {
    "inter": {
        "heavy": ("Inter ExtraBold", "Inter-ExtraBold.otf"),
        "bold": ("Inter Bold", "Inter-Bold.otf"),
        "medium": ("Inter SemiBold", "Inter-SemiBold.otf"),
    },
    "poppins": {
        "heavy": ("Poppins ExtraBold", "Poppins-ExtraBold.ttf"),
        "bold": ("Poppins Bold", "Poppins-Bold.ttf"),
        "medium": ("Poppins SemiBold", "Poppins-SemiBold.ttf"),
    },
    "anton": {k: ("Anton", "Anton-Regular.ttf") for k in ("heavy", "bold", "medium")},
    "dm_serif": {k: ("DM Serif Display", "DMSerifDisplay-Regular.ttf") for k in ("heavy", "bold", "medium")},
}

SIZE_SCALE = {"s": 0.82, "m": 1.0, "l": 1.18}


@lru_cache(maxsize=32)
def _em_ratio(font_path: str) -> float:
    """em size in pixels per unit of ASS font size, as libass computes it.

    libass (like VSFilter) scales fonts so that usWinAscent + usWinDescent equals the ASS
    font size, so one ASS point is ``unitsPerEm / (winAscent + winDescent)`` em-pixels.
    """
    data = Path(font_path).read_bytes()
    num_tables = struct.unpack(">H", data[4:6])[0]
    tables: dict[bytes, int] = {}
    for i in range(num_tables):
        rec = data[12 + 16 * i : 28 + 16 * i]
        tables[rec[:4]] = struct.unpack(">I", rec[8:12])[0]
    upm = struct.unpack(">H", data[tables[b"head"] + 18 : tables[b"head"] + 20])[0]
    os2 = tables.get(b"OS/2")
    if os2 is not None:
        asc, desc = struct.unpack(">HH", data[os2 + 74 : os2 + 78])
        if asc + desc > 0:
            return upm / (asc + desc)
    hhea = tables[b"hhea"]
    asc, desc = struct.unpack(">hh", data[hhea + 4 : hhea + 8])
    return upm / max(1, asc - desc)


@lru_cache(maxsize=256)
def _pil_font(font_path: str, px10: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, px10 / 10)


def word_gap(font_path: str, ass_size: float, spec: StyleSpec, k: float = 1.0) -> float:
    """Horizontal gap between words: the font's space plus room for outlines and the active-word pop."""
    gap = text_width(" ", font_path, ass_size) * 1.05 + spec.outline * k * 1.4
    if spec.active in ("color", "pop"):
        gap += ass_size * 0.045
    if spec.active == "box":
        gap += ass_size * 0.10
    return gap


def text_width(text: str, font_path: str, ass_size: float) -> float:
    px = ass_size * _em_ratio(font_path)
    return float(_pil_font(font_path, max(10, int(round(px * 10)))).getlength(text))


@dataclass
class CaptionWord:
    text: str  # display text (already cased/cleaned)
    raw: str  # as transcribed/edited
    start: float
    end: float
    emph: bool = False
    k: int = 0  # index in the clip's word list


@dataclass
class Page:
    start: float
    end: float
    words: list[CaptionWord]
    lines: list[list[CaptionWord]] = field(default_factory=list)
    scale: float = 1.0

    @property
    def k0(self) -> int:
        return self.words[0].k

    @property
    def k1(self) -> int:
        return self.words[-1].k + 1

    @property
    def raw_text(self) -> str:
        return " ".join(w.raw for w in self.words)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.raw_text,
            "k0": self.k0,
            "k1": self.k1,
        }


def display_text(raw: str, spec: StyleSpec) -> str:
    text = raw.replace("{", "(").replace("}", ")").replace("\\", "/").replace("\n", " ").strip()
    if spec.strip_punct:
        text = text.rstrip(".,;:").lstrip("¿¡")
        if text.endswith("...") or text.endswith("…"):
            text = text.rstrip(".…")
    if spec.upper:
        text = text.upper()
    return text


def resolve_font(opts: CaptionOptions, spec: StyleSpec) -> tuple[str, Path]:
    family = spec.font if opts.font == "auto" else opts.font
    name, file = FONTS[family][spec.weight]
    return name, get_settings().fonts_dir / file


def paginate(
    words: list[Word],
    opts: CaptionOptions,
    clip_duration: float,
    keywords: set[str] | None = None,
) -> list[Page]:
    """Group words into short caption pages, breaking on pauses, punctuation and length."""
    spec = STYLES[opts.style]
    keywords = keywords or set()
    items: list[CaptionWord] = []
    for k, w in enumerate(words):
        text = display_text(w.text, spec)
        if not text:
            continue
        key = norm(w.text).strip(".,;:!?\"'()…")
        emph = opts.highlight_keywords and (
            key in keywords or (any(ch.isdigit() for ch in key) and len(key) >= 2)
        )
        items.append(CaptionWord(text=text, raw=w.text, start=w.start, end=w.end, emph=bool(emph), k=k))

    budget = spec.max_chars * spec.max_lines
    groups: list[list[CaptionWord]] = []
    cur: list[CaptionWord] = []
    for it in items:
        if cur:
            prev = cur[-1]
            chars = sum(len(x.text) for x in cur) + len(cur) - 1
            brk = (
                len(cur) >= spec.max_words
                or it.start - prev.end > 0.6
                or prev.raw.rstrip("\"')").endswith(_TERMINAL)
                or chars + 1 + len(it.text) > budget
                or (prev.raw.endswith((",", ";", ":")) and len(cur) >= max(2, spec.max_words - 1))
            )
            if brk:
                groups.append(cur)
                cur = []
        cur.append(it)
    if cur:
        groups.append(cur)

    # avoid orphans: a lone short word joins the previous page when it directly follows it
    merged: list[list[CaptionWord]] = []
    for g in groups:
        if (
            merged
            and len(g) == 1
            and len(g[0].text) <= 3
            and g[0].start - merged[-1][-1].end < 0.3
            and len(merged[-1]) <= spec.max_words
            and not merged[-1][-1].raw.endswith(_TERMINAL)
        ):
            merged[-1].append(g[0])
        else:
            merged.append(g)

    pages: list[Page] = []
    for i, g in enumerate(merged):
        # restraint: at most one emphasised keyword per page
        seen = False
        for w in g:
            if w.emph and seen:
                w.emph = False
            seen = seen or w.emph
        start = g[0].start
        nxt = merged[i + 1][0].start if i + 1 < len(merged) else None
        end = g[-1].end + 0.6
        if nxt is not None:
            end = nxt if nxt - g[-1].end < 0.9 else min(end, nxt)
        end = min(end, clip_duration)
        if end - start < 0.2:
            end = min(clip_duration, start + 0.2)
        pages.append(Page(start=start, end=max(end, start + 0.05), words=g))
    return pages


def _split_lines(
    words: list[CaptionWord], widths: list[float], space: float, max_width: float, spec: StyleSpec
) -> list[list[int]]:
    idx = list(range(len(words)))
    total = sum(widths) + space * (len(words) - 1)
    chars = sum(len(w.text) for w in words) + len(words) - 1
    if spec.max_lines == 1 or len(words) == 1 or (total <= max_width and chars <= spec.max_chars):
        return [idx]
    best, best_cost = 1, float("inf")
    for cut in range(1, len(words)):
        w1 = sum(widths[:cut]) + space * (cut - 1)
        w2 = sum(widths[cut:]) + space * (len(words) - cut - 1)
        cost = max(w1, w2)
        if words[cut - 1].raw.endswith((",", ";", ":")):
            cost *= 0.94  # prefer breaking after punctuation
        if cost < best_cost:
            best, best_cost = cut, cost
    return [idx[:best], idx[best:]]


@dataclass
class Layout:
    width: int
    height: int
    font_name: str
    font_path: Path
    size: float
    spec: StyleSpec
    accent: str
    center_y: float
    max_width: float


def make_layout(opts: CaptionOptions, platform: str, width: int, height: int) -> Layout:
    spec = STYLES[opts.style]
    name, path = resolve_font(opts, spec)
    safe = SAFE_AREAS.get(platform, SAFE_AREAS["universal"])
    k = height / 1920
    size = spec.size * SIZE_SCALE[opts.size] * k
    max_width = width * (1 - 2 * safe["side"])
    block = size * spec.line_spacing * spec.max_lines
    if opts.position == "top":
        cy = height * (safe["top"] + 0.03) + block / 2
    elif opts.position == "middle":
        cy = height * 0.52
    else:
        cy = height * (1 - safe["bottom"] - 0.03) - block / 2
    return Layout(width, height, name, path, size, spec, opts.color or spec.accent, cy, max_width)


def layout_pages(pages: list[Page], lay: Layout) -> None:
    """Break each page into lines and shrink it if it would exceed the safe width."""
    fp = str(lay.font_path)
    for p in pages:
        widths = [text_width(w.text, fp, lay.size) for w in p.words]
        space = word_gap(fp, lay.size, lay.spec, lay.height / 1920)
        groups = _split_lines(p.words, widths, space, lay.max_width, lay.spec)
        widest = max(sum(widths[i] for i in g) + space * (len(g) - 1) for g in groups)
        margin = lay.spec.outline * 2 + lay.size * 0.12  # outline, active-word scale-up
        p.scale = min(1.0, (lay.max_width - margin) / widest) if widest > 0 else 1.0
        p.lines = [[p.words[i] for i in g] for g in groups]


def _ass_color(hex_rgb: str, alpha: int = 0) -> str:
    r, g, b = int(hex_rgb[1:3], 16), int(hex_rgb[3:5], 16), int(hex_rgb[5:7], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _c(hex_rgb: str) -> str:
    return _ass_color(hex_rgb)[:2] + _ass_color(hex_rgb)[4:] + "&"


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _rounded_rect(w: float, h: float, r: float) -> str:
    r = min(r, w / 2, h / 2)
    c = r * 0.45  # bezier handle offset approximating a quarter circle
    pts = (
        f"m {r:.0f} 0 l {w - r:.0f} 0 b {w - c:.0f} 0 {w:.0f} {c:.0f} {w:.0f} {r:.0f} "
        f"l {w:.0f} {h - r:.0f} b {w:.0f} {h - c:.0f} {w - c:.0f} {h:.0f} {w - r:.0f} {h:.0f} "
        f"l {r:.0f} {h:.0f} b {c:.0f} {h:.0f} 0 {h - c:.0f} 0 {h - r:.0f} "
        f"l 0 {r:.0f} b 0 {c:.0f} {c:.0f} 0 {r:.0f} 0"
    )
    return pts


def build_ass(pages: list[Page], opts: CaptionOptions, platform: str, width: int, height: int) -> str:
    lay = make_layout(opts, platform, width, height)
    layout_pages(pages, lay)
    spec = lay.spec
    k = height / 1920
    white, black, accent = "#FFFFFF", "#000000", lay.accent
    head = "\n".join(
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
            f"Style: W,{lay.font_name},{lay.size:.1f},{_ass_color(white)},{_ass_color(white)},{_ass_color(black)},"
            f"{_ass_color(black, 0x70)},0,0,0,0,100,100,0,0,1,{spec.outline * k:.1f},{spec.shadow * k:.1f},5,0,0,0,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )
    events: list[str] = []

    def ev(layer: int, t0: float, t1: float, text: str) -> None:
        if t1 - t0 >= 0.01:
            events.append(f"Dialogue: {layer},{_ts(t0)},{_ts(t1)},W,,0,0,0,,{text}")

    fp = str(lay.font_path)
    anim = opts.animation
    for p in pages:
        size = lay.size * p.scale
        line_h = size * spec.line_spacing
        block_h = line_h * len(p.lines)
        top = lay.center_y - block_h / 2
        space = word_gap(fp, size, spec, k)
        flat = [w for line in p.lines for w in line]
        for li, line in enumerate(p.lines):
            widths = [text_width(w.text, fp, size) for w in line]
            total = sum(widths) + space * (len(line) - 1)
            x = (lay.width - total) / 2
            cy = top + line_h * (li + 0.5)
            for w, ww in zip(line, widths, strict=True):
                cx = x + ww / 2
                x += ww + space
                wi = flat.index(w)
                t_on = max(p.start, w.start) if wi > 0 else p.start
                t_off = flat[wi + 1].start if wi + 1 < len(flat) else p.end
                t_off = min(max(t_off, t_on + 0.04), p.end)
                base = f"\\an5\\pos({cx:.0f},{cy:.0f})\\fs{size:.1f}\\blur{spec.blur}"
                fade_in = "\\fad(70,0)" if anim else ""
                # the box style already carries the accent; coloured text next to it would fight for attention
                after_col = accent if (w.emph and spec.active != "box") else white

                if spec.active == "reveal":
                    dim, full = "\\1a&H99&\\3a&HB0&\\4a&HD0&", "\\1a&H00&\\3a&H50&\\4a&H90&"
                    col = f"\\1c{_c(after_col)}"
                    soft_in = "\\fad(140,0)" if anim else ""
                    ev(1, p.start, t_on, f"{{{base}{dim}{soft_in}}}{w.text}")
                    if anim:
                        dur = int(min(160, max(60, (t_off - t_on) * 1000)))
                        ev(1, t_on, t_off, f"{{{base}{dim}\\t(0,{dur},{full}{col})}}{w.text}")
                    else:
                        ev(1, t_on, t_off, f"{{{base}{full}{col}}}{w.text}")
                    tail = "\\fad(0,140)" if anim and p.end - t_off > 0.2 else ""
                    ev(1, t_off, p.end, f"{{{base}{full}{col}{tail}}}{w.text}")
                    continue

                # states: upcoming -> active -> spoken
                ev(1, p.start, t_on, f"{{{base}{fade_in}}}{w.text}")
                if spec.active == "color":
                    pop = (
                        "\\fscx100\\fscy100\\t(0,70,\\fscx110\\fscy110)\\t(70,170,\\fscx104\\fscy104)"
                        if anim
                        else "\\fscx104\\fscy104"
                    )
                    ev(2, t_on, t_off, f"{{{base}\\1c{_c(accent)}{pop}}}{w.text}")
                elif spec.active == "pop":
                    tilt = 2.5 if wi % 2 == 0 else -2.5
                    pop = (
                        f"\\fscx80\\fscy80\\frz0\\t(0,80,\\fscx120\\fscy120\\frz{tilt})\\t(80,180,\\fscx112\\fscy112)"
                        if anim
                        else "\\fscx112\\fscy112"
                    )
                    ev(2, t_on, t_off, f"{{{base}\\1c{_c(accent)}{pop}}}{w.text}")
                elif spec.active == "box":
                    pad_x, bh = size * 0.20, size * 1.06
                    bw = ww + 2 * pad_x
                    grow = "\\fscx86\\fscy86\\t(0,90,\\fscx100\\fscy100)" if anim else ""
                    ev(
                        0, t_on, t_off,
                        f"{{\\an5\\pos({cx:.0f},{cy + size * 0.02:.0f})\\p1\\bord0\\shad0\\blur0.4\\1c{_c(accent)}{grow}}}"
                        f"{_rounded_rect(bw, bh, size * 0.2)}{{\\p0}}",
                    )  # fmt: skip
                    ev(2, t_on, t_off, f"{{{base}\\bord0\\shad0}}{w.text}")
                ev(1, t_off, p.end, f"{{{base}\\1c{_c(after_col)}}}{w.text}")
    return head + "\n" + "\n".join(events) + "\n"


def _srt_ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def build_srt(pages: list[Page]) -> str:
    out: list[str] = []
    for i, p in enumerate(pages, 1):
        lines = [" ".join(w.raw for w in line) for line in p.lines] if p.lines else [p.raw_text]
        out.append(f"{i}\n{_srt_ts(p.start)} --> {_srt_ts(p.end)}\n" + "\n".join(lines) + "\n")
    return "\n".join(out)


def apply_page_edit(words: list[Word], k0: int, k1: int, new_text: str) -> list[Word]:
    """Replace words[k0:k1] with the tokens of ``new_text``.

    Timings are kept when the word count is unchanged; otherwise the page's time span is shared
    between the new words in proportion to their length. An empty text removes the page.
    """
    k0 = max(0, min(k0, len(words)))
    k1 = max(k0, min(k1, len(words)))
    old = words[k0:k1]
    tokens = new_text.split()
    if not old:
        return words
    if len(tokens) == len(old):
        new = [Word(t, o.start, o.end, o.prob) for t, o in zip(tokens, old, strict=True)]
    elif not tokens:
        new = []
    else:
        t0, t1 = old[0].start, old[-1].end
        weights = [len(t) + 1.5 for t in tokens]
        total = sum(weights)
        new, cursor = [], t0
        for tok, wgt in zip(tokens, weights, strict=True):
            d = (t1 - t0) * wgt / total
            new.append(Word(tok, round(cursor, 3), round(cursor + d * 0.92, 3), 1.0))
            cursor += d
    return words[:k0] + new + words[k1:]
