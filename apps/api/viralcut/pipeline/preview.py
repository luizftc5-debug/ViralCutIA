"""Caption Studio live preview: one frame rendered by the *same* engine as the export.

The preview is not a CSS imitation. It runs the real pagination, layout and ASS generation of
``captions.py`` and burns the result with ffmpeg/libass onto a 1080x1920 frame — the exact code
path of a final render — then scales the frame down for the browser. What you see is what the
MP4 will contain.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from ..options import CaptionOptions
from . import captions as cap
from .ffmpeg import input_args, run_capture
from .render import _link_fonts
from .transcribe import Word

SAMPLE = {
    "en": "Most people quit after two weeks because they practise the wrong way",
    "pt": "A maioria das pessoas desiste em duas semanas porque treina do jeito errado",
}
_gate = threading.BoundedSemaphore(2)  # previews never pile up on the machine that renders videos


class PreviewBusy(RuntimeError):
    pass


def sample_words(text: str, word_s: float = 0.32) -> list[Word]:
    out, t = [], 0.2
    for tok in text.split()[:40]:
        d = max(0.18, min(0.6, word_s * (0.6 + len(tok) / 8)))
        out.append(Word(tok, round(t, 3), round(t + d, 3), 1.0))
        t += d + 0.04
    return out


def render_preview(
    opts: CaptionOptions,
    platform: str = "universal",
    text: str = "",
    language: str = "en",
    background: tuple[Path, str | None, float, int, int] | None = None,  # (source, family, t, w, h)
    keywords: set[str] | None = None,
    width: int = 540,
) -> bytes:
    """JPEG of the caption page at the moment its second word is spoken (so the highlight shows)."""
    if opts.style == "auto":
        opts = opts.model_copy(update={"style": "high_retention"})
    opts = opts.model_copy(update={"enabled": True})
    words = sample_words(text.strip() or SAMPLE.get(language[:2], SAMPLE["en"]))
    pages = cap.paginate(words, opts, words[-1].end + 1.0, keywords or {"two", "weeks", "duas", "semanas"})
    if not pages:
        raise ValueError("nothing to show")
    page = pages[0]
    at = page.words[min(1, len(page.words) - 1)].start + 0.12
    ass = cap.build_ass(pages, opts, platform, 1080, 1920)
    if not _gate.acquire(timeout=5):
        raise PreviewBusy("previews are busy")
    try:
        with tempfile.TemporaryDirectory(prefix="vc-preview-") as tmp:
            wd = Path(tmp)
            (wd / "p.ass").write_text(ass, encoding="utf-8")
            _link_fonts(wd)
            if background is not None:
                src, family, t, sw, sh = background
                # centre 9:16 window of a real source frame, as the default framing would show it
                cw = min(sw, int(sh * 9 / 16)) // 2 * 2
                chh = min(sh, int(cw * 16 / 9)) // 2 * 2
                grab = run_capture(
                    ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *input_args(src, family, seek=max(0.0, t), threads=1),
                     "-frames:v", "1", "-vf", f"crop={cw}:{chh},scale=1080:1920:flags=bicubic,setsar=1", "-q:v", "2", "bg.jpg"],
                    timeout=20, cwd=wd,
                )  # fmt: skip
                if grab.returncode == 0 and (wd / "bg.jpg").exists():
                    inp = ["-loop", "1", "-framerate", "10", "-t", f"{at + 0.2:.2f}", "-i", "bg.jpg"]
                else:
                    background = None
            if background is None:
                inp = ["-f", "lavfi", "-i", f"color=c=0x1d1f27:s=1080x1920:r=10:d={at + 0.2:.2f}"]
            vf = f"ass=p.ass:fontsdir=fonts,scale={width}:-2"
            cmd = [
                "ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *inp, "-vf", vf,
                "-ss", f"{at:.2f}", "-frames:v", "1", "-q:v", "3", "-f", "image2", "-c:v", "mjpeg", "pipe:1",
            ]  # fmt: skip
            res = run_capture(cmd, timeout=30, cwd=wd)
            if res.returncode != 0 or not res.stdout:
                raise RuntimeError(res.stderr.decode("utf-8", "replace")[-300:])
            return res.stdout
    finally:
        _gate.release()
