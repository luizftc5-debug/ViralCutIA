#!/usr/bin/env python3
"""Render the caption-style preview images used by the web app.

The previews are produced by the real caption engine and libass, so what the style picker
shows is exactly what gets burned into a clip. Run from the repository root:

    apps/api/.venv/bin/python scripts/render_style_previews.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from viralcut.config import get_settings
from viralcut.options import CaptionOptions
from viralcut.pipeline import captions as cap
from viralcut.pipeline.transcribe import Word

OUT = Path(__file__).resolve().parents[1] / "apps" / "web" / "public" / "caption-styles"
SAMPLE = "The best moment was already in your video"
W, H = 1080, 1920
BAND = 560  # height of the cropped preview band, in output pixels


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    words, t = [], 0.1
    for tok in SAMPLE.split():
        words.append(Word(tok, round(t, 2), round(t + 0.3, 2)))
        t += 0.36
    fonts = get_settings().fonts_dir.resolve()
    for style in cap.STYLES:
        opts = CaptionOptions(style=style)
        pages = cap.paginate(words, opts, t + 1, {"moment"})
        ass_text = cap.build_ass(pages, opts, "universal", W, H)
        page = pages[0]
        # freeze on the last word of the first page while it is the active word
        at = page.words[-1].start + 0.22
        cy = cap.make_layout(opts, "universal", W, H).center_y
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "c.ass").write_text(ass_text, encoding="utf-8")
            (Path(tmp) / "fonts").symlink_to(fonts, target_is_directory=True)
            out = OUT / f"{style}.webp"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"gradients=s={W}x{H}:c0=0x2b3138:c1=0x4a4036:c2=0x1d2126:n=3:d=4:speed=0.01:r=30",
                    "-vf", f"ass=c.ass:fontsdir=fonts,crop={W}:{BAND}:0:{int(cy - BAND / 2)},scale=720:-2",
                    "-ss", f"{at:.2f}", "-frames:v", "1", "-quality", "82", str(out),
                ],
                check=True, cwd=tmp,
            )  # fmt: skip
            print("wrote", out.relative_to(OUT.parents[3]), f"({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
