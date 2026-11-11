"""Output variations: alternative versions of one selected clip, built from the same analysis.

Every variation is a faithful edit of the original material — nothing is reordered, invented
or re-voiced — and reuses the transcript, the scene analysis and (when the in/out points are
unchanged) the alignment and face tracking of the main version:

* **Hook-first** — starts on a later sentence of the same clip whose opening measures as a
  clearly stronger hook (Hook Detection Engine), when the rest still makes a clip of the
  requested length.
* **Alternate style** — the same moment edited with a different video style (the one the
  clip's measurements recommend, or a contrasting one) and that style's caption design.
* **Pacing** — the same moment at another editing intensity.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidates import Candidate, clip_bounds
from .hooks import analyze_hook
from .scoring import AnalysisContext
from .styles import STYLES, resolve

HOOK_GAIN = 0.12  # a later opening must beat the original hook score by this much (0..1 scale)

# contrasting partner when the recommended style is the one already used
CONTRAST = {
    "viral": "minimal", "podcast": "viral", "gaming": "reaction", "cinematic": "viral",
    "educational": "viral", "reaction": "storytime", "minimal": "viral", "storytime": "viral", "news": "educational",
}  # fmt: skip
NEXT_INTENSITY = {"balanced": "high", "high": "subtle", "subtle": "balanced"}


@dataclass
class Variation:
    index: int  # 2, 3
    kind: str  # hook_first | style | pacing
    label: str
    why: str
    start: float
    end: float
    style: str
    intensity: str
    caption_style: str | None  # None = keep the job's caption choice

    def to_dict(self) -> dict:
        return {"index": self.index, "kind": self.kind, "label": self.label, "why": self.why}


def hook_first(c: Candidate, ctx: AnalysisContext, min_d: float) -> tuple[float, float, dict] | None:
    """A later start inside ``c`` whose opening is a measurably stronger hook, or None."""
    if c.kind != "speech" or c.s0 < 0 or c.s1 <= c.s0:
        return None
    base = float((c.hook or {}).get("score", c.dims.get("hook", 0.0)))
    best = None
    for s in range(c.s0 + 1, c.s1 + 1):
        st, en = clip_bounds(ctx.sentences, ctx.words, s, c.s1, ctx.media.duration)
        if en - st < min_d or st - c.start > 0.5 * c.duration:
            continue
        h = analyze_hook(st, ctx.sentences, ctx.words, s, c.s1, ctx.language, ctx.audio, ctx.visual, en)
        if h.score >= base + HOOK_GAIN and (best is None or h.score > best[2]["score"]):
            best = (st, en, {"score": round(h.score, 3), "from": round(base, 3), "text": h.text})
    return best


def plan_variations(
    c: Candidate,
    ctx: AnalysisContext,
    count: int,
    style: str,
    intensity: str,
    min_d: float,
    recommended: str | None = None,
) -> list[Variation]:
    """Up to ``count - 1`` alternatives to the main version of ``c`` (the main one is index 1)."""
    out: list[Variation] = []
    if count <= 1:
        return out
    base_style = resolve(style)
    hf = hook_first(c, ctx, min_d)
    if hf is not None:
        st, en, info = hf
        out.append(
            Variation(
                len(out) + 2, "hook_first", "Hook-first",
                f"starts {st - c.start:.1f}s later on a stronger opening (hook {info['from']:.0%} → {info['score']:.0%}): “{info['text'][:80]}”",
                st, en, style, intensity, None,
            )
        )  # fmt: skip
    if len(out) < count - 1:
        alt = recommended if recommended and recommended != base_style else CONTRAST.get(base_style, "viral")
        if base_style == "auto":
            alt = recommended or "viral"
            alt = CONTRAST.get(alt, "minimal")
        why = (
            "the style this clip's measurements recommend"
            if alt == recommended
            else f"a contrasting edit to compare against {STYLES.get(base_style).label if base_style in STYLES else 'the automatic style'}"
        )
        out.append(
            Variation(
                len(out) + 2,
                "style",
                STYLES[alt].label,
                why,
                c.start,
                c.end,
                alt,
                intensity,
                STYLES[alt].caption,
            )
        )
    if len(out) < count - 1:
        nxt = NEXT_INTENSITY.get(intensity, "high")
        out.append(
            Variation(
                len(out) + 2, "pacing", f"{nxt.title()} pacing", f"same moment cut at {nxt} intensity",
                c.start, c.end, style, nxt, None,
            )
        )  # fmt: skip
    return out[: count - 1]
