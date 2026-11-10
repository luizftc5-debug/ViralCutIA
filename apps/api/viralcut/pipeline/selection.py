"""Stage 7b — ranking, de-duplication and diversity."""

from __future__ import annotations

from .candidates import Candidate
from .scoring import AnalysisContext
from .text import TfIdf


def temporal_overlap(a: Candidate, b: Candidate) -> float:
    """Overlap as a share of the shorter clip."""
    inter = min(a.end, b.end) - max(a.start, b.start)
    if inter <= 0:
        return 0.0
    return inter / max(0.1, min(a.duration, b.duration))


def auto_count(media_duration: float) -> int:
    """Upper bound for the 'automatic' clip count (1..10).

    It only caps the list: how many clips are actually returned is decided by the minimum
    score and the de-duplication rules, so weak material yields fewer clips.
    """
    return int(max(1, min(10, media_duration // 25)))


def select(
    candidates: list[Candidate],
    ctx: AnalysisContext | None,
    *,
    max_clips: int,
    min_score: float,
    max_overlap: float = 0.25,
    max_similarity: float = 0.55,
) -> list[Candidate]:
    """Greedy non-maximum suppression over time plus semantic de-duplication.

    Returns fewer clips than requested when the material does not support more.
    """
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    chosen: list[Candidate] = []
    vectors: list[dict[str, float]] = []
    for c in ranked:
        if len(chosen) >= max_clips:
            break
        if c.score < min_score:
            break
        if any(temporal_overlap(c, o) > max_overlap for o in chosen):
            continue
        vec: dict[str, float] = {}
        if ctx is not None and c.kind == "speech" and c.s0 >= 0:
            tokens = [t for s in ctx.sentences[c.s0 : c.s1 + 1] for t in s.content]
            vec = ctx.tfidf.vector(tokens)
            if any(v and TfIdf.cosine(vec, v) > max_similarity for v in vectors):
                continue
        chosen.append(c)
        vectors.append(vec)
    return chosen
