"""Extractive titles and captions.

Without an LLM the copy is built only from sentences that were actually spoken in the clip —
lightly cleaned, never paraphrased or embellished. Results are labelled ``extractive`` so the
UI does not present them as generated copywriting.
"""

from __future__ import annotations

import re

from .candidates import Candidate
from .scoring import AnalysisContext
from .text import (
    DEPENDENT_OPENERS,
    FILLERS,
    HOOK_PATTERNS,
    STRIPPABLE_OPENERS,
    Sentence,
    lang_key,
    norm,
    tokenize,
)

MAX_TITLE = 80


def clean_sentence(text: str, lk: str | None) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # drop leading connectives/fillers ("And so, ..." -> "...")
    for _ in range(3):
        toks = tokenize(text)
        if not toks or lk is None:
            break
        first = toks[0]
        if first in STRIPPABLE_OPENERS[lk] or first in FILLERS[lk]:
            parts = text.split(" ", 1)
            if len(parts) < 2 or len(toks) < 4:
                break
            text = parts[1].lstrip(",;: ")
        else:
            break
    text = text.rstrip(" ,;:")
    if text.endswith(".") and not text.endswith("..."):
        text = text[:-1]
    return text[:1].upper() + text[1:] if text else text


def shorten(text: str, limit: int = MAX_TITLE) -> str:
    if len(text) <= limit:
        return text
    # prefer cutting at a clause boundary, otherwise at a word boundary
    head = text[:limit]
    for sep in (", ", "; ", " — ", " - ", ": "):
        k = head.rfind(sep)
        if k >= limit * 0.5:
            return head[:k].rstrip(" ,;:") + "…"
    k = head.rfind(" ")
    return (head[:k] if k > 0 else head).rstrip(" ,;:") + "…"


def _sentence_strength(s: Sentence, lk: str | None, keywords: set[str]) -> float:
    score = len(set(s.content) & keywords) * 0.5
    if s.is_question:
        score += 1.5
    if lk:
        n = norm(s.text)
        for cat in ("curiosity", "stakes", "value", "interrupt"):
            if HOOK_PATTERNS[cat][lk].search(n):
                score += 1.0
    if not 4 <= s.n_words <= 18:
        score -= 1.5
    if lk and s.tokens:
        # a sentence that leans on the previous one ("That is why…", "It is…") does not work as a title
        first = next((t for t in s.tokens if t not in STRIPPABLE_OPENERS[lk] and t not in FILLERS[lk]), "")
        if first in DEPENDENT_OPENERS[lk]:
            score -= 3.0
    return score


def extractive_copy(c: Candidate, ctx: AnalysisContext) -> tuple[list[str], str]:
    """Return (up to three titles, caption) built from the clip's own sentences."""
    lk = lang_key(ctx.language)
    sents = ctx.sentences[c.s0 : c.s1 + 1] if c.s0 >= 0 else []
    if not sents:
        return [], ""
    keywords = set(ctx.tfidf.keywords)
    titles: list[str] = []

    def add(text: str) -> None:
        t = shorten(clean_sentence(text, lk))
        if len(t) >= 12 and all(norm(t) != norm(x) for x in titles):
            titles.append(t)

    add(sents[0].text)  # the hook as spoken
    ranked = sorted(sents[1:], key=lambda s: _sentence_strength(s, lk, keywords), reverse=True)
    for s in ranked:
        if len(titles) >= 3:
            break
        if _sentence_strength(s, lk, keywords) > 0:
            add(s.text)

    lead = clean_sentence(sents[0].text, lk)
    if not lead.endswith(("?", "!", "…")):
        lead += "."
    body = lead
    if len(sents) > 1 and len(body) < 110:
        nxt = clean_sentence(sents[1].text, lk)
        if nxt:
            body = shorten(f"{body} {nxt}", 180)
            if not body.endswith(("?", "!", "…", ".")):
                body += "."
    # hashtags: only topic words that recur across the video and appear in this clip
    clip_tokens = {t for s in sents for t in s.content}
    tags = [
        k
        for k in ctx.tfidf.keywords[:12]
        if k in clip_tokens and len(k) >= 5 and k.isalpha() and ctx.tfidf.tf[k] >= 3
    ][:3]
    caption = body + ("\n\n" + " ".join(f"#{t}" for t in tags) if tags else "")
    return titles, caption
