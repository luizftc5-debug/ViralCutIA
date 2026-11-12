"""Creator Growth Toolkit: publishing material drawn from what is actually said in each clip.

Everything is extractive (or comes from the optional LLM titles, which are grounded in the
transcript): titles, description, keywords, the opening hook, thumbnail text, category,
platform hashtags, follow-up topics taken from other strong moments of the same video, and
chapters for long-form edits. No performance numbers are produced or promised.
"""

from __future__ import annotations

import re
from collections import Counter

from .text import STOPWORDS, lang_key, norm

PLATFORM_LIMITS = {"youtube_shorts": 100, "tiktok": 150, "reels": 150, "youtube": 100}


def keywords_for(text: str, language: str, global_keywords: list[str], k: int = 8) -> list[str]:
    stop = STOPWORDS.get(lang_key(language) or "", frozenset())
    toks = [t for t in (norm(x).strip(".,!?;:\"'()") for x in text.split()) if len(t) >= 4 and t not in stop]
    counts = Counter(toks)
    rank = {w: i for i, w in enumerate(global_keywords)}
    scored = sorted(counts, key=lambda w: (-(counts[w] + (3 if w in rank else 0)), rank.get(w, 999)))
    return scored[:k]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]


def thumbnail_texts(hook: str, keywords: list[str]) -> list[str]:
    words = [w.strip(".,!?;:\"'") for w in hook.split()]
    out = []
    if words:
        out.append(" ".join(words[:4]).upper())
    nums = [w for w in words if any(ch.isdigit() for ch in w)]
    if nums and keywords:
        out.append(f"{nums[0]} {keywords[0]}".upper())
    if keywords[:2]:
        out.append(" ".join(keywords[:2]).upper() + "?")
    return list(dict.fromkeys(out))[:3]


def hashtags(keywords: list[str], category: str) -> list[str]:
    tags = ["#" + re.sub(r"\W+", "", k) for k in keywords[:5] if re.sub(r"\W+", "", k)]
    if category and category != "general":
        tags.append("#" + category)
    return list(dict.fromkeys(tags))


def build(
    *,
    text: str,
    titles: list[str],
    caption: str,
    hook_text: str,
    language: str,
    global_keywords: list[str],
    category: str,
    content_type: str,
    followups: list[str],
    variants: list[str],
) -> dict:
    kws = keywords_for(text, language, global_keywords)
    sents = _sentences(text)
    desc = caption or " ".join(sents[:2])[:280]
    titles = list(dict.fromkeys(titles))[:3]
    if len(titles) < 3 and sents:
        for s_ in sents:
            t = s_.rstrip(".")
            if 20 <= len(t) <= 90 and t not in titles:
                titles.append(t)
            if len(titles) == 3:
                break
    tags = hashtags(kws, category)
    platforms = {
        "youtube_shorts": f"{titles[0] if titles else ''} #shorts".strip(),
        "tiktok": (desc[:120] + " " + " ".join(tags[:4])).strip(),
        "reels": (desc[:120] + " " + " ".join(tags[:5])).strip(),
    }
    if content_type == "youtube":
        platforms = {"youtube": desc}
    return {
        "titles": titles,
        "description": desc,
        "keywords": kws,
        "hook": hook_text,
        "thumbnail_text": thumbnail_texts(hook_text or (sents[0] if sents else ""), kws),
        "category": category,
        "hashtags": tags,
        "platforms": platforms,
        "followups": followups[:3],
        "versions_to_test": variants,
        "note": "Suggestions drawn from this video's own words. They do not predict or guarantee views or revenue.",
    }


def chapters(
    sentences, breaks: list[float], to_out, start: float, end: float, min_gap: float = 60.0
) -> list[dict]:  # noqa: ANN001
    """YouTube chapters: 0:00 plus topic breaks inside the edit, at least ``min_gap`` apart."""
    out = [{"t": 0.0, "title": ""}]
    for i, s_ in enumerate(sentences):
        if not (start <= s_.start < end) or i >= len(breaks):
            continue
        if breaks[i] >= 0.55:
            t = to_out(s_.start - start)
            if t is not None and t - out[-1]["t"] >= min_gap:
                out.append({"t": round(t, 1), "title": " ".join(s_.text.split()[:7]).rstrip(".,")})
    first = next((s_ for s_ in sentences if s_.start >= start), None)
    out[0]["title"] = " ".join(first.text.split()[:7]).rstrip(".,") if first else "Intro"
    return out if len(out) >= 3 else []  # YouTube needs at least three chapters
