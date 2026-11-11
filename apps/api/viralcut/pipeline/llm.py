"""Optional LLM narrative analysis (Anthropic Claude).

Active only when ``ANTHROPIC_API_KEY`` is configured. The model reads the transcript of the
strongest heuristic candidates and returns structured ratings plus titles and a caption that
must be grounded in what was actually said. Ratings are blended with the measured signals;
technical quality always stays signal-based.

Without a key (or when a call fails) the pipeline keeps its heuristic scores and extractive
copy, and says so in the job's analysis summary — the fallback is never presented as LLM output.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from ..config import get_settings
from .candidates import Candidate
from .scoring import AnalysisContext, finalize

log = logging.getLogger(__name__)

LLM_WEIGHT = 0.6  # share of the LLM rating in each blended text dimension
_DIMS = ("hook", "narrative", "value", "impact", "self_contained")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You evaluate excerpts of a long video as candidate short-form clips (YouTube Shorts, Instagram Reels, TikTok).

Each candidate is a verbatim transcript excerpt. Judge only what is in the transcript: you cannot see the video, and audio/visual quality is measured separately.

Rate every candidate from 0 to 10 on five dimensions:
- hook: do the first one or two sentences give a viewer a reason to keep watching (an open question, a surprising or contrarian claim, clear stakes, an immediately useful promise, the start of a story)? An opening that depends on earlier context, or that warms up slowly, scores low.
- narrative: does the excerpt develop and then pay off? High scores need a setup that goes somewhere and an ending that resolves it; an excerpt that stops mid-thought or ends on an unanswered question scores low.
- value: how much would a viewer learn, feel or take away? Specific, concrete, non-obvious content scores high; filler and generalities score low.
- impact: emotional intensity carried by the words themselves (surprise, tension, humour, vulnerability, conviction).
- self_contained: would someone who never saw the full video understand it? Unresolved references ("as I said", "he", "that thing") score low.

Use the whole scale. Most excerpts of ordinary talk deserve middling scores; reserve 9-10 for excerpts you would confidently publish as they are.

For every candidate also write:
- rationale: one objective sentence naming the main strength and the main weakness.
- hook_quote: the exact words from the excerpt that work as the hook (empty string if there is no real hook).
- titles: up to three publishable titles, each under 80 characters, in the same language as the transcript.
- caption: a one or two sentence post caption in the same language, optionally followed by up to three relevant hashtags.

Titles and captions must be faithful to the excerpt. Use curiosity, clarity and tension that the content actually supports. Do not add facts, numbers, names, outcomes or promises that are not in the transcript, and do not write misleading clickbait. Never predict views or virality."""


class ClipRating(BaseModel):
    index: int
    hook: int
    narrative: int
    value: int
    impact: int
    self_contained: int
    rationale: str
    hook_quote: str
    titles: list[str]
    caption: str


class RatingResponse(BaseModel):
    clips: list[ClipRating]


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "hook": {"type": "integer"},
                    "narrative": {"type": "integer"},
                    "value": {"type": "integer"},
                    "impact": {"type": "integer"},
                    "self_contained": {"type": "integer"},
                    "rationale": {"type": "string"},
                    "hook_quote": {"type": "string"},
                    "titles": {"type": "array", "items": {"type": "string"}},
                    "caption": {"type": "string"},
                },
                "required": [
                    "index",
                    "hook",
                    "narrative",
                    "value",
                    "impact",
                    "self_contained",
                    "rationale",
                    "hook_quote",
                    "titles",
                    "caption",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


def build_user_prompt(candidates: list[Candidate], language: str) -> str:
    lines = [
        f"Transcript language: {language or 'unknown'}.",
        f"There are {len(candidates)} candidates. Return one entry per candidate, using its index.",
        "",
    ]
    for i, c in enumerate(candidates):
        lines.append(f'<candidate index="{i}" duration_seconds="{c.duration:.0f}">')
        lines.append(c.text.strip())
        lines.append("</candidate>")
        lines.append("")
    return "\n".join(lines)


def _make_client():
    import anthropic

    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key, max_retries=2, timeout=180.0)


def request_ratings(candidates: list[Candidate], language: str, client: Any | None = None) -> RatingResponse:
    """Single structured-output call. Raises on API errors, refusals and malformed output."""
    s = get_settings()
    client = client or _make_client()
    response = client.beta.messages.create(
        model=s.llm_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(candidates, language)}],
        output_config={"effort": s.llm_effort, "format": {"type": "json_schema", "schema": _SCHEMA}},
        # if the primary model declines a request, let the API retry it on its default fallback
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined to analyse this transcript")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("the model response was truncated")
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return RatingResponse.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"unexpected response format: {exc}") from exc


def apply_ratings(candidates: list[Candidate], ratings: RatingResponse, ctx: AnalysisContext) -> int:
    """Blend LLM ratings into the candidates' dimensions. Returns how many were updated."""
    updated = 0
    for r in ratings.clips:
        if not 0 <= r.index < len(candidates):
            continue
        c = candidates[r.index]
        for dim in _DIMS:
            llm_value = max(0, min(10, int(getattr(r, dim)))) / 10.0
            c.dims[dim] = LLM_WEIGHT * llm_value + (1 - LLM_WEIGHT) * c.dims.get(dim, 0.0)
        finalize(c, ctx.weights)  # recompute score from blended dimensions
        titles = [t.strip() for t in r.titles if t and t.strip()][:3]
        c.details["llm"] = {
            "ratings": {dim: int(getattr(r, dim)) for dim in _DIMS},
            "rationale": r.rationale.strip(),
            "titles": titles,
            "caption": r.caption.strip(),
        }
        if r.rationale.strip():
            c.explanation = r.rationale.strip()
        # only accept a hook quote that really is in the excerpt
        quote = r.hook_quote.strip()
        if quote and quote.lower() in c.text.lower():
            c.hook["text"] = quote
        c.source = "llm"
        updated += 1
    return updated


def enhance(candidates: list[Candidate], ctx: AnalysisContext, client: Any | None = None) -> dict[str, Any]:
    """Rate candidates with the LLM. Never raises: failures degrade to heuristic-only scoring."""
    s = get_settings()
    info: dict[str, Any] = {"used": False, "model": s.llm_model, "rated": 0}
    speech = [c for c in candidates if c.kind == "speech" and c.text.strip()]
    if not speech:
        info["reason"] = "no speech candidates"
        return info
    try:
        import anthropic
    except ImportError:
        info["reason"] = "anthropic SDK is not installed (pip install 'viralcut[llm]')"
        return info
    try:
        ratings = request_ratings(speech, ctx.language, client)
    except anthropic.AuthenticationError:
        info["reason"] = "the Anthropic API key was rejected"
    except anthropic.RateLimitError:
        info["reason"] = "rate limited by the Anthropic API"
    except anthropic.APIStatusError as exc:
        info["reason"] = f"Anthropic API error ({exc.status_code})"
    except anthropic.APIConnectionError:
        info["reason"] = "could not reach the Anthropic API"
    except Exception as exc:  # refusal, truncation, malformed output
        info["reason"] = str(exc)[:200]
    else:
        info["rated"] = apply_ratings(speech, ratings, ctx)
        info["used"] = info["rated"] > 0
        if not info["used"]:
            info["reason"] = "the model returned no usable ratings"
    if not info["used"]:
        log.warning("LLM analysis unavailable: %s", info.get("reason"))
    return info
