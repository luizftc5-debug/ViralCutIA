"""The optional LLM layer, exercised with a stub client (no network)."""

import json
from types import SimpleNamespace

from helpers import STRONG, WEAK, make_context

from viralcut.pipeline import llm, scoring
from viralcut.pipeline.candidates import speech_candidates


class StubClient:
    def __init__(self, payload=None, stop_reason="end_turn", error=None):
        self.payload, self.stop_reason, self.error = payload, stop_reason, error
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
        )


def _candidates():
    ctx = make_context(STRONG + " | " + WEAK)
    cands = speech_candidates(ctx.sentences, ctx.words, ctx.media.duration, 8, 40)
    for c in cands:
        scoring.score_speech(c, ctx)
    return ctx, sorted(cands, key=lambda c: -c.score)[:2]


def _rating(index, score, **kw):
    base = {
        "index": index,
        "hook": score,
        "narrative": score,
        "value": score,
        "impact": score,
        "self_contained": score,
        "rationale": "Opens on a question and resolves it.",
        "hook_quote": "Why do most people fail",
        "titles": ["A grounded title"],
        "caption": "Caption.",
    }
    base.update(kw)
    return base


def test_ratings_are_blended_and_copy_is_attached():
    ctx, cands = _candidates()
    before = [dict(c.dims) for c in cands]
    client = StubClient({"clips": [_rating(0, 10), _rating(1, 0, hook_quote="words that were never said")]})
    info = llm.enhance(cands, ctx, client)
    assert info["used"] and info["rated"] == 2
    assert cands[0].dims["hook"] == round(0.6 * 1.0 + 0.4 * before[0]["hook"], 4)
    assert cands[1].dims["hook"] == round(0.4 * before[1]["hook"], 4)
    assert cands[0].dims["technical"] == before[0]["technical"], "technical quality stays signal-based"
    assert cands[0].score > cands[1].score
    assert cands[0].details["llm"]["titles"] == ["A grounded title"]
    assert cands[0].explanation == "Opens on a question and resolves it."
    assert cands[0].hook["text"] == "Why do most people fail"
    assert cands[1].hook["text"] != "words that were never said", "ungrounded hook quotes are rejected"
    call = client.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert '<candidate index="0"' in call["messages"][0]["content"]


def test_failures_degrade_to_heuristics():
    ctx, cands = _candidates()
    scores = [c.score for c in cands]
    for client in (
        StubClient(error=RuntimeError("boom")),
        StubClient({"clips": []}, stop_reason="refusal"),
        StubClient({"unexpected": True}),
        StubClient({"clips": [_rating(99, 5)]}),
    ):
        info = llm.enhance(cands, ctx, client)
        assert info["used"] is False and info.get("reason")
        assert [c.score for c in cands] == scores
