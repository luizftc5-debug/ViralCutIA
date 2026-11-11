import json

from helpers import STRONG, STRONG_PT, WEAK, make_context, make_words

from viralcut.pipeline import scoring
from viralcut.pipeline.candidates import Candidate, speech_candidates
from viralcut.pipeline.copywriting import extractive_copy
from viralcut.pipeline.selection import auto_count, select, temporal_overlap
from viralcut.pipeline.text import build_sentences, lang_key, norm, tokenize


def _scored(ctx, lo=8.0, hi=40.0):
    cands = speech_candidates(ctx.sentences, ctx.words, ctx.media.duration, lo, hi)
    for c in cands:
        scoring.score_speech(c, ctx)
    return cands


def test_sentences_split_on_punctuation_and_pauses():
    words = make_words("Hello there. How are you today? fine thanks | new topic starts here")
    sents = build_sentences(words, "en")
    assert [s.text for s in sents] == [
        "Hello there.",
        "How are you today?",
        "fine thanks",
        "new topic starts here",
    ]
    assert sents[1].is_question and sents[0].terminal and not sents[2].terminal
    assert sents[3].gap_before > 1.0


def test_text_normalisation_handles_accents():
    assert norm("Você NÃO") == "voce nao"
    assert tokenize("Coração, não é?") == ["coracao", "nao", "e"]
    assert lang_key("pt") == "pt" and lang_key("en") == "en" and lang_key("ja") is None


def test_candidates_start_and_end_on_sentence_boundaries():
    ctx = make_context(STRONG + " | " + WEAK)
    cands = _scored(ctx)
    assert cands
    starts = {round(s.start, 2) for s in ctx.sentences}
    for c in cands:
        assert 8.0 <= ctx.sentences[c.s1].end - ctx.sentences[c.s0].start <= 40.0
        assert any(abs(c.start - st) <= 0.2 for st in starts), "clip must begin at a sentence start"
        assert c.start <= ctx.sentences[c.s0].start and c.end >= ctx.sentences[c.s1].end, (
            "never cuts into a word"
        )


def test_strong_segment_outranks_filler():
    ctx = make_context(WEAK + " | " + STRONG + " | " + WEAK)
    cands = _scored(ctx)
    best = max(cands, key=lambda c: c.score)
    assert best.text.startswith("Why do most people fail")
    assert best.score >= 75
    assert best.dims["hook"] >= 0.8
    assert best.hook["types"][0]["type"] == "curiosity"
    filler = [c for c in cands if c.text.startswith("And then we went")]
    assert filler and max(c.score for c in filler) < best.score - 15
    assert "opening" in best.explanation.lower()


def test_portuguese_hooks_are_detected():
    ctx = make_context(WEAK.replace("And then", "E depois") + " | " + STRONG_PT, language="pt")
    best = max(_scored(ctx), key=lambda c: c.score)
    assert best.text.startswith("Você sabia")
    assert best.dims["hook"] >= 0.7
    assert any(t["type"] == "curiosity" for t in best.hook["types"])


def test_dependent_opener_is_penalised():
    ctx = make_context("Because they never practice the hard part first. " + STRONG)
    cands = _scored(ctx)
    dep = [c for c in cands if c.s0 == 0]
    clean = [c for c in cands if c.s0 == 1]
    assert max(c.dims["self_contained"] for c in dep) < max(c.dims["self_contained"] for c in clean)
    assert all(c.hook["opener"] == "dependent" for c in dep)


def test_topic_break_inside_clip_lowers_narrative():
    ctx = make_context(STRONG + " | " + STRONG.replace("skill", "language").replace("practice", "study"))
    cands = _scored(ctx, 8.0, 70.0)
    n = len(ctx.sentences) // 2
    single = next(c for c in cands if c.s0 == 0 and c.s1 == n - 1)
    spanning = next(c for c in cands if c.s0 == 0 and c.s1 == n + 1)
    assert spanning.details["unity"] < single.details["unity"]
    assert spanning.dims["narrative"] < single.dims["narrative"]


def test_selection_dedupes_and_respects_min_score():
    ctx = make_context(STRONG + " | " + WEAK + " | " + STRONG.replace("skill", "language"))
    cands = _scored(ctx)
    chosen = select(cands, ctx, max_clips=10, min_score=60)
    assert 1 <= len(chosen) < 10, "returns fewer clips than requested when material is limited"
    assert all(c.score >= 60 for c in chosen)
    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            assert temporal_overlap(a, b) <= 0.08
    assert select(cands, ctx, max_clips=10, min_score=99) == []


def test_semantic_duplicates_are_removed():
    ctx = make_context(STRONG + " | " + STRONG)  # the same story told twice
    chosen = select(_scored(ctx), ctx, max_clips=5, min_score=50)
    assert len([c for c in chosen if c.text.startswith("Why do most people")]) == 1


def test_overlap_and_auto_count():
    a, b = Candidate(0, 10), Candidate(8, 30)
    assert abs(temporal_overlap(a, b) - 0.2) < 1e-6
    assert temporal_overlap(Candidate(0, 5), Candidate(6, 9)) == 0
    assert auto_count(20) == 1 and auto_count(125) == 5 and auto_count(7200) == 10


def test_weights_are_configurable(monkeypatch):
    monkeypatch.setenv(
        "VIRALCUT_SCORE_WEIGHTS",
        json.dumps({"hook": 1, "narrative": 0, "value": 0, "impact": 0, "self_contained": 0, "technical": 0}),
    )
    w = scoring.load_weights()
    assert w["hook"] == 1.0 and w["narrative"] == 0.0
    monkeypatch.setenv("VIRALCUT_SCORE_WEIGHTS", "not json")
    assert scoring.load_weights() == scoring.DEFAULT_WEIGHTS
    assert abs(sum(scoring.DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


def test_extractive_copy_only_uses_spoken_sentences():
    ctx = make_context(STRONG)
    best = max(_scored(ctx), key=lambda c: c.score)
    titles, caption = extractive_copy(best, ctx)
    assert 1 <= len(titles) <= 3
    assert titles[0] == "Why do most people fail at learning a new skill?"
    spoken = norm(best.text)
    for t in titles:
        assert len(t) <= 81
        assert norm(t.rstrip("…")) in spoken, "titles must be verbatim excerpts"
    assert caption.startswith("Why do most people fail")
