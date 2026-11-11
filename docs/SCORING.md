# Viral Potential Score

The score estimates how well a moment **opens, holds attention and stands on its own**. It is not
a prediction of views and makes no claim about how any platform ranks content.

Three kinds of statement appear in this project and are kept apart:

| Kind | Example | Status |
|---|---|---|
| Creation heuristics | "Start on a sentence boundary", "do not cut into a word" | Editing craft; enforced by construction |
| Growth hypotheses | "A stronger first three seconds improves retention", the weights below | Unvalidated product assumptions |
| Observed results | Views, retention, shares reported for a published clip | Collected through the metrics API; none exist yet |

## Dimensions and initial weights

| Dimension | Weight | Measured from |
|---|---:|---|
| Hook strength | 25% | First ~3 s: hook type signals (below), sentence length, dependent openers |
| Narrative retention | 20% | Payoff markers, question answered, development, pacing, topic unity |
| Content value | 20% | Duration-weighted sentence informativeness, lexical density, actionable wording, filler |
| Emotional / visual impact | 15% | Emotion and stakes lexicon, punctuation, vocal dynamics, loudness and motion percentiles |
| Standalone clarity | 10% | Clean start, complete ending, cohesion, starting at a section break, back-references |
| Technical quality | 10% | Transcription confidence, level, signal-to-noise, clipping, source resolution |

`score = 100 × Σ weightᵢ × dimensionᵢ`, each dimension in 0..1.

Override the weights without touching code (they are renormalised):

```bash
VIRALCUT_SCORE_WEIGHTS='{"hook":0.30,"narrative":0.20,"value":0.20,"impact":0.10,"self_contained":0.10,"technical":0.10}'
```

## Hook Detection Engine

`pipeline/hooks.py` looks only at what is really in the source.

| Hook type | Evidence |
|---|---|
| Curiosity gap | An opening question; phrases such as "why", "the real reason", "você sabia", "o segredo" |
| Pattern interruption | Contrarian wording ("stop", "wrong", "pare de"), a loud onset, a shot change at the opening |
| High stakes | Loss, risk, money, failure vocabulary; numbers reinforce it |
| Emotional impact | Emotion vocabulary, exclamation, expressive vocal dynamics |
| Immediate value | "Here's how", counted lists ("three ways", "três formas"), direct address |
| Narrative tension | Story openers ("three years ago", "um dia", "de repente") |
| Strong visual opening | Motion in the first two seconds relative to the whole video |

The three strongest types are combined with diminishing weight, multiplied by a penalty when the
first sentence depends on earlier context ("And…", "Because…", "Ele…").

## What keeps clips coherent

* Candidates begin and end on sentence boundaries, padded without touching neighbouring words.
* Every topic break inside a candidate (long pause, shot change during a pause, lexical shift)
  reduces its narrative and clarity scores, so a clip stays on one idea.
* Value is an average over sentences, so filler dilutes it and length alone earns nothing.
* Selection is greedy non-maximum suppression: at most 8% temporal overlap with an already chosen
  clip, and TF-IDF cosine similarity below 0.55 (the same story told twice is kept once).
* Clips below the minimum score (`VIRALCUT_MIN_SCORE`, default 58, adjustable per job) are not
  rendered. Fewer, better clips are preferred to a padded list.

## Engines and their labels

| `engine` | When | What the UI says |
|---|---|---|
| `heuristic` | Speech found, no LLM configured | "Estimated from speech, audio and picture signals" |
| `llm+signals` | Speech found and `ANTHROPIC_API_KEY` set | LLM ratings (60%) blended with heuristics (40%); technical stays signal-based |
| `signals-only` | No usable speech | Low confidence; narrative, value and clarity are fixed neutral values |

The LLM receives only transcript excerpts of the shortlisted candidates and must return structured
ratings, a one-sentence rationale, an exact hook quote (rejected if it is not in the excerpt),
titles and a caption grounded in the transcript. A failed or refused call degrades to heuristics
and is reported in the job's warnings.

Lexicons exist for English and Portuguese. Other languages fall back to language-agnostic signals
(questions, numbers, audio, picture) and, when configured, the LLM; the job says so.

## Calibrating with real performance

`POST /api/jobs/{job}/clips/{clip}/metrics` stores what a creator observed after publishing
(views, average percent watched, 3-second retention, shares…) next to a snapshot of the features
that produced the score: all six dimensions, hook types, duration, caption style and layout.

That table is the training set for the planned calibration loop:

1. Collect outcomes per platform; normalise by account baseline and hours since publishing.
2. Fit the weights (start with a regularised linear model on the six dimensions; target:
   average percent watched and 3-second retention, not raw views).
3. Ship candidate weights behind `VIRALCUT_SCORE_WEIGHTS` and A/B them: hook type, caption style
   and clip length are already recorded per clip, so they can be compared directly.
4. Report calibrated scores only once they beat the hypothesis weights on held-out clips.

Until that data exists the weights remain hypotheses, and the product says so.
