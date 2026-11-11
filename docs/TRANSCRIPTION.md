# Speech recognition, caption timing and how they were measured

This document records what was wrong with transcription and caption timing, what changed, and the
numbers that justify each change. Every figure comes from a script in this repository that you can
re-run (see [Reproduce](#reproduce)). What is **not** measured is listed explicitly at the end.

## 1. What the audit found

The original pipeline decoded audio correctly (mono, 16 kHz, float32, one pass), detected language
correctly on every file tried, and transcribed clean speech reasonably. The defects were elsewhere,
and each one was reproduced with a number before it was changed.

| # | Defect | Evidence | Root cause | Fix |
|---|---|---|---|---|
| 1 | **Invented words** in music | A music-only file produced `" you"` (Portuguese and English test files alike) | The segment had `no_speech_prob = 0.96`, but faster-whisper drops a segment only when `no_speech_prob > 0.6` **and** `avg_logprob < -1.0`; this one had `-0.75` | Evidence filter (`transcribe.reject_reason`): model no-speech probability, confidence, VAD agreement, repetition loops, stock phrases. Every dropped segment is reported with its reason |
| 2 | **Whole utterances silently lost** | With VAD padding of 300 ms instead of 400 ms Whisper skipped a 7 s utterance, no warning (WER 46 %) | Whisper can end a 30 s window early without saying so | Completeness check: any stretch the VAD marks as speech but that has almost no words is transcribed again on its own. Recovered 17 words, WER 46 % → 10 % in that configuration |
| 3 | **Captions ahead of the voice** | Whisper's word timestamps put the first word of an utterance **early: median 176 ms with `small`, 320 ms with `large-v3-turbo`, 33 to 47 of 73 utterances more than 200 ms early** (section 3.3) | Cross-attention word timestamps are approximate and anchor to segment edges | CTC forced alignment of the recognised words (below) |
| 4 | **Captions outliving the voice** | Pages stayed 0.6 s after the last word | Fixed `+0.6` tail in pagination | Page ends at the last word + 0.18 s (chained with the next page only when the pause is < 0.3 s) |
| 5 | **Captions outside the safe area** | Rasterised worst case: the *active* word of the Gaming preset grows 20 %, and for a very long word that pushed the text to x = 66 px with a 97 px safe margin | The margin allowed a fixed 12 % of the font size, not a percentage of the word's own width | Layout now reserves the active word's peak size (`ACTIVE_PEAK`) |
| 6 | Inconsistent pacing on re-render | Re-rendering a *gaming* clip trimmed pauses at 0.7 s while the first render did not trim | `options.trim_max_pause or 0.7` turns "disabled" (`None`) into 0.7 | Pass `None` through |
| 7 | Single-pass loudness | `loudnorm` in dynamic mode on a short clip can pump and mis-target | One pass with no measurement | Two-pass linear loudness on the clip's exact audio graph |

## 2. What changed

**Transcription** (`pipeline/transcribe.py`)

* *Modes.* `VIRALCUT_TRANSCRIPTION_MODE=auto|fast|balanced|accurate`. `auto` = `large-v3` on a GPU,
  `large-v3-turbo` on a CPU with 4+ cores, `small` below that. An explicit `VIRALCUT_WHISPER_MODEL`
  always wins. If a model cannot be loaded the next smaller one is tried and the job says so.
* *Evidence filter and completeness check* (defects 1 and 2). Dropped segments and re-transcribed
  stretches are stored in `transcript.quality` and shown in the job analysis.
* *Glossary.* Names and jargon (`VIRALCUT_GLOSSARY`, or the per-job field) are passed to Whisper as
  hotwords.
* *Level treatment for the recogniser only.* Very quiet audio gets gain. **Denoising is opt-in
  (`VIRALCUT_ASR_ENHANCE=on`)**: measured on speech 5 dB above noise it did not lower the word error
  rate (PT 9.0 % → 11.0 %, EN 30.5 % → 33.3 %; Whisper is trained on noisy audio and the denoiser's
  artefacts cost about what they save). The audio rendered into the clip is never touched by this.
* *Language-specific prompt* (a short, correctly punctuated and accented sentence) is given to Whisper
  **when the language is set explicitly** (with `auto` the language is not known before decoding, so no
  prompt is used). Measured with `small`, language fixed, 19 PT and 20 EN utterances: PT 6.85 % → 5.74 %
  and EN 6.31 % → 5.61 % with the prompt (6 and 3 fewer wrong words out of 540 and 428). Small, consistent
  in both languages, not statistically established; it never made a result worse. `VIRALCUT_WHISPER_PROMPT=false` disables it.
* Silero VAD stays on (faster-whisper's own); the language is detected over up to four windows.

**Forced alignment** (`pipeline/align.py`) — the WhisperX approach without PyTorch: a CTC acoustic
model through onnxruntime aligns the *recognised* words to the waveform (Viterbi trellis in numpy),
per clip, so only the audio that is rendered pays for it.

| Language | Model | Licence |
|---|---|---|
| Portuguese | `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese` (ONNX 4-bit export, 241 MB) | Apache-2.0 |
| English | `facebook/wav2vec2-base-960h` (ONNX fp32, 378 MB; the int8 export produced a 640 ms outlier, fp32 did not) | Apache-2.0 |

The multilingual MMS aligner (1 130 languages) was evaluated and **not adopted by default because it
is CC-BY-NC (non-commercial)**. `VIRALCUT_ALIGN_REPO_OVERRIDES="xx=owner/repo:onnx/model.onnx"` lets
an operator who is licensed to use it point a language at it. Other languages, an offline host or a
failed quality gate fall back to Whisper's timestamps, then every word boundary next to a pause is
tightened against the loudness envelope. A failed model download is retried after 10 minutes rather
than given up for the life of the worker.

Quality gates: a chunk is rejected if the acoustic model disagrees with Whisper about where the speech
is (median shift > 1.2 s); words the model could not hear are interpolated between their neighbours;
digits and symbols (not in the character vocabulary) are placed in the gap between aligned neighbours.
The text is never altered.

## 3. Results

Corpus: **FLEURS** (Google, CC-BY-4.0): real human speakers reading sentences, Portuguese (Brazil) and
English (US). `scripts/make_speech_corpus.py` builds from it the conditions that break recognisers in
real videos, with exact ground truth for every utterance: a plain talk with pauses, a music bed 10 dB
and 3 dB below the speech, broadband noise 5 dB below the speech, very quiet speech (about −38 dBFS),
speech at 1.3× rate, 20 s of music before the speech starts, and three files with **no speech at all**
(music, noise, silence). Word Error Rate lower-cases and strips punctuation but **keeps accents**
("não" ≠ "nao"); spelled numbers and unit abbreviations are normalised on both sides.

### 3.1 Invented speech

| | Words produced from 6 non-speech files (music, noise, silence × 2 languages) |
|---|---|
| Original code (`small`) | **2**: `"you"` in the Portuguese music file and in the English one |
| Now (`small` and `large-v3-turbo`) | **0** |

Across all 73 speech utterances of the stress corpus, **no word fell outside the time span of a real
utterance** (`words_outside_speech: 0`) with either model, and the language was detected correctly on
all 17 speech files (the 3 non-speech files have no language to detect).

### 3.2 Word Error Rate under stress (same files, same normalisation)

| Condition (PT / EN) | original code, `small` | now, `small` | now, `large-v3-turbo` (default on 4+ cores) |
|---|---|---|---|
| talk with pauses | 4.8 % / 7.6 % | 4.8 % / 7.6 % | 3.3 % / 7.6 % |
| music bed, speech +10 dB | 4.3 % / 7.6 % | 4.3 % / 7.6 % | 3.8 % / 7.6 % |
| music bed, speech +3 dB | 4.3 % / 11.4 % | 4.3 % / 11.4 % | 3.3 % / 12.4 % |
| noise, speech +5 dB | 11.9 % / 35.2 % | 11.9 % / 35.2 % | 6.7 % / 22.9 % |
| very quiet speech | 5.2 % / 7.6 % | 3.8 % / 7.6 % | 3.3 % / 8.6 % |
| 1.3× speech rate | 13.6 % / 2.9 % | 13.6 % / 2.9 % | 3.4 % / 2.9 % |
| 20 s of music, then speech | 5.0 % / 4.5 % | 5.0 % / 4.5 % | 0.0 % / 3.0 % |
| **all conditions** | **6.6 % / 11.8 %** | **6.3 % / 11.8 %** | **3.8 % / 10.0 %** |

(Portuguese: 210 reference words per talk condition; English: 105. A single word is about 1 % in
English, so differences of a point or two between cells are noise.)

Clean read speech, 20 utterances per language sampled across the corpus (FLEURS recordings that were
truncated by my partial download are detected and skipped):

| | original code, `small` | now, `small` | now, `large-v3-turbo` |
|---|---|---|---|
| Portuguese (19 utterances, 540 words) | 6.99 % (one recording was truncated) | 6.85 % | **3.52 %** |
| English (20 utterances, 428 words) | 7.01 % | 6.31 % | **3.27 %** |

Honest reading. **With the same model (`small`) the accuracy is unchanged except for very quiet
speech** (the gain stage: PT 5.2 % → 3.8 %); what the code changes fix with `small` are the invented
words and the lost speech (defects 1 and 2 in section 1), not the error rate on audible speech. **The accuracy
gain comes from the model:** the default `large-v3-turbo` roughly halves the clean-speech error rate
and cuts noisy-Portuguese and fast-speech errors by more than half. Music at +3 dB and speech 5 dB
above noise remain hard (English: 12 % and 23 % with the best model measured).

### 3.3 Caption timing against the real audio

*Start of speech* — for every utterance, the time of its first word against the first audible instant,
taken from a VAD run on the **clean** recording (an estimator independent of both the recogniser and
the aligner). 73 utterances in 14 conditions, `large-v3-turbo`:

| Model | | mean abs. start error | median | 90th percentile | more than 200 ms early |
|---|---|---|---|---|---|
| `large-v3-turbo` | Whisper timestamps | 360 ms | −320 ms | 481 ms | 47 / 73 |
| | **+ CTC alignment and refinement** | **44 ms** | **+14 ms** | **97 ms** | **1 / 73** |
| `small` | Whisper timestamps | 278 ms | −176 ms | 400 ms | 33 / 73 |
| | **+ CTC alignment and refinement** | **63 ms** | **+14 ms** | **60 ms** | **2 / 73** |

(Negative = the caption would appear before the sound. The more accurate model has *worse* raw word
timestamps, which is why the alignment step is not optional with it. A word that ends more than 200 ms
after the last audible instant, the condition that makes a caption linger, occurs 2 / 73 times after
alignment. The reference end of an utterance includes room-noise tails the VAD counts as voice, so end
error is reported one-sided only.)

*End to end, on the rendered MP4* (`tests/test_speech_e2e.py`): the clip is rendered twice, with and
without captions; the frame difference is exactly where a caption is visible; the clip's own audio
gives the speech intervals (VAD on the output). English fixture: the four caption runs start and end
within 0.09 s of the four speech runs, the last caption disappears 0.014 s before the speech ends,
98 % of speech time has a caption on screen and 0 % of caption time falls in silence. The Portuguese
fixture covers 93.5 % of speech with 0 % stray caption time; the one outlier was a first word starting
~0.3 s before the detected speech onset.

### 3.4 Speed (4 CPU cores, int8, no GPU)

| Model | Real-time factor on continuous speech | Notes |
|---|---|---|
| `small` | ≈ 0.15 | |
| `large-v3-turbo` | ≈ 0.29 | Whisper pads every window to 30 s, so very short clips cost ~8 s each (RTF ≈ 0.7–1.0 on 10 s utterances) |
| CTC alignment | ≈ 0.15 (Portuguese, 300 M-parameter model) and ≈ 0.04 (English, base model) of the *clip* audio | only the rendered audio is aligned |

## 4. Manual correction

* Per caption page (existing): fix a page in the studio and re-render.
* **Whole transcript** (new): `GET/PUT /api/jobs/{id}/transcript` — fix words by index or find-and-replace
  across the video. Text changes only; every word keeps the time it was spoken at; the recogniser's
  original output is kept as `transcript.original.json`.
* **Before rendering** (new): start a job with `review_transcript: true` (a toggle under *Advanced*).
  The job pauses in status `review` after the clips are chosen, nothing is rendered, the studio shows the
  transcript for correction, and `POST /api/jobs/{id}/resume` renders with the corrected words.
* *Names and terms* can be given up front (glossary) so they are spelled right the first time.

Timestamps after cutting: every word is mapped through the same frame-snapped time map that cuts the
video and the audio (`trimming.TimeMap`), after alignment and before pagination; the end-to-end sync test
above runs on clips whose pauses were trimmed.

## 5. Reproduce

```bash
# the corpus (needs the FLEURS test split; the first ~60 MB of each language's tarball is enough)
python scripts/make_speech_corpus.py --fleurs pt:DIR_PT --fleurs en:DIR_EN --out corpus/

# WER, language detection, invented words outside speech, and timing before/after alignment
python scripts/benchmark_transcription.py --json corpus/manifest.json --limit 0 --align --model large-v3-turbo

# clean read speech
python scripts/benchmark_transcription.py --tsv test.tsv --audio-dir audio/ --lang pt --limit 20

# the regression suite (real speech end to end; needs the models, ~2 minutes)
cd apps/api && pytest -m integration
```

## 6. What is not measured

* **Spontaneous, overlapping or heavily accented speech.** FLEURS is read speech. Nothing here says
  how the engine does on a podcast with crosstalk. The glossary and the review step exist because it
  will make mistakes on names and jargon.
* **Real music.** The "music" is a synthetic chord/arpeggio/hi-hat bed, a fair stand-in for level and
  spectral masking but not for sung lyrics, which Whisper will transcribe (correctly or not).
* **Word-level ground truth.** Timing is measured at the start of each utterance and end to end on
  the render; inner-word boundaries have no independent reference. They come from the CTC model.
* **Languages other than Portuguese and English** for alignment, and any language but those two for
  the text heuristics.
* **GPU speed and `large-v3`.** No GPU was available; `large-v3` is selected by `auto` on CUDA but was
  not run here.
