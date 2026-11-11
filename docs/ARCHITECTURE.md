# Architecture

ViralCut AI is two deployables that share nothing but HTTP and a database:

```
┌────────────┐   upload / poll / download    ┌──────────────┐        ┌───────────────┐
│  Web app   │ ────────────────────────────▶ │     API      │ ─────▶ │   Database    │
│  Next.js   │ ◀──────────────────────────── │   FastAPI    │        │ SQLite / PG   │
└────────────┘                               └──────┬───────┘        └───────▲───────┘
                                                    │ files                  │ claims jobs,
                                                    ▼                        │ writes progress
                                             ┌──────────────┐        ┌───────┴───────┐
                                             │ Job storage  │ ◀───── │    Worker     │
                                             │ (local disk) │        │   pipeline    │
                                             └──────────────┘        └───────────────┘
```

| Part | Location | Responsibility |
|---|---|---|
| Web app | `apps/web` | Landing page, upload, settings, live progress, results, clip studio |
| API | `apps/api/viralcut/api` | Upload validation, job/clip CRUD, media delivery, ZIP export |
| Worker | `apps/api/viralcut/worker` | Claims queued work and runs the pipeline |
| Pipeline | `apps/api/viralcut/pipeline` | The twelve processing stages (below) |
| Queue | `apps/api/viralcut/jobs.py` | Database-backed job queue, progress, logs, retention |

The API and the worker are the same Python package and Docker image started with different
commands, so they can never drift apart.

## Why these choices

**FastAPI + a database-backed queue instead of Redis/Celery.** A job is a row; claiming it is one
atomic `UPDATE … WHERE status = 'queued'`. That gives persistence, retries, cancellation and
multi-worker safety with one moving part fewer, and it works identically on SQLite (development)
and PostgreSQL (production, verified by running the full test suite against PostgreSQL 16).
The queue is isolated behind `jobs.py`; swapping in Redis/Celery later touches only that module
and `worker/main.py`. The trade-off is polling latency (1 s) and no built-in fan-out of a single
job across machines.

**Polling instead of WebSockets/SSE.** The browser asks for job state once per second while work
is running. It survives proxies, reconnects and page reloads with no extra infrastructure.

**ffmpeg for all media work.** Probing, analysis decodes, rendering and QA run as subprocesses with
a protocol whitelist. Each clip is rendered by exactly one ffmpeg invocation (decode → trim →
reframe → concat → captions → encode), so there are no intermediate generations.

**ffmpeg scene detection instead of PySceneDetect.** One low-resolution decode pass yields both the
motion timeline and coarse shot changes; frame-accurate cuts are then refined only inside the
selected clips. This avoids a second full decode and an extra dependency.

**Heuristics first, LLM optional.** The engine must work on a laptop with no paid API. A language
model, when configured, re-rates the shortlisted candidates and writes titles; without it the
system says so (`engine: "heuristic"`, `copy_source: "extractive"`).

## Pipeline stages

Each stage is a module with a narrow interface and its own tests.

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | Video ingestion | `api/routes.py`, `security.py` | Streamed upload, size limit, extension + magic-byte check |
| 2 | Media probing | `pipeline/probe.py` | ffprobe → duration, display size, fps, codecs, HDR flag |
| 3 | Transcription | `pipeline/transcribe.py` | faster-whisper (mode/hardware-driven model), Silero VAD, evidence filter against invented speech, re-transcription of skipped speech, glossary, model fallback chain |
| 3b | Caption sync | `pipeline/align.py` | Per clip: CTC forced alignment (wav2vec2 via onnxruntime, Portuguese and English) of the recognised words, then loudness-envelope refinement of pause boundaries. Whisper timestamps are the fallback |
| 4 | Scene and audio analysis | `pipeline/audio.py`, `pipeline/visual.py` | Loudness envelope, noise floor, silences; motion timeline, shot changes |
| 5 | Candidate generation | `pipeline/candidates.py`, `pipeline/text.py` | Sentence segmentation; every sentence-aligned span in the duration range |
| 6 | Hook analysis | `pipeline/hooks.py` | Seven hook types measured on the first ~3 s |
| 7 | Scoring and ranking | `pipeline/scoring.py`, `pipeline/selection.py`, `pipeline/llm.py` | Six-dimension score, optional LLM blend, temporal + semantic de-duplication |
| 8 | Smart cropping | `pipeline/reframe.py` | YuNet face tracks per shot → crop path, split, blur or fit layout |
| 8b | Editing decisions | `pipeline/editing.py` | Content profile with evidence, punch-in planning on stressed words, opening tightening |
| 9 | Subtitle generation | `pipeline/captions.py` | Pagination, measured layout, ASS (burn-in) and SRT |
| 10 | Rendering | `pipeline/render.py`, `pipeline/trimming.py` | Silence-aware trimming, filter graph (with eased punch-in), two-pass loudness, H.264/AAC MP4 |
| 11 | Quality assurance | `pipeline/qa.py` | Codec, size, fps, duration, A/V sync, faststart, full decode, integrated loudness, true peak, sample rate, SAR |
| 12 | Export | `api/routes.py` | Range-enabled streaming, downloads, SRT, ZIP with publishing notes |

`pipeline/orchestrator.py` runs them in order, reports real progress per stage, and persists the
analysis (`transcript.json`, `signals.npz`) so a clip can be re-rendered from the studio without
re-analysing the video.

## Job lifecycle

```
uploaded ──start──▶ queued ──claim──▶ processing ──▶ completed
                                           │  └─ review_transcript ─▶ review ──resume──▶ queued
                      ▲                   │  ├──▶ failed      (user-facing error, or retries exhausted)
                      └──── retry ────────┘  └──▶ canceled    (cancel flag polled between and inside stages)
```

* **Progress** is written by the worker as it happens: bytes decoded, seconds transcribed,
  candidates scored, frames analysed, ffmpeg `out_time`. Nothing is simulated.
* **Retry**: an unexpected exception re-queues the job until `VIRALCUT_MAX_ATTEMPTS` is reached.
  Errors caused by the input (unreadable file, no renderable clip) fail immediately.
* **Crash recovery**: a processing job whose heartbeat is older than `VIRALCUT_STALE_JOB_MINUTES`
  is re-queued or failed by any worker's maintenance loop.
* **Clip re-renders** from the studio are queued on the clip row (`status = queued`) and served
  before new jobs because they are short and interactive. The previous version stays downloadable
  until the new one passes QA.

## Reframing

For every selected clip the source range is sampled at 5 fps. Faces are detected with YuNet (the
232 KB ONNX model is bundled; a Haar cascade is the fallback and is reported as less accurate),
associated into per-person tracks, and each shot gets one layout:

* `crop` — a 9:16 window following the main face on a low-passed path, reduced to a few linear
  keyframes and rendered as a flat sum of clipped ramps in ffmpeg's `crop` expression. Nearly
  static subjects get a locked-off frame.
* `split` — two stacked 9:8 panels when two persistent faces cannot share one vertical window.
  Captions move to the seam so they never cover the lower speaker.
* `blur` — the whole frame over a blurred, darkened copy of itself when there is nobody to follow.
  It is also the default for the Gaming style, where the action is rarely where the face is.
* `fit` — the source is already vertical.

## Editing decisions

`pipeline/editing.py` never applies a template. For each clip it classifies the content from
measurable signals (vocabulary density with a minimum number of hits, face presence, motion,
speech rate, vocal dynamics), keeps the scores and the matched evidence, and derives the editing
parameters from the result. A punch-in is planned only on a word that is louder than that
speaker's own baseline (or a number / key term with some lift), spaced out, eased with a
smoothstep, kept away from the opening, the ending and hard cuts, and applied only to face-tracked
`crop` layouts. The plan, with each decision's reason, is stored in `clip.framing["edit"]`.

## Captions

Every word is positioned explicitly. Widths are measured with the real font file using the same
size mapping libass uses (`unitsPerEm / (usWinAscent + usWinDescent)`), so lines are balanced,
never leave the platform safe area (a page that would overflow is scaled down, not clipped), and
a word can be recoloured or animated without shifting its neighbours.

## Storage layout

```
data/
├── viralcut.db                  # SQLite (development)
└── jobs/<job id>/
    ├── source.<ext>             # the upload, stored under a fixed name
    ├── analysis/                # transcript.json, signals.npz, meta.json
    └── clips/<clip id>/v<N>.mp4 | .srt | .jpg
```

`storage.py` defines the storage seam. Only the local backend exists today; an S3 backend would
materialise sources into the job directory before processing and publish results afterwards.

## Scaling notes

* Run more workers (`docker compose up --scale worker=N`); claims are atomic.
* API and workers must share the media volume (or an object store, once implemented).
* Transcription dominates CPU time. A CUDA GPU (`VIRALCUT_WHISPER_DEVICE=cuda`) and a larger
  model improve both speed and accuracy.
* The upload rate limiter is in-process memory; put a shared limiter at the edge when running
  several API replicas.
