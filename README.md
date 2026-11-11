# ViralCut AI

Turn one long video into ranked, vertical short-form clips: selected by content analysis, reframed
to 9:16, captioned word by word and rendered as MP4 files ready for YouTube Shorts, Instagram
Reels and TikTok.

**Upload once. Discover your best moments. Publish everywhere.**

This is a working application, not a mock-up: every score comes from analysing the uploaded
video, and a clip is only shown once its file exists and has passed automated quality checks.
No system can guarantee virality, and this one does not claim to: see
[what the score is and is not](docs/SCORING.md).

## What it does

| Area | Implemented |
|---|---|
| Upload | Drag and drop, MP4/MOV/MKV/WebM, streamed with real progress, validated by content (not just extension), size and duration limits |
| Analysis | faster-whisper transcription (`fast` / `balanced` / `accurate` modes, `large-v3-turbo` by default on 4+ cores) with an evidence filter that discards music/noise hallucinations and re-transcribes speech Whisper skipped, CTC forced alignment of word times (Portuguese, English), loudness/silence analysis, motion and shot-change detection. [Measured results](docs/TRANSCRIPTION.md) |
| Selection | Sentence-aligned candidates, Hook Detection Engine (7 hook types), six-dimension Viral Potential Score with an explanation, temporal and semantic de-duplication, minimum-score cut-off |
| Optional LLM | With an Anthropic API key, Claude re-rates shortlisted moments and writes titles/captions grounded in the transcript. Without it the engine is heuristic and says so |
| Reframing | YuNet face tracking per shot, smoothed crop path, split screen for two speakers, blurred-backdrop layout, vertical sources kept as they are |
| Editing decisions | Content profile from evidence (podcast, gaming, education, entertainment, storytelling), eased punch-in zoom on words the speaker really stresses (face-tracked clips only), pause trimming per profile, leading disfluencies dropped. Every decision is stored with its reason |
| Trimming | Shortens pauses only where the audio is actually silent; keeps a beat after questions |
| Captions | Five presets (**Viral Dynamic**, **Creator High Retention**, **Premium Minimal**, **Gaming / Entertainment**, Dynamic Creator), aligned per-word timing, captions leave with the speech, keyword emphasis, size/position/colour/typeface, platform safe areas (verified by rasterising the worst case), captions placed in the blurred band of letterboxed footage, accents, burned in or off (`Add subtitles? yes/no`), SRT sidecar |
| Rendering | One ffmpeg pass per clip: H.264 High + AAC, 1080×1920 (720×1280 for low-resolution sources; the native source crop is reported, never presented as higher resolution), source frame rate up to 60 fps, BT.709 signalled, VBV-capped, faststart, **two-pass EBU R128 loudness to −14 LUFS / −1.5 dBTP** with only the speech treatments the measurements justify |
| QA | Codec, resolution, orientation, frame rate, duration, A/V sync, faststart, full decode, non-silent audio, **integrated loudness, true peak, 48 kHz stereo and square pixels** verified for every file with ffprobe/ffmpeg |
| Studio | Preview (the rendered MP4, so preview and export are identical), score breakdown, move start/end, fix caption text, change caption style and framing, re-render, download MP4/SRT, export all as ZIP. **Optional transcript review before rendering**: pause after the clips are chosen, correct words everywhere (find and replace), continue |
| Jobs | Persistent queue with real per-stage progress, logs, cancel, controlled retry, crash recovery, automatic deletion after the retention period |

## Quick start

### Docker

```bash
cp .env.example .env
```

```bash
docker compose up --build
```

Open http://localhost:3000. The first job downloads the speech model (about 1.6 GB for the default `large-v3-turbo`, 480 MB for `small`) and the two alignment models (about 240 MB for Portuguese, 380 MB for English).

### Without Docker

Requirements: Python 3.11+, Node.js 20+, and `ffmpeg`/`ffprobe` on the PATH (built with libx264
and libass, as in the Debian, Ubuntu and Homebrew packages).

```bash
./scripts/dev.sh
```

That creates the Python environment, installs both apps, and starts the API with an embedded
worker on port 8000 and the web app on port 3000. To run the pieces yourself:

```bash
cd apps/api && python3 -m venv .venv && .venv/bin/pip install -e ".[dev,llm]"
```

```bash
cd apps/api && .venv/bin/uvicorn viralcut.api.app:app --port 8000
```

```bash
cd apps/api && .venv/bin/viralcut-worker
```

```bash
cd apps/web && npm install && npm run dev
```

### Command line

Process a file without the web app and copy the clips to a folder:

```bash
cd apps/api && .venv/bin/viralcut-cli /path/to/video.mp4 --clips 5 --style high_retention --out ./clips
```

### A test video with no licensing questions

```bash
python scripts/make_test_video.py --out sample.mp4
```

This synthesises a two-minute talk (ffmpeg's built-in voice) that mixes strong moments with
filler. Pass `--face portrait.png` to composite a face and exercise tracking and split-screen,
or `--script text.txt --tts "<command>"` to use your own script and TTS voice.

## Configuration

All settings are environment variables, documented in [.env.example](.env.example). The ones that
matter most:

| Variable | Default | Effect |
|---|---|---|
| `VIRALCUT_TRANSCRIPTION_MODE` | `auto` | `auto`: `large-v3` on a GPU, `large-v3-turbo` on a CPU with 4+ cores, `small` below that. `fast` / `balanced` / `accurate` force a mode |
| `VIRALCUT_WHISPER_MODEL` | empty | An explicit model (`tiny` … `large-v3`, `large-v3-turbo`) always wins over the mode |
| `VIRALCUT_ALIGN_MODE` | `auto` | CTC forced alignment of word timestamps for Portuguese and English (`off` keeps Whisper's timestamps) |
| `ANTHROPIC_API_KEY` | unset | Enables LLM narrative rating and written titles. Only transcript text is sent |
| `VIRALCUT_MIN_SCORE` | `60` | Clips below this score are not rendered |
| `VIRALCUT_SCORE_WEIGHTS` | built in | JSON override of the six score weights |
| `VIRALCUT_RETENTION_HOURS` | `24` | When uploads and clips are deleted |
| `VIRALCUT_MAX_UPLOAD_MB` / `VIRALCUT_MAX_DURATION_S` | `2048` / `10800` | Upload limits |
| `VIRALCUT_DATABASE_URL` | SQLite file | Set to PostgreSQL in production (Compose does this) |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | API address as seen from the browser |

## Repository layout

```
apps/
├── api/                  Python package `viralcut`: FastAPI service, worker, pipeline, CLI
│   ├── viralcut/api/         HTTP routes and serializers
│   ├── viralcut/pipeline/    one module per processing stage
│   ├── viralcut/worker/      queue consumer
│   ├── viralcut/assets/      bundled caption fonts (OFL) and the YuNet face model (MIT)
│   └── tests/                unit, API and integration tests
└── web/                  Next.js app (landing page, studio, results, clip editor)
docs/                     architecture, API, scoring, security
scripts/                  dev runner, test-video generator, caption preview renderer
docker-compose.yml        PostgreSQL + API + worker + web
```

The suggested `workers/` and `packages/shared` directories were folded into `apps/api`: the API
and the worker are one package and one image started with different commands, which removes a
shared-library build step without changing how they scale. Reasons for the other technology
choices are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development

```bash
make test
```

```bash
make lint
```

| Suite | What it covers |
|---|---|
| `apps/api/tests/test_security.py` | File-name sanitising, container sniffing, path traversal |
| `test_analysis.py` | Sentences, hooks (English and Portuguese), scoring, topic unity, de-duplication, extractive copy |
| `test_captions.py`, `test_caption_quality.py` | Pagination, accents, options, SRT, text edits; **timing (no caption outlives its speech), every word shown exactly once, and rasterised safe-area geometry for every preset and position** |
| `test_evaluation.py`, `test_transcribe_guard.py`, `test_align.py` | WER metric, the no-fabrication rules (with the real measured hallucination), recognition modes, CTC alignment against exact synthetic ground truth |
| `test_editing.py`, `test_render_polish.py`, `test_zoom_on_a_face.py` | Content profile and zoom planning, **measured** punch-in (image registration on a real face), two-pass loudness and true peak, colour signalling |
| `test_transcript_review.py` | Manual correction keeps timings and the original, re-render and resume behaviour |
| `test_speech_e2e.py` | **Real human speech (FLEURS, Portuguese and English) end to end**: WER against the reference, forced alignment, burned captions vs. the audio's own voice activity, captions-off render is clean, review flow |
| `test_media.py` | Trimming, reframing layouts, filter graph, **real renders and QA** on generated media |
| `test_llm.py` | LLM blending and failure fallback with a stub client |
| `test_api.py` | Upload validation, full job flow, edit and re-render, cancel, retry, retention |
| `test_pipeline_integration.py` | End to end with real speech recognition (downloads the Whisper model) |
| `apps/web` | ESLint, TypeScript, unit tests, production build |

CI (`.github/workflows/ci.yml`) runs all of the above on every push and pull request.

## Documentation

* [Architecture](docs/ARCHITECTURE.md): components, pipeline stages, job lifecycle, trade-offs
* [HTTP API](docs/API.md)
* [Viral Potential Score](docs/SCORING.md): dimensions, hook detection, calibration plan
* [Security, privacy and retention](docs/SECURITY.md)

## Known limitations

These are real constraints of this version, stated so nobody is surprised by them.

* **The score is heuristic.** Its weights are product hypotheses and have not been calibrated
  against real performance data. Text heuristics cover English and Portuguese; other languages
  rely on language-agnostic signals unless an LLM key is configured.
* **The LLM path has not been run against the live Anthropic API** in this repository's test
  runs (no key was available). It is covered by tests with a stub client only.
* **Transcription quality follows the model and the audio.** Measured Word Error Rates (clean read speech, music, noise, fast speech) are in [docs/TRANSCRIPTION.md](docs/TRANSCRIPTION.md); noisy recordings (speech only 5 dB above broadband noise) are still hard (tens of percent WER in English with the best model measured). Names and jargon benefit from the glossary and the transcript review. One language is assumed per video. There is no speaker diarisation.
* **Reframing follows faces.** There is no active-speaker detection, object or gameplay
  saliency model. Footage without faces uses the blurred-backdrop layout; choose centre crop in
  the studio if you prefer.
* **No laughter, applause or facial-expression models.** Emotional impact is estimated from
  vocabulary, vocal dynamics and motion.
* **HDR sources** go through a basic tone-mapping step that has not been tested on real HDR
  footage.
* **No accounts.** Job links are private capability URLs; see the security document before
  exposing an instance publicly.
* **Storage is local disk.** The storage interface is ready for an S3 backend, which is not
  implemented.
* **Docker images and the Compose stack were written but not built or run** in the environment
  where this was developed (no Docker daemon). The application itself was tested natively on
  SQLite and on PostgreSQL 16.
* **CPU speed.** Processing is CPU-bound without a GPU; transcription dominates on long videos.
  Measured on 4 CPU cores with the `small` model: a 2-minute 1080p video produced 4 clips in
  about 1.5 minutes, and a 10-minute 1080p video produced 8 clips in 4 min 42 s.
* **Not implemented** (stated so nobody assumes otherwise): reordering a clip to open with a later line (a "cold open"; it can mislead, so the engine only removes lead-in), active-speaker tracking, a gameplay saliency model, background-music ducking (the app adds no music, so there is nothing to duck).
* **Verified with synthetic and read-speech material**, not with a labelled set of real creator videos: the content classifier, the zoom thresholds and the denoise trigger are heuristics whose decisions are explained per clip but whose *taste* is unmeasured.
* The interface is in English only.

## Roadmap

1. Accounts, ownership and quotas; signed media URLs; S3 storage backend.
2. Collect published-clip metrics and calibrate the score weights; A/B hook types, caption styles
   and durations.
3. Active-speaker detection and diarisation for multi-person framing and speaker-coloured captions.
4. Saliency-based framing for gameplay and screen recordings.
5. Forced alignment for more languages (a permissive CTC model per language) and GPU workers.
6. More aspect ratios (1:1, 4:5), brand kits (fonts, colours, logo), caption templates.
7. Direct publishing to platforms through their official APIs.
8. Interface localisation, starting with Portuguese.

## Licences

Code: [MIT](LICENSE). Bundled assets keep their own licences, included in
`apps/api/viralcut/assets/licenses/`: Inter, Poppins, Anton and DM Serif Display (SIL Open Font
License) and the YuNet face detector (MIT). Speech recognition uses faster-whisper (MIT) with
OpenAI Whisper weights (MIT), downloaded at first use.

Process only video you own or are licensed to edit.
