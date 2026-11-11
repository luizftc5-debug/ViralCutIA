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
| Analysis | faster-whisper transcription with word timestamps (multilingual), loudness/silence analysis, motion and shot-change detection |
| Selection | Sentence-aligned candidates, Hook Detection Engine (7 hook types), six-dimension Viral Potential Score with an explanation, temporal and semantic de-duplication, minimum-score cut-off |
| Optional LLM | With an Anthropic API key, Claude re-rates shortlisted moments and writes titles/captions grounded in the transcript. Without it the engine is heuristic and says so |
| Reframing | YuNet face tracking per shot, smoothed crop path, split screen for two speakers, blurred-backdrop layout, vertical sources kept as they are |
| Trimming | Shortens pauses only where the audio is actually silent; keeps a beat after questions |
| Captions | Four styles (High Retention, Clean Editorial, Dynamic Creator, Gaming), per-word timing, keyword highlight, size/position/colour/typeface, platform safe areas, accents, burned in or off, SRT sidecar |
| Rendering | One ffmpeg pass per clip: H.264 High + AAC, 1080×1920 (720×1280 for low-resolution sources), source frame rate up to 60 fps, faststart, loudness normalised |
| QA | Codec, resolution, orientation, frame rate, duration, A/V sync, faststart, full decode and non-silent audio verified for every file |
| Studio | Preview, score breakdown, move start/end, fix caption text, change caption style and framing, re-render, download MP4/SRT, export all as ZIP |
| Jobs | Persistent queue with real per-stage progress, logs, cancel, controlled retry, crash recovery, automatic deletion after the retention period |

## Quick start

### Docker

```bash
cp .env.example .env
```

```bash
docker compose up --build
```

Open http://localhost:3000. The first job downloads the speech model (about 480 MB for `small`).

### Without Docker

Requirements: Python 3.11+, Node.js 20+, and `ffmpeg`/`ffprobe` on the PATH (built with libx264
and libass, as in the Debian, Ubuntu and Homebrew packages).

```bash
./scripts/dev.sh
```

That creates the Python environment, installs both apps, and starts the API with an embedded
worker on port 8000 and the web app on port 3000. To check the prerequisites first:

```bash
python3 scripts/doctor.py
```

#### Windows (PowerShell, no Docker)

1. Install [Python 3.12](https://www.python.org/downloads/) (tick "Add python.exe to PATH") and
   [Node.js LTS](https://nodejs.org/).
2. Install a **full** ffmpeg build that includes libx264 and libass, for example from
   [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (`ffmpeg-release-full.7z`, or
   `winget install Gyan.FFmpeg`). Add its `bin` folder to PATH and open a new PowerShell window.
3. From the repository folder:

```powershell
copy .env.example .env
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
```

The script checks everything with `scripts\doctor.py`, creates `apps\api\.venv` and installs the
dependencies on the first run, then opens http://localhost:3000. If `.venv` ends up broken, delete
the `apps\api\.venv` folder and run the script again.

To run the pieces yourself:

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
| `VIRALCUT_WHISPER_MODEL` | `small` | Speech model. `medium` / `large-v3` are more accurate and slower; use a GPU with `VIRALCUT_WHISPER_DEVICE=cuda` |
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
| `test_captions.py` | Pagination, safe-area layout, accents, options, SRT, text edits |
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
* **Transcription quality follows the model.** The default `small` model makes mistakes,
  especially on names, jargon and noisy audio; the studio lets you correct captions. One language
  is assumed per video. There is no speaker diarisation.
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
* The interface is in English only.

## Roadmap

1. Accounts, ownership and quotas; signed media URLs; S3 storage backend.
2. Collect published-clip metrics and calibrate the score weights; A/B hook types, caption styles
   and durations.
3. Active-speaker detection and diarisation for multi-person framing and speaker-coloured captions.
4. Saliency-based framing for gameplay and screen recordings.
5. WhisperX-style forced alignment and larger models on GPU workers.
6. More aspect ratios (1:1, 4:5), brand kits (fonts, colours, logo), caption templates.
7. Direct publishing to platforms through their official APIs.
8. Interface localisation, starting with Portuguese.

## Licences

Code: [MIT](LICENSE). Bundled assets keep their own licences, included in
`apps/api/viralcut/assets/licenses/`: Inter, Poppins, Anton and DM Serif Display (SIL Open Font
License) and the YuNet face detector (MIT). Speech recognition uses faster-whisper (MIT) with
OpenAI Whisper weights (MIT), downloaded at first use.

Process only video you own or are licensed to edit.
