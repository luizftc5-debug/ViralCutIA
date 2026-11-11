# HTTP API

Base URL: `http://localhost:8000`. Interactive OpenAPI docs: `/docs`. All bodies are JSON except
the upload. Errors are `{"detail": "message safe to show to the user"}`.

Job and clip ids are 32 lowercase hex characters. A job id is an unguessable capability: holding
it is what grants access (see [SECURITY.md](SECURITY.md)).

## Flow

```
POST /api/jobs            upload the file            → job (status "uploaded")
POST /api/jobs/{id}/start choose options, enqueue    → job (status "queued")
GET  /api/jobs/{id}       poll once per second       → stages, progress, clips
GET  …/clips/{cid}/video  preview / download         → MP4 (Range supported)
PATCH …/clips/{cid}       studio edit                → clip (status "queued"), poll until "ready"
```

## Meta

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Liveness: `{"status":"ok","version":"0.1.0"}` |
| GET | `/api/config` | Upload limits, caption styles and fonts, and which engines are active |

`engine.narrative_analysis` is `"heuristic"` or `"llm:<model>"`, so a client can tell users which
analysis they are getting before they upload.

## Upload

`POST /api/jobs?filename=<original name>` with the **raw file as the request body** (not
multipart). The body is streamed to disk.

```bash
curl -X POST --data-binary @talk.mp4 "http://localhost:8000/api/jobs?filename=talk.mp4"
```

| Status | Meaning |
|---|---|
| 201 | Stored and probed. Body: the job, with `media` (duration, size, fps, codecs) and `expires_at` |
| 413 | Larger than `VIRALCUT_MAX_UPLOAD_MB` |
| 422 | Wrong extension, content does not match the container, unreadable/unsupported video, too long or too short |
| 429 | Per-IP upload rate limit |

Nothing is kept from a rejected upload.

## Start processing

`POST /api/jobs/{id}/start` — every field is optional.

```json
{
  "captions": {
    "enabled": true,
    "style": "high_retention",
    "highlight_keywords": true,
    "size": "m",
    "position": "bottom",
    "color": "",
    "font": "auto",
    "animation": true
  },
  "language": "auto",
  "clip_count": "auto",
  "duration": "auto",
  "platform": "universal",
  "aspect": "9:16",
  "style": "auto",
  "framing": "auto",
  "min_score": null
}
```

| Field | Values |
|---|---|
| `captions.style` | `high_retention`, `clean_editorial`, `dynamic_creator`, `gaming` |
| `captions.size` / `position` | `s` `m` `l` / `top` `middle` `bottom` |
| `captions.color` | `#RRGGBB` accent, or empty for the style default |
| `captions.font` | `auto`, `inter`, `poppins`, `anton`, `dm_serif` |
| `language` | `auto` or a language code (`en`, `pt`, `es`, …) |
| `clip_count` | `"auto"` or 1–20 (a ceiling, not a quota) |
| `duration` | `auto` (18–75 s), `15-30`, `30-60`, `60-90` |
| `platform` | `universal`, `shorts`, `reels`, `tiktok` (caption safe areas) |
| `style` | `auto`, `dynamic`, `editorial`, `podcast`, `gaming` (pause trimming and default framing) |
| `framing` | `auto`, `face`, `split`, `blur`, `center` |
| `min_score` | 0–100, or `null` for the server default |

Returns 409 while the job is queued or processing. Calling it on a finished, failed or canceled
job discards its clips and runs it again with the new options.

## Job

`GET /api/jobs/{id}`

```json
{
  "id": "fc845b31897b37648baa46620fa70bdf",
  "status": "processing",
  "progress": 0.41,
  "stages": [
    {"key": "analyzing", "label": "Analyzing video", "status": "done", "progress": 1.0},
    {"key": "transcribing", "label": "Transcribing speech", "status": "active", "progress": 0.62},
    {"key": "discovering", "label": "Discovering high-impact moments", "status": "pending", "progress": 0.0}
  ],
  "error": "",
  "filename": "talk.mp4",
  "media": {"duration": 122.0, "width": 1920, "height": 1080, "fps": "30/1", "has_audio": true},
  "analysis": {
    "engine": "heuristic", "mode": "speech", "language": "en",
    "transcription": "faster-whisper:small", "words": 354, "candidates": 263,
    "requested": 4, "selected": 4, "min_score": 60.0, "best_candidate_score": 85.1,
    "llm": {"used": false}, "warnings": []
  },
  "expires_at": "2026-11-12T03:21:40Z",
  "clips": []
}
```

`status`: `uploaded`, `queued`, `processing`, `completed`, `failed`, `canceled`.
Stage `status`: `pending`, `active`, `done`, `skipped` (for example subtitles when there is no speech).
`analysis.warnings` lists every degradation that affected this job in plain language.

| Method | Path | Description |
|---|---|---|
| GET | `/api/jobs/{id}/logs?after=<log id>` | Processing log, incrementally |
| POST | `/api/jobs/{id}/cancel` | Cancel a queued job immediately, or ask a running one to stop |
| DELETE | `/api/jobs/{id}` | Delete the job and every file (409 while processing) |
| GET | `/api/jobs/{id}/export.zip` | All ready clips (MP4 + SRT), `clips.json` and `publishing-notes.txt` |

## Clips

Each entry of `clips` (and `GET /api/jobs/{id}/clips/{cid}`, which adds `transcript`, `pages` and
`source_duration`):

```json
{
  "id": "…", "rank": 1, "status": "ready",
  "start": 0.0, "end": 23.2, "duration": 23.2,
  "score": 85.1, "score_stale": false, "engine": "heuristic",
  "dimensions": [{"key": "hook", "label": "Hook strength", "value": 90, "weight": 0.25}],
  "explanation": "Strong opening (curiosity gap), clear narrative payoff, …",
  "hook": {"text": "Why do most people quit learning a new skill after only 2 weeks?",
           "types": [{"type": "curiosity", "label": "curiosity gap", "strength": 1.0}]},
  "titles": ["Why do most people quit learning a new skill after only 2 weeks?"],
  "caption": "…", "copy_source": "extractive", "language": "en",
  "layouts": ["crop"], "has_captions_burned": true, "has_srt": true,
  "qa": {"passed": true, "width": 1080, "height": 1920, "fps": 30.0, "size_bytes": 5600000,
         "checks": {"video_codec_h264": true, "av_sync": true, "faststart": true}},
  "version": 1,
  "urls": {"video": "/api/…/video?v=1", "download": "/api/…/download?v=1",
           "srt": "/api/…/srt?v=1", "thumbnail": "/api/…/thumbnail?v=1"}
}
```

`urls` is `null` until a file has passed quality assurance: a clip is never offered before it
exists and has been verified. `copy_source` is `extractive` (quoted from the transcript), `llm`
or `none`.

| Method | Path | Description |
|---|---|---|
| GET | `…/clips/{cid}/video` | MP4 for playback (HTTP Range) |
| GET | `…/clips/{cid}/download` | MP4 as an attachment with a readable file name |
| GET | `…/clips/{cid}/srt` | Subtitles (404 when the clip has no speech) |
| GET | `…/clips/{cid}/thumbnail` | JPEG poster frame |
| PATCH | `…/clips/{cid}` | Edit and queue a re-render |

### Editing a clip

```json
{
  "start": 12.4,
  "end": 41.0,
  "captions": {"enabled": true, "style": "clean_editorial", "position": "top"},
  "framing_mode": "blur",
  "smart_trim": false,
  "page_edits": [{"index": 0, "text": "Corrected caption text"}]
}
```

All fields are optional. `page_edits[].index` refers to `pages` of the version currently
rendered; an empty `text` removes that caption. A clip must stay between 3 and 180 seconds. The
response has `status: "queued"`; poll the clip until it is `ready` (its `version` increases) or
shows an `error`. If the new render fails, the previous version is kept.

## Performance feedback

`POST /api/jobs/{id}/clips/{cid}/metrics` records what happened after publishing, for future
calibration of the score (see [SCORING.md](SCORING.md)).

```json
{"platform": "shorts", "views": 12000, "avg_percent_watched": 71.5, "retention_3s": 82.0,
 "likes": 640, "shares": 31, "hours_since_publish": 48}
```

`GET` on the same path lists what was reported. Nothing is ever generated by the server.
