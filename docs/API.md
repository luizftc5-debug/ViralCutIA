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
| GET | `/api/config` | Upload limits, caption styles and fonts, video styles, the processing server's hardware and resource profiles, and which engines are active |
| POST | `/api/caption-preview` | JPEG of a caption page rendered by the export engine (same layout code, same libass). Body: `{"captions": {...}, "platform": "tiktok", "text": "", "job_id": null, "clip_id": null}`; with a job/clip the background is a frame of that video. 429 when two previews are already rendering |

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
| `captions.style` | `auto` (the caption design of the clip's video style), `viral_dynamic` (Bold Dynamic), `energy` (High Energy), `podcast` (Podcast Premium), `kinetic` (Kinetic Typography), `clean_editorial` (Minimal Editorial), `gaming` (Gaming Impact), `cinematic` (Cinematic), `high_retention` (Classic Viral), `dynamic_creator` (Highlight Box) |
| `captions.size` / `position` | `s` `m` `l` `xl` / `top` `middle` `bottom` |
| `captions.color` | `#RRGGBB` highlight colour of the spoken word, or empty for the style default |
| Caption Studio | `text_color`, `keyword_color` (`#RRGGBB` or empty), `weight` (`auto` `medium` `bold` `heavy`), `case` (`auto` `upper` `normal`), `words_per_page` (1–8), `max_lines` (1–3), `line_spacing` (0.8–1.6), `outline` (0–16), `shadow` (0–12), `background` (`auto` `none` `box`), `animation_type` (`auto` `color` `pop` `box` `reveal` `none`), `animation_intensity` (`subtle` `normal` `strong`), `keyword_mode` (`auto` `color` `color_scale` `off`), `offset` (−0.2…0.2 of the frame height, clamped to the safe area). `null`/empty/`auto` = the style's value |
| `captions.font` | `auto`, `inter`, `poppins`, `anton`, `dm_serif` |
| `language` | `auto` or a language code (`en`, `pt`, `es`, …) |
| `clip_count` | `"auto"` or 1–20 (a ceiling, not a quota) |
| `duration` | `auto` (18–75 s), `15-30`, `30-60`, `60-90`, `custom` (with `custom_min` / `custom_max`, 5–180 s, at least 5 s apart) |
| `platform` | `universal`, `shorts`, `reels`, `tiktok` (caption safe areas) |
| `style` | Video style: `auto`, `viral`, `podcast`, `gaming`, `cinematic`, `educational`, `reaction`, `minimal`, `storytime`, `news` (legacy `dynamic` = `viral`, `editorial` = `minimal`). Sets pacing, punch-ins, jump-cut handling, default framing and caption design, and speech treatment (see `pipeline/styles.py`) |
| `intensity` | `subtle`, `balanced` (default), `high` |
| `variations` | 1–3 versions per selected clip (hook-first, alternate style, alternate pacing), from the same analysis |
| `quality` | `high` (CRF 18, profile preset) or `max` (CRF 16, `slow`) |
| `performance` | `auto` (server default), `eco`, `balanced`, `turbo`: resource profile of the worker for this job |
| `framing` | `auto`, `face` (full-screen vertical), `speaker` (speaker focus), `split`, `blur`, `center` |
| `smart_zoom` | `true` (default): eased punch-ins on stressed words, only on face-tracked clips |
| `review_transcript` | `false` (default). `true` pauses the job in status `review` after clip selection, until `POST /jobs/{id}/resume` |
| `glossary` | Comma-separated names and terms the recogniser should prefer (max 400 characters) |
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

## Transcript review

A transcript can be read and corrected while a job is in status `review` (see `review_transcript`)
or `completed`. Corrections change **text only**: every word keeps the time it was spoken at, and the
recogniser's original output is kept (`transcript.original.json`).

| Method | Path | Description |
|---|---|---|
| GET | `/api/jobs/{id}/transcript` | `{language, engine, words:[{index,text,start,end,confidence}], quality, edited}`. `quality` lists the segments that were discarded as non-speech and the stretches that were re-transcribed |
| PUT | `/api/jobs/{id}/transcript` | `{words:[{index,text}], replace:[{find,replace,case_sensitive}]}`. Whole-word find and replace, or by index. On a completed job the clips that contain corrected words are queued for re-rendering; in `review` nothing renders yet. Returns `{changed, affected_clips, words}` |
| POST | `/api/jobs/{id}/resume` | Continue a job paused for review (409 otherwise): the clips render with the corrected words |

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
| GET | `…/clips/{cid}` | Detail. Besides the score and pages it includes `alignment` (how word times were obtained: method, aligned words, median correction) and `qa` (including `integrated_lufs` and `true_peak_db`) |
| PATCH | `…/clips/{cid}` | Edit and queue a re-render |

### Editing a clip

```json
{
  "start": 12.4,
  "end": 41.0,
  "captions": {"enabled": true, "style": "clean_editorial", "position": "top"},
  "framing_mode": "blur",
  "smart_trim": false,
  "style": "cinematic",
  "intensity": "subtle",
  "quality": "max",
  "page_edits": [{"index": 0, "text": "Corrected caption text"}]
}
```

All fields are optional. A clip also returns `applied_style` (the concrete style used and why),
`recommendation` (`{style, label, intensity, reasons}` from the clip's measured signals) and, for an
extra version, `variant` (`{index, kind, label, why, of}`). Edits that leave the in/out points and
words alone (captions, style, quality) reuse the clip's cached alignment, face tracking and
loudness measurement. `page_edits[].index` refers to `pages` of the version currently
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
