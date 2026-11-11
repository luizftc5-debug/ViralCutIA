# Security, privacy and data retention

## Threat model

Uploaded files are untrusted. The service must not let a crafted upload read other files, reach
the network, exhaust the host, or expose one user's media to another.

## Controls

| Risk | Control | Where |
|---|---|---|
| Disguised or malicious files | Extension whitelist **and** container magic bytes must agree; checked on the first bytes, before the rest of the body is accepted | `security.py`, `api/routes.py` |
| ffmpeg reading local files or URLs through playlists/concat scripts | `-protocol_whitelist file,pipe` and a forced demuxer (`mov`/`matroska`) on every ffmpeg/ffprobe call that touches an upload | `pipeline/ffmpeg.py`, `pipeline/probe.py` |
| Path traversal | Client file names are never used as paths: sources are stored as `source.<ext>`; ids must match `^[0-9a-f]{32}$`; every path is resolved and checked to stay inside the storage root | `security.py`, `storage.py` |
| Oversized uploads | Streamed to disk with a running byte count (`VIRALCUT_MAX_UPLOAD_MB`); `Content-Length` rejected early; duration limit after probing | `api/routes.py` |
| Resource exhaustion | One job per worker at a time, ffmpeg timeout, container CPU/memory limits, per-IP upload rate limit | `worker/main.py`, `docker-compose.yml` |
| Unsupported or hostile codecs | Only an explicit list of video codecs is accepted | `pipeline/probe.py` |
| Information leaks in errors | Users see fixed messages; exception details go to the job log and server log only | `pipeline/orchestrator.py`, `api/app.py` |
| Metadata leaks in output | Source metadata and chapters are stripped from rendered clips (location, device) | `pipeline/render.py` |
| Privilege | Containers run as an unprivileged user | `apps/api/Dockerfile`, `apps/web/Dockerfile` |
| Secrets | Read from environment only; `.env` is git-ignored; `.env.example` contains no secrets | `config.py` |
| Cross-origin access | CORS limited to configured origins; `Referrer-Policy: no-referrer`; `X-Content-Type-Options: nosniff` | `api/app.py`, `apps/web/next.config.ts` |

## Access model

There are no user accounts yet. A job id is 128 bits from the OS random source and acts as a
capability: whoever holds the link can view, edit and delete that job, and nobody else can list or
guess it. Consequences:

* Treat job links as private. They are never sent in `Referer` headers and pages are not indexed.
* "Recent videos" are remembered in the browser's local storage only.
* Before offering this to the public, add authentication and per-user ownership checks in
  `api/routes.py` (`_get_job` is the single choke point), and serve media through signed URLs.

## Retention

| Data | Kept for | Removed by |
|---|---|---|
| Uploaded source, analysis artifacts, rendered clips, database rows, logs | `VIRALCUT_RETENTION_HOURS` after upload (default 24 h) | Worker maintenance loop (`purge_expired`), every minute |
| Rejected uploads | Not kept | Deleted before the error response is sent |
| Superseded clip versions | Until the new version passes QA | Deleted in the same transaction that publishes the new version |
| Canceled or failed runs | Partial renders are deleted immediately; the source follows the normal retention | `orchestrator.storage_cleanup_partial` |

Users can delete a job and all of its files at any time (`DELETE /api/jobs/{id}`, "Delete now" in
the UI).

## Third parties

* Speech recognition and face detection run locally. No media leaves the server.
* If `ANTHROPIC_API_KEY` is set, **transcript text** of shortlisted candidates is sent to the
  Anthropic API for rating. Audio and video are never sent. Leave the key unset to keep everything
  on your infrastructure.
* Model weights are downloaded from Hugging Face on first use and cached.

## Content rights

ViralCut processes files the user uploads. It does not download media from third-party platforms.
Operators should require users to confirm they own or are licensed to edit what they upload.

## Not implemented yet

Authentication and quotas per user, malware scanning of uploads, encryption at rest, audit logs,
sandboxing ffmpeg beyond container isolation (seccomp/gVisor), and signed media URLs.
