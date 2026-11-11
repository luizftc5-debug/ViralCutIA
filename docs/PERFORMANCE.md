# Performance: what was measured, what changed

All numbers come from `scripts/benchmark_pipeline.py` (fresh process, empty data directory,
whole process tree sampled every 0.25 s with psutil; CPU time from `getrusage`, exact). Machine:
4 vCPU / 16 GB container, no GPU, ffmpeg 6.1, Whisper `large-v3-turbo` int8 (the `auto` mode on 4+
cores). "Before" is commit `f0dd600` run from a separate worktree under the same conditions;
"after" is the BALANCED profile (default) unless noted. Synthetic test videos made with
`scripts/make_test_video.py` (flite speech, face image, cuts) plus a 60 fps high-motion clip.

Single runs: differences under ~5 % are within run-to-run noise.

## Results

| Scenario | Before: wall / CPU / peak RAM | After: wall / CPU / peak RAM | Change |
|---|---|---|---|
| Horizontal 2 min, 3 clips, captions | 86.2 s / 223 s / 3.71 GB | 74.7 s / 193 s / 3.39 GB | −13 % time, −13 % CPU, −0.3 GB |
| Same video processed again (other options, retry) | ~86 s (everything repeated) | 42.3 s / 104 s / 1.43 GB | −51 % time: transcript and signals reused |
| Studio re-render after a caption-style change | 19.2 s / 43 s / 1.57 GB | 12.8 s / 29 s / 0.67 GB | −33 % time, −57 % RAM |
| Long 8 min, 6 clips | 228.5 s / 592 s / 3.71 GB | 228.6 s / 584 s / 3.47 GB | same time, −0.24 GB |
| Long 8 min, processed again | 230.2 s | 122.1 s | −47 % |
| Vertical 37 s, 2 clips | 30.5 s / 80 s / 2.82 GB | 29.4 s / 76 s / 2.63 GB | −4 % (noise level), −0.2 GB |
| Gaming 2 min 60 fps, 2 clips | 146.9 s / 463 s / 3.64 GB | 152.0 s / 443 s / 3.32 GB | +3 % time (noise level), −4 % CPU, −0.3 GB |
| Horizontal 2 min, no burned captions | 78.8 s / 203 s / 3.66 GB | 81.4 s / 187 s / 3.30 GB¹ | ≈ same, −8 % CPU |

¹ measured with the first version of BALANCED (n−1 threads), see below.

Profiles on the horizontal 2 min video (same job):

| Profile | Wall | CPU | Avg cores used | Peak RAM |
|---|---|---|---|---|
| ECO | 96.7 s | 188 s | 1.95 | 2.76 GB (Whisper released before rendering) |
| BALANCED | 74.7 s | 193 s | 2.58 | 3.39 GB |
| TURBO (2 renders at a time) | 69.2 s | 190 s | 2.75 | 3.78 GB |

Output quality is identical across profiles: same CRF 18, same x264 preset, same resolution,
frame rate and loudness target; every file passes the same QA checks.

## What made the difference (and what did not)

| Change | Measured effect |
|---|---|
| Content-addressed cache of transcript + audio/visual signals | a second job on the same file skips decoding and recognition (−47 to −51 %) |
| Per-clip memo of alignment, face tracking and loudness pass 1 | re-render 19.2 → 12.8 s; the aligner is never loaded for a caption-only edit (RAM 1.57 → 0.67 GB) |
| ONNX Runtime memory arena off for the aligner | memory kept after inference 1012 → 604 MB; inference +7 % |
| Audio decoded into one preallocated buffer | one copy of the audio instead of three (≈0.7 GB less per hour of source at peak) |
| Two renders at a time (TURBO) | render stage 46.4 → 36.7 s (−21 %) for +0.6 GB RAM on 4 cores |
| BALANCED capped at n−1 threads (first attempt) | **rejected**: +20 % wall time on the 60 fps render, no CPU saved. Now all cores at lowered priority (nice 5) |
| x264 `faster` instead of `medium` | **rejected**: −5 % CPU but SSIM 0.99885 → 0.99857 and different file size; `medium` kept |
| x264 `slow` | +14 % CPU, SSIM 0.99890, −7 % size: used only for "Maximum quality" |
| Colour correction before upscaling in the blurred backdrop | **rejected**: no difference beyond noise |

Transcription (~40 s per 2 min of speech on 4 CPU cores) and rendering are the floor of a first
run; a CUDA GPU is the way to lower it (`VIRALCUT_WHISPER_DEVICE=cuda`, FP16 or INT8+FP16 by free
VRAM). GPU paths and hardware encoders were not measured here (no GPU in the test machine), so the
hardware encoder stays off by default.

## Reproduce

```bash
apps/api/.venv/bin/python scripts/benchmark_pipeline.py video.mp4 --clips 3 --rerender --repeat-job --label mine --out report.json
```

```bash
VIRALCUT_PERFORMANCE_PROFILE=eco apps/api/.venv/bin/python scripts/benchmark_pipeline.py video.mp4 --clips 3
```
