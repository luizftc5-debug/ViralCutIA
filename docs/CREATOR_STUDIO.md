# Creator Studio

The Creator Studio is part of the existing pipeline, not a second editing system. After upload, the
New page offers **A. Quick generation** (AI Automatic preset, one click), **B. Creative presets**
(12) and, in Advanced mode, **C. Advanced editors** (13), plus the content type.

## The 13 editors

| Editor | What it really does | Where | Cost |
|---|---|---|---|
| `/corte` | Removes silences above the pause limit and filler words (um, uh, hã…) on the waveform | trimming | free |
| `/tomada` | When a sentence is said twice in a row (or a false start), keeps the last take | trimming | free |
| `/ganchos` | Starts the clip on a later sentence that measures as a clearly stronger hook (≥ +12 points); optional on-screen opening line | selection, ASS | free |
| `/legenda` | Word-synced captions: 9 styles + Caption Studio | ASS | free |
| `/lettering` | Animated title: pop, typewriter (per character), slide; big rank numbers in rankings | ASS | free |
| `/animacao` | Progress bar (animated clip), accent flashes on stressed words / keywords | ASS | free |
| `/3d` | 2.5D camera: eased push + drift; perspective sway only on *strong* (measured ~0.65 CPU-s per video second extra) | ffmpeg | light / heavy on strong |
| `/trilha` | Your music under the voice: starts on the first strong beat (onset/tempo analysis), fades, sidechain ducking | ffmpeg audio | light |
| `/tela-verde` | Chroma key (auto-detected green/blue at the frame edges) or AI person matting (MODNet, Apache-2.0) over blur / colour / image | ffmpeg (+ 2nd pass for AI) | AI: heavy |
| `/prova` | Your screenshots/evidence as cards at a time or when a keyword is spoken | ffmpeg overlay | light |
| `/texto-atras` | A big word behind the person, positioned from the matte so the head overlaps only its lower part | 2-pass composite | heavy |
| `/referencia` | Measures a reference video (shots/min, pause lengths, voice dynamics, text band) and adapts pauses, punch-ins, cuts, audio treatment and caption band | before editing | light |
| `/enquadramento` | ViralCut's face-tracked reframing, split screen and punch-ins (the 13th editor: the capability this repository already had) | reframe | existing |

Heavy editors are off by default. The matting model (25 MB) downloads on first use, loads only
for jobs that need it, streams frames (flat memory) and follows the resource profile; when it finds
no person (< 3 % of the frame) the effects are skipped with a note instead of replacing the whole
picture. Every editor writes what it did into the clip's decision notes.

## Content types

* **Shorts / TikTok / Reels** — 9:16 with each platform's caption safe area.
* **YouTube long-form** — 16:9 (1920×1080 when the source allows), 3–15 min by default or a custom
  range up to 60 min, chapters from the topic segmentation, punch-ins on full-frame footage.
* **Compilation / ranking** — the selected moments rendered with big rank numbers, joined weakest
  to strongest by stream copy (no extra encode), verified like any clip.

## Growth Toolkit

Titles, description, keywords, hashtags, thumbnail text, per-platform text, follow-up topics (other
strong moments of the same video), versions to compare and, for long-form, chapters. All of it is
quoted or derived from the video's own words. Nothing predicts or guarantees views or revenue; the
app never publishes anything.

## Measured on the test machine (4 vCPU, ECO/BALANCED)

| Run | Result |
|---|---|
| 2 clips, podcast preset + lettering, motion graphics, music, proof card, 3D (ECO) | 104 s, both clips QA-passed, all effects visible in the MP4 |
| Chroma key on a green-screen video | 58 s, key 0x00AF3E detected, background replaced |
| AI matte + text behind subject (1 clip, 23 s) | 88 s, 234 matte frames, person 29 % of frame |
| YouTube long-form of an 8 min video | 323 s, 472 s 1920×1080 edit, 15.7 s of silence removed |
| Top-5 compilation | 121 s, 106.5 s video joined without re-encoding |
| Reference-based edit | 33 s, reference measured and applied |

Limits: `/referencia` measures rhythm, pauses, dynamics and a text band; it does not detect zoom
moves or copy fonts. `/3d` is a camera move, not depth estimation. Music and evidence images must be
supplied by you (none are bundled, to keep licensing clean).
