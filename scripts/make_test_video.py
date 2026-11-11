#!/usr/bin/env python3
"""Generate a synthetic, licence-clean test video for the ViralCut pipeline.

Speech is synthesised with ffmpeg's built-in ``flite`` voice (or any TTS command you pass with
``--tts``), so no third-party footage is needed. The script deliberately mixes strong moments
(clear hook, development, payoff) with filler, and varies the picture — one moving face, two
faces, no face, hard cuts — to exercise ranking, reframing and trimming.

    python scripts/make_test_video.py --out sample.mp4 [--face portrait.png] [--size 1920x1080]

``--face`` is any image containing a face (for example scikit-image's public-domain
``astronaut`` picture); without it the video has no faces and reframing falls back to the
blurred-background layout.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import tempfile
from pathlib import Path

SEGMENTS_EN = [
    # (visual, voice, text)
    ("face_pan", "slt",
     "Why do most people quit learning a new skill after only two weeks? It is not because they lack talent. "
     "It is because they practice the wrong way. Most people repeat what they already know, because it feels good. "
     "But real progress only happens at the edge of your ability. So here is the rule. "
     "Practice the hardest part first, every single day, for twenty minutes. That is why top performers improve so fast."),
    ("noface", "slt",
     "Okay, so, before we continue, let me just check my notes here. And, yeah, I think we covered that. "
     "So we can also talk about the schedule later. It was fine. And then we had lunch, and it was okay."),
    ("face_right", "rms",
     "Three years ago, I lost forty thousand dollars in a single afternoon. I had put all my savings into one business idea, "
     "and I never tested it with real customers. The day we launched, nobody bought anything. Not one person. I was terrified. "
     "But that failure taught me the most important lesson of my career. Never build anything before you talk to ten customers. "
     "Today, every product I launch starts with ten conversations, and it changed everything."),
    ("noface", "rms",
     "So anyway, the next slide shows the agenda, and then we also have some other items. "
     "He said that it was on the second page. And they were also looking at that. So let's move on."),
    ("two_faces", "slt",
     "Here are three simple ways to remember anything you read. First, close the book and explain the idea out loud, in your own words. "
     "Second, write down one question the chapter answered. Third, review that question the next morning. "
     "If you do these three things, you will remember far more than people who only highlight. Stop highlighting, and start explaining."),
    ("face_cut", "rms",
     "Stop setting goals. I know that sounds crazy, but listen. Goals tell you where you want to go. "
     "They do not tell you what to do on a Tuesday afternoon when you are tired. Systems do. "
     "A writer who writes two pages every day does not need a goal. In the end, you do not rise to your goals. "
     "You fall to the level of your systems."),
]

COLORS = ["0x1f2a44", "0x3a2f2f", "0x20382c", "0x35303f", "0x403522", "0x243640"]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return float(out.stdout.strip())


def synth(text: str, voice: str, out: Path, tts: str | None, tmp: Path) -> None:
    if tts:
        txt = tmp / (out.stem + ".txt")
        txt.write_text(text, encoding="utf-8")
        raw = tmp / (out.stem + ".raw.wav")
        cmd = tts.format(text=shlex.quote(str(txt)), out=shlex.quote(str(raw)), voice=voice)
        subprocess.run(cmd, shell=True, check=True)  # noqa: S602 - developer-supplied command
        # same pause between segments and same audio format as the built-in voice
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-af", "apad=pad_dur=0.6", "-ar", "48000", "-ac", "2", str(out)])
        return
    txt = tmp / (out.stem + ".txt")
    txt.write_text(text, encoding="utf-8")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"flite=textfile={txt}:voice={voice}",
         "-af", "apad=pad_dur=0.6", "-ar", "48000", "-ac", "2", str(out)])  # fmt: skip


def visual_filter(kind: str, w: int, h: int, d: float, has_face: bool) -> str:
    fh = int(h * 0.62)  # rendered face image height
    if not has_face or kind == "noface":
        # moving bars: visible motion, nothing to track
        return f"[0:v]drawbox=x='mod(t*{w // 6},{w})':y={h // 3}:w={w // 10}:h={h // 3}:color=white@0.25:t=fill[v]"
    if kind == "face_pan":
        x = f"{w * 0.12:.0f}+({w * 0.45:.0f})*t/{d:.2f}"
        return f"[1:v]scale=-2:{fh}[f];[0:v][f]overlay=x='{x}':y={(h - fh) // 2}[v]"
    if kind == "face_right":
        return f"[1:v]scale=-2:{fh}[f];[0:v][f]overlay=x={int(w * 0.62)}:y={(h - fh) // 2}[v]"
    if kind == "two_faces":
        return (
            f"[1:v]scale=-2:{fh},split[f1][f2s];[f2s]hflip[f2];"
            f"[0:v][f1]overlay=x={int(w * 0.06)}:y={(h - fh) // 2}[t];[t][f2]overlay=x={int(w * 0.66)}:y={(h - fh) // 2}[v]"
        )
    # face_cut: hard jump of the subject half-way through (a simulated camera cut)
    half = d / 2
    return (
        f"[1:v]scale=-2:{fh}[f];"
        f"[0:v][f]overlay=x='if(lt(t,{half:.2f}),{int(w * 0.10)},{int(w * 0.60)})':y={(h - fh) // 2},"
        f"hue=h='if(lt(t,{half:.2f}),0,40)'[v]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--face", type=Path, default=None, help="image with a face to composite")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", default="30")
    ap.add_argument("--segments", type=int, default=len(SEGMENTS_EN), help="use only the first N segments")
    ap.add_argument("--script", type=Path, default=None, help="text file: one paragraph per segment (overrides the English script)")
    ap.add_argument("--tts", default=None, help="custom TTS command with {text} (input file), {out} (wav) and {voice} placeholders")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))

    segments = SEGMENTS_EN[: args.segments]
    if args.script:
        paragraphs = [p.strip() for p in args.script.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        kinds = [s[0] for s in SEGMENTS_EN]
        segments = [(kinds[i % len(kinds)], "slt", p) for i, p in enumerate(paragraphs)]

    with tempfile.TemporaryDirectory(prefix="vc-testvideo-") as t:
        tmp = Path(t)
        parts: list[Path] = []
        for i, (kind, voice, text) in enumerate(segments):
            wav = tmp / f"s{i}.wav"
            synth(text, voice, wav, args.tts, tmp)
            d = duration(wav)
            part = tmp / f"p{i}.mp4"
            inputs = ["-f", "lavfi", "-i", f"color=c={COLORS[i % len(COLORS)]}:s={w}x{h}:r={args.fps}:d={d:.3f}"]
            if args.face:
                inputs += ["-loop", "1", "-i", str(args.face)]
            inputs += ["-i", str(wav)]
            audio_idx = 2 if args.face else 1
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
                 "-filter_complex", visual_filter(kind, w, h, d, args.face is not None),
                 "-map", "[v]", "-map", f"{audio_idx}:a", "-t", f"{d:.3f}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", args.fps,
                 "-c:a", "aac", "-b:a", "160k", "-ar", "48000", str(part)])  # fmt: skip
            parts.append(part)
        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", "-movflags", "+faststart", str(args.out)])  # fmt: skip
    print(f"wrote {args.out} ({duration(args.out):.1f}s)")


if __name__ == "__main__":
    main()
