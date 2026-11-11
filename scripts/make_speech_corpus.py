#!/usr/bin/env python3
"""Build realistic stress-test audio from reference utterances, with exact ground truth.

Starting from FLEURS-style recordings (real human speech with transcripts) this assembles
the conditions that make real videos hard for speech recognition and records, for every
output file, the reference text and the exact time span of every utterance:

    talk          utterances separated by room tone and pauses
    music_10db    same talk over a synthetic music bed, speech 10 dB above the music
    music_3db     speech only 3 dB above the music
    noise_5db     speech 5 dB above broadband noise
    quiet         speech at very low level (about -38 dBFS peak)
    fast          speech sped up 1.3x without changing pitch
    music_intro   20 s of music before the speech starts (breaks naive language detection)
    non_speech    music only, noise only and silence only: **no words may be produced**

    python scripts/make_speech_corpus.py --fleurs pt:DIR_PT --fleurs en:DIR_EN --out corpus/

``DIR`` contains ``test.tsv`` and the wav files. Output: ``corpus/*.wav`` and
``corpus/manifest.json`` consumable by ``scripts/benchmark_transcription.py --json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 16_000


def read_audio(path: Path) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )  # fmt: skip
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def scale_to_rms(x: np.ndarray, target: float) -> np.ndarray:
    return x * (target / max(rms(x), 1e-9))


def synth_music(seconds: float, seed: int = 7) -> np.ndarray:
    """A plain but real music-like bed: chord progression, bass, arpeggio and a hi-hat."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    bpm = 104
    beat = 60 / bpm
    prog = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]  # Am F C G
    out = np.zeros(n)
    for i, chord in enumerate(prog * int(np.ceil(seconds / (beat * 8)))):
        a, b = int(i * beat * 2 * SR), int(min(n, (i + 1) * beat * 2 * SR))
        if a >= n:
            break
        tt = t[a:b] - t[a]
        env = np.minimum(1, tt / 0.04) * np.exp(-tt * 0.9)
        for note in chord:
            f = 440 * 2 ** ((note - 69) / 12)
            out[a:b] += env * (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * 2 * f * tt)) * 0.18
        out[a:b] += env * np.sin(2 * np.pi * (440 * 2 ** ((chord[0] - 12 - 69) / 12)) * tt) * 0.25
    # arpeggio on eighth notes
    step = int(beat / 2 * SR)
    for k, a in enumerate(range(0, n - step, step)):
        note = prog[(a // int(beat * 2 * SR)) % 4][k % 3] + 12
        f = 440 * 2 ** ((note - 69) / 12)
        tt = np.arange(step) / SR
        out[a : a + step] += np.sin(2 * np.pi * f * tt) * np.exp(-tt * 9) * 0.12
    # hi-hat: filtered noise bursts on off-beats
    for a in range(int(beat / 2 * SR), n - 2000, step):
        burst = rng.normal(0, 1, 1600) * np.exp(-np.arange(1600) / 260)
        out[a : a + 1600] += np.diff(np.r_[0, burst]) * 0.05
    return out.astype(np.float32)


def room_tone(seconds: float, level_db: float = -62.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, int(seconds * SR))
    x = np.convolve(x, np.ones(8) / 8, mode="same")  # soften
    return scale_to_rms(x, 10 ** (level_db / 20)).astype(np.float32)


def noise(seconds: float, seed: int = 11) -> np.ndarray:
    """Pink-ish broadband noise (crowd / fan / street)."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, int(seconds * SR))
    spec = np.fft.rfft(white)
    freqs = np.maximum(np.fft.rfftfreq(len(white), 1 / SR), 30)
    spec /= np.sqrt(freqs)
    return np.fft.irfft(spec, len(white)).astype(np.float32)


def atempo(x: np.ndarray, factor: float) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", f"atempo={factor}",
         "-f", "f32le", "-"],
        input=x.tobytes(), capture_output=True, check=True,
    )  # fmt: skip
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def voiced_edges(x: np.ndarray) -> tuple[float, float]:
    """First and last audible instant (seconds) of a *clean* utterance.

    Uses the Silero VAD on the clean signal as an estimator that is independent of the
    recogniser and of the aligner being measured (a fixed loudness threshold is fooled by room
    noise and breaths).
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    ts = get_speech_timestamps(
        x.astype(np.float32),
        VadOptions(threshold=0.5, min_silence_duration_ms=1500, speech_pad_ms=0, min_speech_duration_ms=120),
    )
    if not ts:
        return 0.0, len(x) / SR
    return ts[0]["start"] / SR, ts[-1]["end"] / SR


def assemble(utts: list[dict], pauses: list[float], lead: float = 0.8, tail: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    pieces, spans, t = [room_tone(lead)], [], lead
    for u, pause in zip(utts, pauses, strict=True):
        pieces.append(u["audio"])
        v0, v1 = voiced_edges(u["audio"])
        spans.append(
            {
                "t0": round(t, 3),
                "t1": round(t + len(u["audio"]) / SR, 3),
                "v0": round(t + v0, 3),  # ground truth: first audible instant of this utterance
                "v1": round(t + v1, 3),  # ground truth: last audible instant
                "text": u["text"],
            }
        )
        t += len(u["audio"]) / SR
        pieces.append(room_tone(pause, seed=int(t * 100) % 97))
        t += pause
    pieces.append(room_tone(tail))
    return np.concatenate(pieces), spans


def mix(speech: np.ndarray, bed: np.ndarray, snr_db: float) -> np.ndarray:
    bed = np.resize(bed, len(speech))
    active = speech[np.abs(speech) > 0.01]
    s = rms(active) if len(active) else rms(speech)
    bed = scale_to_rms(bed, s / 10 ** (snr_db / 20))
    out = speech + bed
    peak = float(np.max(np.abs(out)))
    return out / peak * 0.85 if peak > 0.95 else out


def load_pool(spec: str, n: int) -> tuple[str, list[dict]]:
    lang, folder = spec.split(":", 1)
    d = Path(folder)
    index = {p.name: p for p in d.rglob("*.wav")}
    rows = []
    with open(d / "test.tsv", encoding="utf-8", newline="") as fh:
        for r in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) > 2 and r[1] in index:
                rows.append((r[0], index[r[1]], r[2], int(r[5]) if len(r) > 5 and r[5].isdigit() else 0))
    seen: set[str] = set()
    pool: list[dict] = []
    for sid, path, text, n_samples in rows[::3]:  # skip around: different sentences, different speakers
        if sid in seen:
            continue
        a = read_audio(path)
        if n_samples and len(a) < 0.97 * n_samples:
            continue  # a partially downloaded recording would not match its reference transcript
        seen.add(sid)
        # normalise each utterance's level the way a recording would be (speech around -22 dBFS RMS)
        a = scale_to_rms(a, 10 ** (-22 / 20))
        pool.append({"audio": a, "text": text})
        if len(pool) >= n:
            break
    return lang, pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fleurs", action="append", required=True, help="LANG:DIR (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-talk", type=int, default=6, help="utterances per talk")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    def emit(name: str, lang: str, audio: np.ndarray, spans: list[dict], kind: str, note: str = "") -> None:
        write_wav(args.out / f"{name}.wav", audio)
        manifest.append(
            {
                "audio": str(args.out / f"{name}.wav"),
                "name": name,
                "lang": lang,
                "kind": kind,
                "note": note,
                "text": " ".join(s["text"] for s in spans),
                "spans": spans,
                "duration": round(len(audio) / SR, 2),
            }
        )

    for spec in args.fleurs:
        lang, pool = load_pool(spec, args.per_talk * 2)
        talk_utts, other = pool[: args.per_talk], pool[args.per_talk :]
        pauses = [0.6, 1.4, 0.5, 2.2, 0.8, 0.5][: len(talk_utts)]
        speech, spans = assemble(talk_utts, pauses)
        music = synth_music(len(speech) / SR + 2)
        emit(f"{lang}_talk", lang, speech, spans, "talk", "clean read speech with pauses")
        emit(f"{lang}_music_10db", lang, mix(speech, music, 10), spans, "music", "music bed, speech +10 dB")
        emit(f"{lang}_music_3db", lang, mix(speech, music, 3), spans, "music", "music bed, speech +3 dB")
        emit(f"{lang}_noise_5db", lang, mix(speech, noise(len(speech) / SR + 1), 5), spans, "noise", "broadband noise, speech +5 dB")
        emit(f"{lang}_quiet", lang, speech / max(1e-9, float(np.max(np.abs(speech)))) * 0.0125, spans, "quiet", "peak about -38 dBFS")
        fast_utts = [{"audio": atempo(u["audio"], 1.3), "text": u["text"]} for u in other[:4]]
        fs, fspans = assemble(fast_utts, [0.3, 0.3, 0.3, 0.3])
        emit(f"{lang}_fast", lang, fs, fspans, "fast", "1.3x speech rate")
        intro = 20.0
        m_intro = synth_music(intro + 1)[: int(intro * SR)]
        kept = [s for s in spans if s["t1"] <= 45] or spans[:1]  # only utterances that are entirely present
        cut = int((kept[-1]["t1"] + 0.8) * SR)
        body = mix(speech[:cut], music, 12)
        spans_intro = [
            {**s, "t0": s["t0"] + intro, "t1": s["t1"] + intro, "v0": s["v0"] + intro, "v1": s["v1"] + intro} for s in kept
        ]
        emit(f"{lang}_music_intro", lang, np.concatenate([m_intro, body]), spans_intro, "intro", "20 s music, then speech")
        # no speech at all: nothing may be transcribed
        emit(f"{lang}_nonspeech_music", lang, scale_to_rms(synth_music(24), 10 ** (-24 / 20)), [], "nonspeech", "music only")
        emit(f"{lang}_nonspeech_noise", lang, scale_to_rms(noise(24), 10 ** (-30 / 20)), [], "nonspeech", "noise only")
        emit(f"{lang}_nonspeech_silence", lang, room_tone(24), [], "nonspeech", "room tone only")

    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(manifest)} files to {args.out}")


if __name__ == "__main__":
    main()
