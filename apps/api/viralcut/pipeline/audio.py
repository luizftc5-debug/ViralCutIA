"""Stage 4a — audio decoding and signal analysis (loudness envelope, silence, onsets)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg import CancelFn, ProgressFn, input_args, read_pipe

SAMPLE_RATE = 16_000
HOP_S = 0.05


def decode_audio(
    path: Path,
    family: str | None,
    duration: float,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    *,
    seek: float | None = None,
    limit: float | None = None,
) -> np.ndarray:
    """Decode the first audio stream to mono float32 PCM at 16 kHz (the rate Whisper expects).

    ``seek``/``limit`` decode only a stretch of the file (used to align a single clip).
    """
    args = [
        *input_args(path, family, seek=seek, duration=limit),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
    ]
    chunks: list[bytes] = []
    expected = max(1.0, duration * SAMPLE_RATE * 4)
    got = 0
    for buf in read_pipe(args, chunk_size=SAMPLE_RATE * 4 * 10, exact=False, should_cancel=should_cancel):
        chunks.append(buf)
        got += len(buf)
        if on_progress:
            on_progress(min(1.0, got / expected))
    raw = b"".join(chunks)
    raw = raw[: len(raw) - (len(raw) % 4)]
    return np.frombuffer(raw, dtype=np.float32).copy()


@dataclass
class AudioSignals:
    """Loudness envelope sampled every ``HOP_S`` seconds."""

    rms_db: np.ndarray  # dBFS per hop
    noise_floor_db: float
    speech_db: float  # typical level of the louder (active) portions
    clip_ratio: float  # share of samples at/near full scale
    hop: float = HOP_S

    @property
    def duration(self) -> float:
        return len(self.rms_db) * self.hop

    @property
    def silence_threshold_db(self) -> float:
        # Adaptive: a third of the way from noise floor to speech level, never above -30 dBFS.
        return min(-30.0, self.noise_floor_db + max(6.0, (self.speech_db - self.noise_floor_db) * 0.35))

    def _slice(self, t0: float, t1: float) -> np.ndarray:
        a = max(0, int(t0 / self.hop + 1e-6))
        b = max(a + 1, min(len(self.rms_db), int(np.ceil(t1 / self.hop - 1e-6))))
        return self.rms_db[a:b]

    def level(self, t0: float, t1: float) -> float:
        s = self._slice(t0, t1)
        return float(np.mean(s)) if len(s) else self.noise_floor_db

    def peak(self, t0: float, t1: float) -> float:
        s = self._slice(t0, t1)
        return float(np.max(s)) if len(s) else self.noise_floor_db

    def dynamics(self, t0: float, t1: float) -> float:
        """Standard deviation (dB) of the active part of the envelope — a proxy for vocal expressiveness."""
        s = self._slice(t0, t1)
        s = s[s > self.silence_threshold_db]
        return float(np.std(s)) if len(s) > 4 else 0.0

    def is_silent(self, t0: float, t1: float) -> bool:
        s = self._slice(t0, t1)
        if not len(s):
            return True
        return bool(np.percentile(s, 90) < self.silence_threshold_db)

    def silent_ratio(self, t0: float, t1: float) -> float:
        s = self._slice(t0, t1)
        return float(np.mean(s < self.silence_threshold_db)) if len(s) else 1.0

    def percentile_of(self, value_db: float) -> float:
        """Where a level sits within the whole video's loudness distribution (0..1)."""
        return float(np.mean(self.rms_db <= value_db)) if len(self.rms_db) else 0.5

    def to_arrays(self) -> dict:
        return {
            "rms_db": self.rms_db.astype(np.float32),
            "meta": np.array(
                [self.noise_floor_db, self.speech_db, self.clip_ratio, self.hop], dtype=np.float64
            ),
        }

    @classmethod
    def from_arrays(cls, rms_db: np.ndarray, meta: np.ndarray) -> AudioSignals:
        return cls(
            rms_db=rms_db,
            noise_floor_db=float(meta[0]),
            speech_db=float(meta[1]),
            clip_ratio=float(meta[2]),
            hop=float(meta[3]),
        )


def analyze_audio(samples: np.ndarray) -> AudioSignals:
    hop = int(SAMPLE_RATE * HOP_S)
    n = len(samples) // hop
    if n == 0:
        return AudioSignals(np.full(1, -90.0, dtype=np.float32), -90.0, -90.0, 0.0)
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    rms_db = (20 * np.log10(np.maximum(rms, 1e-5))).astype(np.float32)  # floor at -100 dBFS
    noise = float(np.percentile(rms_db, 10))
    active = rms_db[rms_db > noise + 6]
    speech = float(np.percentile(active, 70)) if len(active) > 10 else float(np.percentile(rms_db, 90))
    clip_ratio = float(np.mean(np.abs(samples) > 0.985))
    return AudioSignals(rms_db, noise, speech, clip_ratio)


def silent_signals(duration: float) -> AudioSignals:
    """Placeholder envelope for sources without an audio stream."""
    n = max(1, int(duration / HOP_S))
    return AudioSignals(np.full(n, -100.0, dtype=np.float32), -100.0, -100.0, 0.0)


# --------------------------------------------------------------------------- recogniser input


def asr_filter_chain(signals: AudioSignals, mode: str = "auto") -> tuple[str, dict]:
    """ffmpeg filter chain that prepares audio for the speech recogniser, and why.

    Only a condition that measurably hurts recognition is treated automatically: very low level
    (the recogniser's front end then sees mostly noise floor). Clean, well-levelled speech
    passes through untouched.
    """
    snr = signals.speech_db - signals.noise_floor_db
    report: dict = {
        "mode": mode,
        "speech_db": round(signals.speech_db, 1),
        "noise_floor_db": round(signals.noise_floor_db, 1),
        "snr_db": round(snr, 1),
        "applied": [],
    }
    if mode == "off":
        return "", report
    filters: list[str] = []
    force = mode == "on"
    if force or signals.speech_db < -32.0:
        gain = float(np.clip(-23.0 - signals.speech_db, 0.0, 24.0)) if signals.speech_db > -90 else 0.0
        if gain >= 3.0:
            filters.append(f"volume={gain:.1f}dB")
            report["applied"].append(f"gain +{gain:.0f} dB (speech was at {signals.speech_db:.0f} dBFS)")
    # Spectral noise reduction is opt-in only: measured on real speech it did not lower the word
    # error rate (Whisper is trained on noisy audio and the denoiser's artefacts cost as much
    # as they save), so ``auto`` never applies it. See docs/TRANSCRIPTION.md.
    if force:
        filters.append("highpass=f=70")
        filters.append("afftdn=nr=12:nf=-45:tn=1")
        report["applied"].append(f"high-pass 70 Hz + spectral noise reduction (SNR {snr:.0f} dB)")
    if filters:
        filters.append("alimiter=limit=0.95")  # a boosted peak must never clip
    return ",".join(filters), report


def enhance_for_asr(
    samples: np.ndarray, signals: AudioSignals, mode: str = "auto"
) -> tuple[np.ndarray, dict]:
    """Return ``(audio for the recogniser, report)``. The input array is never modified."""
    chain, report = asr_filter_chain(signals, mode)
    if not chain or len(samples) == 0:
        return samples, report
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1",
         "-i", "pipe:0", "-af", chain, "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"],
        input=samples.astype(np.float32).tobytes(), capture_output=True, check=False,
    )  # fmt: skip
    out = np.frombuffer(proc.stdout[: len(proc.stdout) - len(proc.stdout) % 4], dtype=np.float32)
    if proc.returncode != 0 or len(out) < len(samples) * 0.95:
        report["applied"] = []
        report["error"] = "enhancement failed; original audio used"
        return samples, report
    return out.copy(), report
