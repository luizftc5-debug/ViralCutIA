"""Stage 3 — speech-to-text with word-level timestamps (faster-whisper)."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import get_settings
from .audio import SAMPLE_RATE
from .ffmpeg import CancelFn, Cancelled, ProgressFn

log = logging.getLogger(__name__)

# Smaller models tried, in order, when the configured one cannot be loaded (e.g. offline host).
_FALLBACK_CHAIN = ["small", "base", "tiny"]


class TranscriptionUnavailable(RuntimeError):
    pass


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Transcript:
    language: str = ""
    language_probability: float = 0.0
    engine: str = "none"
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "engine": self.engine,
            "words": [asdict(w) for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Transcript:
        return cls(
            language=d.get("language", ""),
            language_probability=d.get("language_probability", 0.0),
            engine=d.get("engine", "none"),
            words=[Word(**w) for w in d.get("words", [])],
        )


_model_lock = threading.Lock()
_models: dict[str, object] = {}


def _resolve_device() -> tuple[str, str]:
    s = get_settings()
    device = s.whisper_device
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = s.whisper_compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def load_model(name: str | None = None):
    """Load (and cache) a Whisper model, falling back to smaller ones if it is unavailable.

    Returns ``(model, name_actually_loaded)``.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - dependency missing
        raise TranscriptionUnavailable(f"faster-whisper is not installed: {exc}") from exc

    s = get_settings()
    wanted = name or s.whisper_model
    chain = [wanted] + [m for m in _FALLBACK_CHAIN if m != wanted and _smaller(m, wanted)]
    device, compute = _resolve_device()
    last: Exception | None = None
    with _model_lock:
        for candidate in chain:
            if candidate in _models:
                return _models[candidate], candidate
            try:
                kwargs = {"device": device, "compute_type": compute}
                if s.models_dir:
                    kwargs["download_root"] = str(s.models_dir)
                model = WhisperModel(candidate, **kwargs)
                _models[candidate] = model
                if candidate != wanted:
                    log.warning("Whisper model %r unavailable, fell back to %r", wanted, candidate)
                return model, candidate
            except Exception as exc:
                last = exc
                log.warning("could not load Whisper model %r: %s", candidate, exc)
    raise TranscriptionUnavailable(f"no Whisper model could be loaded ({last})")


_ORDER = ["tiny", "base", "small", "medium", "large"]


def _size_rank(name: str) -> int:
    for i, key in enumerate(_ORDER):
        if key in name:
            return i
    return len(_ORDER)


def _smaller(candidate: str, wanted: str) -> bool:
    return _size_rank(candidate) < _size_rank(wanted)


def transcribe(
    samples: np.ndarray,
    language: str | None = None,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
) -> Transcript:
    """Transcribe mono 16 kHz float32 PCM. ``language=None`` lets the model detect it."""
    duration = len(samples) / SAMPLE_RATE
    if duration < 0.2:
        return Transcript(engine="none")
    model, name = load_model()
    s = get_settings()
    segments, info = model.transcribe(
        samples,
        language=language or None,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=s.whisper_beam_size,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    for seg in segments:  # lazy generator: decoding happens here
        if should_cancel and should_cancel():
            raise Cancelled()
        for w in seg.words or []:
            text = w.word.strip()
            if not text:
                continue
            if words and not any(ch.isalnum() for ch in text):
                # a stray symbol ("%", "…") belongs to the word before it, not on a caption of its own
                words[-1].text += text
                words[-1].end = max(words[-1].end, round(float(w.end), 3))
                continue
            start = float(w.start)
            end = max(float(w.end), start + 0.02)
            if words and start < words[-1].end:  # keep the sequence monotonic
                start = words[-1].end
                end = max(end, start + 0.02)
            words.append(
                Word(text=text, start=round(start, 3), end=round(end, 3), prob=round(float(w.probability), 3))
            )
        if on_progress:
            on_progress(min(1.0, float(seg.end) / duration))
    if on_progress:
        on_progress(1.0)
    return Transcript(
        language=info.language or "",
        language_probability=round(float(info.language_probability or 0.0), 3),
        engine=f"faster-whisper:{name}",
        words=words,
    )
