"""Stage 3 — speech-to-text with word-level timestamps (faster-whisper).

What this stage guarantees
--------------------------
* **No invented dialogue.** Whisper happily "recognises" words in music, noise and silence
  ("you", "Thanks for watching", "Legendas pela comunidade Amara.org"). Every segment is
  therefore checked against evidence before it is kept: the model's own no-speech
  probability, its confidence, whether the Silero voice-activity detector heard speech there
  and whether the text is a known stock hallucination. Dropped segments are reported in
  ``Transcript.quality`` instead of silently disappearing.
* **Accuracy that follows the hardware.** ``fast`` / ``balanced`` / ``accurate`` modes (or an
  explicit model name) choose the model and decoding effort; the default ``auto`` picks
  ``large-v3`` on a GPU, ``large-v3-turbo`` on a CPU with 4+ cores and ``small`` otherwise.
* **Level and noise treatment for the recogniser only** (``audio.enhance_for_asr``): the
  waveform that is rendered into the clip is never touched.
* **Punctuation and accents** are preserved: words are returned exactly as recognised.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..models import DownloadWatch, StatusFn, report
from ..resources import active_plan, detect_hardware
from .audio import SAMPLE_RATE, AudioSignals, analyze_audio, enhance_for_asr
from .ffmpeg import CancelFn, Cancelled, ProgressFn

log = logging.getLogger(__name__)

# Smaller models tried, in order, when the configured one cannot be loaded (e.g. offline host).
_FALLBACK_CHAIN = ["medium", "small", "base", "tiny"]
_ORDER = ["tiny", "base", "small", "medium", "large"]

# Whisper conditions its decoder on this text. A short, correctly punctuated and accented
# sentence in the right language keeps the output in that style (capitals, commas, accents).
_PROMPTS = {
    "pt": "Olá, tudo bem? Vamos começar. Hoje é um ótimo dia, não é?",
    "en": "Hello, how are you? Let's get started. It's a great day, isn't it?",
}

# Stock phrases Whisper produces from silence/music (lower-case, punctuation stripped).
_HALLUCINATIONS = {
    "you", "thank you", "thanks", "thanks for watching", "thank you for watching", "bye", "bye bye",
    "thank you so much for watching", "please subscribe", "subscribe to my channel", "subtitles by the amara.org community",
    "legendas pela comunidade amara.org", "legendas pela comunidade amara org", "obrigado", "obrigada",
    "obrigado por assistir", "obrigado por ter assistido", "tchau", "tchau tchau", "inscreva-se no canal",
    "inscreva se no canal", "música", "musica", "music", "applause", "aplausos", "silêncio", "silence",
    "sous-titres réalisés para la communauté d'amara.org", "mbc 뉴스", "ご視聴ありがとうございました",
}  # fmt: skip


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
    quality: dict = field(default_factory=dict)  # mode, model, enhancement, dropped segments, mean confidence

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "engine": self.engine,
            "words": [asdict(w) for w in self.words],
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Transcript:
        return cls(
            language=d.get("language", ""),
            language_probability=d.get("language_probability", 0.0),
            engine=d.get("engine", "none"),
            words=[Word(**w) for w in d.get("words", [])],
            quality=d.get("quality", {}),
        )


_model_lock = threading.Lock()
_models: dict[tuple[str, str, str], object] = {}  # (model, device, compute type) -> loaded model

# Verdict of the one-time GPU self-test (and of a real run that failed on the GPU). While
# ``checked`` is false a GPU is only a candidate: nothing is decoded on it before ``verify_cuda``.
_cuda: dict = {"checked": False, "ok": False, "reason": ""}
_cuda_lock = threading.Lock()

NoteFn = Callable[[str], None]


def _cuda_devices() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _supported_compute(device: str) -> set[str]:
    """Numeric precisions this device can run (empty when it cannot be asked)."""
    try:
        import ctranslate2

        return set(ctranslate2.get_supported_compute_types(device))
    except Exception:
        return set()


def _pick_compute(device: str, wanted: str = "auto") -> str:
    """A precision the device supports. Asking for one it lacks is an error in CTranslate2
    ("Requested int8_float16 compute type, but the target device or backend do not support
    efficient int8_float16 computation"), which made every model fail to load on older cards."""
    supported = _supported_compute(device)
    if wanted != "auto":
        if not supported or wanted in supported:
            return wanted
        log.warning("compute type %r is not supported on %s here; choosing one that is", wanted, device)
    if device == "cuda":
        # FP16 weights of large-v3 need ~3.5 GB of VRAM at inference; on a small card INT8
        # weights with FP16 compute fit in about half with a negligible accuracy cost
        free = detect_hardware().gpu.get("vram_free_gb")
        order = (
            ["int8_float16", "float16"] if free is not None and free < 5.0 else ["float16", "int8_float16"]
        )
        order += ["int8", "float32"]  # cards without fast half precision (GTX 10xx, MX series)
    else:
        order = ["int8", "float32"]
    for compute in order:
        if not supported or compute in supported:
            return compute
    return "default"


def _resolve_device() -> tuple[str, str]:
    """Device and precision for recognition, restricted to what this machine really supports."""
    s = get_settings()
    device = s.whisper_device
    if device == "auto":
        device = "cuda" if _cuda_devices() > 0 else "cpu"
    if device == "cuda" and _cuda["checked"] and not _cuda["ok"]:
        device = "cpu"  # the self-test or a real run failed on the GPU: stay on the processor
    return device, _pick_compute(device, s.whisper_compute_type)


def _add_nvidia_wheels_to_path() -> None:
    """Windows: let CTranslate2 find cuBLAS/cuDNN installed as pip wheels (scripts\\ativar-gpu.bat)."""
    if sys.platform != "win32":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
        roots = list(spec.submodule_search_locations or []) if spec else []
        for root in roots:
            for bin_dir in Path(root).glob("*/bin"):
                if str(bin_dir) not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                    os.add_dll_directory(str(bin_dir))
    except Exception:  # pragma: no cover - only an optimisation
        log.debug("could not register NVIDIA wheel folders", exc_info=True)


def _missing_cuda_libraries() -> str:
    """Windows: which NVIDIA runtime libraries cannot be loaded ('' when all are, or unknown).

    A missing cuDNN does not raise an exception inside CTranslate2, it ends the process; knowing
    in advance avoids even starting the self-test.
    """
    if sys.platform != "win32":
        return ""
    try:
        import ctypes

        def loadable(names: tuple[str, ...]) -> bool:
            for name in names:
                try:
                    ctypes.WinDLL(name, winmode=0)  # standard search order, PATH included
                    return True
                except OSError:
                    continue
            return False

        missing = []
        if not loadable(("cublas64_12.dll", "cublas64_13.dll", "cublas64_11.dll")):
            missing.append("cuBLAS")
        if not loadable(("cudnn_ops64_9.dll", "cudnn64_9.dll", "cudnn_ops_infer64_8.dll", "cudnn64_8.dll")):
            missing.append("cuDNN")
        return " and ".join(missing)
    except Exception:  # pragma: no cover - the self-test decides
        return ""


# One real recognition of three seconds of noise on the GPU. Run in a child process: when a
# CUDA library is missing or incompatible the failure can be the death of the process.
_GPU_SELF_TEST = """
import sys
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x8003)  # fail quietly, never open an error dialog
import numpy as np
from faster_whisper import WhisperModel
model = WhisperModel(sys.argv[1], device="cuda", compute_type=sys.argv[2])
audio = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.05).astype("float32")
segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False, condition_on_previous_text=False)
list(segments)
print("GPU-OK")
"""


def _run_gpu_self_test(model_path: str, compute: str) -> tuple[bool, str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        res = subprocess.run(
            [sys.executable, "-c", _GPU_SELF_TEST, model_path, compute],
            capture_output=True, text=True, timeout=240, creationflags=flags, check=False,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return False, "the graphics card did not answer in time"
    except OSError as exc:
        return False, str(exc)
    if res.returncode == 0 and "GPU-OK" in res.stdout:
        return True, ""
    lines = [ln.strip() for ln in (res.stderr or "").splitlines() if ln.strip()]
    return False, (lines[-1] if lines else f"the test process ended with code {res.returncode}")[:300]


def verify_cuda(on_status: StatusFn | None = None) -> bool:
    """Prove once that the GPU can really run recognition before trusting it with a video.

    ``get_cuda_device_count() > 0`` only says that a driver is installed. Recognition also needs
    the cuBLAS and cuDNN libraries and a precision the card supports. The verdict is kept for the
    life of the process; ``_cuda["reason"]`` says why a GPU is not used.
    """
    with _cuda_lock:
        if _cuda["checked"]:
            return bool(_cuda["ok"])
        report("Checking whether the graphics card can run the speech model…", on_status)
        _add_nvidia_wheels_to_path()
        ok, reason = False, ""
        missing = _missing_cuda_libraries()
        if missing:
            reason = f"the NVIDIA {missing} libraries are not installed"
        else:
            try:
                path = _fetch_model("tiny", on_status)
            except Exception as exc:
                reason = f"the test model could not be downloaded ({exc})"
            else:
                ok, reason = _run_gpu_self_test(
                    path, _pick_compute("cuda", get_settings().whisper_compute_type)
                )
        _cuda.update(checked=True, ok=ok, reason=reason)
        if ok:
            log.info("GPU self-test passed: recognition runs on the graphics card")
        else:
            log.warning("GPU not used for recognition: %s. Running on the processor.", reason)
        return ok


def mark_cuda_broken(reason: str) -> None:
    with _cuda_lock:
        _cuda.update(checked=True, ok=False, reason=reason)


@dataclass(frozen=True)
class ASRPlan:
    mode: str
    model: str
    beam_size: int
    vad: bool
    device: str
    compute: str = "int8"


def plan_asr(mode: str | None = None) -> ASRPlan:
    """Resolve the recognition mode into a concrete model and decoding effort for this machine."""
    s = get_settings()
    device, compute = _resolve_device()
    mode = (mode or s.transcription_mode or "auto").lower()
    cpus = os.cpu_count() or 4
    if mode == "auto":
        # measured: large-v3-turbo halves the word error rate of `small` at roughly twice the
        # compute (about 0.3x real time on 4 CPU cores), so it is the default wherever it is usable
        mode = "accurate" if (device == "cuda" or cpus >= 4) else "balanced"
    if mode == "fast":
        model, beam = ("small" if device == "cuda" else "base"), 1
    elif mode == "accurate":
        model, beam = ("large-v3" if device == "cuda" else "large-v3-turbo"), max(5, s.whisper_beam_size)
        if device == "cuda" and "float16" not in compute:
            # without half precision large-v3 needs 3 GB or more of VRAM: a small card gets turbo
            free = detect_hardware().gpu.get("vram_free_gb")
            if free is not None and free < 5.0:
                model = "large-v3-turbo"
    else:  # balanced
        model, beam = "small", s.whisper_beam_size
    if s.whisper_model:  # an explicit model always wins; the mode still sets the decoding effort
        model = s.whisper_model
    return ASRPlan(mode=mode, model=model, beam_size=beam, vad=s.whisper_vad, device=device, compute=compute)


def ready_plan(on_status: StatusFn | None = None) -> ASRPlan:
    """The plan recognition will really use: a GPU is included only after its self-test passed."""
    plan = plan_asr()
    if plan.device == "cuda" and not verify_cuda(on_status):
        plan = plan_asr()
    return plan


def _size_rank(name: str) -> int:
    for i, key in enumerate(_ORDER):
        if key in name:
            return i
    return len(_ORDER)


def _smaller(candidate: str, wanted: str) -> bool:
    return _size_rank(candidate) < _size_rank(wanted)


_MODEL_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]


def _fetch_model(name: str, on_status: StatusFn | None = None) -> str:
    """Folder with the model files, downloading them (with a progress report) when missing."""
    if os.path.isdir(name):
        return name
    from faster_whisper import utils

    s = get_settings()
    cache_dir = str(s.models_dir) if s.models_dir else None
    try:
        path = utils.download_model(name, local_files_only=True, cache_dir=cache_dir)
        if os.path.isfile(os.path.join(path, "model.bin")):
            return path  # already on disk: no network needed
    except Exception:
        pass  # not downloaded yet, or only partly (an interrupted download resumes below)
    repo = getattr(utils, "_MODELS", {}).get(name, name)
    with DownloadWatch(
        f"Downloading the speech model ({name}), first time only", repo, on_status, _MODEL_FILES
    ):
        return utils.download_model(name, cache_dir=cache_dir)


def load_model(name: str | None = None, on_status: StatusFn | None = None, plan: ASRPlan | None = None):
    """Load (and cache) a Whisper model, falling back to smaller ones if it cannot be fetched.

    Returns ``(model, name_actually_loaded)``. A model that cannot be *downloaded* (offline host)
    is replaced by a smaller one. A model that cannot be *started on the GPU* is a problem of the
    device, not of the model: the error is raised so the caller switches to the processor instead
    of downloading four more models that would fail the same way.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - dependency missing
        raise TranscriptionUnavailable(f"faster-whisper is not installed: {exc}") from exc

    plan = plan or plan_asr()
    wanted = name or plan.model
    chain = [wanted] + [m for m in _FALLBACK_CHAIN if m != wanted and _smaller(m, wanted)]
    device, compute = plan.device, plan.compute
    last: Exception | None = None
    with _model_lock:
        for candidate in chain:
            key = (candidate, device, compute)
            if key in _models:
                return _models[key], candidate
            try:
                path = _fetch_model(candidate, on_status)
            except Exception as exc:
                last = exc
                log.warning("speech model %r is not available: %s", candidate, exc)
                continue
            report("Loading the speech model…", on_status)
            kwargs = {"device": device, "compute_type": compute}
            if device == "cpu":
                kwargs["cpu_threads"] = active_plan().asr_threads
            try:
                model = WhisperModel(path, **kwargs)
            except Exception as exc:
                if device == "cuda":
                    raise
                last = exc
                log.warning("could not load Whisper model %r: %s", candidate, exc)
                continue
            _models[key] = model
            if candidate != wanted:
                log.warning("Whisper model %r unavailable, fell back to %r", wanted, candidate)
            return model, candidate
    raise TranscriptionUnavailable(f"no Whisper model could be loaded ({last})")


def release_models() -> None:
    """Drop cached Whisper models (ECO profile: memory goes back to the system between jobs)."""
    import gc

    with _model_lock:
        _models.clear()
    gc.collect()


def cache_key(digest: str, language: str | None, glossary: str = "") -> str:
    """Everything that changes a transcript of the same audio bytes, for the content-addressed cache."""
    from ..cache import make_key

    s = get_settings()
    plan = ready_plan()
    terms = sorted({g.strip().lower() for g in f"{s.glossary},{glossary}".split(",") if g.strip()})
    return make_key(
        "asr-v1", digest, language or "auto", plan.model, plan.beam_size, plan.vad, plan.device,
        s.whisper_vad_min_silence_ms, s.whisper_vad_speech_pad_ms, s.whisper_prompt, s.asr_enhance, terms,
    )  # fmt: skip


# --------------------------------------------------------------------------- evidence filter


def _canon(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s'-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _speech_fraction(chunks: list[tuple[float, float]], t0: float, t1: float) -> float:
    """Share of [t0, t1] that the voice-activity detector marked as speech."""
    span = max(1e-6, t1 - t0)
    covered = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in chunks)
    return min(1.0, covered / span)


def reject_reason(
    text: str,
    *,
    avg_logprob: float,
    no_speech_prob: float,
    compression_ratio: float,
    speech_frac: float,
    duration: float,
    n_words: int,
) -> str | None:
    """Why a recognised segment should be discarded as invented, or ``None`` to keep it.

    Each rule names the evidence it relies on, so a dropped segment can always be explained.
    """
    canon = _canon(text)
    if not canon:
        return "empty"
    if no_speech_prob >= 0.85:
        return f"model says no speech (p={no_speech_prob:.2f})"
    if no_speech_prob > 0.6 and avg_logprob < -0.6:
        return f"likely no speech (p={no_speech_prob:.2f}) and low confidence ({avg_logprob:.2f})"
    if avg_logprob < -1.6:
        return f"very low confidence ({avg_logprob:.2f})"
    if compression_ratio > 2.8:
        return f"repetition loop (compression ratio {compression_ratio:.1f})"
    if canon in _HALLUCINATIONS and (speech_frac < 0.5 or no_speech_prob > 0.3 or avg_logprob < -0.7):
        return "stock hallucination phrase with weak audio evidence"
    if speech_frac < 0.12 and (avg_logprob < -0.5 or n_words <= 3):
        return f"no speech detected in this stretch of audio (VAD {speech_frac:.0%})"
    if n_words <= 2 and duration < 0.6 and avg_logprob < -0.5:
        return "isolated low-confidence blip"
    return None


# --------------------------------------------------------------------------- completeness


def find_speech_gaps(
    chunks: list[tuple[float, float]], words: list[Word], *, min_len: float = 1.2, min_coverage: float = 0.30
) -> list[tuple[float, float]]:
    """Stretches the voice-activity detector marks as speech but that have (almost) no words.

    Whisper can skip a stretch of audio without saying so (measured: a 0.1 s change of the VAD
    padding made it drop a whole 7 s utterance). Any such stretch is re-transcribed on its own.
    """
    gaps: list[tuple[float, float]] = []
    for a, b in chunks:
        if b - a < min_len:
            continue
        covered = sum(max(0.0, min(w.end, b) - max(w.start, a)) for w in words if w.end > a and w.start < b)
        if covered < min_coverage * (b - a):
            if gaps and a - gaps[-1][1] < 0.6:
                gaps[-1] = (gaps[-1][0], b)
            else:
                gaps.append((a, b))
    return gaps


def _merge_recovered(words: list[Word], extra: list[Word]) -> list[Word]:
    """Add recovered words that do not collide with words already placed in time."""
    out = list(words)
    for w in extra:
        mid = (w.start + w.end) / 2
        if not any(o.start <= mid <= o.end for o in out):
            out.append(w)
    out.sort(key=lambda x: x.start)
    prev_end = 0.0
    for w in out:  # keep the sequence monotonic
        if w.start < prev_end:
            w.start = prev_end
        w.end = max(w.end, w.start + 0.02)
        prev_end = w.end
    return out


# --------------------------------------------------------------------------- main entry


def device_label(device: str) -> str:
    return "graphics card" if device == "cuda" else "processor"


def transcribe(
    samples: np.ndarray,
    language: str | None = None,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    signals: AudioSignals | None = None,
    glossary: str = "",
    on_status: StatusFn | None = None,
    on_note: NoteFn | None = None,
) -> Transcript:
    """Transcribe mono 16 kHz float32 PCM. ``language=None`` lets the model detect it.

    ``on_status`` receives what is happening right now (a download, loading the model); it is
    shown under the stage. ``on_note`` receives lasting remarks for the job log. Recognition
    that fails on the GPU is started again on the processor instead of failing the video.
    """
    duration = len(samples) / SAMPLE_RATE
    if duration < 0.2:
        return Transcript(engine="none")
    plan = ready_plan(on_status)
    if plan.device != "cuda" and _cuda["checked"] and not _cuda["ok"] and _cuda["reason"] and on_note:
        on_note(f"Graphics card not used for recognition: {_cuda['reason']}. Running on the processor.")
    try:
        return _transcribe(plan, samples, language, on_progress, should_cancel, signals, glossary, on_status)
    except (Cancelled, TranscriptionUnavailable):
        raise
    except Exception as exc:
        if plan.device != "cuda":
            raise
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        reason = (lines[-1] if lines else type(exc).__name__)[:300]
        log.warning("recognition failed on the GPU (%s); starting again on the processor", exc)
        mark_cuda_broken(reason)
        release_models()
        if on_note:
            on_note(
                f"The graphics card failed during recognition ({reason}). Starting again on the processor."
            )
        return _transcribe(
            plan_asr(), samples, language, on_progress, should_cancel, signals, glossary, on_status
        )


def _transcribe(
    plan: ASRPlan,
    samples: np.ndarray,
    language: str | None,
    on_progress: ProgressFn | None,
    should_cancel: CancelFn | None,
    signals: AudioSignals | None,
    glossary: str,
    on_status: StatusFn | None,
) -> Transcript:
    duration = len(samples) / SAMPLE_RATE
    s = get_settings()

    # loading the model, the audio treatment and speech detection report no progress (on the first
    # run the model is also downloaded): say what is happening and honour Cancel between the steps
    def check_cancel() -> None:
        if should_cancel and should_cancel():
            raise Cancelled()

    log.info(
        "loading speech model %s (%s, %s, %.0f s of audio)", plan.model, plan.device, plan.compute, duration
    )
    t_start = time.monotonic()
    model, name = load_model(plan.model, on_status, plan)
    log.info("speech model %s ready in %.0f s", name, time.monotonic() - t_start)
    check_cancel()

    # level / noise treatment for the recogniser only (the rendered audio is never touched)
    report("Preparing the audio and finding where people speak…", on_status)
    signals = signals or analyze_audio(samples)
    asr_audio, enhancement = enhance_for_asr(samples, signals, s.asr_enhance)
    check_cancel()

    chunks: list[tuple[float, float]] = []
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        ts = get_speech_timestamps(
            asr_audio, VadOptions(threshold=0.5, min_silence_duration_ms=300, speech_pad_ms=100)
        )
        chunks = [(c["start"] / SAMPLE_RATE, c["end"] / SAMPLE_RATE) for c in ts]
    except Exception as exc:  # pragma: no cover - VAD is part of faster-whisper
        log.warning("voice activity detection unavailable: %s", exc)
    log.info(
        "speech detected in %d stretch(es), %.0f s in total; decoding",
        len(chunks),
        sum(b - a for a, b in chunks),
    )
    check_cancel()
    report(f"Recognising speech on the {device_label(plan.device)} ({name})", on_status)

    prompt = None
    if s.whisper_prompt and language in _PROMPTS:
        prompt = _PROMPTS[language]
    terms = [g.strip() for g in f"{s.glossary},{glossary}".split(",") if g.strip()]
    glossary = ", ".join(dict.fromkeys(terms))  # names and jargon the recogniser should prefer

    kwargs: dict = {
        "language": language or None,
        "word_timestamps": True,
        "vad_filter": plan.vad,
        "vad_parameters": {
            "threshold": 0.5,
            "min_silence_duration_ms": s.whisper_vad_min_silence_ms,
            "speech_pad_ms": s.whisper_vad_speech_pad_ms,
        },
        "beam_size": plan.beam_size,
        "condition_on_previous_text": False,  # a mistake must not propagate through the whole video
        "no_speech_threshold": 0.6,
        "log_prob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "hallucination_silence_threshold": 2.0,
        "language_detection_segments": 4 if not language else 1,
        "language_detection_threshold": 0.5,
        "initial_prompt": prompt,
    }
    if glossary:
        kwargs["hotwords"] = glossary
    segments, info = model.transcribe(asr_audio, **kwargs)

    words: list[Word] = []
    dropped: list[dict] = []
    logprobs: list[float] = []

    def consume(segs, offset: float = 0.0, progress: bool = True) -> list[Word]:  # noqa: ANN001
        got: list[Word] = []
        for seg in segs:  # lazy generator: decoding happens here
            if should_cancel and should_cancel():
                raise Cancelled()
            seg_words = [w for w in (seg.words or []) if w.word.strip()]
            t0 = offset + (float(seg_words[0].start) if seg_words else float(seg.start))
            t1 = offset + (float(seg_words[-1].end) if seg_words else float(seg.end))
            why = reject_reason(
                seg.text,
                avg_logprob=float(seg.avg_logprob),
                no_speech_prob=float(seg.no_speech_prob),
                compression_ratio=float(seg.compression_ratio),
                speech_frac=_speech_fraction(chunks, t0, t1) if chunks or plan.vad else 1.0,
                duration=t1 - t0,
                n_words=len(seg_words),
            )
            if why:
                dropped.append(
                    {"start": round(t0, 2), "end": round(t1, 2), "text": seg.text.strip(), "reason": why}
                )
                log.info("dropped segment %.1f-%.1f %r: %s", t0, t1, seg.text.strip(), why)
            else:
                logprobs.append(float(seg.avg_logprob))
                for w in seg_words:
                    text = w.word.strip()
                    if got and not any(ch.isalnum() for ch in text):
                        # a stray symbol ("%", "…") belongs to the word before it, not on a caption of its own
                        got[-1].text += text
                        got[-1].end = max(got[-1].end, round(offset + float(w.end), 3))
                        continue
                    start = offset + float(w.start)
                    end = max(offset + float(w.end), start + 0.02)
                    if got and start < got[-1].end:  # keep the sequence monotonic
                        start = got[-1].end
                        end = max(end, start + 0.02)
                    got.append(
                        Word(
                            text=text,
                            start=round(start, 3),
                            end=round(end, 3),
                            prob=round(float(w.probability), 3),
                        )
                    )
            if progress and on_progress:
                on_progress(min(1.0, float(seg.end) / duration))
        return got

    words = consume(segments)

    # completeness: speech the detector heard but Whisper skipped is transcribed again on its own
    recovered: list[dict] = []
    lang_used = language or info.language or None
    for a, b in find_speech_gaps(chunks, words)[:12]:
        lo, hi = max(0.0, a - 0.4), min(duration, b + 0.4)
        piece = asr_audio[int(lo * SAMPLE_RATE) : int(hi * SAMPLE_RATE)]
        retry = dict(kwargs, language=lang_used, vad_filter=False, language_detection_segments=1)
        retry["initial_prompt"] = _PROMPTS.get((lang_used or "")[:2]) if s.whisper_prompt else None
        try:
            extra = consume(model.transcribe(piece, **retry)[0], offset=lo, progress=False)
        except Cancelled:
            raise
        except Exception as exc:  # pragma: no cover - a failed retry must not lose the first pass
            log.warning("re-transcription of %.1f-%.1f failed: %s", a, b, exc)
            continue
        before = len(words)
        words = _merge_recovered(words, extra)
        recovered.append({"start": round(a, 2), "end": round(b, 2), "words": len(words) - before})
        log.info("recovered %d words in skipped speech %.1f-%.1f", len(words) - before, a, b)
    if on_progress:
        on_progress(1.0)
    return Transcript(
        language=info.language or "",
        language_probability=round(float(info.language_probability or 0.0), 3),
        engine=f"faster-whisper:{name}",
        words=words,
        quality={
            "mode": plan.mode,
            "requested_model": plan.model,
            "model": name,
            "device": plan.device,
            "compute": plan.compute,
            "beam_size": plan.beam_size,
            "vad": plan.vad,
            "enhancement": enhancement,
            "dropped_segments": dropped,
            "recovered_gaps": recovered,
            "mean_logprob": round(float(np.mean(logprobs)), 3) if logprobs else None,
        },
    )
