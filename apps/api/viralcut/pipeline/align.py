"""Stage 3b — forced alignment of the recognised words to the waveform.

Whisper's own word timestamps come from cross-attention weights: good enough to rank content,
but they drift by a few hundred milliseconds, cluster at segment edges and stretch words into
the pauses around them. Captions make every one of those errors visible.

This stage re-times the *already recognised* words against the audio with a CTC acoustic model
(the approach of WhisperX), run through onnxruntime so no PyTorch install is needed:

* Portuguese: ``jonatasgrosman/wav2vec2-large-xlsr-53-portuguese`` (Apache-2.0)
* English:    ``facebook/wav2vec2-base-960h`` (Apache-2.0)

Both are fetched from the Hugging Face Hub on first use (4-bit / int8 ONNX exports). Other
languages, an offline host, or a failed quality gate keep Whisper's timestamps, then every word
boundary is tightened against the loudness envelope (``refine_with_energy``), so captions never
start before the sound or linger in the silence after it. The text is never altered here.

The multilingual MMS aligner (1 130 languages) is CC-BY-NC and therefore *not* a default;
``VIRALCUT_ALIGN_REPO_OVERRIDES`` can point a language at any compatible CTC export you are
licensed to use.
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..resources import active_plan
from .audio import AudioSignals
from .transcribe import Word

log = logging.getLogger(__name__)

SR = 16_000
FRAME_S = 0.02  # wav2vec2 stride: 320 samples
MAX_WINDOW_S = 28.0
WINDOW_PAD_S = 0.5
NEG = -1e30


@dataclass(frozen=True)
class AlignerSpec:
    repo: str
    onnx_file: str
    upper: bool  # the vocabulary is upper-case (LibriSpeech-style) instead of lower-case
    licence: str


ALIGNERS: dict[str, AlignerSpec] = {
    "pt": AlignerSpec("onnx-community/wav2vec2-large-xlsr-53-portuguese-ONNX", "onnx/model_q4.onnx", False, "Apache-2.0"),
    "en": AlignerSpec("onnx-community/wav2vec2-base-960h-ONNX", "onnx/model.onnx", True, "Apache-2.0"),
}  # fmt: skip


@dataclass
class AlignStats:
    method: str = "whisper"  # ctc | whisper
    words: int = 0
    aligned: int = 0
    interpolated: int = 0
    kept_whisper: int = 0
    median_shift_s: float = 0.0
    chunks: int = 0
    chunks_rejected: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "words": self.words,
            "aligned": self.aligned,
            "interpolated": self.interpolated,
            "kept_whisper": self.kept_whisper,
            "median_shift_s": round(self.median_shift_s, 3),
            "chunks": self.chunks,
            "chunks_rejected": self.chunks_rejected,
            "note": self.note,
        }


# --------------------------------------------------------------------------- model loading


class Aligner:
    def __init__(self, spec: AlignerSpec, onnx_path: Path, vocab: dict[str, int]) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = active_plan().onnx_threads
        so.inter_op_num_threads = 1
        # windows have different lengths: an arena would keep the largest allocation forever
        so.enable_cpu_mem_arena = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.spec = spec
        self.vocab = vocab
        self.blank = vocab.get("<pad>", 0)
        self.delim = vocab.get("|")
        self._lock = threading.Lock()

    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float32)
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        with self._lock:
            logits = self.session.run(None, {"input_values": x[None, :]})[0][0]
        logits = logits.astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


_aligners: dict[str, Aligner | None] = {}
_aligner_failed_at: dict[str, float] = {}
_aligner_lock = threading.Lock()
RETRY_AFTER_S = 600.0  # a failed download (offline host, disk) is retried, not given up on for good


def _repo_override(lang: str) -> tuple[str, str] | None:
    raw = get_settings().align_repo_overrides
    for item in filter(None, (p.strip() for p in raw.split(","))):
        # "pt=owner/repo:onnx/model.onnx"
        key, _, rest = item.partition("=")
        if key.strip() == lang and ":" in rest:
            repo, _, file = rest.partition(":")
            return repo.strip(), file.strip()
    return None


def load_aligner(language: str) -> Aligner | None:
    """Return the aligner for ``language`` (``None`` when unsupported, disabled or unavailable)."""
    s = get_settings()
    lang = (language or "")[:2].lower()
    if s.align_mode == "off":
        return None
    spec = ALIGNERS.get(lang)
    override = _repo_override(lang)
    if spec is None and override is None:
        return None
    if override is not None:
        spec = AlignerSpec(override[0], override[1], spec.upper if spec else False, "user-supplied")
    assert spec is not None
    with _aligner_lock:
        if _aligners.get(lang) is not None:
            return _aligners[lang]
        if lang in _aligner_failed_at and time.monotonic() - _aligner_failed_at[lang] < RETRY_AFTER_S:
            return None
        try:
            from huggingface_hub import hf_hub_download

            kw = {"cache_dir": str(s.models_dir)} if s.models_dir else {}
            onnx = hf_hub_download(spec.repo, spec.onnx_file, **kw)
            vocab_path = hf_hub_download(spec.repo, "vocab.json", **kw)
            import json

            vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
            _aligners[lang] = Aligner(spec, Path(onnx), vocab)
            _aligner_failed_at.pop(lang, None)
        except Exception as exc:  # offline, disk, incompatible export...
            log.warning("forced aligner for %r unavailable (%s); using Whisper timestamps", lang, exc)
            _aligners[lang] = None
            _aligner_failed_at[lang] = time.monotonic()
        return _aligners[lang]


# --------------------------------------------------------------------------- CTC Viterbi


def ctc_force_align(logp: np.ndarray, tokens: list[int], blank: int) -> np.ndarray | None:
    """Most likely CTC path for ``tokens``. Returns, per frame, the index into the extended
    sequence ``[blank, t0, blank, t1, ..., blank]`` or ``None`` if the audio is too short."""
    T = logp.shape[0]
    n = len(tokens)
    if n == 0 or T == 0:
        return None
    L = 2 * n + 1
    ext = np.full(L, blank, dtype=np.int64)
    ext[1::2] = tokens
    repeats = sum(1 for a, b in zip(tokens, tokens[1:], strict=False) if a == b)
    if T < n + repeats:
        return None
    can_skip = np.zeros(L, dtype=bool)
    can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    dp = np.full(L, NEG)
    dp[0] = logp[0, blank]
    dp[1] = logp[0, ext[1]]
    back = np.zeros((T, L), dtype=np.uint8)
    emit = logp[:, ext]  # [T, L]
    for t in range(1, T):
        stay = dp
        prev1 = np.concatenate(([NEG], dp[:-1]))
        prev2 = np.concatenate(([NEG, NEG], dp[:-2]))
        prev2 = np.where(can_skip, prev2, NEG)
        cand = np.stack((stay, prev1, prev2))
        idx = cand.argmax(axis=0)
        back[t] = idx
        dp = np.take_along_axis(cand, idx[None], axis=0)[0] + emit[t]
    s = L - 1 if dp[L - 1] >= dp[L - 2] else L - 2
    if dp[s] <= NEG / 2:
        return None
    path = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
        if s < 0:
            s = 0
    return path


# --------------------------------------------------------------------------- text -> tokens


def _clean_chars(text: str, vocab: dict[str, int], upper: bool) -> list[int]:
    """Characters of ``text`` the acoustic model knows. Digits and symbols are not alignable."""
    out: list[int] = []
    text = unicodedata.normalize("NFC", text).replace("’", "'")
    for ch in text:
        c = ch.upper() if upper else ch.lower()
        if c in vocab and c not in ("|", "<pad>", "<s>", "</s>", "<unk>"):
            out.append(vocab[c])
            continue
        base = "".join(x for x in unicodedata.normalize("NFD", c) if unicodedata.category(x) != "Mn")
        if base in vocab and base != "|" and base.isalpha():
            out.append(vocab[base])
    return out


def _chunks(words: list[Word], max_s: float = MAX_WINDOW_S) -> list[tuple[int, int]]:
    """Split the word list into windows of at most ``max_s`` seconds, preferring the longest gaps."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(words)
    while i < n:
        j = i + 1
        best, best_gap = None, -1.0
        while j < n and words[j].end - words[i].start <= max_s:
            gap = words[j].start - words[j - 1].end
            if gap >= best_gap and words[j].end - words[i].start >= max_s * 0.5:
                best, best_gap = j, gap
            j += 1
        cut = n if j >= n else (best or j)
        out.append((i, cut))
        i = cut
    return out


# --------------------------------------------------------------------------- alignment


def align_words(
    words: list[Word], samples: np.ndarray, language: str, origin: float = 0.0
) -> tuple[list[Word], AlignStats]:
    """Re-time ``words`` (times on the source timeline) using ``samples``, whose first sample
    is at ``origin`` seconds on that timeline. Always returns a list of the same length."""
    stats = AlignStats(words=len(words))
    if not words or len(samples) < SR:
        stats.note = "nothing to align"
        return words, stats
    aligner = load_aligner(language)
    if aligner is None:
        stats.note = "no forced aligner for this language or aligner disabled"
        return words, stats

    out = [Word(w.text, w.start, w.end, w.prob) for w in words]
    shifts: list[float] = []
    total = len(samples) / SR
    for lo, hi in _chunks(words):
        stats.chunks += 1
        chunk = words[lo:hi]
        w0 = max(0.0, chunk[0].start - origin - WINDOW_PAD_S)
        w1 = min(total, chunk[-1].end - origin + WINDOW_PAD_S)
        if w1 - w0 < 0.3:
            stats.chunks_rejected += 1
            continue
        audio = samples[int(w0 * SR) : int(w1 * SR)]
        try:
            logp = aligner.log_probs(audio)
        except Exception as exc:  # pragma: no cover - runtime/model problems
            log.warning("alignment inference failed: %s", exc)
            stats.chunks_rejected += 1
            continue
        tokens: list[int] = []
        owner: list[int] = []  # token -> index into chunk (-1 for delimiters)
        alignable: list[bool] = []
        for k, w in enumerate(chunk):
            ids = _clean_chars(w.text, aligner.vocab, aligner.spec.upper)
            alignable.append(bool(ids))
            if not ids:
                continue
            if tokens and aligner.delim is not None:
                tokens.append(aligner.delim)
                owner.append(-1)
            tokens.extend(ids)
            owner.extend([k] * len(ids))
        if len(tokens) < 3:
            stats.chunks_rejected += 1
            continue
        path = ctc_force_align(logp, tokens, aligner.blank)
        if path is None:
            stats.chunks_rejected += 1
            continue
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        conf: dict[int, list[float]] = {}
        for t, s in enumerate(path):
            if s % 2 == 1:
                k = owner[(s - 1) // 2]
                if k < 0:
                    continue
                first.setdefault(k, t)
                last[k] = t
                conf.setdefault(k, []).append(float(logp[t, tokens[(s - 1) // 2]]))
        t_off = origin + w0
        new: dict[int, tuple[float, float]] = {}
        for k in first:
            new[k] = (t_off + first[k] * FRAME_S, t_off + (last[k] + 1) * FRAME_S)
        # quality gate: the acoustic model must agree with Whisper about where this speech is
        diffs = [new[k][0] - chunk[k].start for k in new]
        if not diffs or abs(float(np.median(diffs))) > 1.2:
            stats.chunks_rejected += 1
            continue
        poor = [k for k in new if np.mean(conf[k]) < -9.0]  # the model could not hear these characters
        for k in poor:
            new.pop(k)
        for k, (a, b) in new.items():
            out[lo + k].start, out[lo + k].end = a, b
            shifts.append(a - chunk[k].start)
        stats.aligned += len(new)
        # unalignable tokens (digits, symbols) and poorly heard words: place between neighbours
        for k in range(len(chunk)):
            if k in new:
                continue
            _interpolate(out, lo, hi, k, new)
            stats.interpolated += 1

    stats.kept_whisper = stats.words - stats.aligned - stats.interpolated
    if stats.aligned:
        stats.method = "ctc"
        stats.median_shift_s = float(np.median(shifts)) if shifts else 0.0
    out = _make_monotonic(out)
    return out, stats


def _interpolate(out: list[Word], lo: int, hi: int, k: int, aligned: dict[int, tuple[float, float]]) -> None:
    """Give an unalignable word the stretch of audio left between its aligned neighbours."""
    prev_k = next((j for j in range(k - 1, -1, -1) if j in aligned), None)
    next_k = next((j for j in range(k + 1, hi - lo) if j in aligned), None)
    if prev_k is None and next_k is None:
        return
    t0 = aligned[prev_k][1] if prev_k is not None else out[lo + k].start
    t1 = aligned[next_k][0] if next_k is not None else out[lo + k].end
    run_lo = prev_k + 1 if prev_k is not None else 0
    run_hi = next_k if next_k is not None else hi - lo
    group = [j for j in range(run_lo, run_hi) if j not in aligned]
    if not group or t1 - t0 <= 0.04 * len(group):
        return
    weights = [max(1, len(out[lo + j].text)) for j in group]
    total = float(sum(weights))
    cur = t0
    for j, wgt in zip(group, weights, strict=True):
        d = (t1 - t0) * wgt / total
        out[lo + j].start, out[lo + j].end = cur, cur + d
        cur += d


def _make_monotonic(words: list[Word]) -> list[Word]:
    prev_end = 0.0
    for w in words:
        if w.start < prev_end:
            w.start = prev_end
        if w.end < w.start + 0.04:
            w.end = w.start + 0.04
        prev_end = w.end
        w.start, w.end = round(w.start, 3), round(w.end, 3)
    return words


# --------------------------------------------------------------------------- energy refinement


@dataclass
class RefineStats:
    moved_starts: int = 0
    moved_ends: int = 0
    extra: dict = field(default_factory=dict)


def _frame(t: float, hop: float) -> int:
    return int(t / hop + 1e-6)  # 1.0 / 0.05 is 19.999999999999996 in floating point


def refine_with_energy(
    words: list[Word], audio: AudioSignals, *, max_snap: float = 0.6
) -> tuple[list[Word], RefineStats]:
    """Tighten word boundaries that border a real pause using the loudness envelope.

    Only boundaries next to a gap (or the start/end of the speech) are touched: inside
    continuous speech there is no energy evidence to move a boundary. A word never ends later
    than the last audible frame in its neighbourhood and never starts before the first one, so
    a caption cannot be shown ahead of its sound or stay up through the silence after it.
    """
    st = RefineStats()
    if not words:
        return words, st
    thr = audio.silence_threshold_db
    hop = audio.hop
    env = audio.rms_db

    def voiced(i: int) -> bool:
        return 0 <= i < len(env) and bool(env[i] > thr)

    out = [Word(w.text, w.start, w.end, w.prob) for w in words]
    n = len(out)
    for i, w in enumerate(out):
        nxt_start = out[i + 1].start if i + 1 < n else w.end + max_snap + 1
        prv_end = out[i - 1].end if i > 0 else max(0.0, w.start - max_snap - 1)
        # --- end: walk back while the frames at the tail are silent, up to max_snap
        if nxt_start - w.end > 0.08 or i + 1 == n:
            a = _frame(w.end, hop)
            lim = max(_frame(w.start + 0.06, hop), a - _frame(max_snap, hop))
            j = a - 1
            while j > lim and not voiced(j):
                j -= 1
            new_end = (j + 1) * hop
            if new_end < w.end - hop * 1.5:
                w.end = round(max(new_end, w.start + 0.06), 3)
                st.moved_ends += 1
        # --- start: walk forward while the frames at the head are silent
        if w.start - prv_end > 0.08 or i == 0:
            a = _frame(w.start, hop)
            lim = min(_frame(w.end - 0.06, hop), a + _frame(max_snap, hop))
            j = a
            while j < lim and not voiced(j):
                j += 1
            new_start = j * hop
            if new_start > w.start + hop * 1.5:
                w.start = round(min(new_start, w.end - 0.06), 3)
                st.moved_starts += 1
    return _make_monotonic(out), st
