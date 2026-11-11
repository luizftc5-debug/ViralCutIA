"""Forced alignment and energy refinement, tested with exact synthetic ground truth."""

import numpy as np
import pytest

from viralcut.pipeline import align as AL
from viralcut.pipeline.audio import AudioSignals
from viralcut.pipeline.transcribe import Word

VOCAB = {"<pad>": 0, "|": 4, "a": 7, "b": 8, "c": 9, "d": 10, "e": 11, "ç": 12}


def _emissions(frames: int, spikes: dict[int, int]) -> np.ndarray:
    logp = np.full((frames, 13), np.log(1e-4))
    logp[:, 0] = np.log(0.99)  # blank everywhere...
    for f, tok in spikes.items():  # ...except where a character is emitted
        logp[f, :] = np.log(1e-4)
        logp[f, tok] = np.log(0.99)
    return logp


def test_viterbi_recovers_exact_spike_frames():
    logp = _emissions(60, {10: 7, 25: 8, 40: 9})
    path = AL.ctc_force_align(logp, [7, 8, 9], blank=0)
    assert path is not None
    for frame, k in ((10, 0), (25, 1), (40, 2)):
        assert path[frame] == 2 * k + 1


def test_viterbi_handles_repeated_characters_and_rejects_too_short_audio():
    logp = _emissions(30, {5: 7, 12: 7, 20: 8})
    path = AL.ctc_force_align(logp, [7, 7, 8], blank=0)
    assert path is not None and path[5] == 1 and path[12] == 3 and path[20] == 5
    assert AL.ctc_force_align(_emissions(2, {}), [7, 8, 9, 10], blank=0) is None


def test_clean_chars_keeps_letters_and_drops_digits_and_symbols():
    ids = AL._clean_chars("A-çb 19%!", VOCAB, upper=False)
    assert ids == [VOCAB["a"], VOCAB["ç"], VOCAB["b"]]
    assert AL._clean_chars("19%", VOCAB, upper=False) == []  # unalignable: interpolated later


def test_chunks_split_long_speech_at_the_longest_gap():
    words = [Word(f"w{i}", i * 1.0, i * 1.0 + 0.5) for i in range(60)]
    words[40] = Word("w40", 41.5, 42.0)  # a 1 s gap before word 40
    chunks = AL._chunks(words, max_s=28.0)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(words)
    assert all(words[b - 1].end - words[a].start <= 28.0 + 1e-6 for a, b in chunks)
    assert sum(b - a for a, b in chunks) == len(words)


class _FakeAligner:
    """Stands in for the ONNX model: emits each character at a known time."""

    def __init__(self, truth: dict[str, tuple[float, float]]):
        self.spec = AL.ALIGNERS["pt"]
        self.vocab = VOCAB
        self.blank, self.delim = 0, 4
        self.truth = truth

    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        frames = len(audio) // 320
        spikes: dict[int, int] = {}
        for word, (t0, t1) in self.truth.items():
            chars = [VOCAB[c] for c in word]
            for k, tok in enumerate(chars):
                t = t0 + (t1 - t0) * (k + 0.5) / len(chars) - self.window_start
                spikes[int(t / 0.02)] = tok
        return _emissions(frames, spikes)

    window_start = 0.0


def test_align_words_corrects_a_systematic_early_bias(monkeypatch):
    truth = {"abc": (2.00, 2.60), "de": (2.90, 3.30), "cab": (4.00, 4.50)}
    fake = _FakeAligner(truth)
    fake.window_start = 1.5  # the chunk window starts at first word - 0.5 s pad = 1.75 -> see below
    # Whisper believes every word starts ~0.25 s earlier than it really does
    words = [Word("abc", 1.75, 2.35), Word("de", 2.65, 3.05), Word("cab", 3.75, 4.25)]
    fake.window_start = max(0.0, words[0].start - AL.WINDOW_PAD_S)
    monkeypatch.setattr(AL, "load_aligner", lambda lang: fake)
    samples = np.zeros(16000 * 6, dtype=np.float32)
    out, stats = AL.align_words(words, samples, "pt", origin=0.0)
    assert stats.method == "ctc" and stats.aligned == 3 and stats.chunks_rejected == 0
    for w, (t0, t1) in zip(out, truth.values(), strict=True):
        assert abs(w.start - t0) < 0.12 and abs(w.end - t1) < 0.12
    assert [w.text for w in out] == ["abc", "de", "cab"], "text is never altered"
    assert stats.median_shift_s > 0.15


def test_align_words_falls_back_to_whisper_when_no_aligner(monkeypatch):
    monkeypatch.setattr(AL, "load_aligner", lambda lang: None)
    words = [Word("abc", 1.0, 1.4)]
    out, stats = AL.align_words(words, np.zeros(16000 * 3, dtype=np.float32), "xx")
    assert out == words and stats.method == "whisper"


def test_digits_are_placed_between_their_neighbours(monkeypatch):
    truth = {"ab": (1.0, 1.4), "cd": (3.0, 3.4)}
    fake = _FakeAligner(truth)
    words = [Word("ab", 0.9, 1.3), Word("19", 1.5, 2.9), Word("cd", 2.9, 3.3)]
    fake.window_start = max(0.0, words[0].start - AL.WINDOW_PAD_S)
    monkeypatch.setattr(AL, "load_aligner", lambda lang: fake)
    out, stats = AL.align_words(words, np.zeros(16000 * 5, dtype=np.float32), "pt")
    assert stats.interpolated == 1
    assert out[0].end <= out[1].start < out[1].end <= out[2].start + 1e-6


def _signals(voiced: list[tuple[float, float]], seconds: float = 6.0) -> AudioSignals:
    hop = 0.05
    env = np.full(int(seconds / hop), -80.0, dtype=np.float32)
    for a, b in voiced:
        env[round(a / hop) : round(b / hop)] = -20.0
    return AudioSignals(env, noise_floor_db=-70.0, speech_db=-20.0, clip_ratio=0.0)


def test_energy_refinement_trims_words_that_lead_or_linger_around_a_pause():
    audio = _signals([(1.0, 2.0), (3.0, 3.5)])
    words = [Word("one", 0.7, 2.4), Word("two", 2.6, 3.9)]
    out, st = AL.refine_with_energy(words, audio)
    assert out[0].start == pytest.approx(1.0, abs=0.06) and out[0].end == pytest.approx(2.0, abs=0.06)
    assert out[1].start == pytest.approx(3.0, abs=0.06) and out[1].end == pytest.approx(3.5, abs=0.06)
    assert st.moved_starts == 2 and st.moved_ends == 2


def test_energy_refinement_leaves_continuous_speech_alone():
    audio = _signals([(1.0, 4.0)])
    words = [Word("a", 1.0, 1.9), Word("b", 1.9, 3.0), Word("c", 3.0, 4.0)]
    out, _ = AL.refine_with_energy(words, audio)
    assert [(w.start, w.end) for w in out] == [(1.0, 1.9), (1.9, 3.0), (3.0, 4.0)]


def test_a_failed_aligner_download_is_retried_later_not_given_up_on(monkeypatch):
    import huggingface_hub

    calls: list[int] = []

    def offline(*args, **kwargs):
        calls.append(1)
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    AL._aligners.pop("pt", None)
    AL._aligner_failed_at.pop("pt", None)
    try:
        assert AL.load_aligner("pt") is None and len(calls) == 1
        assert AL.load_aligner("pt") is None and len(calls) == 1, (
            "no hammering the network inside the retry window"
        )
        AL._aligner_failed_at["pt"] -= AL.RETRY_AFTER_S + 1
        assert AL.load_aligner("pt") is None and len(calls) == 2, "retried once the window has passed"
    finally:
        AL._aligners.pop("pt", None)
        AL._aligner_failed_at.pop("pt", None)
