"""Regression tests for 'never fabricate dialogue' and for how the recogniser is configured."""

import numpy as np
import pytest

from viralcut.config import get_settings
from viralcut.pipeline import audio as A
from viralcut.pipeline import transcribe as T


def _reject(text, **kw):
    base = dict(
        avg_logprob=-0.3, no_speech_prob=0.02, compression_ratio=1.5, speech_frac=1.0, duration=3.0, n_words=6
    )
    base.update(kw)
    return T.reject_reason(text, **base)


def test_the_real_music_hallucination_is_rejected():
    # measured on a music-only file: Whisper returned " you" with these values and the stock
    # faster-whisper rule (no_speech > 0.6 AND logprob < -1.0) let it through
    assert _reject(
        "you", avg_logprob=-0.75, no_speech_prob=0.96, compression_ratio=0.27, duration=0.1, n_words=1
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(no_speech_prob=0.9),
        dict(no_speech_prob=0.7, avg_logprob=-0.8),
        dict(avg_logprob=-1.8),
        dict(compression_ratio=3.5),
        dict(speech_frac=0.0, n_words=2),
        dict(n_words=1, duration=0.3, avg_logprob=-0.8),
    ],
)
def test_weak_evidence_is_rejected(kwargs):
    assert _reject("alguma coisa", **kwargs)


def test_stock_phrases_need_audio_evidence_to_survive():
    assert _reject("Legendas pela comunidade Amara.org", speech_frac=0.2)
    assert _reject("Thanks for watching!", no_speech_prob=0.5)
    # a creator genuinely ending a video: strong, confident, clearly voiced -> kept
    assert (
        _reject("Thank you.", avg_logprob=-0.2, no_speech_prob=0.01, speech_frac=1.0, n_words=2, duration=1.4)
        is None
    )


def test_confident_speech_is_kept():
    assert _reject("Quanto menor a tensão, mais positiva é a força vital presente.") is None
    assert _reject("A very short but real sentence.", n_words=6, duration=1.5, avg_logprob=-0.45) is None


def test_modes_pick_models_by_hardware(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("VIRALCUT_WHISPER_MODEL", "")
    monkeypatch.setattr(T, "_resolve_device", lambda: ("cpu", "int8"))
    try:
        monkeypatch.setenv("VIRALCUT_TRANSCRIPTION_MODE", "fast")
        get_settings.cache_clear()
        assert T.plan_asr().model == "base" and T.plan_asr().beam_size == 1
        monkeypatch.setenv("VIRALCUT_TRANSCRIPTION_MODE", "accurate")
        get_settings.cache_clear()
        assert T.plan_asr().model == "large-v3-turbo"
        monkeypatch.setattr(T, "_resolve_device", lambda: ("cuda", "float16"))
        assert T.plan_asr().model == "large-v3"
        monkeypatch.setenv("VIRALCUT_TRANSCRIPTION_MODE", "auto")
        get_settings.cache_clear()
        assert T.plan_asr().model == "large-v3"  # GPU
        monkeypatch.setattr(T, "_resolve_device", lambda: ("cpu", "int8"))
        monkeypatch.setattr(T.os, "cpu_count", lambda: 8)
        assert T.plan_asr().model == "large-v3-turbo"
        monkeypatch.setattr(T.os, "cpu_count", lambda: 2)
        assert T.plan_asr().model == "small", "a very small CPU stays on the efficient model"
        monkeypatch.setattr(T, "_resolve_device", lambda: ("cuda", "float16"))
        monkeypatch.setenv("VIRALCUT_WHISPER_MODEL", "medium")  # an explicit model always wins
        get_settings.cache_clear()
        assert T.plan_asr().model == "medium"
    finally:
        get_settings.cache_clear()


def test_quiet_audio_gets_gain_but_clean_audio_is_untouched():
    rng = np.random.default_rng(1)
    quiet = (rng.normal(0, 0.004, 16000 * 3)).astype(np.float32)
    sig = A.analyze_audio(quiet)
    chain, rep = A.asr_filter_chain(sig, "auto")
    assert "volume=" in chain and "afftdn" not in chain, "auto never denoises; denoising did not help WER"
    loud = (rng.normal(0, 0.1, 16000 * 3)).astype(np.float32)
    chain, rep = A.asr_filter_chain(A.analyze_audio(loud), "auto")
    assert chain == "" and rep["applied"] == []
    assert "afftdn" in A.asr_filter_chain(A.analyze_audio(loud), "on")[0]
    assert A.asr_filter_chain(sig, "off")[0] == ""


def test_enhancement_does_not_mutate_input():
    x = (np.random.default_rng(2).normal(0, 0.004, 16000 * 2)).astype(np.float32)
    before = x.copy()
    out, rep = A.enhance_for_asr(x, A.analyze_audio(x), "auto")
    assert np.array_equal(x, before)
    assert rep["applied"] and float(np.sqrt(np.mean(out**2))) > float(np.sqrt(np.mean(x**2)))

