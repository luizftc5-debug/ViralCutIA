"""Builders for synthetic analysis inputs."""

from __future__ import annotations

import numpy as np

from viralcut.pipeline.audio import silent_signals
from viralcut.pipeline.probe import MediaInfo
from viralcut.pipeline.scoring import AnalysisContext
from viralcut.pipeline.text import build_sentences
from viralcut.pipeline.transcribe import Word
from viralcut.pipeline.visual import VisualSignals

STRONG = (
    "Why do most people fail at learning a new skill? The answer is not talent. It is the way they practice. "
    "Most people repeat what they already know. But real progress only happens at the edge of your ability. "
    "So here is the rule. Practice the hard part first, every single day. That is why top performers improve so fast."
)
WEAK = "And then we went to the store. It was fine. He said that it was okay and we also bought some milk and bread. So yeah."
STRONG_PT = (
    "Você sabia que a maioria das pessoas desiste de estudar depois de duas semanas? O problema não é falta de talento. "
    "O problema é o método. Quase todo mundo só relê o que já sabe. Mas o aprendizado de verdade acontece quando você erra. "
    "Então a regra é simples. Comece sempre pela parte mais difícil. Por isso os melhores alunos evoluem tão rápido."
)


def make_words(
    text: str, start: float = 1.0, step: float = 0.33, pause: float = 0.45, section_pause: float = 1.4
) -> list[Word]:
    """Evenly spaced words; a pause after each sentence; '|' in the text marks a longer section break."""
    words: list[Word] = []
    t = start
    for tok in text.split():
        if tok == "|":
            t += section_pause
            continue
        words.append(Word(tok, round(t, 3), round(t + step * 0.85, 3), 0.95))
        t += step
        if tok.endswith((".", "?", "!")):
            t += pause
    return words


def make_context(text: str, language: str = "en", width: int = 1920, height: int = 1080) -> AnalysisContext:
    words = make_words(text)
    duration = words[-1].end + 1.0
    sentences = build_sentences(words, language)
    media = MediaInfo(
        duration=duration, width=width, height=height, fps="30/1", video_codec="h264", has_audio=True
    )
    audio = silent_signals(duration)
    audio.rms_db[:] = -20.0
    audio.speech_db, audio.noise_floor_db = -20.0, -55.0
    visual = VisualSignals(np.zeros(int(duration * 4) + 1, dtype=np.float32), [])
    return AnalysisContext.build(words, sentences, language, audio, visual, media)
