"""Objective metrics for the speech and caption pipeline (no model needed, pure functions).

* ``wer`` — Word Error Rate against a reference transcript, with the alignment so recurring
  failures (substitutions, insertions of invented words, deletions) can be listed.
* ``timing_report`` — how well word timestamps agree with the audio itself: a word that is
  mostly silence in the waveform is mis-timed or invented.

Normalisation lower-cases and strips punctuation but **keeps accents**: "não" and "nao" are
different words, and the product promises accents are preserved.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

_PUNCT = re.compile(r"[^\w\s']+", re.UNICODE)
_SPACES = re.compile(r"\s+")
# Numbers are written as digits by both reference corpora and Whisper; keep them as tokens.
_APOS = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})


_EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_EN_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_EN_SCALE = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}

# Spelling / unit variants that are the same word to a listener (applied to both sides).
_EQUIV = {
    "km": "kilometers", "kilometres": "kilometers", "kilometre": "kilometers", "kilometer": "kilometers",
    "mi": "miles", "mile": "miles",
    "percent": "", "per": "", "cent": "", "porcento": "", "por": "por",
    "okay": "ok", "mr": "mister", "dr": "doctor",
}  # fmt: skip


_EQUIV_PT = {"km": "quilometros", "kms": "quilometros", "quilômetros": "quilometros", "quilômetro": "quilometro",
             "quilometros": "quilometros", "m": "metros"}  # fmt: skip


def _spoken_numbers_to_digits(tokens: list[str]) -> list[str]:
    """English only: 'one hundred and five' -> '105' (the form Whisper writes)."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in _EN_UNITS or t in _EN_TENS or t in _EN_SCALE:
            total, cur, j, used = 0, 0, i, False
            while j < len(tokens):
                u = tokens[j]
                if u in _EN_UNITS:
                    cur += _EN_UNITS[u]
                elif u in _EN_TENS:
                    cur += _EN_TENS[u]
                elif u == "hundred":
                    cur = max(cur, 1) * 100
                elif u in _EN_SCALE:
                    total += max(cur, 1) * _EN_SCALE[u]
                    cur = 0
                elif u == "and" and j + 1 < len(tokens) and tokens[j + 1] in {**_EN_UNITS, **_EN_TENS}:
                    pass
                else:
                    break
                used = True
                j += 1
            if used:
                out.append(str(total + cur))
                i = j
                continue
        out.append(t)
        i += 1
    return out


def normalize_text(text: str, *, keep_accents: bool = True, lang: str = "") -> list[str]:
    text = unicodedata.normalize("NFC", text).translate(_APOS).lower()
    if not keep_accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = text.replace("-", " ").replace("—", " ").replace("–", " ")
    text = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text)  # 3,000 / 3.000 -> 3000
    text = text.replace("%", " percent ")
    text = _PUNCT.sub(" ", text)
    tokens = [t for t in _SPACES.split(text.strip()) if t]
    if lang[:2] in ("en", ""):
        tokens = _spoken_numbers_to_digits(tokens)
    table = {**_EQUIV, **_EQUIV_PT} if lang[:2] == "pt" else _EQUIV
    return [e for t in tokens if (e := table.get(t, t))]


@dataclass
class WerResult:
    wer: float
    ref_words: int
    substitutions: int
    deletions: int
    insertions: int
    errors: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, reference, hypothesis)

    @property
    def hits(self) -> int:
        return self.ref_words - self.substitutions - self.deletions

    def to_dict(self) -> dict:
        return {
            "wer": round(self.wer, 4),
            "ref_words": self.ref_words,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
        }


def wer(
    reference: str | list[str], hypothesis: str | list[str], *, keep_accents: bool = True, lang: str = ""
) -> WerResult:
    """Word error rate with a full Levenshtein alignment (substitution/insertion/deletion cost 1)."""
    ref = (
        normalize_text(reference, keep_accents=keep_accents, lang=lang)
        if isinstance(reference, str)
        else reference
    )
    hyp = (
        normalize_text(hypothesis, keep_accents=keep_accents, lang=lang)
        if isinstance(hypothesis, str)
        else hypothesis
    )
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    prev = list(range(m + 1))
    back: list[list[int]] = [[2] * (m + 1)]  # 0 = match/sub (diag), 1 = deletion (up), 2 = insertion (left)
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        row = [1] + [0] * m
        for j in range(1, m + 1):
            sub = prev[j - 1] + (ref[i - 1] != hyp[j - 1])
            dele = prev[j] + 1
            ins = cur[j - 1] + 1
            best = min(sub, dele, ins)
            cur[j] = best
            row[j] = 0 if best == sub else (1 if best == dele else 2)
        back.append(row)
        prev = cur
    s = d = ins_count = 0
    errors: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j] if i > 0 and j > 0 else (1 if i > 0 else 2)
        if op == 0:
            if ref[i - 1] != hyp[j - 1]:
                s += 1
                errors.append(("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif op == 1:
            d += 1
            errors.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ins_count += 1
            errors.append(("ins", "", hyp[j - 1]))
            j -= 1
    errors.reverse()
    rate = (s + d + ins_count) / n if n else (0.0 if m == 0 else float(m))
    return WerResult(rate, n, s, d, ins_count, errors)


def top_errors(results: list[WerResult], limit: int = 15) -> list[tuple[tuple[str, str, str], int]]:
    """Most frequent individual errors across many utterances (recurring failure modes)."""
    counter: Counter[tuple[str, str, str]] = Counter()
    for r in results:
        counter.update(r.errors)
    return counter.most_common(limit)


# --------------------------------------------------------------------------- timing sanity


def timing_report(words: list[tuple[str, float, float]], rms_db, hop: float, silence_db: float) -> dict:
    """Compare word timestamps with the waveform envelope.

    ``rms_db`` is the loudness envelope (one value per ``hop`` seconds). A correctly timed word
    covers audible speech; a word that falls mostly on silence is mis-timed (or invented).
    """
    import numpy as np

    if not words:
        return {"words": 0, "on_silence": 0, "on_silence_ratio": 0.0, "mean_speech_coverage": 0.0}
    cover: list[float] = []
    bad = 0
    for _text, t0, t1 in words:
        a = max(0, int(t0 / hop))
        b = max(a + 1, min(len(rms_db), int(np.ceil(t1 / hop))))
        seg = np.asarray(rms_db[a:b])
        frac = float(np.mean(seg > silence_db)) if len(seg) else 0.0
        cover.append(frac)
        if frac < 0.25:
            bad += 1
    return {
        "words": len(words),
        "on_silence": bad,
        "on_silence_ratio": round(bad / len(words), 4),
        "mean_speech_coverage": round(float(np.mean(cover)), 4),
    }
