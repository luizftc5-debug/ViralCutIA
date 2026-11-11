"""Manual transcript correction: fix what the recogniser got wrong, keep when it was said.

Corrections change *text only*. Every word keeps the time it was spoken at; when a correction
changes how many words a stretch has, the stretch's time span is shared between the new words
in proportion to their length. Nothing here invents speech: the user is the author of the text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .transcribe import Word

_STRIP = ".,;:!?\"'()[]{}…¿¡-–—"


def _key(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower().strip(_STRIP)
    return t


def _split_punct(text: str) -> tuple[str, str, str]:
    """(leading punctuation, core, trailing punctuation)."""
    m = re.match(rf"^([{re.escape(_STRIP)}]*)(.*?)([{re.escape(_STRIP)}]*)$", text, re.S)
    assert m is not None
    return m.group(1), m.group(2), m.group(3)


@dataclass
class Replacement:
    find: str
    replace: str
    case_sensitive: bool = False


def _redistribute(old: list[Word], tokens: list[str]) -> list[Word]:
    t0, t1 = old[0].start, old[-1].end
    if len(tokens) == len(old):
        return [Word(t, o.start, o.end, o.prob) for t, o in zip(tokens, old, strict=True)]
    weights = [len(t) + 1.5 for t in tokens]
    total = sum(weights)
    out, cursor = [], t0
    for i, (tok, w) in enumerate(zip(tokens, weights, strict=True)):
        d = (t1 - t0) * w / total
        last = i == len(tokens) - 1
        out.append(Word(tok, round(cursor, 3), round(t1 if last else cursor + d * 0.94, 3), 1.0))
        cursor += d
    return out


def apply_replacements(words: list[Word], replacements: list[Replacement]) -> tuple[list[Word], list[int]]:
    """Whole-word find and replace across the transcript. Returns the new words and the indices
    (in the *new* list) that changed. Punctuation around a match is preserved."""
    cur = list(words)
    changed: set[int] = set()
    for rep in replacements:
        needle = [_key(t) if not rep.case_sensitive else t.strip(_STRIP) for t in rep.find.split()]
        new_tokens = rep.replace.split()
        if not needle:
            continue
        out: list[Word] = []
        out_changed: set[int] = set()
        i = 0
        while i < len(cur):
            window = cur[i : i + len(needle)]
            hay = [(_key(w.text) if not rep.case_sensitive else w.text.strip(_STRIP)) for w in window]
            if len(window) == len(needle) and hay == needle:
                lead, _, _ = _split_punct(window[0].text)
                _, _, trail = _split_punct(window[-1].text)
                tokens = list(new_tokens)
                if tokens:
                    tokens[0] = lead + tokens[0]
                    if not tokens[-1].endswith(tuple(_STRIP)):
                        tokens[-1] = tokens[-1] + trail
                    start = len(out)
                    out.extend(_redistribute(window, tokens))
                    out_changed.update(range(start, len(out)))
                i += len(needle)
            else:
                out.append(cur[i])
                if i in changed:
                    out_changed.add(len(out) - 1)
                i += 1
        # indices of earlier replacements shift when counts change: recompute by identity
        cur, changed = out, out_changed
    return cur, sorted(changed)


def apply_word_edits(words: list[Word], edits: dict[int, str]) -> tuple[list[Word], list[int]]:
    """Replace the text of individual words by index (timings untouched)."""
    out = list(words)
    changed = []
    for k, text in sorted(edits.items()):
        if 0 <= k < len(out) and text.strip() and out[k].text != text.strip():
            w = out[k]
            out[k] = Word(text.strip(), w.start, w.end, 1.0)
            changed.append(k)
    return out, changed
