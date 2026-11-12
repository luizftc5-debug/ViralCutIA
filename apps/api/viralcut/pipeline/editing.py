"""Stage 8b — editing decisions: what kind of content is this, and where does a move help?

Nothing here is a template. Every decision is computed from measurements of this clip (what
was said, how loud, how fast, where the pauses and the subject are) and carries the evidence
that justified it, so the studio can show *why* a zoom or a cut exists and tests can check it.

* ``classify_content`` — podcast/interview, gaming, education, entertainment, storytelling or
  general, from vocabulary, speech rhythm, vocal dynamics, motion and faces. It returns the
  scores and the matched evidence, not just a label.
* ``plan_zooms`` — controlled punch-ins on words the speaker really stresses (louder than their
  own baseline, a number, a key term), spaced out, eased in and out, never across a cut and
  never on layouts where zooming would hide content (split screen, letterboxed gameplay).
* ``tighten_opening`` — drops a pure disfluency ("uh", "hum") from the very start of a clip so
  the first frame is the first real word. Nothing is reordered or invented.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from .audio import AudioSignals
from .transcribe import Word
from .visual import VisualSignals

# --------------------------------------------------------------------------- content profile


def _canon(w: str) -> str:
    w = unicodedata.normalize("NFD", w.lower())
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9']+", "", w)


_LEX: dict[str, dict[str, set[str]]] = {
    "gaming": {
        "en": {
            "game",
            "gameplay",
            "boss",
            "loot",
            "noob",
            "clutch",
            "headshot",
            "respawn",
            "level",
            "ranked",
            "fps",
            "streamer",
            "stream",
            "lobby",
            "squad",
            "teammate",
            "teammates",
            "controller",
            "console",
            "minecraft",
            "fortnite",
            "valorant",
            "cod",
            "gta",
            "speedrun",
            "nerf",
            "buff",
            "grind",
            "quest",
        },
        "pt": {
            "jogo",
            "jogos",
            "gameplay",
            "boss",
            "loot",
            "noob",
            "clutch",
            "headshot",
            "respawn",
            "nivel",
            "ranqueada",
            "fps",
            "streamer",
            "live",
            "lobby",
            "time",
            "partida",
            "controle",
            "console",
            "minecraft",
            "fortnite",
            "valorant",
            "gta",
            "speedrun",
            "mapa",
            "missao",
            "fase",
            "jogador",
        },
    },
    "education": {
        "en": {
            "learn",
            "lesson",
            "step",
            "steps",
            "first",
            "second",
            "third",
            "example",
            "rule",
            "tip",
            "tips",
            "because",
            "means",
            "define",
            "concept",
            "method",
            "how",
            "why",
            "explain",
            "understand",
            "study",
            "practice",
            "skill",
            "important",
            "remember",
            "framework",
            "principle",
        },
        "pt": {
            "aprender",
            "aula",
            "passo",
            "passos",
            "primeiro",
            "segundo",
            "terceiro",
            "exemplo",
            "regra",
            "dica",
            "dicas",
            "porque",
            "significa",
            "conceito",
            "metodo",
            "como",
            "explicar",
            "entender",
            "estudar",
            "praticar",
            "habilidade",
            "importante",
            "lembre",
            "principio",
            "estudo",
        },
    },
    "storytelling": {
        # narrative-specific markers only: function words such as "was", "then" or "quando" are in
        # every kind of speech and would classify ordinary talk as a story
        "en": {
            "ago",
            "suddenly",
            "remember",
            "remembered",
            "happened",
            "story",
            "finally",
            "once",
            "lost",
            "woke",
            "realized",
            "turned",
            "decided",
            "eventually",
            "years",
            "childhood",
            "stood",
        },
        "pt": {
            "atras",
            "repente",
            "lembro",
            "lembrei",
            "aconteceu",
            "historia",
            "finalmente",
            "perdi",
            "acordei",
            "percebi",
            "decidi",
            "eventualmente",
            "anos",
            "infancia",
            "naquele",
            "naquela",
        },
    },
    "podcast": {
        "en": {
            "guest",
            "podcast",
            "episode",
            "interview",
            "welcome",
            "thanks",
            "question",
            "opinion",
            "think",
            "agree",
            "exactly",
            "right",
            "yeah",
            "mean",
            "talk",
            "conversation",
        },
        "pt": {
            "convidado",
            "podcast",
            "episodio",
            "entrevista",
            "bem-vindo",
            "obrigado",
            "pergunta",
            "opiniao",
            "acho",
            "concordo",
            "exatamente",
            "verdade",
            "sim",
            "quer",
            "dizer",
            "conversa",
        },
    },
    "entertainment": {
        "en": {
            "haha",
            "lol",
            "crazy",
            "insane",
            "omg",
            "wow",
            "funny",
            "joke",
            "hilarious",
            "wild",
            "no way",
            "literally",
            "dude",
            "bro",
            "unbelievable",
            "amazing",
        },
        "pt": {
            "kkk",
            "kkkk",
            "rs",
            "loucura",
            "surreal",
            "nossa",
            "caramba",
            "engracado",
            "piada",
            "mano",
            "cara",
            "inacreditavel",
            "incrivel",
            "absurdo",
            "bizarro",
            "mds",
        },
    },
}


@dataclass
class ContentProfile:
    name: str  # podcast | gaming | education | entertainment | storytelling | general
    confidence: float
    scores: dict[str, float]
    evidence: list[str]
    # editing parameters derived from the profile
    max_pause: float | None
    zoom: dict  # {"allowed", "min_gap", "amp", "ramp", "max_share"}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": round(self.confidence, 2),
            "scores": {k: round(v, 2) for k, v in self.scores.items()},
            "evidence": self.evidence,
            "max_pause": self.max_pause,
            "zoom": self.zoom,
        }


# max_pause: longest pause kept between words. None disables pause trimming.
_PROFILE_PARAMS: dict[str, dict] = {
    "podcast": {"max_pause": 0.7, "zoom": {"allowed": True, "min_gap": 8.0, "amp": 0.05, "ramp": 0.45, "max_share": 0.25}},
    "gaming": {"max_pause": None, "zoom": {"allowed": False, "min_gap": 99.0, "amp": 0.0, "ramp": 0.3, "max_share": 0.0}},
    "education": {"max_pause": 0.6, "zoom": {"allowed": True, "min_gap": 6.0, "amp": 0.06, "ramp": 0.4, "max_share": 0.3}},
    "entertainment": {"max_pause": 0.5, "zoom": {"allowed": True, "min_gap": 4.5, "amp": 0.08, "ramp": 0.3, "max_share": 0.35}},
    "storytelling": {"max_pause": 0.9, "zoom": {"allowed": True, "min_gap": 9.0, "amp": 0.07, "ramp": 0.6, "max_share": 0.3}},
    "general": {"max_pause": 0.7, "zoom": {"allowed": True, "min_gap": 6.0, "amp": 0.06, "ramp": 0.4, "max_share": 0.3}},
}  # fmt: skip

_STYLE_TO_PROFILE = {"podcast": "podcast", "gaming": "gaming"}


def classify_content(
    words: list[Word],
    language: str,
    audio: AudioSignals,
    visual: VisualSignals,
    start: float,
    end: float,
    *,
    face_ratio: float = 0.0,
    split_layout: bool = False,
    user_style: str = "auto",
) -> ContentProfile:
    """Classify a clip from measurable signals. An explicit user style always wins."""
    lang = "pt" if language[:2] == "pt" else "en"
    n = max(1, len(words))
    tokens = [_canon(w.text) for w in words]
    dur = max(1.0, end - start)
    evidence: list[str] = []
    scores: dict[str, float] = {}

    for name, per_lang in _LEX.items():
        lex = per_lang[lang]
        hits = [t for t in tokens if t in lex]
        density = len(hits) / n
        # ~11 % matching words saturates; a single stray hit ("yeah") is not evidence of a category
        scores[name] = min(1.0, density * 9.0) * min(1.0, len(hits) / 4.0)
        if len(hits) >= 3 and density >= 0.03:
            evidence.append(f"{name}: {len(hits)} vocabulary hits ({', '.join(sorted(set(hits))[:5])})")

    motion = visual.motion_level(start, end)
    motion_pct = visual.motion_percentile(motion)
    dynamics = audio.dynamics(start, end)
    wpm = 60.0 * len(words) / dur
    questions = sum(1 for w in words if w.text.rstrip().endswith("?"))
    exclaims = sum(1 for w in words if w.text.rstrip().endswith("!"))

    # picture: no faces + lots of motion is how gameplay and screen recordings look
    if face_ratio < 0.15 and motion_pct > 0.6:
        scores["gaming"] += 0.35
        evidence.append(f"gaming: no face in {1 - face_ratio:.0%} of the picture with above-average motion")
    # picture + rhythm: a talking head or two, low motion, long continuous speech
    if face_ratio > 0.6 and motion_pct < 0.55:
        scores["podcast"] += 0.25
        evidence.append(f"podcast: face visible {face_ratio:.0%} of the time with a calm picture")
    if split_layout:
        scores["podcast"] += 0.3
        evidence.append("podcast: two people share the frame")
    if questions >= 2 and n / max(1, questions) < 40:
        scores["podcast"] += 0.12
    if exclaims >= 2 or (dynamics > 7.5 and motion_pct > 0.5):
        scores["entertainment"] += 0.25
        evidence.append(
            f"entertainment: expressive delivery (vocal dynamics {dynamics:.1f} dB, {exclaims} exclamations)"
        )
    if wpm < 135 and scores["storytelling"] > 0.15:
        scores["storytelling"] += 0.1  # a measured pace carries a narrative

    forced = _STYLE_TO_PROFILE.get(user_style) or (user_style if user_style in _PROFILE_PARAMS else None)
    best = max(scores, key=lambda k: scores[k])
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
    if forced:
        name, conf = forced, 1.0
        evidence.insert(0, f"style chosen by the user: {user_style}")
    elif ranked[0] < 0.28 or margin < 0.06:
        name, conf = "general", max(0.0, 1.0 - ranked[0])
        evidence.append("no category stood out clearly: general-purpose editing")
    else:
        name, conf = best, min(1.0, 0.45 + margin + ranked[0] * 0.3)
    params = _PROFILE_PARAMS[name]
    return ContentProfile(
        name=name,
        confidence=conf,
        scores=scores,
        evidence=evidence,
        max_pause=params["max_pause"],
        zoom=dict(params["zoom"]),
    )


# --------------------------------------------------------------------------- opening


_DISFLUENCIES = {"um", "uh", "erm", "hmm", "hm", "ah", "eh", "ahn", "ne", "hum", "uhm", "umm"}


def tighten_opening(words: list[Word]) -> tuple[list[Word], str | None]:
    """Remove up to two pure disfluencies from the very start of the clip (never real words)."""
    i = 0
    while i < min(2, len(words) - 4) and _canon(words[i].text) in _DISFLUENCIES:
        i += 1
    if i == 0:
        return words, None
    removed = " ".join(w.text for w in words[:i])
    return words[i:], f"opening tightened: removed leading disfluency “{removed}”"


# --------------------------------------------------------------------------- zoom planning


@dataclass
class Zoom:
    t0: float  # output-timeline seconds
    t1: float
    amp: float  # 0.06 = 6 % punch-in
    ramp: float
    reason: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "t0": round(self.t0, 3),
            "t1": round(self.t1, 3),
            "amp": round(self.amp, 3),
            "ramp": self.ramp,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def word_emphasis(words: list[Word], audio: AudioSignals) -> list[dict]:
    """Per-word vocal emphasis relative to this speaker in this clip.

    ``lift_db``: how much louder than the clip's median word. ``stretch``: duration relative to
    what the word's length predicts (stressed words are held longer).
    """
    if not words:
        return []
    levels = np.array([audio.level(w.start, w.end) for w in words], dtype=np.float64)
    base = float(np.median(levels))
    spread = float(np.median(np.abs(levels - base))) or 1.0
    per_char = np.array([(w.end - w.start) / max(2, len(w.text)) for w in words])
    typical = float(np.median(per_char)) or 1e-3
    out = []
    for w, lv, pc in zip(words, levels, per_char, strict=True):
        out.append(
            {
                "lift_db": float(lv - base),
                "z": float((lv - base) / (1.4826 * spread)),
                "stretch": float(pc / typical),
                "word": w.text,
            }
        )
    return out


def plan_zooms(
    words: list[Word],  # source timeline (clip-relative not required: only used with `to_out`)
    to_out,  # noqa: ANN001 - callable: source second -> output second or None
    audio: AudioSignals,
    keywords: set[str],
    profile: ContentProfile,
    out_duration: float,
    cuts_out: list[float],
    *,
    enabled: bool = True,
) -> list[Zoom]:
    """Choose punch-in moments on the output timeline. Empty when zoom would not help."""
    z = profile.zoom
    if not enabled or not z["allowed"] or out_duration < 6.0 or len(words) < 8:
        return []
    emph = word_emphasis(words, audio)
    candidates: list[tuple[float, int, str, dict]] = []
    for i, (w, e) in enumerate(zip(words, emph, strict=True)):
        key = _canon(w.text)
        is_num = any(ch.isdigit() for ch in w.text)
        is_kw = key in keywords
        stressed = e["z"] >= 1.3 and e["lift_db"] >= 3.0
        if not (stressed or ((is_num or is_kw) and e["lift_db"] >= 1.5)):
            continue
        score = e["z"] + 0.6 * e["lift_db"] / 3 + (0.8 if is_num else 0) + (0.6 if is_kw else 0)
        score += 0.3 * max(0.0, e["stretch"] - 1.0)
        why = "vocal emphasis" + (" on a number" if is_num else " on a key term" if is_kw else "")
        candidates.append(
            (score, i, why, {"word": w.text, "lift_db": round(e["lift_db"], 1), "z": round(e["z"], 1)})
        )
    candidates.sort(key=lambda c: c[0], reverse=True)

    chosen: list[Zoom] = []
    ramp = float(z["ramp"])
    budget = z["max_share"] * out_duration
    used = 0.0
    top = candidates[0][0] if candidates else 1.0
    for score, i, why, ev in candidates:
        w = words[i]
        t_start = to_out(w.start)
        if t_start is None:
            continue
        t0 = max(0.0, t_start - 0.15)
        hold = max(0.5, (w.end - w.start) + 0.45)
        t1 = min(out_duration, t0 + ramp + hold + ramp)
        if t0 < 0.8 or t1 > out_duration - 0.35:
            continue  # leave the opening and the ending alone
        if any(t0 - ramp < c < t1 + ramp for c in cuts_out):
            continue  # never ramp across a hard cut
        if any(abs(t0 - o.t0) < z["min_gap"] or (t0 < o.t1 + 1.5 and t1 > o.t0 - 1.5) for o in chosen):
            continue
        if used + (t1 - t0) > budget:
            continue
        amp = float(z["amp"]) * (0.8 + 0.4 * min(1.0, score / top))
        chosen.append(Zoom(t0, t1, round(amp, 3), ramp, why, ev))
        used += t1 - t0
    chosen.sort(key=lambda c: c.t0)
    return chosen


def zoom_expression(zooms: list[tuple[float, float, float, float]], local_offset: float) -> str:
    """ffmpeg expression for the zoom factor at the frame time ``t`` of a segment.

    ``zooms`` are ``(t0, t1, amp, ramp)`` on the output timeline; ``local_offset`` is the output
    time at which the segment starts, so ``t + local_offset`` is the clip time. Each move is a
    smoothstep in and a smoothstep out (no mechanical linear ramps).
    """

    def smooth(x: str) -> str:
        c = f"clip({x},0,1)"
        return f"({c}*{c}*(3-2*{c}))"

    terms = []
    for t0, t1, amp, ramp in zooms:
        a0, a1 = t0 - local_offset, t0 + ramp - local_offset
        b0, b1 = t1 - ramp - local_offset, t1 - local_offset
        up = smooth(f"(t-{a0:.4f})/{ramp:.4f}")
        down = smooth(f"(t-{b0:.4f})/{ramp:.4f}")
        terms.append(f"{amp:.4f}*{up}*(1-{down})")
        _ = (a1, b1)
    return "1+" + "+".join(terms) if terms else "1"
