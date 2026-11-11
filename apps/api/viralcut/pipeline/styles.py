"""Video style presets: named bundles of *real* editing parameters.

A style does not paint a template over the video. It sets the knobs the editing engine already
uses — how much silence is removed (pacing), how often and how far the camera punches in on a
stressed word (zoom), how jump cuts are handled (transition), the default framing and caption
design, and the speech treatment of the audio — and the engine still applies them only where
the measurements justify a move (a zoom still needs a stressed word, a trim still needs real
silence). ``auto`` picks the style from the clip's measured content profile and says why.

Editing intensity scales the same knobs: ``subtle`` keeps more of the original rhythm,
``high`` cuts tighter and moves more, within hard limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Content profile (pipeline/editing.py) -> style chosen by "auto"
PROFILE_TO_STYLE = {
    "podcast": "podcast",
    "gaming": "gaming",
    "education": "educational",
    "entertainment": "reaction",
    "storytelling": "storytime",
    "general": "viral",
}

# Legacy option values (before the style library) keep working
LEGACY = {"dynamic": "viral", "editorial": "minimal"}


@dataclass(frozen=True)
class VideoStyle:
    key: str
    label: str
    description: str
    profile: str | None  # content profile forced by the style (None = detected from the clip)
    max_pause: float | None  # longest pause kept between words; None = no pause trimming
    zoom: dict  # {"allowed", "amp", "min_gap", "ramp", "max_share"} (see editing.plan_zooms)
    transition: str  # cut | punch | smooth (how jump cuts are handled, see render/assign_transitions)
    framing: str  # default framing mode when the user chose "auto"
    caption: str  # default caption style key (captions.STYLES)
    audio: str  # natural | voice | punchy (speech treatment, see render.plan_audio_polish)
    best_for: list[str] = field(default_factory=list)


def _z(allowed: bool, amp: float, min_gap: float, ramp: float, share: float) -> dict:
    return {"allowed": allowed, "amp": amp, "min_gap": min_gap, "ramp": ramp, "max_share": share}


STYLES: dict[str, VideoStyle] = {
    "viral": VideoStyle(
        "viral", "Viral Fast-Paced", "Tight pacing, punch-ins on stressed words, jump cuts hidden by quick punches.",
        None, 0.4, _z(True, 0.08, 4.5, 0.3, 0.35), "punch", "auto", "viral_dynamic", "punchy",
        ["talking head", "advice", "hot takes"],
    ),
    "podcast": VideoStyle(
        "podcast", "Podcast Clips", "Speaker tracking and split screen for two voices, gentle punch-ins, clear levelled speech.",
        "podcast", 0.65, _z(True, 0.05, 7.0, 0.45, 0.25), "cut", "auto", "podcast", "voice",
        ["interviews", "conversations"],
    ),
    "gaming": VideoStyle(
        "gaming", "Gaming Highlights", "Whole gameplay kept visible over a blurred backdrop, commentary rhythm untouched, impact captions.",
        "gaming", None, _z(False, 0.0, 99.0, 0.3, 0.0), "cut", "blur", "gaming", "punchy",
        ["gameplay", "screen recordings"],
    ),
    "cinematic": VideoStyle(
        "cinematic", "Cinematic Storytelling", "Long breaths kept, slow eased moves, no jump-cut effects, refined serif captions.",
        "storytelling", 1.1, _z(True, 0.05, 10.0, 0.8, 0.25), "smooth", "auto", "cinematic", "natural",
        ["narratives", "vlogs", "documentary"],
    ),
    "educational": VideoStyle(
        "educational", "Educational / Authority", "Clean structure, measured pacing, emphasis on numbers and key terms, clear voice.",
        "education", 0.6, _z(True, 0.06, 6.0, 0.4, 0.3), "cut", "auto", "clean_editorial", "voice",
        ["lessons", "explainers", "expert advice"],
    ),
    "reaction": VideoStyle(
        "reaction", "Reaction / Entertainment", "Emotional peaks emphasised, quick comedic timing, energetic captions.",
        "entertainment", 0.45, _z(True, 0.09, 4.0, 0.25, 0.4), "punch", "auto", "energy", "punchy",
        ["reactions", "comedy", "challenges"],
    ),
    "minimal": VideoStyle(
        "minimal", "Minimal Premium", "Natural editing, almost no movement, sophisticated typography.",
        None, 0.9, _z(True, 0.04, 12.0, 0.7, 0.15), "smooth", "auto", "clean_editorial", "natural",
        ["brands", "premium content", "interviews"],
    ),
    "storytime": VideoStyle(
        "storytime", "Storytime", "Opens on the most curious line, keeps the narrative rhythm, punches in on the payoff.",
        "storytelling", 0.75, _z(True, 0.07, 7.0, 0.5, 0.3), "cut", "auto", "kinetic", "natural",
        ["personal stories", "anecdotes"],
    ),
    "news": VideoStyle(
        "news", "News / Commentary", "Fact-first hooks, brisk pacing, editorial captions, levelled clear voice.",
        None, 0.5, _z(True, 0.06, 6.0, 0.35, 0.3), "cut", "auto", "high_retention", "voice",
        ["commentary", "news", "analysis"],
    ),
}  # fmt: skip

# How intensity scales each knob. Pause: multiplier on the longest kept pause (smaller = tighter).
INTENSITY = {
    "subtle": {"pause": 1.4, "amp": 0.6, "gap": 1.5, "share": 0.6},
    "balanced": {"pause": 1.0, "amp": 1.0, "gap": 1.0, "share": 1.0},
    "high": {"pause": 0.72, "amp": 1.35, "gap": 0.7, "share": 1.4},
}
MIN_PAUSE, MAX_PAUSE = 0.28, 1.6  # never cut into breaths, never keep dead air
MAX_ZOOM_AMP = 0.12  # beyond ~12 % the crop loses too much resolution and context


def resolve(key: str | None) -> str:
    key = LEGACY.get(key or "auto", key or "auto")
    return key if key in STYLES or key == "auto" else "auto"


@dataclass
class EditParams:
    style: str  # the concrete style applied (never "auto")
    chosen_by: str  # user | auto
    intensity: str
    max_pause: float | None
    zoom: dict
    transition: str
    audio: str
    caption: str
    framing: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "style": self.style,
            "label": STYLES[self.style].label,
            "chosen_by": self.chosen_by,
            "intensity": self.intensity,
            "max_pause": self.max_pause,
            "zoom": self.zoom,
            "transition": self.transition,
            "audio": self.audio,
            "caption": self.caption,
            "framing": self.framing,
            "reason": self.reason,
        }


def edit_params(
    style_key: str,
    intensity: str,
    profile_name: str,
    profile_evidence: list[str] | None = None,
    profile_params: tuple[float | None, dict] | None = None,
) -> EditParams:
    """Concrete editing parameters for a clip: the user's style, or the one its content calls for.

    With ``auto`` the pacing and zoom come from the detected content profile itself
    (``profile_params``: its measured-and-tuned ``max_pause`` and ``zoom``); the matching style
    contributes its transition, audio treatment and caption design.
    """
    key = resolve(style_key)
    auto = key == "auto"
    if auto:
        key = PROFILE_TO_STYLE.get(profile_name, "viral")
        ev = "; ".join((profile_evidence or [])[:2])
        reason = f"detected {profile_name} content" + (f" ({ev})" if ev else "") + f" → {STYLES[key].label}"
        chosen_by = "auto"
    else:
        reason = f"{STYLES[key].label} chosen by the user"
        chosen_by = "user"
    st = STYLES[key]
    k = INTENSITY.get(intensity, INTENSITY["balanced"])
    base_pause, base_zoom = profile_params if (auto and profile_params) else (st.max_pause, st.zoom)
    pause = None if base_pause is None else min(MAX_PAUSE, max(MIN_PAUSE, base_pause * k["pause"]))
    z = dict(base_zoom)
    if z["allowed"]:
        z["amp"] = round(min(MAX_ZOOM_AMP, z["amp"] * k["amp"]), 3)
        z["min_gap"] = round(z["min_gap"] * k["gap"], 2)
        z["max_share"] = round(min(0.5, z["max_share"] * k["share"]), 2)
    transition = st.transition
    if intensity == "subtle" and transition == "punch":
        transition = "cut"  # a subtle edit does not decorate its cuts
    return EditParams(
        key, chosen_by, intensity, pause, z, transition, st.audio, st.caption, st.framing, reason
    )


def recommend(dims: dict, profile_name: str, wpm: float, motion_pct: float, face_ratio: float) -> dict:
    """Recommend a style for a selected clip from its measured signals, with the reasons.

    Uses only what was measured (score dimensions, content profile, speech rate, motion, faces);
    it does not promise views. The user can always override it.
    """
    why: list[str] = []
    key = PROFILE_TO_STYLE.get(profile_name, "viral")
    if profile_name != "general":
        why.append(f"content reads as {profile_name}")
    hook = float(dims.get("hook", 0.0))
    narrative = float(dims.get("narrative", 0.0))
    value = float(dims.get("value", 0.0))
    if profile_name == "general":
        if narrative >= 0.6 and wpm < 150:
            key = "storytime"
            why.append(f"strong narrative arc ({narrative:.0%}) at a measured pace ({wpm:.0f} wpm)")
        elif value >= 0.6:
            key = "educational"
            why.append(f"high information value ({value:.0%})")
        elif hook >= 0.6 or wpm >= 165:
            key = "viral"
            why.append(f"strong opening ({hook:.0%}) and fast delivery ({wpm:.0f} wpm) suit a tight edit")
        else:
            key = "minimal"
            why.append("no single strong trait: a restrained edit keeps it natural")
    if key == "podcast" and face_ratio < 0.3:
        key = "educational"
        why.append("conversation without a visible speaker: no speaker tracking to show")
    if motion_pct > 0.75 and face_ratio < 0.2 and key not in ("gaming",):
        key = "gaming"
        why.append(f"busy picture ({motion_pct:.0%} motion percentile) with no face to follow")
    intensity = "high" if (hook >= 0.7 and wpm >= 160) else "subtle" if wpm < 120 else "balanced"
    return {
        "style": key,
        "label": STYLES[key].label,
        "intensity": intensity,
        "reasons": why or ["general-purpose content"],
    }
