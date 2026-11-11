"""Creator Studio: the 13 specialised editors and the content presets.

Each editor is a configuration block that the existing pipeline stages read; there is no second
editing system. Where an editor acts:

| Editor            | Stage that applies it                          | What it really does |
|-------------------|------------------------------------------------|---------------------|
| /corte            | step_framing (trimming.plan_keeps)             | removes silences longer than the pause limit and filler words, on the waveform |
| /tomada           | step_framing                                   | drops earlier takes of a sentence the speaker repeated (keeps the last take) |
| /ganchos          | clip selection + overlays                      | starts the clip on a measurably stronger hook sentence; optional on-screen hook line |
| /legenda          | step_captions                                  | word-synced captions (CaptionOptions: styles + Caption Studio) |
| /lettering        | overlays (ASS)                                 | animated title (pop / typewriter / slide) |
| /animacao         | overlays (ASS)                                 | progress bar, emphasis bursts on stressed words / keywords |
| /3d               | render (video post chain)                      | 2.5D camera: slow dolly push with perspective sway (no depth estimation) |
| /trilha           | render (audio graph)                           | user-supplied music: beat-aligned start, fades, sidechain ducking under speech |
| /tela-verde       | render (composite)                             | chroma key, or AI person matting (MODNet) over blur / colour / image |
| /prova            | render (image overlays)                        | user-supplied screenshots/evidence cards at a time or when a keyword is spoken |
| /texto-atras      | render (two-pass composite)                    | big text placed *behind* the person using the AI matte |
| /referencia       | before editing (reference.py)                  | measures a reference video (shot length, pauses, dynamics, text band) and adapts the edit |
| /enquadramento    | step_framing (reframe.py)                      | ViralCut's face-tracked reframing and punch-ins (13th editor: the capability already in this repository) |

Heavy editors (`/tela-verde` in AI mode, `/texto-atras`, `/3d`) are off by default, load their
model only when used, and follow the job's resource profile.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_HEX = r"^(#[0-9a-fA-F]{6})?$"


class _Ed(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    enabled: bool = False


class Corte(_Ed):
    enabled: bool = True
    max_pause: float | None = Field(default=None, ge=0.2, le=2.0)  # None = the video style decides
    remove_fillers: bool = True


class Tomada(_Ed):
    enabled: bool = True
    similarity: float = Field(default=0.6, ge=0.4, le=0.95)


class Ganchos(_Ed):
    enabled: bool = True
    start_on_hook: bool = False  # start on a later sentence when it is a clearly stronger opening
    hook_text: bool = False  # show the opening line as on-screen text for the first seconds


class Lettering(_Ed):
    text: str = Field(default="", max_length=80)  # empty = the clip's best title
    animation: Literal["pop", "typewriter", "slide"] = "pop"
    position: Literal["top", "middle"] = "top"
    duration: float = Field(default=2.8, ge=1.0, le=8.0)
    color: str = Field(default="", pattern=_HEX)


class Animacao(_Ed):
    progress_bar: bool = True
    emphasis: bool = True
    color: str = Field(default="", pattern=_HEX)


class Depth3D(_Ed):
    strength: Literal["subtle", "medium", "strong"] = "subtle"


class Trilha(_Ed):
    asset: str = ""  # id of an uploaded music file (user-supplied / licensed)
    volume_db: float = Field(default=-20.0, ge=-40.0, le=-6.0)
    ducking: bool = True
    beat_sync: bool = True


class TelaVerde(_Ed):
    mode: Literal["chroma", "ai"] = "ai"
    background: Literal["blur", "color", "image"] = "blur"
    color: str = Field(default="#101820", pattern=_HEX)
    asset: str = ""  # background image id when background == "image"


class ProofItem(BaseModel):
    asset: str
    at: float | None = Field(default=None, ge=0)  # seconds into the clip; None = when `keyword` is spoken
    keyword: str = Field(default="", max_length=40)
    duration: float = Field(default=3.0, ge=1.0, le=10.0)


class Prova(_Ed):
    items: list[ProofItem] = Field(default_factory=list, max_length=10)


class TextoAtras(_Ed):
    text: str = Field(default="", max_length=24)  # empty = the strongest keyword of the clip
    color: str = Field(default="#FFFFFF", pattern=_HEX)
    size: float = Field(default=1.0, ge=0.5, le=1.6)


class Referencia(_Ed):
    asset: str = ""  # id of an uploaded reference video


class Enquadramento(_Ed):
    enabled: bool = True  # mirrors JobOptions.framing / smart_zoom (single source of truth there)


class Legenda(_Ed):
    enabled: bool = True  # mirrors CaptionOptions.enabled


class EditorsConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    corte: Corte = Field(default_factory=Corte)
    referencia: Referencia = Field(default_factory=Referencia)
    lettering: Lettering = Field(default_factory=Lettering)
    depth_3d: Depth3D = Field(default_factory=Depth3D, alias="3d")
    trilha: Trilha = Field(default_factory=Trilha)
    tela_verde: TelaVerde = Field(default_factory=TelaVerde, alias="tela-verde")
    animacao: Animacao = Field(default_factory=Animacao)
    tomada: Tomada = Field(default_factory=Tomada)
    legenda: Legenda = Field(default_factory=Legenda)
    prova: Prova = Field(default_factory=Prova)
    ganchos: Ganchos = Field(default_factory=Ganchos)
    texto_atras: TextoAtras = Field(default_factory=TextoAtras, alias="texto-atras")
    enquadramento: Enquadramento = Field(default_factory=Enquadramento)

    @field_validator("*", mode="before")
    @classmethod
    def _none_is_default(cls, v):  # noqa: ANN001
        return {} if v is None else v

    @property
    def needs_matte(self) -> bool:
        return (self.tela_verde.enabled and self.tela_verde.mode == "ai") or self.texto_atras.enabled

    def assets(self) -> list[str]:
        ids = [self.trilha.asset, self.tela_verde.asset, self.referencia.asset]
        ids += [i.asset for i in self.prova.items]
        return [i for i in ids if i]


# Registry shown in the Creator Studio. `controls` describe the UI for each field.
EDITORS: list[dict] = [
    {"key": "corte", "command": "/corte", "label": "Cut", "heavy": False,
     "description": "Removes dead air and filler words on the waveform; keeps natural breaths.",
     "controls": [{"name": "max_pause", "type": "number", "min": 0.2, "max": 2.0, "step": 0.05, "label": "Longest pause kept (s)", "auto": "Video style"},
                  {"name": "remove_fillers", "type": "bool", "label": "Remove filler words (um, uh, é, tipo…)"}]},
    {"key": "referencia", "command": "/referencia", "label": "Edit like a reference", "heavy": False,
     "description": "Measures a reference video (shot length, pauses, energy, caption band) and adapts this edit to it.",
     "controls": [{"name": "asset", "type": "asset", "kind": "reference", "label": "Reference video"}]},
    {"key": "lettering", "command": "/lettering", "label": "Lettering", "heavy": False,
     "description": "An animated title on screen at the start of the clip.",
     "controls": [{"name": "text", "type": "text", "label": "Title (empty = best title)"},
                  {"name": "animation", "type": "select", "options": ["pop", "typewriter", "slide"], "label": "Animation"},
                  {"name": "position", "type": "select", "options": ["top", "middle"], "label": "Position"},
                  {"name": "duration", "type": "number", "min": 1, "max": 8, "step": 0.5, "label": "Seconds on screen"},
                  {"name": "color", "type": "color", "label": "Colour"}]},
    {"key": "3d", "command": "/3d", "label": "3D camera", "heavy": True,
     "description": "2.5D camera move: slow dolly push with a gentle perspective sway. Costs extra render time.",
     "controls": [{"name": "strength", "type": "select", "options": ["subtle", "medium", "strong"], "label": "Strength"}]},
    {"key": "trilha", "command": "/trilha", "label": "Music", "heavy": False,
     "description": "Your own (licensed) music under the voice: starts on a beat, fades, ducks while someone speaks.",
     "controls": [{"name": "asset", "type": "asset", "kind": "music", "label": "Music file"},
                  {"name": "volume_db", "type": "number", "min": -40, "max": -6, "step": 1, "label": "Music level (dB)"},
                  {"name": "ducking", "type": "bool", "label": "Lower the music under speech"},
                  {"name": "beat_sync", "type": "bool", "label": "Start on the first strong beat"}]},
    {"key": "tela-verde", "command": "/tela-verde", "label": "Green screen", "heavy": True,
     "description": "Replaces the background: chroma key for green/blue screens, or AI person matting for any video.",
     "controls": [{"name": "mode", "type": "select", "options": ["ai", "chroma"], "label": "Method"},
                  {"name": "background", "type": "select", "options": ["blur", "color", "image"], "label": "New background"},
                  {"name": "color", "type": "color", "label": "Background colour"},
                  {"name": "asset", "type": "asset", "kind": "image", "label": "Background image"}]},
    {"key": "animacao", "command": "/animacao", "label": "Motion graphics", "heavy": False,
     "description": "Progress bar and emphasis bursts synced to stressed words and keywords.",
     "controls": [{"name": "progress_bar", "type": "bool", "label": "Progress bar"},
                  {"name": "emphasis", "type": "bool", "label": "Emphasis bursts"},
                  {"name": "color", "type": "color", "label": "Colour"}]},
    {"key": "tomada", "command": "/tomada", "label": "Best take", "heavy": False,
     "description": "When a sentence is said twice in a row, keeps only the last (usually the good) take.",
     "controls": [{"name": "similarity", "type": "number", "min": 0.4, "max": 0.95, "step": 0.05, "label": "How similar counts as a retake"}]},
    {"key": "legenda", "command": "/legenda", "label": "Captions", "heavy": False,
     "description": "Word-synced captions with premium styles (Caption Studio).", "controls": []},
    {"key": "prova", "command": "/prova", "label": "Proof", "heavy": False,
     "description": "Shows your screenshots, documents or references as cards at a time or when a word is spoken.",
     "controls": [{"name": "items", "type": "proofs", "label": "Evidence images"}]},
    {"key": "ganchos", "command": "/ganchos", "label": "Hooks", "heavy": False,
     "description": "Finds the strongest authentic opening; can start the clip on it and show it on screen.",
     "controls": [{"name": "start_on_hook", "type": "bool", "label": "Start on the strongest opening"},
                  {"name": "hook_text", "type": "bool", "label": "Show the opening line on screen"}]},
    {"key": "texto-atras", "command": "/texto-atras", "label": "Text behind subject", "heavy": True,
     "description": "A large word placed behind the person (AI matting). Best on static, well-lit talking heads.",
     "controls": [{"name": "text", "type": "text", "label": "Text (empty = main keyword)"},
                  {"name": "color", "type": "color", "label": "Colour"},
                  {"name": "size", "type": "number", "min": 0.5, "max": 1.6, "step": 0.1, "label": "Size"}]},
    {"key": "enquadramento", "command": "/enquadramento", "label": "Reframing", "heavy": False,
     "description": "Face tracking to fill the frame, split screen for two people, punch-ins on stressed words.",
     "controls": []},
]  # fmt: skip


# --------------------------------------------------------------------------- presets

# Each preset sets real parameters: video style + intensity (pacing, moves, transitions, audio),
# caption design, framing, content type and the editors it switches on.
PRESETS: dict[str, dict] = {
    "auto": {"label": "AI Automatic", "description": "Analyses each clip and picks the editing strategy its content calls for.",
             "options": {"style": "auto", "intensity": "balanced", "captions": {"style": "auto"}},
             "editors": {}},
    "gaming_highlights": {"label": "Gaming Viral Highlights", "description": "Gameplay kept whole, high-energy captions, emphasis bursts, punchy opening.",
                          "options": {"style": "gaming", "intensity": "high", "captions": {"style": "gaming"}},
                          "editors": {"animacao": {"enabled": True, "progress_bar": False}, "ganchos": {"start_on_hook": True}, "lettering": {"enabled": True, "animation": "pop"}}},
    "gaming_news": {"label": "Gaming News & Rankings", "description": "Fact-first opening, editorial titles, evidence cards and a progress bar.",
                    "options": {"style": "news", "intensity": "balanced", "captions": {"style": "energy"}},
                    "editors": {"lettering": {"enabled": True, "animation": "slide"}, "animacao": {"enabled": True}, "prova": {"enabled": True}}},
    "podcast": {"label": "Podcast Viral Clips", "description": "Speaker tracking, tight pacing, starts on the strongest statement.",
                "options": {"style": "podcast", "intensity": "balanced", "captions": {"style": "podcast"}},
                "editors": {"ganchos": {"start_on_hook": True, "hook_text": True}, "corte": {"remove_fillers": True}}},
    "faceless": {"label": "Faceless YouTube", "description": "Narration-led: kinetic captions, evidence cards, slow 3D camera on your visuals.",
                 "options": {"style": "educational", "intensity": "balanced", "captions": {"style": "kinetic"}, "framing": "blur"},
                 "editors": {"animacao": {"enabled": True}, "prova": {"enabled": True}, "3d": {"enabled": True, "strength": "subtle"}}},
    "curiosities": {"label": "Viral Curiosities", "description": "Curiosity-driven opening on screen, typewriter title, reveals.",
                    "options": {"style": "storytime", "intensity": "balanced", "captions": {"style": "kinetic"}},
                    "editors": {"ganchos": {"start_on_hook": True, "hook_text": True}, "lettering": {"enabled": True, "animation": "typewriter"}}},
    "documentary": {"label": "Storytelling & Documentary", "description": "Cinematic pacing, serif captions, slow camera, music ducked under the voice.",
                    "options": {"style": "cinematic", "intensity": "subtle", "captions": {"style": "cinematic"}},
                    "editors": {"3d": {"enabled": True, "strength": "subtle"}, "trilha": {"enabled": True, "volume_db": -24}}},
    "rankings": {"label": "Top 5 / Top 10 Rankings", "description": "Moments counted down with rank titles, ordered weakest to strongest.",
                 "options": {"style": "viral", "intensity": "high", "captions": {"style": "energy"}, "content_type": "compilation", "clip_count": 5},
                 "editors": {"lettering": {"enabled": True, "animation": "pop"}, "animacao": {"enabled": True, "progress_bar": False}}},
    "news": {"label": "News & Trending Topics", "description": "Fast context, editorial captions, sliding titles, evidence cards.",
             "options": {"style": "news", "intensity": "balanced", "captions": {"style": "high_retention"}},
             "editors": {"lettering": {"enabled": True, "animation": "slide"}, "prova": {"enabled": True}}},
    "educational": {"label": "Educational Content", "description": "Clear pacing, numbers and key terms emphasised, progress bar.",
                    "options": {"style": "educational", "intensity": "balanced", "captions": {"style": "clean_editorial"}},
                    "editors": {"animacao": {"enabled": True}}},
    "reactions": {"label": "Entertainment & Reactions", "description": "Comedic timing, quick punches on cuts, expressive captions.",
                  "options": {"style": "reaction", "intensity": "high", "captions": {"style": "energy"}},
                  "editors": {"animacao": {"enabled": True, "progress_bar": False}}},
    "premium": {"label": "Premium Cinematic", "description": "Minimal, restrained editing with refined typography and a slow camera.",
                "options": {"style": "minimal", "intensity": "subtle", "captions": {"style": "cinematic"}},
                "editors": {"3d": {"enabled": True, "strength": "subtle"}}},
}  # fmt: skip


def apply_preset(options: dict, preset: str) -> dict:
    """Return job options with ``preset`` applied (explicit editor assets are kept)."""
    p = PRESETS.get(preset)
    if p is None:
        return options
    out = dict(options)
    for k, v in p["options"].items():
        if k == "clip_count" and out.get("clip_count") not in (None, "auto"):
            continue  # a number the user chose wins over the preset's default
        if k == "captions":
            out["captions"] = {**(out.get("captions") or {}), **v}
        else:
            out[k] = v
    editors = {k: dict(v) for k, v in (out.get("editors") or {}).items()}
    for key, cfg in p["editors"].items():
        editors[key] = {**editors.get(key, {}), **cfg}
    out["editors"] = editors
    out["preset"] = preset
    return out
