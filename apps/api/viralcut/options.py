"""User-facing processing options (validated at the API boundary, consumed by the pipeline)."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CaptionStyle = Literal[
    "auto",  # the caption design of the clip's video style
    "viral_dynamic", "energy", "podcast", "kinetic", "clean_editorial", "gaming", "cinematic", "high_retention",
    "dynamic_creator",
]  # fmt: skip
FramingMode = Literal["auto", "face", "speaker", "split", "blur", "center"]
VideoStyleKey = Literal[
    "auto", "viral", "podcast", "gaming", "cinematic", "educational", "reaction", "minimal", "storytime", "news",
    "dynamic", "editorial",  # legacy values, mapped to viral / minimal
]  # fmt: skip

DURATION_RANGES: dict[str, tuple[float, float]] = {
    "15-30": (15.0, 30.0),
    "30-60": (30.0, 60.0),
    "60-90": (60.0, 90.0),
    "auto": (18.0, 75.0),
}
CUSTOM_LIMITS = (5.0, 180.0)

# Longest pause kept between words, per editing style (None = smart trimming disabled).
TRIM_MAX_PAUSE: dict[str, float | None] = {
    "dynamic": 0.45,
    "editorial": 1.0,
    "podcast": 0.7,
    "gaming": None,
    "auto": 0.7,
}  # legacy table; the style library (pipeline/styles.py) now derives this per clip

# UI overlays of each platform, as fractions of the 9:16 frame that captions must avoid.
SAFE_AREAS: dict[str, dict[str, float]] = {
    "shorts": {"top": 0.10, "bottom": 0.20, "side": 0.08},
    "reels": {"top": 0.11, "bottom": 0.21, "side": 0.08},
    "tiktok": {"top": 0.10, "bottom": 0.22, "side": 0.09},
    "universal": {"top": 0.11, "bottom": 0.22, "side": 0.09},
}

LANGUAGES = {"auto", "en", "pt", "es", "fr", "de", "it", "ja", "ko", "zh", "hi", "ar", "ru", "nl", "tr", "pl"}


class CaptionOptions(BaseModel):
    enabled: bool = True
    style: CaptionStyle = "high_retention"
    highlight_keywords: bool = True
    size: Literal["s", "m", "l", "xl"] = "m"
    position: Literal["top", "middle", "bottom"] = "bottom"
    color: str = ""  # highlight colour of the spoken word, #RRGGBB; empty = style default
    font: Literal["auto", "inter", "poppins", "anton", "dm_serif"] = "auto"
    animation: bool = True
    # --- Caption Studio overrides (empty / "auto" / None = the style's own value)
    text_color: str = ""  # colour of words not being spoken
    keyword_color: str = ""  # colour of highlighted keywords
    weight: Literal["auto", "medium", "bold", "heavy"] = "auto"
    case: Literal["auto", "upper", "normal"] = "auto"
    words_per_page: int | None = Field(default=None, ge=1, le=8)
    max_lines: int | None = Field(default=None, ge=1, le=3)
    line_spacing: float | None = Field(default=None, ge=0.8, le=1.6)
    outline: float | None = Field(default=None, ge=0, le=16)
    shadow: float | None = Field(default=None, ge=0, le=12)
    background: Literal["auto", "none", "box"] = "auto"  # a rounded translucent panel behind each page
    animation_type: Literal["auto", "color", "pop", "box", "reveal", "none"] = "auto"
    animation_intensity: Literal["subtle", "normal", "strong"] = "normal"
    keyword_mode: Literal["auto", "color", "color_scale", "off"] = "auto"
    offset: float = Field(default=0.0, ge=-0.2, le=0.2)  # vertical nudge, fraction of the frame height

    @field_validator("color", "text_color", "keyword_color")
    @classmethod
    def _hex(cls, v: str) -> str:
        v = (v or "").strip()
        if v and not re.fullmatch(r"#[0-9a-fA-F]{6}", v):
            raise ValueError("colours must be #RRGGBB")
        return v.upper()


class JobOptions(BaseModel):
    captions: CaptionOptions = Field(default_factory=CaptionOptions)
    language: str = "auto"
    clip_count: int | Literal["auto"] = "auto"
    duration: Literal["15-30", "30-60", "60-90", "auto", "custom"] = "auto"
    custom_min: float = Field(default=20.0, ge=CUSTOM_LIMITS[0], le=CUSTOM_LIMITS[1])
    custom_max: float = Field(default=45.0, ge=CUSTOM_LIMITS[0], le=CUSTOM_LIMITS[1])
    platform: Literal["shorts", "reels", "tiktok", "universal"] = "universal"
    aspect: Literal["9:16"] = "9:16"
    style: VideoStyleKey = "auto"  # video style preset (pipeline/styles.py)
    intensity: Literal["subtle", "balanced", "high"] = "balanced"
    variations: int = Field(default=1, ge=1, le=3)  # versions rendered per selected clip
    quality: Literal["high", "max"] = "high"
    framing: FramingMode = "auto"
    smart_zoom: bool = True  # controlled punch-ins on words the speaker really stresses
    review_transcript: bool = (
        False  # pause after clip selection so the transcript can be corrected before rendering
    )
    glossary: str = Field(
        default="", max_length=400
    )  # names / terms the recogniser should prefer, comma separated
    min_score: float | None = Field(default=None, ge=0, le=100)
    performance: Literal["auto", "eco", "balanced", "turbo"] = "auto"  # resource profile for this job

    @field_validator("glossary")
    @classmethod
    def _glossary(cls, v: str) -> str:
        return ", ".join(t.strip() for t in re.split(r"[,\n;]", v or "") if t.strip())[:400]

    @field_validator("language")
    @classmethod
    def _lang(cls, v: str) -> str:
        v = (v or "auto").lower()
        if v not in LANGUAGES:
            raise ValueError("unsupported language")
        return v

    @field_validator("clip_count")
    @classmethod
    def _count(cls, v):  # noqa: ANN001
        if v == "auto":
            return v
        if not 1 <= int(v) <= 20:
            raise ValueError("clip_count must be between 1 and 20")
        return int(v)

    @model_validator(mode="after")
    def _custom_range(self) -> JobOptions:
        if self.duration == "custom" and self.custom_max - self.custom_min < 5.0:
            raise ValueError("a custom clip length needs at least 5 seconds between minimum and maximum")
        return self

    @property
    def duration_range(self) -> tuple[float, float]:
        if self.duration == "custom":
            return (self.custom_min, self.custom_max)
        return DURATION_RANGES[self.duration]

    @property
    def video_style(self) -> str:
        from .pipeline.styles import resolve

        return resolve(self.style)

    @property
    def trim_max_pause(self) -> float | None:
        return TRIM_MAX_PAUSE.get(self.style, 0.7)

    @property
    def default_framing(self) -> str:
        if self.framing != "auto":
            return self.framing
        from .pipeline.styles import STYLES

        st = STYLES.get(self.video_style)
        return st.framing if st is not None else "auto"
