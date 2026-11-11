"""User-facing processing options (validated at the API boundary, consumed by the pipeline)."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CaptionStyle = Literal["viral_dynamic", "high_retention", "clean_editorial", "dynamic_creator", "gaming"]
FramingMode = Literal["auto", "face", "split", "blur", "center"]

DURATION_RANGES: dict[str, tuple[float, float]] = {
    "15-30": (15.0, 30.0),
    "30-60": (30.0, 60.0),
    "60-90": (60.0, 90.0),
    "auto": (18.0, 75.0),
}

# Longest pause kept between words, per editing style (None = smart trimming disabled).
TRIM_MAX_PAUSE: dict[str, float | None] = {
    "dynamic": 0.45,
    "editorial": 1.0,
    "podcast": 0.7,
    "gaming": None,
    "auto": 0.7,
}

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
    size: Literal["s", "m", "l"] = "m"
    position: Literal["top", "middle", "bottom"] = "bottom"
    color: str = ""  # accent colour as #RRGGBB; empty = style default
    font: Literal["auto", "inter", "poppins", "anton", "dm_serif"] = "auto"
    animation: bool = True

    @field_validator("color")
    @classmethod
    def _hex(cls, v: str) -> str:
        v = (v or "").strip()
        if v and not re.fullmatch(r"#[0-9a-fA-F]{6}", v):
            raise ValueError("color must be #RRGGBB")
        return v.upper()


class JobOptions(BaseModel):
    captions: CaptionOptions = Field(default_factory=CaptionOptions)
    language: str = "auto"
    clip_count: int | Literal["auto"] = "auto"
    duration: Literal["15-30", "30-60", "60-90", "auto"] = "auto"
    platform: Literal["shorts", "reels", "tiktok", "universal"] = "universal"
    aspect: Literal["9:16"] = "9:16"
    style: Literal["dynamic", "editorial", "podcast", "gaming", "auto"] = "auto"
    framing: FramingMode = "auto"
    smart_zoom: bool = True  # controlled punch-ins on words the speaker really stresses
    review_transcript: bool = (
        False  # pause after clip selection so the transcript can be corrected before rendering
    )
    glossary: str = Field(
        default="", max_length=400
    )  # names / terms the recogniser should prefer, comma separated
    min_score: float | None = Field(default=None, ge=0, le=100)

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

    @property
    def duration_range(self) -> tuple[float, float]:
        return DURATION_RANGES[self.duration]

    @property
    def trim_max_pause(self) -> float | None:
        return TRIM_MAX_PAUSE[self.style]

    @property
    def default_framing(self) -> str:
        if self.framing != "auto":
            return self.framing
        return "blur" if self.style == "gaming" else "auto"
