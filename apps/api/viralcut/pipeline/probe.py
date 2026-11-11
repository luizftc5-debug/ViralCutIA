"""Stage 2 — media probing with ffprobe."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from ..security import FAMILY_DEMUXER
from .ffmpeg import FFmpegError, run_capture

# Codecs the pipeline is tested to decode. Anything else is rejected up front with a clear message.
SUPPORTED_VIDEO_CODECS = {"h264", "hevc", "vp8", "vp9", "av1", "mpeg4", "prores", "mjpeg", "mpeg2video"}


class ProbeError(ValueError):
    """Message is safe to show to the user."""


@dataclass
class MediaInfo:
    duration: float
    width: int  # display width (rotation and pixel aspect applied)
    height: int
    fps: str  # rational string, e.g. "30000/1001"
    video_codec: str
    has_audio: bool
    audio_codec: str = ""
    audio_channels: int = 0
    sample_rate: int = 0
    rotation: int = 0
    sar: str = "1:1"
    pix_fmt: str = ""
    bit_rate: int = 0
    color_transfer: str = ""
    color_space: str = ""
    container: str = ""

    @property
    def fps_float(self) -> float:
        return float(Fraction(self.fps))

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in {"smpte2084", "arib-std-b67"}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MediaInfo:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _fraction(value: str | None) -> Fraction | None:
    try:
        f = Fraction(value or "")
        return f if f > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: Path, family: str | None = None) -> MediaInfo:
    cmd = ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe"]
    if family in FAMILY_DEMUXER:
        cmd += ["-f", FAMILY_DEMUXER[family]]
    cmd += ["-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        res = run_capture(cmd, timeout=60)
    except Exception as exc:  # timeout etc.
        raise ProbeError("The video could not be analyzed.") from exc
    if res.returncode != 0:
        raise ProbeError(
            "The file is not a readable video (it may be corrupted or use an unsupported codec)."
        )
    try:
        data = json.loads(res.stdout or b"{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError("ffprobe returned invalid JSON") from exc

    streams = data.get("streams", [])
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise ProbeError("No video stream was found in the file.")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    codec = video.get("codec_name", "")
    if codec not in SUPPORTED_VIDEO_CODECS:
        raise ProbeError(f"Unsupported video codec: {codec or 'unknown'}.")

    duration = 0.0
    for candidate in (fmt.get("duration"), video.get("duration")):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue
    if duration <= 0:
        raise ProbeError("The video has no measurable duration.")

    w, h = int(video.get("width") or 0), int(video.get("height") or 0)
    if w < 16 or h < 16:
        raise ProbeError("The video resolution is invalid.")

    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                rotation = 0
    sar_txt = video.get("sample_aspect_ratio") or "1:1"
    sar = _fraction(sar_txt.replace(":", "/")) or Fraction(1)
    if sar != 1:
        w = int(round(w * float(sar) / 2) * 2)
    if rotation in (90, 270):
        w, h = h, w

    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate")) or Fraction(30)
    if fps > 240 or fps < 1:
        fps = _fraction(video.get("r_frame_rate")) or Fraction(30)
        if fps > 240 or fps < 1:
            fps = Fraction(30)
    fps = fps.limit_denominator(1001)

    return MediaInfo(
        duration=duration,
        width=w,
        height=h,
        fps=f"{fps.numerator}/{fps.denominator}",
        video_codec=codec,
        has_audio=audio is not None,
        audio_codec=(audio or {}).get("codec_name", ""),
        audio_channels=int((audio or {}).get("channels") or 0),
        sample_rate=int((audio or {}).get("sample_rate") or 0),
        rotation=rotation,
        sar=f"{sar.numerator}:{sar.denominator}",
        pix_fmt=video.get("pix_fmt", ""),
        bit_rate=int(fmt.get("bit_rate") or 0),
        color_transfer=video.get("color_transfer", ""),
        color_space=video.get("color_space", ""),
        container=fmt.get("format_name", ""),
    )
