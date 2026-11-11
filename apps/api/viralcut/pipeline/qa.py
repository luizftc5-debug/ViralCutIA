"""Stage 11 — quality assurance of rendered files.

A clip is only marked ready when its file passes every check here.
"""

from __future__ import annotations

import json
import re
import struct
from fractions import Fraction
from pathlib import Path

from .ffmpeg import run_capture


class QAError(RuntimeError):
    pass


def _moov_before_mdat(path: Path) -> bool:
    """True when the MP4 index precedes the media data (progressive download / faststart)."""
    with path.open("rb") as fh:
        pos, size = 0, path.stat().st_size
        while pos + 8 <= size:
            fh.seek(pos)
            header = fh.read(16)
            if len(header) < 8:
                break
            box_size, box_type = struct.unpack(">I4s", header[:8])
            if box_size == 1 and len(header) >= 16:
                box_size = struct.unpack(">Q", header[8:16])[0]
            if box_type == b"moov":
                return True
            if box_type == b"mdat":
                return False
            if box_size < 8:
                break
            pos += box_size
    return False


def verify_clip(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_duration: float,
    expected_fps: str,
    source_has_audio: bool,
) -> dict:
    """Run all checks and return a report: ``{"passed": bool, "checks": {...}, ...}``."""
    checks: dict[str, bool] = {}
    report: dict = {"passed": False, "checks": checks}
    if not path.exists() or path.stat().st_size < 1024:
        raise QAError("rendered file is missing or empty")

    res = run_capture(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        timeout=60,
    )
    if res.returncode != 0:
        raise QAError("rendered file cannot be probed")
    data = json.loads(res.stdout or b"{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None:
        raise QAError("rendered file has no video stream")

    w, h = int(video.get("width", 0)), int(video.get("height", 0))
    v_dur = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    a_dur = float(audio.get("duration") or 0) if audio else 0.0
    fps = (
        Fraction(video.get("avg_frame_rate") or "0/1")
        if video.get("avg_frame_rate", "0/0") != "0/0"
        else Fraction(0)
    )
    frame = 1 / float(Fraction(expected_fps))

    checks["video_codec_h264"] = video.get("codec_name") == "h264"
    checks["pixel_format_yuv420p"] = video.get("pix_fmt") == "yuv420p"
    checks["resolution"] = (w, h) == (expected_width, expected_height)
    checks["vertical_9_16"] = h > w and abs(w / h - 9 / 16) < 0.01
    checks["frame_rate"] = abs(float(fps) - float(Fraction(expected_fps))) < 0.05
    checks["duration"] = abs(v_dur - expected_duration) <= max(0.12, 3 * frame)
    checks["audio_stream_aac"] = audio is not None and audio.get("codec_name") == "aac"
    checks["av_sync"] = audio is not None and abs(v_dur - a_dur) <= 0.12
    checks["faststart"] = _moov_before_mdat(path)

    # full decode: catches truncated/corrupt streams, and measures loudness of the result
    dec = run_capture(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-xerror", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0?", "-af", "volumedetect,ebur128=peak=true", "-f", "null", "-"],
        timeout=600,
    )  # fmt: skip
    err = dec.stderr.decode("utf-8", "replace")
    checks["decodes_cleanly"] = dec.returncode == 0
    mean = re.search(r"mean_volume:\s*(-?[0-9.]+|-inf) dB", err)
    mean_db = float(mean.group(1)) if mean and mean.group(1) != "-inf" else -120.0
    # the source had sound, so the clip must not be silent
    checks["audio_not_silent"] = (mean_db > -70.0) if source_has_audio else True

    # EBU R128 summary (the last block ffmpeg prints): integrated loudness and true peak
    summary = err[err.rfind("Summary:") :] if "Summary:" in err else ""
    lufs_m = re.search(r"I:\s*(-?[0-9.]+|-inf) LUFS", summary)
    peak_m = re.search(r"Peak:\s*(-?[0-9.]+|-inf) dBFS", summary)
    lufs = float(lufs_m.group(1)) if lufs_m and lufs_m.group(1) != "-inf" else None
    peak = float(peak_m.group(1)) if peak_m and peak_m.group(1) != "-inf" else None
    audible = source_has_audio and lufs is not None and lufs > -60.0
    # never clip: true peak must leave headroom. Loudness must be in the range platforms play
    # back at without turning it up or down much (the target is -14 LUFS; the band is generous
    # so a legitimately peaky recording is not failed, but a broken normaliser is).
    checks["true_peak_safe"] = (peak is None or peak <= -0.5) if audible else True
    checks["loudness_in_range"] = (-20.0 <= lufs <= -9.0) if audible else True
    checks["audio_48k_stereo"] = (
        audio is not None and int(audio.get("sample_rate", 0)) == 48000 and int(audio.get("channels", 0)) == 2
    )
    checks["square_pixels"] = video.get("sample_aspect_ratio", "1:1") in ("1:1", "0:1", None)

    report.update(
        {
            "width": w,
            "height": h,
            "fps": float(fps),
            "duration": round(v_dur, 3),
            "audio_duration": round(a_dur, 3),
            "mean_volume_db": round(mean_db, 1),
            "integrated_lufs": lufs,
            "true_peak_db": peak,
            "audio_sample_rate": int((audio or {}).get("sample_rate", 0)),
            "color_space": video.get("color_space"),
            "video_bit_rate": int(video.get("bit_rate") or 0),
            "size_bytes": path.stat().st_size,
            "bit_rate": int(data.get("format", {}).get("bit_rate") or 0),
            "passed": all(checks.values()),
        }
    )
    return report
