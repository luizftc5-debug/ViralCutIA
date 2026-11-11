"""Stage 10 — rendering.

One ffmpeg invocation per clip: decode the source range once, cut the kept segments, apply the
per-shot layout, concatenate, burn captions (optional) and encode H.264/AAC MP4 with faststart.
There are no intermediate encodes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from ..config import get_settings
from .editing import zoom_expression
from .ffmpeg import CancelFn, ProgressFn, input_args, run_capture, run_ffmpeg
from .probe import MediaInfo
from .reframe import Shot
from .trimming import TimeMap

AUDIO_RATE = 48_000
JOIN_FADE = 0.008  # seconds of fade at each audio joint to avoid clicks


@dataclass
class Segment:
    f0: int  # first frame (inclusive), relative to the clip start, on the output frame grid
    f1: int  # last frame (exclusive)
    layout: dict
    out_start: float = 0.0  # where this segment starts on the output timeline (seconds)
    zooms: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )  # (t0, t1, amp, ramp), output time


def output_size(media: MediaInfo) -> tuple[int, int]:
    """1080x1920 when the source can sustain it, otherwise 720x1280 (no fake upscaling to 'HD')."""
    return (1080, 1920) if min(media.width, media.height) >= 700 else (720, 1280)


def output_fps(media: MediaInfo) -> Fraction:
    fps = Fraction(media.fps)
    limit = Fraction(get_settings().max_fps).limit_denominator(1001)
    while fps > limit:
        fps /= 2  # 120 -> 60: an integer divisor keeps motion cadence even
    return fps


def build_segments(
    keeps: list[tuple[float, float]], shots: list[Shot], clip_start: float, fps: Fraction
) -> list[Segment]:
    """Intersect kept intervals with shots and snap everything to the frame grid."""
    f = float(fps)
    out: list[Segment] = []
    for k0, k1 in keeps:
        a, b = k0 - clip_start, k1 - clip_start
        for shot in shots:
            s0, s1 = max(a, shot.start), min(b, shot.end)
            if s1 - s0 <= 0:
                continue
            f0, f1 = int(round(s0 * f)), int(round(s1 * f))
            if out and f0 < out[-1].f1:
                f0 = out[-1].f1
            if f1 - f0 < 1:
                continue
            if out and out[-1].f1 == f0 and out[-1].layout is shot.layout:
                out[-1].f1 = f1
            else:
                out.append(Segment(f0, f1, shot.layout))
    # absorb segments shorter than 3 frames into a neighbour to avoid single-frame flashes
    cleaned: list[Segment] = []
    for seg in out:
        if seg.f1 - seg.f0 < 3 and cleaned and cleaned[-1].f1 == seg.f0:
            cleaned[-1].f1 = seg.f1
        else:
            cleaned.append(seg)
    return cleaned


def segments_time_map(segments: list[Segment], fps: Fraction) -> TimeMap:
    f = float(fps)
    return TimeMap([(s.f0 / f, s.f1 / f) for s in segments])


def assign_zooms(segments: list[Segment], fps: Fraction, zooms: list) -> list:
    """Attach each punch-in to the segment that fully contains it. Only face-tracked ``crop``
    layouts zoom: a split screen, a letterboxed picture or a vertical source would lose content."""
    f = float(fps)
    cursor, applied = 0.0, []
    for seg in segments:
        d = (seg.f1 - seg.f0) / f
        seg.out_start, seg.zooms = cursor, []
        if seg.layout["type"] == "crop":
            for z in zooms:
                if z.t0 >= cursor + 0.05 and z.t1 <= cursor + d - 0.05:
                    seg.zooms.append((z.t0, z.t1, z.amp, z.ramp))
                    applied.append(z)
        cursor += d
    return applied


def segment_boundaries(segments: list[Segment], fps: Fraction) -> list[float]:
    f = float(fps)
    out, cursor = [], 0.0
    for seg in segments[:-1]:
        cursor += (seg.f1 - seg.f0) / f
        out.append(cursor)
    return out


def _crop_x_expr(keys: list, offset: float, max_x: int) -> str:
    """Piecewise-linear pan as a flat sum of clipped ramps (no nesting, any number of keyframes)."""
    if len(keys) == 1:
        return str(int(keys[0][1]))
    terms = [f"{keys[0][1]}"]
    for (t0, x0), (t1, x1) in zip(keys, keys[1:], strict=False):
        d = t1 - t0
        if d <= 0 or x1 == x0:
            continue
        slope = (x1 - x0) / d
        terms.append(f"{slope:.4f}*clip(t+{offset:.4f}-{t0:.4f},0,{d:.4f})")
    return f"clip({'+'.join(terms).replace('+-', '-')},0,{max_x})"


def _layout_chain(
    layout: dict,
    src: str,
    dst: str,
    tag: str,
    media: MediaInfo,
    ow: int,
    oh: int,
    offset: float,
    zooms: list[tuple[float, float, float, float]] | None = None,
    out_start: float = 0.0,
) -> str:
    kind = layout["type"]
    if kind == "crop":
        cw, ch, y = layout["w"], layout["h"], layout["y"]
        x = _crop_x_expr(layout["keys"], offset, max(0, media.width - cw))
        if zooms:
            # one resample: the source window is scaled straight to (frame x zoom), then the
            # centre of it is kept. The zoom factor is a smoothstep in/out of each emphasis.
            z = zoom_expression(zooms, out_start)
            return (
                f"[{src}]crop={cw}:{ch}:'{x}':{y},"
                f"scale=w='trunc({ow}*({z})/2)*2':h='trunc({oh}*({z})/2)*2':eval=frame:flags=lanczos,"
                f"crop={ow}:{oh}:'(iw-{ow})/2':'(ih-{oh})/2',setsar=1[{dst}]"
            )
        return f"[{src}]crop={cw}:{ch}:'{x}':{y},scale={ow}:{oh}:flags=lanczos,setsar=1[{dst}]"
    if kind == "split":
        t, b = layout["top"], layout["bottom"]
        half = oh // 2
        return (
            f"[{src}]split[{tag}ti][{tag}bi];"
            f"[{tag}ti]crop={t['w']}:{t['h']}:{t['x']}:{t['y']},scale={ow}:{half}:flags=lanczos[{tag}tp];"
            f"[{tag}bi]crop={b['w']}:{b['h']}:{b['x']}:{b['y']},scale={ow}:{oh - half}:flags=lanczos[{tag}bp];"
            f"[{tag}tp][{tag}bp]vstack,drawbox=x=0:y={half - 2}:w={ow}:h=4:color=black:t=fill,setsar=1[{dst}]"
        )
    if kind == "fit":
        return (
            f"[{src}]scale={ow}:{oh}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[{dst}]"
        )
    # blur: whole frame over a blurred, darkened copy of itself
    bw, bh = ow // 4, oh // 4
    return (
        f"[{src}]split[{tag}bg0][{tag}fg0];"
        f"[{tag}bg0]scale={bw}:{bh}:force_original_aspect_ratio=increase:flags=fast_bilinear,crop={bw}:{bh},"
        f"gblur=sigma=10,scale={ow}:{oh}:flags=bilinear,eq=brightness=-0.14:saturation=0.85[{tag}bg];"
        f"[{tag}fg0]scale={ow}:{oh}:force_original_aspect_ratio=decrease:flags=lanczos[{tag}fg];"
        f"[{tag}bg][{tag}fg]overlay=(W-w)/2:(H-h)/2,setsar=1[{dst}]"
    )


@dataclass
class AudioPolish:
    """Conservative speech treatment applied before loudness normalisation, and why."""

    highpass: bool = True  # 70 Hz: rumble and handling noise carry no speech
    denoise_nr: float | None = None  # afftdn reduction in dB, only for genuinely noisy recordings
    noise_floor_db: float = -60.0
    compress: bool = False  # gentle 2:1 leveller, only when the speech level swings widely
    measured: dict | None = None  # loudnorm pass-1 measurement of this exact clip
    report: dict = field(default_factory=dict)


LOUDNESS_TARGET = {"I": -14.0, "TP": -1.5, "LRA": 11.0}  # Shorts / Reels / TikTok playback level


def plan_audio_polish(audio, t0: float, t1: float) -> AudioPolish:  # noqa: ANN001 - AudioSignals
    """Decide, from measurements of this clip, which treatments are worth applying."""
    import numpy as np

    pol = AudioPolish()
    snr = float(audio.speech_db - audio.noise_floor_db)
    seg = audio._slice(t0, t1)
    active = seg[seg > audio.silence_threshold_db]
    spread = float(np.percentile(active, 95) - np.percentile(active, 20)) if len(active) > 20 else 0.0
    pol.report = {
        "snr_db": round(snr, 1),
        "level_spread_db": round(spread, 1),
        "applied": ["high-pass 70 Hz"],
    }
    if snr < 14.0 and audio.noise_floor_db > -62.0:
        pol.denoise_nr = float(np.clip(4.0 + (14.0 - snr) * 0.6, 4.0, 9.0))
        pol.noise_floor_db = float(np.clip(audio.noise_floor_db, -60.0, -28.0))
        pol.report["applied"].append(f"noise reduction {pol.denoise_nr:.0f} dB (SNR only {snr:.0f} dB)")
    if spread > 20.0:
        pol.compress = True
        pol.report["applied"].append(f"gentle 2:1 levelling (speech level swings {spread:.0f} dB)")
    return pol


def _polish_chain(polish: AudioPolish | None) -> str:
    if polish is None:
        return ""
    parts = []
    if polish.highpass:
        parts.append("highpass=f=70")
    if polish.denoise_nr:
        parts.append(f"afftdn=nr={polish.denoise_nr:.1f}:nf={polish.noise_floor_db:.0f}:tn=1")
    if polish.compress:
        parts.append("acompressor=threshold=0.063:ratio=2:attack=15:release=250:makeup=1.5")
    return ",".join(parts)


def _loudnorm(polish: AudioPolish | None, measuring: bool) -> str:
    t = LOUDNESS_TARGET
    base = f"loudnorm=I={t['I']}:TP={t['TP']}:LRA={t['LRA']}"
    if measuring:
        return base + ":print_format=json"
    m = polish.measured if polish else None
    if m:
        return (
            base + f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
            f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true"
        )
    return base  # single-pass (dynamic) fallback


def video_parts(
    segments: list[Segment], media: MediaInfo, fps: Fraction, ow: int, oh: int, ass_name: str | None
) -> list[str]:
    f = float(fps)
    n = len(segments)
    parts: list[str] = []
    pre = f"fps={fps.numerator}/{fps.denominator}"
    if media.sar != "1:1":
        pre = f"scale={media.width}:{media.height}," + pre
    if media.is_hdr:
        # basic HDR -> SDR conversion so colours are not washed out (no dynamic metadata handling)
        pre += ",zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable,zscale=t=bt709:m=bt709:r=tv"
    pre += ",setsar=1"
    if n == 1:
        parts.append(f"[0:v]{pre}[v0]")
    else:
        parts.append(f"[0:v]{pre},split={n}" + "".join(f"[v{i}]" for i in range(n)))
    for i, seg in enumerate(segments):
        parts.append(f"[v{i}]trim=start_frame={seg.f0}:end_frame={seg.f1},setpts=PTS-STARTPTS[t{i}]")
        parts.append(
            _layout_chain(
                seg.layout, f"t{i}", f"s{i}", f"L{i}", media, ow, oh, seg.f0 / f, seg.zooms, seg.out_start
            )
        )
    last = "s0"
    if n > 1:
        parts.append("".join(f"[s{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vcat]")
        last = "vcat"
    tail = f"ass=filename={ass_name}:fontsdir=fonts," if ass_name else ""
    parts.append(f"[{last}]{tail}format=yuv420p[vout]")
    return parts


def audio_parts(
    segments: list[Segment],
    media: MediaInfo,
    fps: Fraction,
    normalize_audio: bool,
    polish: AudioPolish | None = None,
    measuring: bool = False,
) -> list[str]:
    """Audio graph ending in ``[aout]``. With ``measuring`` it ends in the loudness analyser."""
    f = float(fps)
    n = len(segments)
    parts: list[str] = []
    # audio (input 1 is generated silence when the source has no audio stream)
    a_in = "0:a:0" if media.has_audio else "1:a:0"
    a_pre = f"aresample={AUDIO_RATE},aformat=channel_layouts=stereo"
    if n == 1:
        parts.append(f"[{a_in}]{a_pre}[a0]")
    else:
        parts.append(f"[{a_in}]{a_pre},asplit={n}" + "".join(f"[a{i}]" for i in range(n)))
    for i, seg in enumerate(segments):
        t0, t1 = seg.f0 / f, seg.f1 / f
        d = t1 - t0
        fades = ""
        if n > 1 and d > 4 * JOIN_FADE:
            fades = f",afade=t=in:st=0:d={JOIN_FADE},afade=t=out:st={d - JOIN_FADE:.4f}:d={JOIN_FADE}"
        parts.append(f"[a{i}]atrim=start={t0:.6f}:end={t1:.6f},asetpts=PTS-STARTPTS{fades}[as{i}]")
    a_last = "as0"
    if n > 1:
        parts.append("".join(f"[as{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[acat]")
        a_last = "acat"
    if normalize_audio and media.has_audio:
        chain = [c for c in (_polish_chain(polish), _loudnorm(polish, measuring)) if c]
        tail = "" if measuring else f",aresample={AUDIO_RATE},alimiter=limit=0.9:level=disabled"
        parts.append(f"[{a_last}]{','.join(chain)}{tail}[aout]")
    else:
        parts.append(f"[{a_last}]anull[aout]")
    return parts


def build_filtergraph(
    segments: list[Segment],
    media: MediaInfo,
    fps: Fraction,
    ow: int,
    oh: int,
    ass_name: str | None,
    normalize_audio: bool,
    polish: AudioPolish | None = None,
) -> str:
    return ";".join(
        video_parts(segments, media, fps, ow, oh, ass_name)
        + audio_parts(segments, media, fps, normalize_audio, polish)
    )


_JSON_BLOCK = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def measure_loudness(
    source: Path,
    family: str | None,
    media: MediaInfo,
    clip_start: float,
    segments: list[Segment],
    fps: Fraction,
    polish: AudioPolish | None,
    span: float,
) -> dict | None:
    """Loudness pass 1: run the *exact* audio graph of the clip into the EBU R128 analyser."""
    graph = ";".join(audio_parts(segments, media, fps, True, polish, measuring=True))
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-v", "info",
        *input_args(Path(source).resolve(), family, seek=clip_start, duration=span),
        "-filter_complex", graph, "-map", "[aout]", "-f", "null", "-",
    ]  # fmt: skip
    res = run_capture(cmd, timeout=get_settings().ffmpeg_timeout_s)
    if res.returncode != 0:
        return None
    m = _JSON_BLOCK.search(res.stderr.decode("utf-8", "replace"))
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        vals = {
            k: float(data[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
        }
    except (ValueError, KeyError):
        return None
    if vals["input_i"] < -69.0:  # effectively silent: there is nothing to normalise
        return None
    return {k: f"{v:.2f}" for k, v in vals.items()}


def _link_fonts(workdir: Path) -> None:
    target = workdir / "fonts"
    if target.exists():
        return
    src = get_settings().fonts_dir.resolve()
    try:
        os.symlink(src, target, target_is_directory=True)
    except OSError:
        shutil.copytree(src, target)


def render_clip(
    source: Path,
    family: str | None,
    media: MediaInfo,
    clip_start: float,
    segments: list[Segment],
    out_path: Path,
    ass_path: Path | None = None,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    polish: AudioPolish | None = None,
) -> dict:
    """Render one clip. Returns the parameters used (for QA and logs)."""
    s = get_settings()
    fps = output_fps(media)
    ow, oh = output_size(media)
    f = float(fps)
    total_frames = sum(seg.f1 - seg.f0 for seg in segments)
    out_duration = total_frames / f
    span = segments[-1].f1 / f + 0.5  # source seconds to decode

    workdir = out_path.parent
    ass_name = None
    if ass_path is not None:
        _link_fonts(workdir)
        ass_name = ass_path.name  # referenced relative to cwd: no filtergraph escaping needed
        assert ass_path.parent.resolve() == workdir.resolve()

    if s.normalize_audio and media.has_audio and polish is not None:
        polish.measured = measure_loudness(source, family, media, clip_start, segments, fps, polish, span)
        polish.report["loudness_pass"] = (
            "two-pass linear" if polish.measured else "single-pass (no measurement)"
        )
        if polish.measured:
            polish.report["measured_lufs"] = float(polish.measured["input_i"])
            polish.report["measured_true_peak_db"] = float(polish.measured["input_tp"])
    graph = build_filtergraph(segments, media, fps, ow, oh, ass_name, s.normalize_audio, polish)
    tmp_out = out_path.with_suffix(".tmp.mp4")
    args = [*input_args(Path(source).resolve(), family, seek=clip_start, duration=span)]
    if not media.has_audio:
        args += ["-f", "lavfi", "-t", f"{out_duration + 1:.3f}", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
    args += [
        "-filter_complex", graph,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", s.x264_preset, "-crf", str(s.x264_crf),
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-maxrate", f"{16 if oh >= 1920 else 8}M", "-bufsize", f"{32 if oh >= 1920 else 16}M",  # cap spikes, keep CRF quality
        *_color_tags(media),
        "-g", str(max(12, int(round(f * 2)))),
        "-c:a", "aac", "-b:a", s.audio_bitrate, "-ar", str(AUDIO_RATE), "-ac", "2",
        "-t", f"{out_duration:.6f}",
        "-map_metadata", "-1", "-map_chapters", "-1",  # strip source metadata (location, device, ...)
        "-movflags", "+faststart",
        "-f", "mp4",
    ]  # fmt: skip
    if s.ffmpeg_threads > 0:
        args += ["-threads", str(s.ffmpeg_threads)]
    args.append(tmp_out.name)
    try:
        run_ffmpeg(
            args,
            total_duration=out_duration,
            on_progress=on_progress,
            should_cancel=should_cancel,
            cwd=workdir,
            timeout=s.ffmpeg_timeout_s,
        )
        os.replace(tmp_out, out_path)
    finally:
        tmp_out.unlink(missing_ok=True)
    return {
        "width": ow,
        "height": oh,
        "fps": f"{fps.numerator}/{fps.denominator}",
        "duration": round(out_duration, 3),
        "segments": len(segments),
        "crf": s.x264_crf,
        "preset": s.x264_preset,
        "zooms": sum(len(seg.zooms) for seg in segments),
        "audio": polish.report if polish is not None else {},
        "source_crop_px": _native_crop(segments, media),
    }


def _native_crop(segments: list[Segment], media: MediaInfo) -> str:
    """Pixels the source really provides for the 9:16 frame (so 1080x1920 is never oversold)."""
    seg = next((x for x in segments if x.layout["type"] == "crop"), None)
    if seg is not None:
        return f"{seg.layout['w']}x{seg.layout['h']}"
    return f"{media.width}x{media.height}"


def _color_tags(media: MediaInfo) -> list[str]:
    """Explicit BT.709 signalling so players do not guess (HDR is tone-mapped to it above)."""
    if media.color_space in ("smpte170m", "bt470bg"):  # SD sources keep their own matrix
        tag = "smpte170m" if media.color_space == "smpte170m" else "bt470bg"
        return ["-colorspace", tag, "-color_primaries", tag, "-color_trc", "smpte170m", "-color_range", "tv"]
    return ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def make_thumbnail(video: Path, out: Path, at: float = 0.5) -> None:
    args = [
        "-ss",
        f"{max(0.0, at):.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        "scale=360:-2",
        "-q:v",
        "4",
        str(out),
    ]
    run_ffmpeg(args, timeout=60)
