"""Stage 10 — rendering.

One ffmpeg invocation per clip: decode the source range once, cut the kept segments, apply the
per-shot layout, concatenate, burn captions (optional) and encode H.264/AAC MP4 with faststart.
There are no intermediate encodes.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ..config import get_settings
from .ffmpeg import CancelFn, ProgressFn, input_args, run_ffmpeg
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
    layout: dict, src: str, dst: str, tag: str, media: MediaInfo, ow: int, oh: int, offset: float
) -> str:
    kind = layout["type"]
    if kind == "crop":
        cw, ch, y = layout["w"], layout["h"], layout["y"]
        x = _crop_x_expr(layout["keys"], offset, max(0, media.width - cw))
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


def build_filtergraph(
    segments: list[Segment],
    media: MediaInfo,
    fps: Fraction,
    ow: int,
    oh: int,
    ass_name: str | None,
    normalize_audio: bool,
) -> str:
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
        parts.append(_layout_chain(seg.layout, f"t{i}", f"s{i}", f"L{i}", media, ow, oh, seg.f0 / f))
    last = "s0"
    if n > 1:
        parts.append("".join(f"[s{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vcat]")
        last = "vcat"
    tail = f"ass=filename={ass_name}:fontsdir=fonts," if ass_name else ""
    parts.append(f"[{last}]{tail}format=yuv420p[vout]")

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
        parts.append(f"[{a_last}]loudnorm=I=-14:TP=-1.5:LRA=11,aresample={AUDIO_RATE}[aout]")
    else:
        parts.append(f"[{a_last}]anull[aout]")
    return ";".join(parts)


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

    graph = build_filtergraph(segments, media, fps, ow, oh, ass_name, s.normalize_audio)
    tmp_out = out_path.with_suffix(".tmp.mp4")
    args = [*input_args(Path(source).resolve(), family, seek=clip_start, duration=span)]
    if not media.has_audio:
        args += ["-f", "lavfi", "-t", f"{out_duration + 1:.3f}", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
    args += [
        "-filter_complex", graph,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", s.x264_preset, "-crf", str(s.x264_crf),
        "-profile:v", "high", "-pix_fmt", "yuv420p",
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
    }


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
