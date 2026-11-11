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
from ..resources import active_plan
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


@dataclass
class ImageCard:
    """/prova: an evidence image shown as a card between t0 and t1 (output timeline)."""

    path: Path
    t0: float
    t1: float
    input: int = -1


@dataclass
class MusicBed:
    """/trilha: a user-supplied music file under the voice."""

    path: Path
    volume_db: float = -20.0
    duck: bool = True
    offset: float = 0.0  # seconds skipped at the start of the music (beat alignment)
    input: int = -1


@dataclass
class Background:
    """/tela-verde: what replaces the background."""

    kind: str = "blur"  # blur | color | image
    color: str = "#101820"
    image: Path | None = None
    input: int = -1


@dataclass
class RenderExtras:
    """Editor effects applied in the render pass (all optional)."""

    video_post: list[str] = field(default_factory=list)  # e.g. the /3d camera filter
    cards: list[ImageCard] = field(default_factory=list)
    music: MusicBed | None = None
    chroma: dict | None = None  # {"key": "0x00FF00", "similarity": .., "blend": ..}
    background: Background | None = None
    behind_ass: str | None = None  # /texto-atras text layer (file name in the work dir)
    matte_input: int = -1  # composite pass: input index of the person matte

    def inputs(self, start: int, fps: Fraction, duration: float) -> list[str]:
        """ffmpeg input arguments for the extra media, assigning their input indices."""
        args: list[str] = []
        i = start
        for c in self.cards:
            c.input = i
            args += [
                "-loop",
                "1",
                "-framerate",
                f"{fps.numerator}/{fps.denominator}",
                "-t",
                f"{c.t1 + 0.1:.3f}",
                "-i",
                str(c.path),
            ]
            i += 1
        if self.background is not None and self.background.kind == "image" and self.background.image:
            self.background.input = i
            args += [
                "-loop",
                "1",
                "-framerate",
                f"{fps.numerator}/{fps.denominator}",
                "-t",
                f"{duration + 0.5:.3f}",
                "-i",
                str(self.background.image),
            ]
            i += 1
        if self.music is not None:
            self.music.input = i
            args += ["-stream_loop", "-1", "-i", str(self.music.path)]
            i += 1
        return args


def _background(
    bg: Background, src: str, dst: str, ow: int, oh: int, fps: Fraction, duration: float
) -> list[str]:
    if bg.kind == "color":
        c = bg.color.lstrip("#")
        return [
            f"[{src}]nullsink",
            f"color=c=0x{c}:s={ow}x{oh}:r={fps.numerator}/{fps.denominator}:d={duration + 0.5:.3f},format=yuv420p[{dst}]",
        ]
    if bg.kind == "image" and bg.input >= 0:
        return [
            f"[{src}]nullsink",
            f"[{bg.input}:v]scale={ow}:{oh}:force_original_aspect_ratio=increase,crop={ow}:{oh},setsar=1,format=yuv420p[{dst}]",
        ]
    bw, bh = max(2, ow // 6 // 2 * 2), max(2, oh // 6 // 2 * 2)
    return [
        f"[{src}]scale={bw}:{bh}:flags=fast_bilinear,gblur=sigma=8,eq=brightness=-0.08,scale={ow}:{oh}:flags=bilinear,setsar=1[{dst}]"
    ]


def post_parts(
    last: str, ex: RenderExtras, ow: int, oh: int, fps: Fraction, duration: float, matte: str | None = None
) -> tuple[list[str], str]:
    """Editor effects on the assembled picture, before captions. Returns (filters, last label)."""
    parts: list[str] = []
    cur = last
    if matte is not None:  # AI matte composite: background treatment and text *behind* the person
        parts.append(f"[{cur}]format=yuv420p,split[pfg][pbg]")
        bg = "pbg"
        if ex.background is not None:
            parts += _background(ex.background, "pbg", "pbg2", ow, oh, fps, duration)
            bg = "pbg2"
        if ex.behind_ass:
            parts.append(f"[{bg}]ass=filename={ex.behind_ass}:fontsdir=fonts[pbgt]")
            bg = "pbgt"
        parts.append(f"[pfg][{matte}]alphamerge[pperson]")
        parts.append(f"[{bg}][pperson]overlay=format=auto,format=yuv420p[pcomp]")
        cur = "pcomp"
    elif ex.chroma is not None:  # green / blue screen
        k = ex.chroma
        parts.append(f"[{cur}]split[kfg][kbg]")
        parts += _background(ex.background or Background(), "kbg", "kbg2", ow, oh, fps, duration)
        parts.append(
            f"[kfg]chromakey={k['key']}:{k.get('similarity', 0.13)}:{k.get('blend', 0.06)},despill=type={'green' if k.get('green', True) else 'blue'}[kcut]"
        )
        parts.append("[kbg2][kcut]overlay=format=auto,format=yuv420p[kcomp]")
        cur = "kcomp"
    if ex.video_post:
        parts.append(f"[{cur}]{','.join(ex.video_post)}[vpost]")
        cur = "vpost"
    for i, c in enumerate(ex.cards):
        cw, ch = int(ow * 0.82) // 2 * 2, int(oh * (0.42 if oh > ow else 0.6)) // 2 * 2
        y = f"H*{0.16 if oh > ow else 0.12}"
        d = c.t1 - c.t0
        parts.append(
            f"[{c.input}:v]scale={cw}:{ch}:force_original_aspect_ratio=decrease,pad=iw+12:ih+12:6:6:white,format=yuva420p,"
            f"fade=t=in:st=0:d=0.25:alpha=1,fade=t=out:st={max(0.0, d - 0.3):.3f}:d=0.3:alpha=1,setpts=PTS-STARTPTS+{c.t0:.3f}/TB[card{i}]"
        )
        parts.append(
            f"[{cur}][card{i}]overlay=x=(W-w)/2:y={y}:enable='between(t,{c.t0:.3f},{c.t1:.3f})':eof_action=pass[vcard{i}]"
        )
        cur = f"vcard{i}"
    return parts, cur


def music_parts(voice: str, m: MusicBed, duration: float) -> list[str]:
    """Music under the voice: fade in/out, optional sidechain ducking, peak-limited mix."""
    fade_out = max(0.0, duration - 1.5)
    parts = [
        f"[{m.input}:a]atrim=start={m.offset:.3f},asetpts=PTS-STARTPTS,aresample={AUDIO_RATE},aformat=channel_layouts=stereo,"
        f"volume={m.volume_db:.1f}dB,afade=t=in:st=0:d=0.6,afade=t=out:st={fade_out:.3f}:d=1.5[mus]"
    ]
    if m.duck:
        parts.append(f"[{voice}]asplit[mvo][msc]")
        parts.append("[mus][msc]sidechaincompress=threshold=0.02:ratio=6:attack=30:release=450[mduck]")
        parts.append(
            "[mvo][mduck]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.9:level=disabled[aout]"
        )
    else:
        parts.append(
            f"[{voice}][mus]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.9:level=disabled[aout]"
        )
    return parts


def output_size(media: MediaInfo, aspect: str = "9:16") -> tuple[int, int]:
    """1080x1920 (or 1920x1080 for 16:9) when the source can sustain it, otherwise 720p
    (no fake upscaling to 'HD')."""
    hd = min(media.width, media.height) >= 700
    if aspect == "16:9":
        return (1920, 1080) if hd else (1280, 720)
    return (1080, 1920) if hd else (720, 1280)


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
        if seg.layout["type"] == "crop" or seg.layout.get("full"):
            for z in zooms:
                if z.t0 >= cursor + 0.05 and z.t1 <= cursor + d - 0.05:
                    seg.zooms.append((z.t0, z.t1, z.amp, z.ramp))
                    applied.append(z)
        cursor += d
    return applied


def jump_cuts(segments: list[Segment]) -> list[int]:
    """Indices of segments that start after removed source time (a jump cut, not a real shot change)."""
    return [i for i in range(1, len(segments)) if segments[i].f0 > segments[i - 1].f1]


def add_punch_transitions(
    segments: list[Segment], fps: Fraction, amp: float, min_spacing: float = 1.6
) -> int:
    """Hide jump cuts with a quick punch: the incoming shot starts slightly closer and eases back.

    Applied to every other eligible jump cut (repetition becomes a tic), only on face-tracked
    ``crop`` segments long enough to settle, never overlapping a planned emphasis zoom. Timing is
    unchanged, so captions stay in sync.
    """
    f = float(fps)
    ramp, hold = 0.32, 0.18
    added, last, toggle = 0, -99.0, False
    for i in jump_cuts(segments):
        seg = segments[i]
        d = (seg.f1 - seg.f0) / f
        if (seg.layout.get("type") != "crop" and not seg.layout.get("full")) or d < ramp + hold + 0.4:
            continue
        t0 = seg.out_start
        if t0 - last < min_spacing or any(z[0] < t0 + ramp + hold + 0.3 for z in seg.zooms):
            continue
        toggle = not toggle
        if not toggle:
            continue
        seg.zooms.insert(0, (t0 - ramp, t0 + hold + ramp, round(amp, 3), ramp))
        last = t0
        added += 1
    return added


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
    if kind == "fit" and layout.get("full") and zooms:
        # same-aspect source kept whole: punch-ins scale it up and keep the centre (one resample)
        z = zoom_expression(zooms, out_start)
        return (
            f"[{src}]scale=w='trunc({ow}*({z})/2)*2':h='trunc({oh}*({z})/2)*2':eval=frame:flags=lanczos,"
            f"crop={ow}:{oh}:'(iw-{ow})/2':'(ih-{oh})/2',setsar=1[{dst}]"
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
    presence: bool = False  # +2 dB around 3 kHz: consonant clarity for voice-led styles
    join_fade: float = JOIN_FADE  # audio fade at each cut (longer for the smooth transition)
    measured: dict | None = None  # loudnorm pass-1 measurement of this exact clip
    report: dict = field(default_factory=dict)


LOUDNESS_TARGET = {"I": -14.0, "TP": -1.5, "LRA": 11.0}  # Shorts / Reels / TikTok playback level


# Speech-level swing (dB, 95th - 20th percentile of active speech) above which a gentle leveller
# is applied, per audio treatment of the video style. "natural" keeps the original dynamics.
COMPRESS_ABOVE = {"natural": 24.0, "voice": 14.0, "punchy": 16.0}


def plan_audio_polish(audio, t0: float, t1: float, mode: str = "natural") -> AudioPolish:  # noqa: ANN001 - AudioSignals
    """Decide, from measurements of this clip, which treatments are worth applying.

    ``mode`` comes from the video style: ``voice`` / ``punchy`` level speech sooner and add a
    small presence lift; ``natural`` only intervenes when the level swings very widely.
    """
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
    limit = COMPRESS_ABOVE.get(mode, 20.0)
    if spread > limit:
        pol.compress = True
        pol.report["applied"].append(
            f"gentle 2:1 levelling (speech level swings {spread:.0f} dB > {limit:.0f} dB)"
        )
    if mode in ("voice", "punchy") and len(active) > 20:
        pol.presence = True
        pol.report["applied"].append("presence lift +2 dB at 3 kHz (speech clarity)")
    pol.report["treatment"] = mode
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
    if polish.presence:
        parts.append("equalizer=f=3000:t=q:w=1.0:g=2")
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
    segments: list[Segment],
    media: MediaInfo,
    fps: Fraction,
    ow: int,
    oh: int,
    ass_name: str | None,
    extras: RenderExtras | None = None,
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
    if extras is not None:
        dur = sum(seg.f1 - seg.f0 for seg in segments) / f
        post, last = post_parts(last, extras, ow, oh, fps, dur)
        parts += post
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
        jf = polish.join_fade if polish is not None else JOIN_FADE
        if n > 1 and d > 4 * jf:
            fades = f",afade=t=in:st=0:d={jf},afade=t=out:st={d - jf:.4f}:d={jf}"
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
    extras: RenderExtras | None = None,
) -> str:
    audio = audio_parts(segments, media, fps, normalize_audio, polish)
    if extras is not None and extras.music is not None:
        audio[-1] = audio[-1].rsplit("[aout]", 1)[0] + "[voice]"
        audio += music_parts("voice", extras.music, sum(s.f1 - s.f0 for s in segments) / float(fps))
    return ";".join(video_parts(segments, media, fps, ow, oh, ass_name, extras) + audio)


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
        "-vn", "-filter_complex", graph, "-map", "[aout]", "-f", "null", "-",
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
    *,
    quality: str = "high",
    loudness_memo=None,  # noqa: ANN001 - orchestrator.ClipCache: reuses pass 1 when the audio graph is unchanged
    aspect: str = "9:16",
    extras: RenderExtras | None = None,
) -> dict:
    """Render one clip. Returns the parameters used (for QA and logs).

    ``quality="intermediate"`` writes a near-lossless fast encode used as the input of the AI
    matte composite (only when /tela-verde AI or /texto-atras is on).
    """
    s = get_settings()
    plan = active_plan()
    fps = output_fps(media)
    ow, oh = output_size(media, aspect)
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
        # pass 1 depends only on the audio graph: a caption or framing edit does not change it
        lkey = (
            "loud-v1",
            round(clip_start, 4),
            [(g.f0, g.f1) for g in segments],
            str(fps),
            _polish_chain(polish),
        )
        hit = loudness_memo.get(*lkey) if loudness_memo is not None else None
        if hit is not None:
            polish.measured = hit.get("measured")
        else:
            polish.measured = measure_loudness(source, family, media, clip_start, segments, fps, polish, span)
            if loudness_memo is not None:
                loudness_memo.put({"measured": polish.measured}, *lkey)
        polish.report["loudness_pass"] = (
            "two-pass linear" if polish.measured else "single-pass (no measurement)"
        )
        if polish.measured:
            polish.report["measured_lufs"] = float(polish.measured["input_i"])
            polish.report["measured_true_peak_db"] = float(polish.measured["input_tp"])
    if extras is not None and (extras.behind_ass or extras.chroma is not None):
        _link_fonts(workdir)
    # extra media get their input indices first: the filter graph refers to them by index
    extra_inputs = (
        extras.inputs(2 if not media.has_audio else 1, fps, out_duration) if extras is not None else []
    )
    graph = build_filtergraph(segments, media, fps, ow, oh, ass_name, s.normalize_audio, polish, extras)
    # "max" export: lower CRF and a slower preset (measured trade-off in docs/PERFORMANCE.md)
    crf = max(12, s.x264_crf - 2) if quality == "max" else 10 if quality == "intermediate" else s.x264_crf
    preset = "slow" if quality == "max" else "ultrafast" if quality == "intermediate" else plan.x264_preset
    rate = (16 if max(ow, oh) >= 1920 else 8) * (4 if quality == "intermediate" else 1)
    tmp_out = out_path.with_suffix(".tmp.mp4")
    args = [*input_args(Path(source).resolve(), family, seek=clip_start, duration=span)]
    if not media.has_audio:
        args += ["-f", "lavfi", "-t", f"{out_duration + 1:.3f}", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
    args += extra_inputs
    args += [
        "-filter_complex", graph,
        "-map", "[vout]", "-map", "[aout]",
        "-filter_complex_threads", str(plan.threads),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-maxrate", f"{rate}M", "-bufsize", f"{rate * 2}M",  # cap spikes, keep CRF quality
        *_color_tags(media),
        "-g", str(max(12, int(round(f * 2)))),
        "-c:a", "aac", "-b:a", s.audio_bitrate, "-ar", str(AUDIO_RATE), "-ac", "2",
        "-t", f"{out_duration:.6f}",
        "-map_metadata", "-1", "-map_chapters", "-1",  # strip source metadata (location, device, ...)
        "-movflags", "+faststart",
        "-f", "mp4",
    ]  # fmt: skip
    args += ["-threads", str(plan.threads)]  # encoder threads: the profile's budget, not every core
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
        "crf": crf,
        "preset": preset,
        "quality": quality,
        "threads": plan.threads,
        "profile": plan.profile,
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


def composite_clip(
    base: Path,
    matte: Path,
    out_path: Path,
    media_fps: Fraction,
    ow: int,
    oh: int,
    duration: float,
    extras: RenderExtras,
    ass_path: Path | None,
    *,
    quality: str = "high",
    color_media: MediaInfo | None = None,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
) -> None:
    """Second pass of the AI-matte editors: base clip + person matte -> final clip.

    The base already carries the edit, framing and normalised audio; this pass only composites
    (background, text behind the subject), then applies the remaining effects, captions and
    music, and encodes once more at the export settings.
    """
    s = get_settings()
    plan = active_plan()
    workdir = out_path.parent
    _link_fonts(workdir)
    fps = media_fps
    extra_inputs = extras.inputs(2, fps, duration)
    parts = [f"[1:v]scale={ow}:{oh}:flags=bicubic,fps={fps.numerator}/{fps.denominator},format=gray[matte]"]
    post, last = post_parts("0:v", extras, ow, oh, fps, duration, matte="matte")
    parts += post
    tail = f"ass=filename={ass_path.name}:fontsdir=fonts," if ass_path is not None else ""
    parts.append(f"[{last}]{tail}format=yuv420p[vout]")
    if extras.music is not None:
        parts.append("[0:a]anull[voice]")
        parts += music_parts("voice", extras.music, duration)
    else:
        parts.append("[0:a]anull[aout]")
    args = ["-threads", str(plan.threads), "-i", base.name, "-threads", str(plan.threads), "-i", matte.name]
    args += extra_inputs
    crf = max(12, s.x264_crf - 2) if quality == "max" else s.x264_crf
    preset = "slow" if quality == "max" else plan.x264_preset
    rate = 16 if max(ow, oh) >= 1920 else 8
    tmp_out = out_path.with_suffix(".tmp.mp4")
    args += [
        "-filter_complex", ";".join(parts), "-map", "[vout]", "-map", "[aout]",
        "-filter_complex_threads", str(plan.threads),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-maxrate", f"{rate}M", "-bufsize", f"{rate * 2}M",
        *(_color_tags(color_media) if color_media is not None else []),
        "-g", str(max(12, int(round(float(fps) * 2)))),
        "-c:a", "aac", "-b:a", s.audio_bitrate, "-ar", str(AUDIO_RATE), "-ac", "2",
        "-t", f"{duration:.6f}", "-map_metadata", "-1", "-movflags", "+faststart", "-f", "mp4",
        "-threads", str(plan.threads), tmp_out.name,
    ]  # fmt: skip
    try:
        run_ffmpeg(
            args,
            total_duration=duration,
            on_progress=on_progress,
            should_cancel=should_cancel,
            cwd=workdir,
            timeout=s.ffmpeg_timeout_s,
        )
        os.replace(tmp_out, out_path)
    finally:
        tmp_out.unlink(missing_ok=True)
