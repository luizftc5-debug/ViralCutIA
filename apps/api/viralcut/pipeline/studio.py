"""Creator Studio glue: user assets, and turning the editors' configuration into render inputs.

Kept out of the orchestrator so each editor's effect on a clip is in one readable place:
``overlay_events`` (ASS layers), ``render_extras`` (ffmpeg inputs and filters), ``edit_cuts``
(intervals removed by /corte and /tomada).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

from ..security import is_valid_id
from . import effects
from .editors import EditorsConfig
from .render import Background, ImageCard, MusicBed, RenderExtras, output_fps
from .text import norm
from .transcribe import Word

log = logging.getLogger(__name__)

ASSET_KINDS = {"music", "image", "reference"}


# --------------------------------------------------------------------------- assets


def assets_dir(job_dir: Path) -> Path:
    d = job_dir / "assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_assets(job_dir: Path) -> dict[str, dict]:
    try:
        return json.loads((job_dir / "assets" / "index.json").read_text())
    except (OSError, ValueError):
        return {}


def save_asset_index(job_dir: Path, index: dict) -> None:
    (assets_dir(job_dir) / "index.json").write_text(json.dumps(index))


def asset_path(job_dir: Path, asset_id: str, kind: str | None = None) -> Path | None:
    if not asset_id or not is_valid_id(asset_id):
        return None
    meta = list_assets(job_dir).get(asset_id)
    if meta is None or (kind and meta.get("kind") != kind):
        return None
    p = job_dir / "assets" / meta["file"]
    return p if p.exists() else None


# --------------------------------------------------------------------------- /corte, /tomada


def edit_cuts(
    words: list[Word], language: str, ed: EditorsConfig, notes: list[str]
) -> list[tuple[float, float]]:
    cuts: list[tuple[float, float]] = []
    if ed.corte.enabled and ed.corte.remove_fillers:
        f = effects.filler_cuts(words, language)
        if f:
            notes.append(f"/corte: {len(f)} filler word(s) removed")
        cuts += f
    if ed.tomada.enabled:
        r = effects.retake_cuts(words, language, ed.tomada.similarity)
        for a, b, text in r:
            notes.append(f"/tomada: earlier take removed ({b - a:.1f}s): “{text[:60]}”")
        cuts += [(a, b) for a, b, _ in r]
    return cuts


# --------------------------------------------------------------------------- overlays


def _keyword(words: list[Word], keywords: list[str]) -> str:
    spoken = {norm(w.text).strip(".,!?;:\"'") for w in words}
    for k in keywords:
        if k in spoken and len(k) >= 3:
            return k
    longest = sorted((w.text.strip(".,!?;:\"'") for w in words), key=len, reverse=True)
    return longest[0] if longest else ""


def overlay_events(
    cp,  # noqa: ANN001 - orchestrator.ClipPlan
    ed: EditorsConfig,
    width: int,
    height: int,
    keywords: list[str],
    accent: str,
    caption_top: bool,
) -> tuple[list[str], list[str]]:
    """(events over the picture, events behind the subject) for this clip's enabled editors."""
    over: list[str] = []
    behind: list[str] = []
    dur = cp.duration
    t_title = 0.0
    title_text = (ed.lettering.text or cp.title) if ed.lettering.enabled else ""
    same = title_text and norm(title_text)[:24] == norm(cp.hook_text)[:24]
    if ed.ganchos.enabled and ed.ganchos.hook_text and cp.hook_text and not same:
        over += effects.hook_events(cp.hook_text, width, height, duration=min(2.6, dur * 0.4))
        cp.edit_notes.append("/ganchos: opening line shown on screen for the first seconds")
    if ed.lettering.enabled:
        text = ed.lettering.text or cp.title
        if cp.rank_label:
            # rankings: the number is the headline, the title sits under it
            over += effects.title_events(
                cp.rank_label,
                width,
                height,
                duration=min(ed.lettering.duration, dur),
                animation="pop",
                position="top",
                accent=ed.lettering.color or accent,
                size_scale=1.9,
                layer=11,
            )
        if text:
            pos = ed.lettering.position
            if pos == "top" and (caption_top or cp.rank_label):
                pos = "middle"
            over += effects.title_events(
                text, width, height, t0=t_title, duration=min(ed.lettering.duration, dur),
                animation=ed.lettering.animation, position=pos, accent=ed.lettering.color or accent,
            )  # fmt: skip
            cp.edit_notes.append(f"/lettering: “{text[:50]}” ({ed.lettering.animation})")
    if same and ed.ganchos.hook_text:
        cp.edit_notes.append("/ganchos: the title already shows the opening line; not repeated")
    elif cp.rank_label and not ed.lettering.enabled:  # a ranking always shows its number
        over += effects.title_events(
            cp.rank_label, width, height, duration=min(2.5, dur), animation="pop", accent=accent
        )
    if ed.animacao.enabled:
        color = ed.animacao.color or accent
        if ed.animacao.progress_bar:
            from ..options import SAFE_AREAS

            top = SAFE_AREAS["youtube" if width > height else "universal"]["top"]
            over += effects.progress_bar_events(dur, width, height, color, top)
            if not ed.animacao.emphasis:
                cp.edit_notes.append("/animacao: progress bar")
        if ed.animacao.emphasis:
            times = sorted({round(z.t0 + z.ramp, 2) for z in cp.zooms})
            kw = {k for k in keywords[:8]}
            last = -9.0
            for w in cp.out_words or []:
                if norm(w.text).strip(".,!?;:\"'") in kw and w.start - last >= 3.0:
                    times.append(w.start)
                    last = w.start
            times = sorted(t for t in times if 0.3 < t < dur - 0.4)[:10]
            over += effects.emphasis_events(times, width, height, color)
            cp.edit_notes.append(
                f"/animacao: progress bar {'on' if ed.animacao.progress_bar else 'off'}, {len(times)} emphasis flash(es)"
            )
    if ed.texto_atras.enabled:
        text = ed.texto_atras.text or _keyword(cp.words, keywords)
        behind += effects.behind_text_events(
            text, dur, width, height, ed.texto_atras.color, ed.texto_atras.size
        )
        cp.edit_notes.append(f"/texto-atras: “{text.upper()}” placed behind the subject")
    return over, behind


# --------------------------------------------------------------------------- render extras


def detect_key_color(source: Path, family: str | None, t: float) -> dict | None:
    """Median colour of the frame border when it is a saturated green or blue (a real screen)."""
    from .ffmpeg import input_args

    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *input_args(source, family, seek=t, threads=1),
           "-frames:v", "1", "-vf", "scale=96:54,format=rgb24", "-f", "rawvideo", "-"]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
    if res.returncode != 0 or len(res.stdout) < 96 * 54 * 3:
        return None
    f = np.frombuffer(res.stdout[: 96 * 54 * 3], dtype=np.uint8).reshape(54, 96, 3).astype(int)
    border = np.concatenate(
        [f[:6].reshape(-1, 3), f[-6:].reshape(-1, 3), f[:, :8].reshape(-1, 3), f[:, -8:].reshape(-1, 3)]
    )
    r, g, b = np.median(border, axis=0)
    if g > 90 and g > r * 1.35 and g > b * 1.25:
        return {
            "key": f"0x{int(r):02X}{int(g):02X}{int(b):02X}",
            "similarity": 0.16,
            "blend": 0.07,
            "green": True,
        }
    if b > 90 and b > r * 1.35 and b > g * 1.15:
        return {
            "key": f"0x{int(r):02X}{int(g):02X}{int(b):02X}",
            "similarity": 0.16,
            "blend": 0.07,
            "green": False,
        }
    return None


def music_offset(path: Path, cache_file: Path) -> dict:
    from .music import analyze_music

    try:
        return json.loads(cache_file.read_text())
    except (OSError, ValueError):
        pass
    info = analyze_music(path)
    try:
        cache_file.write_text(json.dumps(info))
    except OSError:
        pass
    return info


def render_extras(cp, art, ed: EditorsConfig, job_dir: Path, source: Path) -> RenderExtras | None:  # noqa: ANN001
    """ffmpeg inputs and filters of the enabled editors for this clip (None when there are none)."""
    ex = RenderExtras()
    fps = float(output_fps(art.media))
    from .render import output_size

    ow, oh = output_size(art.media, cp.aspect)
    if ed.depth_3d.enabled:
        ex.video_post.append(effects.depth_filter(ed.depth_3d.strength, cp.duration, ow, oh, fps))
        cp.edit_notes.append(f"/3d: {ed.depth_3d.strength} camera move")
    if ed.prova.enabled:
        for item in ed.prova.items:
            p = asset_path(job_dir, item.asset, "image")
            if p is None:
                continue
            at = item.at
            if at is None and item.keyword:
                key = norm(item.keyword)
                at = next((w.start for w in cp.out_words or [] if key and key in norm(w.text)), None)
            if at is None:
                at = min(cp.duration * 0.35 + 3.2 * len(ex.cards), max(0.0, cp.duration - item.duration))
            t1 = min(cp.duration, at + item.duration)
            if t1 - at >= 0.8:
                ex.cards.append(ImageCard(p, round(at, 3), round(t1, 3)))
        if ex.cards:
            cp.edit_notes.append(
                f"/prova: {len(ex.cards)} evidence card(s) at " + ", ".join(f"{c.t0:.1f}s" for c in ex.cards)
            )
    if ed.trilha.enabled:
        p = asset_path(job_dir, ed.trilha.asset, "music")
        if p is not None:
            info: dict = {}
            if ed.trilha.beat_sync:
                try:
                    info = music_offset(p, job_dir / "assets" / f"{ed.trilha.asset}.beats.json")
                except Exception as exc:  # a beat analysis failure never blocks the render
                    log.warning("music analysis failed: %s", exc)
                    cp.edit_notes.append("/trilha: beat analysis failed; the music starts at its beginning")
            off = float(info.get("first_beat", 0.0))
            ex.music = MusicBed(p, ed.trilha.volume_db, ed.trilha.ducking, off)
            cp.edit_notes.append(
                f"/trilha: music at {ed.trilha.volume_db:.0f} dB"
                + (", ducked under speech" if ed.trilha.ducking else "")
                + (
                    f", starts on the first strong beat ({off:.2f}s, ~{info.get('tempo', 0):.0f} BPM)"
                    if info
                    else ""
                )
            )
        else:
            cp.edit_notes.append("/trilha: no music file uploaded, nothing added")
    if ed.tela_verde.enabled:
        tv = ed.tela_verde
        img = asset_path(job_dir, tv.asset, "image") if tv.background == "image" else None
        ex.background = Background(
            "image" if img else ("color" if tv.background == "color" else "blur"), tv.color or "#101820", img
        )
        if tv.mode == "chroma":
            key = detect_key_color(source, art.family, cp.start + 0.5)
            if key is not None:
                ex.chroma = key
                cp.edit_notes.append(
                    f"/tela-verde: chroma key {key['key']} → {ex.background.kind} background"
                )
            else:
                ex.background = None
                cp.edit_notes.append(
                    "/tela-verde: no green or blue screen detected at the frame edges; use the AI method instead"
                )
        else:
            cp.edit_notes.append(f"/tela-verde: AI person matting → {ex.background.kind} background")
    if not (ex.video_post or ex.cards or ex.music or ex.chroma or ex.background or ed.texto_atras.enabled):
        return None
    return ex
