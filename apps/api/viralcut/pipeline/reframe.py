"""Stage 8 — smart reframing to 9:16.

For each selected clip the source range is sampled, faces are detected (YuNet, with a Haar
cascade fallback), shots are split on frame-accurate scene changes and each shot receives a
layout:

* ``crop``  — a 9:16 window that follows the main subject on a smoothed path
* ``split`` — two stacked panels when two people cannot share one vertical window
* ``blur``  — the full frame over a blurred backdrop when no subject is found (also the
  default for gaming, where the action is rarely where the face is)
* ``fit``   — the source already is (close to) vertical

Layouts are expressed in source pixels and clip-relative seconds; ``render`` turns them into
an ffmpeg filter graph.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from .ffmpeg import CancelFn, ProgressFn, input_args, read_pipe
from .probe import MediaInfo

log = logging.getLogger(__name__)

SAMPLE_FPS = 5.0
DETECT_WIDTH = 640
TARGET_ASPECT = 9 / 16


@dataclass
class Shot:
    start: float  # clip-relative seconds
    end: float
    layout: dict[str, Any]

    def to_dict(self) -> dict:
        return {"start": round(self.start, 4), "end": round(self.end, 4), "layout": self.layout}


@dataclass
class FramingPlan:
    shots: list[Shot]
    detector: str = "none"
    face_ratio: float = 0.0  # share of sampled frames with at least one face
    mode: str = "auto"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "detector": self.detector,
            "face_ratio": round(self.face_ratio, 3),
            "layouts": sorted({s.layout["type"] for s in self.shots}),
            "shots": [s.to_dict() for s in self.shots],
            "notes": self.notes,
        }


class FaceDetector:
    """YuNet (bundled ONNX) when OpenCV supports it, otherwise the classic Haar cascade."""

    def __init__(self) -> None:
        import cv2

        self.cv2 = cv2
        self.name = "none"
        self._yunet = None
        self._haar = None
        self._size: tuple[int, int] | None = None
        model = get_settings().face_model_path
        if hasattr(cv2, "FaceDetectorYN") and model.exists():
            try:
                self._yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.6, 0.3, 50)
                self.name = "yunet"
            except Exception as exc:  # pragma: no cover - depends on OpenCV build
                log.warning("YuNet unavailable (%s); falling back to Haar cascade", exc)
        if self._yunet is None:
            try:
                path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(str(path))
                if not cascade.empty():
                    self._haar = cascade
                    self.name = "haar"
            except Exception as exc:  # pragma: no cover
                log.warning("no face detector available: %s", exc)

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Return faces as (x, y, w, h, score) in the frame's pixel coordinates."""
        h, w = frame.shape[:2]
        if self._yunet is not None:
            if self._size != (w, h):
                self._yunet.setInputSize((w, h))
                self._size = (w, h)
            _, faces = self._yunet.detect(frame)
            if faces is None:
                return []
            return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])) for f in faces]
        if self._haar is not None:
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            found = self._haar.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=6, minSize=(int(h * 0.08), int(h * 0.08))
            )
            return [(float(x), float(y), float(fw), float(fh), 0.7) for x, y, fw, fh in found]
        return []


def crop_size(width: int, height: int) -> tuple[int, int]:
    """Largest 9:16 window inside the frame (even dimensions)."""
    if width / height >= TARGET_ASPECT:
        ch = height
        cw = int(round(height * TARGET_ASPECT / 2) * 2)
    else:
        cw = width
        ch = int(round(width / TARGET_ASPECT / 2) * 2)
    return min(cw, width), min(ch, height)


def is_vertical(media: MediaInfo) -> bool:
    return media.width / media.height <= TARGET_ASPECT * 1.04


@dataclass
class _Track:
    t: list[float] = field(default_factory=list)
    cx: list[float] = field(default_factory=list)
    cy: list[float] = field(default_factory=list)
    w: list[float] = field(default_factory=list)
    h: list[float] = field(default_factory=list)

    def add(self, t: float, cx: float, cy: float, w: float, h: float) -> None:
        self.t.append(t)
        self.cx.append(cx)
        self.cy.append(cy)
        self.w.append(w)
        self.h.append(h)


def build_tracks(
    samples: list[tuple[float, list[tuple[float, float, float, float, float]]]], width: int, height: int
) -> list[_Track]:
    """Greedy nearest-neighbour association of detections into per-person tracks."""
    tracks: list[_Track] = []
    for t, faces in samples:
        used: set[int] = set()
        for x, y, w, h, score in sorted(faces, key=lambda f: f[2] * f[3], reverse=True):
            if score < 0.6 or h < height * 0.045:
                continue
            cx, cy = x + w / 2, y + h / 2
            best, best_d = None, 1e9
            for i, tr in enumerate(tracks):
                if i in used or t - tr.t[-1] > 2.0:
                    continue
                d = abs(cx - tr.cx[-1]) + 0.5 * abs(cy - tr.cy[-1])
                if d < best_d and d < max(width * 0.12, 1.5 * w):
                    best, best_d = i, d
            if best is None:
                tracks.append(_Track())
                best = len(tracks) - 1
            used.add(best)
            tracks[best].add(t, cx, cy, w, h)
    return tracks


def smooth_path(times: np.ndarray, values: np.ndarray, grid: np.ndarray, window_s: float = 1.2) -> np.ndarray:
    """Interpolate onto a regular grid and low-pass it (two passes of a moving average)."""
    if len(times) == 1:
        return np.full(len(grid), values[0])
    y = np.interp(grid, times, values)
    step = grid[1] - grid[0] if len(grid) > 1 else 1.0
    k = max(1, int(round(window_s / step)))
    if k > 1 and len(y) > 2:
        pad = np.pad(y, (k, k), mode="edge")
        kernel = np.ones(k) / k
        for _ in range(2):
            pad = np.convolve(pad, kernel, mode="same")
        y = pad[k:-k]
    return y


def simplify(times: np.ndarray, values: np.ndarray, tolerance: float) -> list[tuple[float, float]]:
    """Ramer–Douglas–Peucker reduction of a 1-D path to a few linear keyframes."""
    n = len(times)
    if n <= 2:
        return [(float(times[i]), float(values[i])) for i in range(n)]
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        ts = times[a + 1 : b]
        line = values[a] + (values[b] - values[a]) * (ts - times[a]) / (times[b] - times[a])
        err = np.abs(values[a + 1 : b] - line)
        k = int(np.argmax(err))
        if err[k] > tolerance:
            idx = a + 1 + k
            keep[idx] = True
            stack.extend([(a, idx), (idx, b)])
    return [(float(times[i]), float(values[i])) for i in range(n) if keep[i]]


def _crop_layout(
    keys: list[tuple[float, float]], cw: int, ch: int, width: int, height: int, y: float | None = None
) -> dict:
    max_x = max(0, width - cw)
    cleaned = [(round(t, 3), int(round(min(max_x, max(0.0, x))))) for t, x in keys]
    y_val = int(round((height - ch) / 2 if y is None else min(max(0.0, y), height - ch)))
    return {"type": "crop", "w": cw, "h": ch, "y": y_val, "keys": cleaned}


def _panel(cx: float, cy: float, face_h: float, width: int, height: int) -> dict:
    """A 9:8 region around one face for the split layout."""
    rh = min(float(height), max(face_h * 3.4, height * 0.62))
    rw = rh * 9 / 8
    if rw > width:
        rw = float(width)
        rh = rw * 8 / 9
    rw_i, rh_i = int(rw // 2 * 2), int(rh // 2 * 2)
    x = int(round(min(max(0.0, cx - rw_i / 2), width - rw_i)))
    y = int(round(min(max(0.0, cy - rh_i * 0.42), height - rh_i)))
    return {"x": x, "y": y, "w": rw_i, "h": rh_i}


def layout_for_shot(
    samples: list[tuple[float, list]],
    t0: float,
    t1: float,
    media: MediaInfo,
    mode: str,
) -> dict:
    width, height = media.width, media.height
    cw, ch = crop_size(width, height)
    center = _crop_layout([(t0, (width - cw) / 2)], cw, ch, width, height)
    if mode == "center":
        return center
    if mode == "blur":
        return {"type": "blur"}
    shot_samples = [(t, f) for t, f in samples if t0 <= t < t1]
    n = len(shot_samples)
    if n == 0:
        return {"type": "blur"} if mode == "auto" else center
    tracks = build_tracks(shot_samples, width, height)
    significant = []
    for tr in tracks:
        presence = len(tr.t) / n
        size = float(np.median(tr.h)) / height
        if presence >= 0.35 and size >= 0.05:
            significant.append((presence * size, presence, tr))
    significant.sort(key=lambda x: x[0], reverse=True)
    if not significant:
        # no person to follow: keep the whole frame visible instead of guessing a crop
        return {"type": "blur"} if mode == "auto" else center

    if len(significant) >= 2 and mode in ("auto", "split") and width / height > 1.0:
        (_, p1, a), (_, p2, b) = significant[0], significant[1]
        ax, bx = float(np.median(a.cx)), float(np.median(b.cx))
        span = abs(ax - bx) + 0.8 * (float(np.median(a.w)) + float(np.median(b.w)))
        if span > cw * 0.92 and (mode == "split" or min(p1, p2) >= 0.5):
            left, right = (a, b) if ax <= bx else (b, a)
            return {
                "type": "split",
                "top": _panel(
                    float(np.median(left.cx)),
                    float(np.median(left.cy)),
                    float(np.median(left.h)),
                    width,
                    height,
                ),
                "bottom": _panel(
                    float(np.median(right.cx)),
                    float(np.median(right.cy)),
                    float(np.median(right.h)),
                    width,
                    height,
                ),
            }
        if span <= cw * 0.92:
            mid = (ax + bx) / 2
            return _crop_layout([(t0, mid - cw / 2)], cw, ch, width, height)

    main = significant[0][2]
    times = np.array(main.t)
    grid = np.arange(t0, max(t1, t0 + 1 / SAMPLE_FPS), 1 / SAMPLE_FPS)
    if cw < width:
        path = smooth_path(times, np.array(main.cx), grid)
        if float(path.max() - path.min()) < width * 0.05:
            keys = [(t0, float(np.median(path)) - cw / 2)]
        else:
            keys = [(t, x - cw / 2) for t, x in simplify(grid, path, tolerance=width * 0.008)]
        return _crop_layout(keys, cw, ch, width, height)
    # source narrower than 9:16: the window slides vertically instead; keep the face in the upper third
    cy = float(np.median(main.cy))
    return _crop_layout([(t0, 0.0)], cw, ch, width, height, y=cy - ch * 0.38)


def _same_layout(a: dict, b: dict, width: int) -> bool:
    if a["type"] != b["type"]:
        return False
    if a["type"] in ("blur", "fit"):
        return True
    if a["type"] == "crop":
        return (
            len(a["keys"]) == 1
            and len(b["keys"]) == 1
            and abs(a["keys"][0][1] - b["keys"][0][1]) <= width * 0.02
            and a["y"] == b["y"]
        )
    return False


def merge_shots(shots: list[Shot], width: int) -> list[Shot]:
    out: list[Shot] = []
    for s in shots:
        if out and _same_layout(out[-1].layout, s.layout, width):
            out[-1].end = s.end
        else:
            out.append(s)
    return out


_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def plan_framing(
    path: Path,
    family: str | None,
    media: MediaInfo,
    start: float,
    end: float,
    mode: str = "auto",
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    max_shots: int = 40,
) -> FramingPlan:
    duration = max(0.05, end - start)
    if is_vertical(media) and mode in ("auto", "face", "split", "center"):
        cw, ch = crop_size(media.width, media.height)
        if cw == media.width and ch == media.height or abs(media.width / media.height - TARGET_ASPECT) < 0.01:
            return FramingPlan(
                [Shot(0.0, duration, {"type": "fit"})], mode=mode, notes=["source is already vertical"]
            )
    if mode == "blur":
        return FramingPlan([Shot(0.0, duration, {"type": "blur"})], mode=mode)
    if mode == "center":
        cw, ch = crop_size(media.width, media.height)
        layout = _crop_layout([(0.0, (media.width - cw) / 2)], cw, ch, media.width, media.height)
        return FramingPlan([Shot(0.0, duration, layout)], mode=mode)

    detector = FaceDetector()
    dw = DETECT_WIDTH if media.width >= DETECT_WIDTH else int(media.width // 2 * 2)
    dh = int(round(dw * media.height / media.width / 2) * 2)
    scale = media.width / dw

    samples: list[tuple[float, list]] = []
    with tempfile.TemporaryDirectory(prefix="vc-reframe-") as tmp:
        cuts_file = Path(tmp) / "cuts.txt"
        # eq(n,0) always lets the first frame through: ffmpeg 7+ aborts ("Could not open encoder
        # before EOF") when the null output receives no frames, i.e. a clip without any shot change.
        # That frame is at t=0, which the cut filter below ignores.
        graph = (
            f"[0:v]split[a][b];"
            f"[a]fps={SAMPLE_FPS},scale={dw}:{dh}:flags=bilinear,format=bgr24[det];"
            f"[b]scale=160:-2:flags=fast_bilinear,select='gt(scene,0.32)+eq(n,0)',metadata=print:file={cuts_file.name}[cut]"
        )
        args = [
            *input_args(Path(path).resolve(), family, seek=start, duration=duration),
            "-an", "-sn", "-filter_complex", graph,
            "-map", "[cut]", "-f", "null", os.devnull,
            "-map", "[det]", "-f", "rawvideo",
        ]  # fmt: skip
        expected = max(1.0, duration * SAMPLE_FPS)
        # the cuts file is addressed relative to ffmpeg's working directory, which keeps the
        # filter argument free of characters that would need filtergraph escaping
        for i, buf in enumerate(
            read_pipe(args, chunk_size=dw * dh * 3, should_cancel=should_cancel, cwd=Path(tmp))
        ):
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(dh, dw, 3)
            faces = detector.detect(frame) if detector.name != "none" else []
            samples.append(
                (
                    i / SAMPLE_FPS,
                    [(x * scale, y * scale, w * scale, h * scale, sc) for x, y, w, h, sc in faces],
                )
            )
            if on_progress and i % 5 == 0:
                on_progress(min(1.0, i / expected))
        cuts: list[float] = []
        if cuts_file.exists():
            for m in _PTS_RE.finditer(cuts_file.read_text(errors="replace")):
                t = float(m.group(1))
                if 0.3 < t < duration - 0.3 and (not cuts or t - cuts[-1] >= 0.4):
                    cuts.append(t)

    notes: list[str] = []
    if len(cuts) + 1 > max_shots:
        notes.append(f"{len(cuts)} shot changes: per-shot framing skipped for this fast-cut clip")
        cuts = []
        if mode == "auto":
            return FramingPlan(
                [Shot(0.0, duration, {"type": "blur"})], detector=detector.name, mode=mode, notes=notes
            )

    bounds = [0.0, *cuts, duration]
    shots = [
        Shot(bounds[i], bounds[i + 1], layout_for_shot(samples, bounds[i], bounds[i + 1], media, mode))
        for i in range(len(bounds) - 1)
    ]
    shots = merge_shots(shots, media.width)
    face_ratio = sum(1 for _, f in samples if any(x[4] >= 0.6 for x in f)) / max(1, len(samples))
    if detector.name == "haar":
        notes.append("basic Haar face detector in use (YuNet model unavailable): tracking is less accurate")
    elif detector.name == "none":
        notes.append("no face detector available: clips use the blurred-background layout")
    return FramingPlan(shots, detector=detector.name, face_ratio=face_ratio, mode=mode, notes=notes)
