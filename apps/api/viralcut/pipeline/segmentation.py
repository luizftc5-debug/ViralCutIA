"""AI person matting for /tela-verde (AI mode) and /texto-atras.

MODNet (Apache-2.0, ONNX export ``Xenova/modnet``, 25 MB) runs on ONNX Runtime — already a
dependency for the forced aligner — on a 288x512 (or 512x288) copy of the rendered clip at up to
10 matte frames per second, with temporal smoothing. The matte is written as a small lossless
grey video and upscaled inside ffmpeg during the final composite. Measured: ~56 ms per frame at
256x256 with 2 threads, so a 30 s clip costs roughly 20–40 s of CPU.

The model is downloaded once, loaded only when a job enables one of these editors, and released
afterwards unless the TURBO profile keeps models resident. Decoding and writing stream frame by
frame, so memory stays flat whatever the clip length.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..models import DownloadWatch, report
from ..resources import active_plan, lower_priority
from .ffmpeg import CancelFn, Cancelled, FFmpegError, ProgressFn, read_pipe

log = logging.getLogger(__name__)

MODEL_REPO, MODEL_FILE = "Xenova/modnet", "onnx/model.onnx"
MATTE_FPS = 10.0
_lock = threading.Lock()
_session = None


class MattingUnavailable(RuntimeError):
    pass


def _load():  # noqa: ANN202
    global _session
    with _lock:
        if _session is not None:
            return _session
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            s = get_settings()
            kw = {"cache_dir": str(s.models_dir)} if s.models_dir else {}
            try:
                path = hf_hub_download(MODEL_REPO, MODEL_FILE, local_files_only=True, **kw)
            except Exception:
                what = "Downloading the person cut-out model, first time only"
                with DownloadWatch(what, MODEL_REPO, patterns=[MODEL_FILE]):
                    path = hf_hub_download(MODEL_REPO, MODEL_FILE, **kw)
                report("")
            so = ort.SessionOptions()
            so.intra_op_num_threads = active_plan().onnx_threads
            so.inter_op_num_threads = 1
            so.enable_cpu_mem_arena = False
            _session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        except Exception as exc:  # offline, disk, incompatible runtime
            raise MattingUnavailable(f"person matting model unavailable: {exc}") from exc
        return _session


def release() -> None:
    global _session
    with _lock:
        _session = None


def matte_size(width: int, height: int) -> tuple[int, int]:
    """Matte resolution: short side 288, long side 512 (MODNet needs multiples of 32)."""
    return (288, 512) if height >= width else (512, 288)


def matte_video(
    src: Path,
    out: Path,
    duration: float,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
) -> dict:
    """Write the person matte of ``src`` (an rendered clip) to ``out`` (FFV1 grey, MATTE_FPS)."""
    session = _load()
    from .probe import probe

    media = probe(src)
    mw, mh = matte_size(media.width, media.height)
    fps = min(MATTE_FPS, media.fps_float or MATTE_FPS)
    args = [
        "-threads", str(active_plan().threads), "-i", str(src), "-an",
        "-vf", f"fps={fps},scale={mw}:{mh}:flags=bilinear,format=rgb24", "-f", "rawvideo",
    ]  # fmt: skip
    writer = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{mw}x{mh}", "-r", f"{fps}", "-i", "pipe:0", "-c:v", "ffv1", "-level", "3", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )  # fmt: skip
    lower_priority(writer.pid)
    name = session.get_inputs()[0].name
    prev: np.ndarray | None = None
    expected = max(1.0, duration * fps)
    n = 0
    coverage: list[float] = []
    tops: list[float] = []  # where the top of the person is (fraction of the height)
    try:
        assert writer.stdin is not None
        for buf in read_pipe(args, chunk_size=mw * mh * 3, should_cancel=should_cancel):
            x = np.frombuffer(buf, dtype=np.uint8).reshape(mh, mw, 3).astype(np.float32)
            x = (x / 127.5 - 1.0).transpose(2, 0, 1)[None]
            m = session.run(None, {name: x})[0][0, 0]
            # temporal smoothing: the matte must not flicker between frames
            m = m if prev is None else 0.65 * m + 0.35 * prev
            prev = m
            coverage.append(float((m > 0.5).mean()))
            rows = np.where((m > 0.5).mean(axis=1) > 0.12)[0]
            if len(rows):
                tops.append(rows[0] / mh)
            writer.stdin.write((np.clip(m, 0, 1) * 255).astype(np.uint8).tobytes())
            n += 1
            if on_progress and n % 5 == 0:
                on_progress(min(1.0, n / expected))
            if should_cancel and should_cancel():
                raise Cancelled()
        writer.stdin.close()
        if writer.wait(timeout=120) != 0:
            raise FFmpegError("matte encode failed: " + (writer.stderr.read() or b"").decode()[-300:])
    except BaseException:
        writer.kill()
        writer.wait()
        out.unlink(missing_ok=True)
        raise
    finally:
        if not active_plan().keep_asr_model:
            release()
    cov = float(np.median(coverage)) if coverage else 0.0
    top = float(np.median(tops)) if tops else None
    return {"frames": n, "fps": fps, "size": f"{mw}x{mh}", "person_coverage": round(cov, 3),
            "person_top": round(top, 3) if top is not None else None}  # fmt: skip
