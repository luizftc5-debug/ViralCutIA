"""How recognition chooses between the graphics card and the processor, and what it reports.

Measured on a real machine (Windows, older NVIDIA card, no CUDA libraries): the driver alone made
the GPU look usable, a precision the card lacks was requested, all five models were downloaded
and failed the same way (5 GB), and the video was cut without captions. None of that may happen.
"""

import time

import numpy as np
import pytest

from viralcut import models as M
from viralcut.config import get_settings
from viralcut.pipeline import transcribe as T

OLD_CARD = {"float32", "int8", "int8_float32"}  # what CTranslate2 reports on a GTX 10xx / MX card


@pytest.fixture
def gpu_candidate(monkeypatch):
    """A machine where CTranslate2 sees a CUDA device: a driver is installed, nothing more is known."""
    monkeypatch.setenv("VIRALCUT_WHISPER_MODEL", "")
    monkeypatch.setenv("VIRALCUT_TRANSCRIPTION_MODE", "accurate")
    get_settings.cache_clear()
    monkeypatch.setattr(T, "_cuda_devices", lambda: 1)
    monkeypatch.setattr(T, "_supported_compute", lambda device: OLD_CARD)
    monkeypatch.setattr(T, "_cuda", {"checked": False, "ok": False, "reason": ""})
    monkeypatch.setattr(T, "_models", {})
    yield
    get_settings.cache_clear()


def test_a_card_without_half_precision_gets_a_precision_it_supports(monkeypatch):
    monkeypatch.setattr(T, "_supported_compute", lambda device: OLD_CARD)
    assert T._pick_compute("cuda") == "int8"
    monkeypatch.setattr(T, "_supported_compute", lambda device: OLD_CARD | {"float16", "int8_float16"})
    assert T._pick_compute("cuda") in ("float16", "int8_float16")
    # an explicit setting the card cannot run is replaced instead of failing every video
    monkeypatch.setattr(T, "_supported_compute", lambda device: OLD_CARD)
    assert T._pick_compute("cuda", "int8_float16") == "int8"
    assert T._pick_compute("cpu", "int8") == "int8"


def test_a_gpu_that_fails_its_self_test_is_not_used(gpu_candidate, monkeypatch):
    assert T.plan_asr().device == "cuda", "only a candidate until it is verified"
    monkeypatch.setattr(T, "_missing_cuda_libraries", lambda: "")
    monkeypatch.setattr(T, "_fetch_model", lambda name, on_status=None: "/nowhere")
    runs = []

    def self_test(path, compute):
        runs.append(compute)
        return False, "Library cublas64_12.dll is not found or cannot be loaded"

    monkeypatch.setattr(T, "_run_gpu_self_test", self_test)
    plan = T.ready_plan()
    assert (plan.device, plan.model, plan.compute) == ("cpu", "large-v3-turbo", "int8")
    assert runs == ["int8"], "tested with a precision the card supports"
    T.ready_plan()
    assert len(runs) == 1, "the verdict is kept: one test per process"
    assert "cublas64_12" in T._cuda["reason"]


def test_missing_nvidia_libraries_skip_the_self_test(gpu_candidate, monkeypatch):
    monkeypatch.setattr(T, "_missing_cuda_libraries", lambda: "cuBLAS and cuDNN")
    monkeypatch.setattr(T, "_run_gpu_self_test", lambda *a: pytest.fail("must not start"))
    monkeypatch.setattr(T, "_fetch_model", lambda *a, **k: pytest.fail("nothing to download"))
    assert T.ready_plan().device == "cpu"
    assert "cuBLAS and cuDNN" in T._cuda["reason"]


def test_a_verified_gpu_is_used_and_a_small_one_gets_the_turbo_model(gpu_candidate, monkeypatch):
    monkeypatch.setattr(T, "_missing_cuda_libraries", lambda: "")
    monkeypatch.setattr(T, "_fetch_model", lambda name, on_status=None: "/nowhere")
    monkeypatch.setattr(T, "_run_gpu_self_test", lambda path, compute: (True, ""))
    plan = T.ready_plan()
    assert (plan.device, plan.compute, plan.model) == ("cuda", "int8", "large-v3")

    class SmallCard:
        gpu = {"name": "GTX 1050 Ti", "vram_gb": 4.0, "vram_free_gb": 3.4}

    monkeypatch.setattr(T, "detect_hardware", lambda: SmallCard)
    assert T.ready_plan().model == "large-v3-turbo", "large-v3 without half precision does not fit in 4 GB"


def test_a_gpu_load_error_does_not_download_smaller_models(gpu_candidate, monkeypatch):
    import faster_whisper

    fetched = []
    monkeypatch.setattr(T, "_fetch_model", lambda name, on_status=None: fetched.append(name) or "/nowhere")

    def refuse(path, **kw):
        raise ValueError(
            "Requested int8_float16 compute type, but the target device or backend do not support it"
        )

    monkeypatch.setattr(faster_whisper, "WhisperModel", refuse)
    plan = T.ASRPlan("accurate", "large-v3", 5, True, "cuda", "int8_float16")
    with pytest.raises(ValueError):
        T.load_model(plan.model, None, plan)
    assert fetched == ["large-v3"], "the device is the problem: medium, small, base and tiny would fail too"


def test_recognition_that_fails_on_the_gpu_starts_again_on_the_processor(gpu_candidate, monkeypatch):
    monkeypatch.setattr(T, "_cuda", {"checked": True, "ok": True, "reason": ""})
    devices = []

    def fake(plan, samples, *rest):
        devices.append(plan.device)
        if plan.device == "cuda":
            raise RuntimeError("CUDA failed with error out of memory")
        return T.Transcript(engine="faster-whisper:test")

    monkeypatch.setattr(T, "_transcribe", fake)
    notes = []
    out = T.transcribe(np.zeros(16000, dtype=np.float32), on_note=notes.append)
    assert devices == ["cuda", "cpu"] and out.engine == "faster-whisper:test"
    assert "out of memory" in notes[0] and "processor" in notes[0]
    assert T.plan_asr().device == "cpu", "later videos of this session go straight to the processor"


@pytest.mark.integration
def test_the_self_test_survives_a_machine_without_cuda():
    """The real child process, on a machine whose GPU cannot run anything (CI has none)."""
    path = T._fetch_model("tiny")
    ok, reason = T._run_gpu_self_test(path, "int8")
    if not ok:  # expected here; a runner with a working GPU legitimately passes
        assert reason and "\n" not in reason


def test_download_progress_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "hub_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(M, "_repo_bytes", lambda repo, patterns: 4_000_000)
    seen = []
    with M.DownloadWatch("Downloading the speech model", "org/name", seen.append, interval=0.05):
        blobs = M.repo_dir("org/name") / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "model.incomplete").write_bytes(b"x" * 2_000_000)
        time.sleep(0.5)
    assert seen[0] == "Downloading the speech model…"
    assert "Downloading the speech model: 2 MB of 4 MB" in seen


def test_a_model_already_on_disk_is_not_downloaded_again(tmp_path, monkeypatch):
    from faster_whisper import utils

    folder = tmp_path / "snapshot"
    folder.mkdir()
    calls = []

    def download(name, local_files_only=False, cache_dir=None, **kw):
        calls.append(local_files_only)
        return str(folder)

    monkeypatch.setattr(utils, "download_model", download)
    monkeypatch.setattr(M, "_repo_bytes", lambda repo, patterns: None)
    T._fetch_model("tiny")
    assert calls == [True, False], "an interrupted download (no model.bin) is resumed"
    (folder / "model.bin").write_bytes(b"x")
    calls.clear()
    T._fetch_model("tiny")
    assert calls == [True], "complete on disk: the network is not touched"
