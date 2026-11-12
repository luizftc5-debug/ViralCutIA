from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session", autouse=True)
def _isolated_env(tmp_path_factory):
    """Point the app at a throwaway data directory before anything imports settings."""
    data = tmp_path_factory.mktemp("data")
    os.environ["VIRALCUT_DATA_DIR"] = str(data)
    os.environ["VIRALCUT_DATABASE_URL"] = os.environ.get("VIRALCUT_TEST_DATABASE_URL", "")
    os.environ.setdefault("VIRALCUT_WHISPER_MODEL", "tiny")
    os.environ["VIRALCUT_UPLOADS_PER_HOUR_PER_IP"] = "0"
    os.environ["VIRALCUT_X264_PRESET"] = "veryfast"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from viralcut.config import get_settings
    from viralcut.db import reset_engine

    get_settings.cache_clear()
    reset_engine()
    yield
    reset_engine()


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def _has_flite() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return " flite " in out


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def silent_video(media_dir: Path) -> Path:
    """Landscape 640x360 test pattern with a tone track and two hard cuts, 14 s."""
    out = media_dir / "pattern.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=5",
        "-f", "lavfi", "-i", "smptebars=s=640x360:r=30:d=4",
        "-f", "lavfi", "-i", "testsrc=s=640x360:r=30:d=5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=14",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(out),
    )  # fmt: skip
    return out


@pytest.fixture(scope="session")
def no_audio_video(media_dir: Path) -> Path:
    out = media_dir / "noaudio.webm"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=s=480x270:r=25:d=8", "-c:v", "libvpx", "-b:v", "400k", "-an", str(out)
    )
    return out


@pytest.fixture(scope="session")
def vertical_video(media_dir: Path) -> Path:
    out = media_dir / "vertical.mov"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=s=720x1280:r=30:d=8", "-f", "lavfi", "-i", "sine=frequency=330:duration=8",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
    )  # fmt: skip
    return out


@pytest.fixture(scope="session")
def speech_video(media_dir: Path) -> Path:
    """Synthetic talk (flite TTS) with strong and weak segments. Skipped without libflite."""
    if not _has_flite():
        pytest.skip("ffmpeg was built without libflite")
    out = media_dir / "talk.mp4"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_test_video.py"),
            "--out",
            str(out),
            "--size",
            "640x360",
            "--segments",
            "3",
        ],
        check=True,
    )
    return out


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from viralcut.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture()
def worker():
    from viralcut.worker.main import Worker

    return Worker()


@pytest.fixture(autouse=True)
def _fresh_content_cache():
    """Each test starts with an empty content-addressed cache (tests that need reuse do it themselves)."""
    import shutil

    from viralcut.config import get_settings

    shutil.rmtree(get_settings().cache_dir, ignore_errors=True)
    yield
