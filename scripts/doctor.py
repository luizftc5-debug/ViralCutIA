#!/usr/bin/env python3
"""Check that this machine can run ViralCut AI without Docker.

Standard library only, so it runs before the Python environment exists:

    python scripts/doctor.py            (Windows: py scripts\\doctor.py)

Exit code 0 = everything required is present. Warnings do not fail the check.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "apps" / "api"
VENV_PY = (
    API / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
)

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  [FAIL] {msg}")


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        )
        return out.stdout + out.stderr
    except (OSError, subprocess.TimeoutExpired):
        return ""


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 11):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(
            f"Python {v.major}.{v.minor} found; 3.11 or newer is required (3.12 recommended)"
        )


def check_node() -> None:
    node = shutil.which("node")
    npm = shutil.which("npm")
    if not node or not npm:
        fail("Node.js and npm are required (https://nodejs.org, LTS 20 or newer)")
        return
    version = run([node, "--version"]).strip().lstrip("v")
    major = int(version.split(".")[0]) if version[:1].isdigit() else 0
    if major >= 20:
        ok(f"Node.js {version}")
    else:
        fail(f"Node.js {version or 'unknown'} found; version 20 or newer is required")


def check_ffmpeg() -> None:
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        if path:
            ok(f"{name} found: {path}")
        else:
            fail(
                f"{name} not on PATH (install a full build, see README 'Without Docker')"
            )
    if not shutil.which("ffmpeg"):
        return

    encoders = run(["ffmpeg", "-hide_banner", "-encoders"])
    if " libx264 " in encoders:
        ok("ffmpeg has libx264 (H.264 encoder)")
    else:
        fail("ffmpeg lacks libx264; rendering cannot produce H.264 MP4")

    filters = run(["ffmpeg", "-hide_banner", "-filters"])
    if " ass " in filters or " subtitles " in filters:
        ok("ffmpeg has libass (caption burn-in)")
    else:
        fail("ffmpeg lacks libass (the 'ass' filter); captions cannot be burned in")

    if " loudnorm " in filters and " atrim " in filters:
        ok("ffmpeg has loudnorm and atrim filters")
    else:
        fail("ffmpeg is missing loudnorm/atrim filters")


def check_project_files() -> None:
    model = API / "viralcut" / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
    fonts = API / "viralcut" / "assets" / "fonts"
    if model.exists():
        ok("face model present (YuNet)")
    else:
        fail(f"face model missing: {model}")
    if fonts.is_dir() and any(fonts.glob("*.ttf")):
        ok("caption fonts present")
    else:
        fail(f"caption fonts missing: {fonts}")


def check_venv() -> None:
    if not VENV_PY.exists():
        warn(
            "API virtualenv not created yet (scripts/dev.ps1 or scripts/dev.sh creates it)"
        )
        return
    res = subprocess.run(
        [str(VENV_PY), "-c", "import viralcut.config, fastapi, faster_whisper, cv2"],
        cwd=API,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
        ok("API virtualenv imports viralcut, fastapi, faster-whisper and OpenCV")
    else:
        fail(
            "API virtualenv is incomplete; delete apps/api/.venv and run the launcher again"
        )


def check_web_deps() -> None:
    if (ROOT / "apps" / "web" / "node_modules").is_dir():
        ok("web dependencies installed")
    else:
        warn("web dependencies not installed yet (the launcher runs npm install)")


def check_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            warn(
                f"port {port} is already in use; stop the other program or the stack will not start"
            )
        else:
            ok(f"port {port} is free")


def main() -> int:
    print("ViralCut AI - prerequisite check (no Docker)")
    print("Required:")
    check_python()
    check_node()
    check_ffmpeg()
    check_project_files()
    print("Environment:")
    check_venv()
    check_web_deps()
    check_port(8000)
    check_port(3000)
    print()
    if failures:
        print(
            f"{len(failures)} required item(s) missing. Fix them and run this check again."
        )
        return 1
    print("All required items are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
