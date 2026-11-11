"""Upload validation and path-safety helpers. Uploaded bytes are untrusted."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# container sniffing: extension -> allowed container families
_EXT_FAMILY = {".mp4": "isobmff", ".mov": "isobmff", ".mkv": "matroska", ".webm": "matroska"}
# demuxer forced on ffmpeg/ffprobe so a renamed playlist/concat script can never be interpreted
FAMILY_DEMUXER = {"isobmff": "mov", "matroska": "matroska"}

_ISOBMFF_BOXES = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot"}


class UploadRejected(ValueError):
    """Raised with a message that is safe to show to the user."""


def is_valid_id(value: str) -> bool:
    return bool(_ID_RE.match(value or ""))


def sanitize_filename(name: str, fallback: str = "video") -> str:
    """Return a display-safe base name. The result is never used to build a storage path."""
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable() and ch not in '<>:"|?*\x00')
    name = re.sub(r"\s+", " ", name).strip().rstrip(" .")
    if not name:
        return fallback
    if name.startswith("."):
        name = fallback + name  # nothing left of the stem: keep only the extension
    stem, dot, ext = name.rpartition(".")
    if dot and stem:
        name = f"{stem[:120]}.{ext[:8]}"
    return name[:140]


def safe_stem(name: str, fallback: str = "clip") -> str:
    """ASCII-ish slug for download file names."""
    stem = Path(sanitize_filename(name)).stem
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return stem[:60] or fallback


def sniff_container(head: bytes) -> str | None:
    """Identify the container family from the first bytes of the file."""
    if len(head) >= 4 and head[:4] == b"\x1a\x45\xdf\xa3":
        return "matroska"
    if len(head) >= 8 and head[4:8] in _ISOBMFF_BOXES:
        return "isobmff"
    return None


def validate_extension_and_magic(filename: str, head: bytes, allowed: tuple[str, ...]) -> tuple[str, str]:
    """Return (extension, container family) or raise UploadRejected."""
    ext = Path(sanitize_filename(filename)).suffix.lower()
    if ext not in allowed:
        raise UploadRejected(
            f"Unsupported file type. Allowed formats: {', '.join(e.lstrip('.').upper() for e in allowed)}."
        )
    family = sniff_container(head)
    if family is None or family != _EXT_FAMILY.get(ext):
        raise UploadRejected("The file content does not match a supported video container.")
    return ext, family


def resolve_inside(base: Path, *parts: str) -> Path:
    """Join and resolve, refusing anything that escapes ``base`` (path traversal guard)."""
    base_r = base.resolve()
    target = base_r.joinpath(*parts).resolve()
    if target != base_r and base_r not in target.parents:
        raise PermissionError("path escapes storage root")
    return target
