from pathlib import Path

import pytest

from viralcut.security import (
    UploadRejected,
    is_valid_id,
    resolve_inside,
    safe_stem,
    sanitize_filename,
    sniff_container,
    validate_extension_and_magic,
)
from viralcut.storage import LocalStorage

MP4_HEAD = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 56
MKV_HEAD = b"\x1a\x45\xdf\xa3" + b"\x00" * 60
ALLOWED = (".mp4", ".mov", ".mkv", ".webm")


def test_sanitize_filename_strips_paths_and_control_chars():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\Users\\me\\My Video?.mp4") == "My Video.mp4"
    assert sanitize_filename("a\x00b\n.mp4") == "ab.mp4"
    assert sanitize_filename("") == "video"
    assert len(sanitize_filename("x" * 500 + ".mp4")) <= 140


def test_safe_stem_is_ascii_slug():
    assert safe_stem("Minha Aula — Coração #1.mp4") == "Minha-Aula-Coracao-1"
    assert safe_stem("???.mp4") == "video"


def test_sniff_container():
    assert sniff_container(MP4_HEAD) == "isobmff"
    assert sniff_container(MKV_HEAD) == "matroska"
    assert sniff_container(b"#EXTM3U\nhttp://evil.example/x.ts" + b" " * 40) is None
    assert sniff_container(b"") is None


def test_extension_and_magic_must_agree():
    assert validate_extension_and_magic("a.mp4", MP4_HEAD, ALLOWED) == (".mp4", "isobmff")
    assert validate_extension_and_magic("a.WEBM", MKV_HEAD, ALLOWED) == (".webm", "matroska")
    with pytest.raises(UploadRejected):
        validate_extension_and_magic("a.exe", MP4_HEAD, ALLOWED)
    with pytest.raises(UploadRejected):
        validate_extension_and_magic("a.mp4", MKV_HEAD, ALLOWED)  # renamed file
    with pytest.raises(UploadRejected):
        validate_extension_and_magic("playlist.mp4", b"#EXTM3U" + b" " * 60, ALLOWED)


def test_ids_and_path_traversal(tmp_path: Path):
    assert is_valid_id("0123456789abcdef0123456789abcdef")
    assert not is_valid_id("../0123456789abcdef0123456789ab")
    assert not is_valid_id("0123456789ABCDEF0123456789abcdef")
    with pytest.raises(PermissionError):
        resolve_inside(tmp_path, "..", "outside")
    storage = LocalStorage(tmp_path)
    with pytest.raises(PermissionError):
        storage.job_dir("../../etc")
    with pytest.raises(PermissionError):
        storage.clip_dir("0123456789abcdef0123456789abcdef", "../x")
