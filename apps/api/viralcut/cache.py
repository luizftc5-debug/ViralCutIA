"""Content-addressed cache for expensive, deterministic results (transcripts, analysis signals).

A key is a hash of *the bytes of the source video* plus every parameter that changes the result
(model, decoding options, glossary, code version...). The same upload processed again — a
retry, a second job with different clip lengths or styles, a duplicate upload — reuses the
result instead of decoding and transcribing the whole video again. Changing any parameter
changes the key, so a stale result is never served.

Entries contain transcripts, so they follow the same retention as uploads: the worker's
maintenance loop deletes entries older than ``retention_hours`` and keeps the cache under
``cache_max_mb`` (oldest first).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .config import get_settings

log = logging.getLogger(__name__)

_CHUNK = 4 << 20


def file_digest(path: Path) -> str:
    """BLAKE2b of the file contents, remembered in a sidecar so a file is read once."""
    path = Path(path)
    st = path.stat()
    sidecar = path.with_name(path.name + ".digest")
    stamp = f"{st.st_size}:{st.st_mtime_ns}"
    try:
        saved_stamp, saved = sidecar.read_text().split()
        if saved_stamp == stamp:
            return saved
    except (OSError, ValueError):
        pass
    h = hashlib.blake2b(digest_size=20)
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    digest = h.hexdigest()
    try:
        sidecar.write_text(f"{stamp} {digest}")
    except OSError:
        pass
    return digest


def make_key(*parts: Any) -> str:
    return hashlib.blake2b(
        json.dumps(parts, sort_keys=True, default=str).encode(), digest_size=20
    ).hexdigest()


def _path(namespace: str, key: str, suffix: str) -> Path:
    d = get_settings().cache_dir / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}{suffix}"


def get_json(namespace: str, key: str) -> Any | None:
    p = _path(namespace, key, ".json")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        os.utime(p)  # recently used entries survive the size cap
    except OSError:
        pass
    return data


def put_json(namespace: str, key: str, value: Any) -> None:
    p = _path(namespace, key, ".json")
    tmp = p.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:  # a full disk must not fail the job: the cache is an optimisation
        log.warning("cache write failed (%s): %s", namespace, exc)
        tmp.unlink(missing_ok=True)


def get_npz(namespace: str, key: str):  # noqa: ANN201 - numpy NpzFile
    import numpy as np

    p = _path(namespace, key, ".npz")
    if not p.exists():
        return None
    try:
        with np.load(p) as z:
            data = {k: z[k] for k in z.files}
        os.utime(p)
        return data
    except (OSError, ValueError):
        return None


def put_npz(namespace: str, key: str, **arrays: Any) -> None:
    import numpy as np

    p = _path(namespace, key, ".npz")
    tmp = p.with_name(p.stem + ".tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, p)
    except OSError as exc:
        log.warning("cache write failed (%s): %s", namespace, exc)
        tmp.unlink(missing_ok=True)


def purge(max_age_hours: float | None = None, max_mb: int | None = None) -> int:
    """Delete expired entries, then the oldest ones until the cache fits its size cap."""
    s = get_settings()
    max_age = (s.retention_hours if max_age_hours is None else max_age_hours) * 3600
    cap = (s.cache_max_mb if max_mb is None else max_mb) * 2**20
    root = s.cache_dir
    if not root.exists():
        return 0
    now = time.time()
    entries: list[tuple[float, int, Path]] = []
    removed = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if now - st.st_mtime > max_age or p.suffix == ".tmp" and now - st.st_mtime > 3600:
            p.unlink(missing_ok=True)
            removed += 1
        else:
            entries.append((st.st_mtime, st.st_size, p))
    total = sum(e[1] for e in entries)
    for _, size, p in sorted(entries):
        if total <= cap:
            break
        p.unlink(missing_ok=True)
        total -= size
        removed += 1
    return removed
