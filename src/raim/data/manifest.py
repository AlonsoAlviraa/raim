"""R14 data snapshot manifest: sha256 of every raw data file used by a study. A report cites the
manifest sha; CI / verify() recompute file hashes and fail on any change."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha_file(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(buf):
            h.update(chunk)
    return h.hexdigest()


def build(roots: list[str | Path], cache: Path | None = None) -> dict:
    c = json.loads(cache.read_text()) if cache and cache.exists() else {}
    files = {}
    for r in roots:
        for p in sorted(Path(r).rglob("*")):
            if not p.is_file() or p.name.startswith(".") or p.suffix == ".pkl":
                continue
            st = p.stat()
            key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
            if key not in c:
                c[key] = sha_file(p)
            files[str(p)] = c[key]
    if cache:
        cache.write_text(json.dumps(c))
    body = json.dumps(files, sort_keys=True).encode()
    return {"manifest_sha": hashlib.sha256(body).hexdigest(), "n_files": len(files), "files": files}


def verify(manifest: dict, sample: int | None = None) -> list[str]:
    items = list(manifest["files"].items())
    if sample:
        items = items[:: max(len(items) // sample, 1)]
    return [f for f, h in items if not Path(f).exists() or sha_file(Path(f)) != h]
