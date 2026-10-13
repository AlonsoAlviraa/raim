"""R1: pre-registration. A family spec (YAML) is hashed and logged BEFORE any data is loaded or
any variant is run. Reports must cite the prereg sha; it must exist in the log with a timestamp
earlier than the first trial of that family (and, when available, be committed in git)."""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import yaml

LOG = Path("registry/prereg_log.jsonl")


def grid_size(grid) -> int:
    if isinstance(grid, list):
        return len(grid)
    if isinstance(grid, dict):
        return len(list(itertools.product(*[v if isinstance(v, list) else [v] for v in grid.values()])))
    raise ValueError("grid must be a list of dicts or a dict of lists")


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(spec: dict, rules: dict) -> None:
    r = rules["r1_prereg"]
    miss = [f for f in r["required_fields"] if f not in spec]
    if miss:
        raise ValueError(f"prereg missing fields: {miss}")
    n = grid_size(spec["grid"])
    if n > r["max_variants_per_family"]:
        raise ValueError(f"grid has {n} variants > {r['max_variants_per_family']} per family")


def register(spec_path: str | Path, rules: dict, log: Path = LOG, now: float | None = None) -> str:
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    validate(spec, rules)
    sha = file_sha(spec_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    for e in entries(log):
        if e["sha"] == sha:
            return sha
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sha": sha, "family": spec["family"], "path": str(spec_path),
                             "registered_ts": time.time() if now is None else now,
                             "n_variants": grid_size(spec["grid"]),
                             "retroactive": bool(spec.get("retroactive", False))}) + "\n")
    return sha


def entries(log: Path = LOG) -> list[dict]:
    return [json.loads(x) for x in log.read_text().splitlines() if x] if log.exists() else []


def lookup(sha: str, log: Path = LOG) -> dict | None:
    return next((e for e in entries(log) if e["sha"] == sha), None)


def git_commit_ts(path: str | Path) -> float | None:
    try:
        out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%ct", "--", str(path)],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[-1]) if out else None
    except Exception:
        return None


def check(sha: str | None, first_trial_ts: float | None, log: Path = LOG) -> dict:
    e = lookup(sha, log) if sha else None
    if e is None:
        return {"ok": False, "reason": "no prereg sha / not in log"}
    before = first_trial_ts is None or e["registered_ts"] < first_trial_ts
    ok = before and not e["retroactive"]
    return {"ok": ok, "registered_ts": e["registered_ts"], "retroactive": e["retroactive"],
            "n_variants": e["n_variants"], "before_first_trial": before,
            "reason": "" if ok else "registered after data/trials (retroactive)"}
