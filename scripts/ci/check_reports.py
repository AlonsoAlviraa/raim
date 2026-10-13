"""CI: reject any research report that lacks a prereg sha (present in registry/prereg_log.jsonl)
or a data manifest sha (matching the committed manifest). Usage: check_reports.py <dir> [...]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from raim.research.verdict import ReportError, validate_report  # noqa: E402


def check_dir(d: Path, prereg_log: Path, manifest_sha: str | None) -> list[str]:
    known = {json.loads(x)["sha"] for x in prereg_log.read_text().splitlines() if x} \
        if prereg_log.exists() else set()
    errs = []
    for f in sorted(Path(d).glob("*.json")):
        if f.name in ("summary.json", "data_manifest.json", "data_manifest_sha.json"):
            continue
        rep = json.loads(f.read_text())
        try:
            validate_report(rep)
        except ReportError as e:
            errs.append(f"{f.name}: {e}")
            continue
        if rep["prereg_sha"] not in known:
            errs.append(f"{f.name}: prereg_sha not in {prereg_log}")
        if manifest_sha and rep["data_manifest_sha"] != manifest_sha:
            errs.append(f"{f.name}: data_manifest_sha does not match the committed manifest")
    return errs


def main(argv):
    dirs = [Path(a) for a in argv[1:]] or [ROOT / "docs/research/rules_rescore_2026-10"]
    errs = []
    for d in dirs:
        ms = d / "data_manifest_sha.json"
        sha = json.loads(ms.read_text())["manifest_sha"] if ms.exists() else None
        errs += check_dir(d, ROOT / "registry/prereg_log.jsonl", sha)
    for e in errs:
        print("FAIL", e)
    print("ok" if not errs else f"{len(errs)} report(s) rejected")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
