"""CI: the verdict thresholds file may NOT change in the same change set as research results.

Usage: python scripts/ci/check_rules_separation.py <base_ref>   (or pipe changed paths on stdin)
Exit 1 when configs/research_rules.yaml changes together with reports/ or docs/research/ results."""
from __future__ import annotations

import subprocess
import sys

RULES = "configs/research_rules.yaml"
RULE_FILES = (RULES, "configs/research_rules_aggressive.yaml")
RESULT_PREFIXES = ("reports/", "docs/research/", "registry/")


def violations(paths: list[str]) -> list[str]:
    paths = [p.strip().replace("\\", "/") for p in paths if p.strip()]
    if not any(r in paths for r in RULE_FILES):
        return []
    return [p for p in paths if p.startswith(RESULT_PREFIXES)]


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        out = subprocess.run(["git", "diff", "--name-only", f"{argv[1]}...HEAD"],
                             capture_output=True, text=True, check=True).stdout
        paths = out.splitlines()
    else:
        paths = sys.stdin.read().splitlines()
    bad = violations(paths)
    if bad:
        print(f"FAIL: {RULES} changed together with results: {bad}")
        return 1
    print("ok: rules file not changed alongside results")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
