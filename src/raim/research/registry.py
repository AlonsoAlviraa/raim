"""Trial registry + Deflated Sharpe Ratio + frozen validator.

- Every evaluated configuration is appended to a registry, so the number of trials N used for
  deflation is honest (Bailey & López de Prado 2014, "The Deflated Sharpe Ratio").
- Gençay (arXiv:2608.27734): a deliberately leaky oracle can pass DSR/PBO, so DSR complements,
  never replaces, the structural anti-look-ahead tests.
- Qu et al. (arXiv:2609.27051): a FROZEN referee admits far fewer false discoveries. The
  validator code + held-out data are pinned by hash; any change makes verify() fail.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from statistics import NormalDist

_N = NormalDist()
EULER = 0.5772156649015329
VALIDATOR_FILES = ("registry.py", "walkforward.py", "markout.py", "gate.py", "robustness.py",
                   "verdict.py", "pbo.py", "spa.py", "mintrl.py", "stability.py", "costs_curve.py",
                   "regime_split.py", "bootstrap.py", "capacity.py", "prereg.py", "vault.py")


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    if n_trials <= 1:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_var) * ((1 - EULER) * a + EULER * b)


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, sr_var: float,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(true SR > 0) after selecting the best of n_trials (per-period SR, not annualized)."""
    sr0 = expected_max_sharpe(n_trials, sr_var)
    den = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    return _N.cdf((sr - sr0) * math.sqrt(max(n_obs - 1, 1)) / den)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TrialRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, params: dict, metrics: dict) -> str:
        h = _sha(json.dumps(params, sort_keys=True, default=str).encode())[:12]
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "trial": h, "params": params,
                                 "metrics": metrics}, default=str) + "\n")
        return h

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x]

    def n_trials(self) -> int:
        return len(self.all())


class FrozenValidator:
    def __init__(self, lock_path: str | Path) -> None:
        self.lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))

    @staticmethod
    def code_hash() -> str:
        here = Path(__file__).parent
        return _sha(b"".join((here / f).read_bytes() for f in VALIDATOR_FILES if (here / f).exists()))

    @staticmethod
    def freeze(lock_path: str | Path, data_path: str | Path) -> None:
        Path(lock_path).write_text(json.dumps({
            "validator_sha": FrozenValidator.code_hash(),
            "data_sha": _sha(Path(data_path).read_bytes()),
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")

    def verify(self, data_path: str | Path) -> None:
        if self.code_hash() != self.lock["validator_sha"]:
            raise RuntimeError("validator code changed since freeze: results not comparable")
        if _sha(Path(data_path).read_bytes()) != self.lock["data_sha"]:
            raise RuntimeError("held-out data changed since freeze")
