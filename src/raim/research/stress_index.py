"""Continuous stress-dial baseline (Thawait, arXiv:2610.05740): LOB state has effective
dimension ~1.4 and one stress component explains ~84-85% of variance; fine partitions do not
transfer OOS, only ~4 macro bands do. Here: PC1 of the (train-standardized, causal) HMM
observation vector — spread, vol, VPIN, |OFI|, VR, Hawkes n (NOT spread alone: quoted spread
can narrow under stress) — oriented so that higher = more stress, cut into <= 4 bands by
TRAIN quantiles. Its band posterior is a drop-in replacement for the HMM posterior."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StressIndex:
    mean: np.ndarray
    w: np.ndarray
    edges: np.ndarray
    n_bands: int
    explained: float

    @classmethod
    def fit(cls, x: np.ndarray, n_bands: int = 4, vol_col: int = 1) -> StressIndex:
        """x: TRAIN rows only (already standardized)."""
        n_bands = min(int(n_bands), 4)
        x = x[np.isfinite(x).all(1)]
        mu = x.mean(0)
        xc = x - mu
        _, sv, vt = np.linalg.svd(xc, full_matrices=False)
        w = vt[0]
        score = xc @ w
        if np.corrcoef(score, x[:, vol_col])[0, 1] < 0:
            w, score = -w, -score
        edges = np.quantile(score, np.linspace(0, 1, n_bands + 1)[1:-1])
        explained = float(sv[0] ** 2 / (sv ** 2).sum())
        return cls(mu, w, edges, n_bands, explained)

    def score(self, row) -> float:
        r = np.asarray(row, dtype=float)
        if not np.isfinite(r).all():
            return float("inf")   # missing data -> treat as maximal stress (conservative)
        return float((r - self.mean) @ self.w)

    def posterior(self, row) -> list[float]:
        band = int(np.searchsorted(self.edges, self.score(row), side="right"))
        p = [0.0] * 4
        p[min(band + (4 - self.n_bands), 3)] = 1.0
        return p
