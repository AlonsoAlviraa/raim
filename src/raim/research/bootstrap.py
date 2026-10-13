"""R13 + R7 helper: stationary block bootstrap (Politis & Romano 1994)."""
from __future__ import annotations

import numpy as np


def stationary_indices(T: int, B: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    new = rng.random((B, T)) < p
    starts = rng.integers(0, T, (B, T))
    for t in range(1, T):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def _streak(neg: np.ndarray) -> np.ndarray:
    cur = np.zeros(neg.shape[0], dtype=np.int64)
    best = cur.copy()
    for t in range(neg.shape[1]):
        cur = np.where(neg[:, t], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def block_bootstrap(daily, n: int = 10000, mean_block: float = 20, seed: int = 7,
                    chunk: int = 2000) -> dict:
    x = np.asarray(daily, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    rng = np.random.default_rng(seed)
    srs, dds, streaks, losing = [], [], [], []
    done = 0
    while done < n:
        b = min(chunk, n - done)
        s = x[stationary_indices(T, b, mean_block, rng)]
        sd = s.std(1)
        srs.append(np.where(sd > 0, s.mean(1) / np.where(sd > 0, sd, 1), 0) * np.sqrt(365))
        eq = np.cumprod(1 + s, axis=1)
        dds.append((eq / np.maximum.accumulate(eq, axis=1) - 1).min(1))
        uw = eq < np.maximum.accumulate(eq, axis=1)
        streaks.append(_streak(uw))
        losing.append(_streak(s < 0))
        done += b
    sr, dd, st = np.concatenate(srs), np.concatenate(dds), np.concatenate(streaks)
    ls = np.concatenate(losing)
    return {"n": int(n), "sharpe_p5": float(np.percentile(sr, 5)), "sharpe_p50": float(np.median(sr)),
            "sharpe_p95": float(np.percentile(sr, 95)),
            "dd_p95": float(np.percentile(dd, 5)),          # 95% of paths have a milder max DD
            "dd_p99": float(np.percentile(dd, 1)),
            "underwater_streak_p95_days": float(np.percentile(st, 95)),
            "losing_streak_p95_days": float(np.percentile(ls, 95))}
