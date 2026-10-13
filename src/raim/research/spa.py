"""R7: White's Reality Check (2000) and Hansen's SPA test (2005), stationary bootstrap.

diffs: T x K matrix of (variant - benchmark) per-period returns. H0: no variant beats the
benchmark. Returns both p-values; the rule uses the SPA (consistent) p-value."""
from __future__ import annotations

import numpy as np

from raim.research.bootstrap import stationary_indices


def hansen_spa(diffs: np.ndarray, B: int = 2000, mean_block: float = 10, seed: int = 11) -> dict:
    d = np.asarray(diffs, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    T, K = d.shape
    rng = np.random.default_rng(seed)
    idx = stationary_indices(T, B, mean_block, rng)
    dbar = d.mean(0)
    boot = np.stack([d[idx[b]].mean(0) for b in range(B)])            # B x K
    omega = np.sqrt(T) * boot.std(0)
    omega = np.where(omega > 0, omega, 1e-12)
    t_spa = max(float(np.max(np.sqrt(T) * dbar / omega)), 0.0)
    keep = np.sqrt(T) * dbar / omega >= -np.sqrt(2 * np.log(np.log(max(T, 3))))
    mu_c = dbar * keep
    z = np.sqrt(T) * (boot - mu_c) / omega
    t_star = np.maximum(z.max(1), 0.0)
    v = float(np.max(np.sqrt(T) * dbar))
    v_star = (np.sqrt(T) * (boot - dbar)).max(1)
    return {"spa_p": float((t_star >= t_spa).mean()), "rc_p": float((v_star >= v).mean()),
            "t_spa": t_spa, "best_mean_diff": float(dbar.max()), "K": K, "T": T}
