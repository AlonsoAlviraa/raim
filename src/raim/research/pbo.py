"""R6: Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado & Zhu 2015).

M is a T x N matrix of per-period returns (one column per variant). Rows are split into S
contiguous blocks; for every split of the blocks into two halves the in-sample best variant is
ranked out of sample. PBO = share of splits where that variant falls at or below the OOS median."""
from __future__ import annotations

from itertools import combinations

import numpy as np


def cscv_pbo(M: np.ndarray, S: int = 16) -> dict:
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    if N < 2 or T < 2 * S:
        raise ValueError("need >=2 variants and >=2*S rows")
    if S % 2:
        raise ValueError("S must be even")
    edges = np.linspace(0, T, S + 1).astype(int)
    s1 = np.stack([M[a:b].sum(0) for a, b in zip(edges[:-1], edges[1:])])        # S x N
    s2 = np.stack([(M[a:b] ** 2).sum(0) for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.diff(edges).astype(float)
    tot1, tot2, totn = s1.sum(0), s2.sum(0), cnt.sum()

    def sharpe(a1, a2, n):
        mu = a1 / n
        var = np.maximum(a2 / n - mu ** 2, 1e-18)
        return mu / np.sqrt(var)

    logits = []
    for J in combinations(range(S), S // 2):
        J = list(J)
        i1, i2, n_in = s1[J].sum(0), s2[J].sum(0), cnt[J].sum()
        sr_is = sharpe(i1, i2, n_in)
        sr_oos = sharpe(tot1 - i1, tot2 - i2, totn - n_in)
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1) + 1
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
    lam = np.array(logits)
    return {"pbo": float((lam <= 0).mean()), "n_splits": len(lam), "logit_median": float(np.median(lam))}
