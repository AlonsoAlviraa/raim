"""S5 BTC -> altcoin lead-lag on 1-minute bars (Kurihara & Matsumoto 2026, Asia-Pacific Financial
Markets, doi:10.1007/s10690-026-09589-z). The paper uses LightGBM on [r_BTC,t-1, r_ALT,t-1];
we use the linear analogue (OLS fitted on the training window only) to keep the trial count and
the overfitting surface small. Long-only spot, as in the paper."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_ols(r_btc: pd.Series, r_alt: pd.Series) -> np.ndarray:
    X = np.column_stack([np.ones(len(r_btc) - 1), r_btc.values[:-1], r_alt.values[:-1]])
    y = r_alt.values[1:]
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return beta


def leadlag_weights(r_btc: pd.Series, r_alt: pd.Series, beta: np.ndarray, entry: float,
                    hold: float) -> pd.Series:
    pred = beta[0] + beta[1] * r_btc + beta[2] * r_alt       # forecast of r_alt,t+1 at t
    p = pred.values
    out = np.zeros(len(p))
    s = 0
    for i in range(len(p)):
        if not np.isfinite(p[i]):
            out[i] = s
            continue
        s = 1 if (s == 0 and p[i] > entry) or (s == 1 and p[i] > hold) else 0
        out[i] = s
    return pd.Series(out, index=r_btc.index)
