"""S6 Cointegration pairs on perpetuals (Danilo, Jamhamed & Martin 2024, CREM WP 2024-11,
RePEc:tut:cremwp:2024-11; copula variant Tadi & Witzany, arXiv:2305.06961).
Rolling OLS hedge ratio on log prices over W bars (past only), z-score of the spread over W,
enter |z|>z_in, exit |z|<z_out or |z|>z_stop. Dollar legs scaled to 0.5 gross each."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_beta(y: pd.Series, x: pd.Series, w: int) -> pd.Series:
    cov = y.rolling(w).cov(x)
    var = x.rolling(w).var()
    return cov / var


def pair_weights(py: pd.Series, px: pd.Series, w: int, z_in: float, z_out: float = 0.5,
                 z_stop: float = 4.0) -> pd.DataFrame:
    ly, lx = np.log(py), np.log(px)
    b = rolling_beta(ly, lx, w)
    spread = ly - b * lx
    z = (spread - spread.rolling(w).mean()) / spread.rolling(w).std()
    zv, bv = z.values, b.values
    st = np.zeros(len(zv))
    s = 0
    for i in range(len(zv)):
        if not np.isfinite(zv[i]):
            st[i] = 0
            s = 0
            continue
        if s == 0:
            if zv[i] > z_in:
                s = -1
            elif zv[i] < -z_in:
                s = 1
        elif abs(zv[i]) < z_out or abs(zv[i]) > z_stop:
            s = 0
        st[i] = s
    st = pd.Series(st, index=py.index)
    bb = pd.Series(bv, index=py.index).clip(-3, 3).fillna(0.0)
    gross = 1 + bb.abs()
    return pd.DataFrame({"Y": 0.5 * st * 2 / gross * 1.0, "X": -0.5 * st * 2 * bb / gross})
