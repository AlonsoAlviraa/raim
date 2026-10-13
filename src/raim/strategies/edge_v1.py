"""edge-v1 market-neutral cross-sectional signals (specs: configs/prereg/edge_v1/*.yaml,
committed before the data was downloaded). Info <= t only; the engine adds the +1 bar delay."""
from __future__ import annotations

import numpy as np
import pandas as pd


def xs_long_short(score: pd.DataFrame, univ: pd.DataFrame, reb: int, q: float = 0.2,
                  long_high: bool = True) -> pd.DataFrame:
    s = score.where(univ)
    r = s.rank(axis=1, pct=True)
    n = s.notna().sum(axis=1)
    top, bot = (r > 1 - q), (r <= q)
    lng, sht = (top, bot) if long_high else (bot, top)
    w = lng.astype(float).div(lng.sum(axis=1).replace(0, np.nan), axis=0) * 0.5 \
        - sht.astype(float).div(sht.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    w = w.where(n >= 10, 0.0).fillna(0.0)                    # need a real cross-section
    is_reb = pd.Series(np.arange(len(w)) % reb == 0, index=w.index)
    return w.where(is_reb, np.nan).ffill().fillna(0.0)


def e1_weights(P, univ, W):
    f = P["funding"].rolling(W, min_periods=max(W // 2, 1)).mean()
    return xs_long_short(f, univ, 7, long_high=False).where(P["close"].notna(), 0.0)


def e2_weights(P, univ, k, cond):
    c, qv = P["close"], P["qv"]
    r = c / c.shift(k) - 1
    m = univ.copy()
    if cond == "low_vshock":
        vs = np.log((qv.rolling(k).mean()) / qv.shift(k).rolling(60, min_periods=30).median())
        vs = vs.where(univ)
        m = univ & vs.lt(vs.median(axis=1), axis=0)
    return xs_long_short(r, m, k, long_high=False).where(c.notna(), 0.0)


def e3_weights(P, univ, W, resid):
    c, qv, tb = P["close"], P["qv"], P["tbq"]
    flow = (2 * tb - qv).rolling(W).sum() / qv.rolling(W).sum()
    if resid:
        r = (c / c.shift(W) - 1).where(univ)
        f = flow.where(univ & r.notna())
        fm, rm = f.sub(f.mean(axis=1), axis=0), r.where(f.notna()).sub(r.where(f.notna()).mean(axis=1), axis=0)
        b = (fm * rm).sum(axis=1) / (rm ** 2).sum(axis=1).replace(0, np.nan)
        flow = fm - rm.mul(b, axis=0)
    return xs_long_short(flow, univ, W, long_high=True).where(c.notna(), 0.0)
