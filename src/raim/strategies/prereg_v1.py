"""Signal functions for the prereg-v1 families (specs: configs/prereg/v1/*.yaml, committed
before any data access). All weights use information <= t; the engine adds the +1 bar delay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from raim.strategies import convex
from raim.strategies.trend import donchian_state

LOOKBACKS_ALL = [5, 10, 20, 30, 60, 90, 150, 250, 360]


def btc_filter(close: pd.DataFrame, sma: int) -> pd.Series:
    b = close["BTCUSDT"]
    return (b > b.rolling(sma, min_periods=sma).mean()).astype(float)


def _cap(w: pd.DataFrame, max_gross: float = 1.0) -> pd.DataFrame:
    g = w.abs().sum(axis=1)
    return w.mul((max_gross / g).where(g > max_gross, 1.0), axis=0)


def p1_weights(close, univ, L, sma, tv=0.5):
    w = convex.alt_tsmom_weights(close, univ, L, tv)
    return w.mul(btc_filter(close, sma), axis=0)


def p2_weights(close, univ, tv, filt):
    w = convex.donchian_pit_weights(close, univ, LOOKBACKS_ALL, tv).clip(lower=0.0)
    if filt == "btc200":
        w = w.mul(btc_filter(close, 200), axis=0)
    return w


def p3_weights(close, univ, k, L, reb, tv=0.5):
    mom = (close / close.shift(L) - 1).where(univ)
    vol = np.log(close).diff().rolling(30, min_periods=20).std() * np.sqrt(365)
    rank = mom.where(mom > 0).rank(axis=1, ascending=False)
    pick = (rank <= k).astype(float)
    raw = _cap(pick * (tv / vol).clip(upper=2.0) / k).fillna(0.0)
    is_reb = pd.Series(np.arange(len(close)) % reb == 0, index=close.index)
    w = raw.where(is_reb, np.nan).ffill().fillna(0.0)
    w = w.mul(btc_filter(close, 200), axis=0)
    return w.where(close.notna(), 0.0)


def p4_weights(close, assets, rule, tv=0.5):
    syms = ["BTCUSDT"] if assets == "BTC" else ["BTCUSDT", "ETHUSDT"]
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for s in syms:
        c = close[s]
        if rule == "sma100":
            sig = (c > c.rolling(100, min_periods=100).mean()).astype(float)
        elif rule == "sma200":
            sig = (c > c.rolling(200, min_periods=200).mean()).astype(float)
        else:
            cs = c.dropna()
            sig = (sum(donchian_state(cs, n) for n in (20, 60, 120, 250)) / 4).clip(lower=0)
            sig = sig.reindex(c.index).fillna(0.0)
        vol = np.log(c).diff().rolling(90, min_periods=30).std() * np.sqrt(365)
        w[s] = (sig * (tv / vol).clip(upper=2.0) / len(syms)).fillna(0.0)
    return _cap(w).where(close.notna(), 0.0)
