"""S3 Donchian-ensemble trend following with volatility sizing (Zarattini, Pagani & Barbon 2025,
'Catching Crypto Trends', SSRN 5209907) - daily, long-only spot rotation over a fixed universe.
S4 Time-series momentum with ATR trailing stop on 6h bars (simplified AdaptiveTrend, Nguyen 2026,
arXiv:2602.11708) - perps, long-only or long/short with 70/30 asymmetry.

All functions are causal: values at t use closes <= t only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def donchian_state(close: pd.Series, n: int) -> pd.Series:
    """1 after close breaks the prior-n-bar high; exit when close < channel midpoint
    (trailing). Channel uses bars t-n..t-1 (strictly past)."""
    hi = close.shift(1).rolling(n).max()
    lo = close.shift(1).rolling(n).min()
    mid = (hi + lo) / 2
    c, h, m = close.values, hi.values, mid.values
    out = np.zeros(len(c))
    s = 0
    for i in range(len(c)):
        if np.isnan(h[i]):
            out[i] = 0
            continue
        if s == 0 and c[i] >= h[i]:
            s = 1
        elif s == 1 and c[i] < m[i]:
            s = 0
        out[i] = s
    return pd.Series(out, index=close.index)


def donchian_ensemble_weights(closes: pd.DataFrame, lookbacks: list[int], target_vol: float,
                              vol_window: int = 90, max_gross: float = 1.0) -> pd.DataFrame:
    sig = {}
    for col in closes:
        c = closes[col].dropna()
        e = sum(donchian_state(c, n) for n in lookbacks) / len(lookbacks)
        sig[col] = e.reindex(closes.index).fillna(0.0)
    sig = pd.DataFrame(sig)
    vol = np.log(closes).diff().rolling(vol_window).std() * np.sqrt(365)
    n = closes.notna().sum(axis=1).clip(lower=1)
    w = sig.mul(1.0 / n, axis=0) * (target_vol / vol).clip(upper=3.0)
    w = w.fillna(0.0)
    g = w.abs().sum(axis=1)
    scale = (max_gross / g).where(g > max_gross, 1.0)
    return w.mul(scale, axis=0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, k: int) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(k).mean()


def tsmom_atr_state(close, high, low, L: int, theta: float, alpha: float, k: int = 14,
                    allow_short: bool = False) -> pd.Series:
    mom = close / close.shift(L) - 1
    a = atr(high, low, close, k)
    c, mo, av = close.values, mom.values, a.values
    out = np.zeros(len(c))
    s, stop = 0, np.nan
    for i in range(len(c)):
        if np.isnan(mo[i]) or np.isnan(av[i]):
            continue
        if s == 0:
            if mo[i] > theta:
                s, stop = 1, c[i] - alpha * av[i]
            elif allow_short and mo[i] < -theta:
                s, stop = -1, c[i] + alpha * av[i]
        elif s == 1:
            stop = max(stop, c[i] - alpha * av[i])
            if c[i] < stop:
                s = 0
        elif s == -1:
            stop = min(stop, c[i] + alpha * av[i])
            if c[i] > stop:
                s = 0
        out[i] = s
    return pd.Series(out, index=close.index)


def tsmom_portfolio_weights(bars: dict[str, pd.DataFrame], L: int, theta: float, alpha: float,
                            allow_short: bool, long_frac: float = 0.7) -> pd.DataFrame:
    idx = None
    for b in bars.values():
        idx = b.index if idx is None else idx.union(b.index)
    st = pd.DataFrame({s: tsmom_atr_state(b.close, b.high, b.low, L, theta, alpha,
                                          allow_short=allow_short) for s, b in bars.items()}
                      ).reindex(idx).fillna(0.0)
    n = len(bars)
    if allow_short:
        w = st.clip(lower=0) * (long_frac / n) * 2 + st.clip(upper=0) * ((1 - long_frac) / n) * 2
    else:
        w = st / n
    return w
