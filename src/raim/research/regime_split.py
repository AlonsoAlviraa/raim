"""R12: BTC-trend regimes (causal: yesterday's close vs its 200d SMA and the SMA's 20d slope)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def btc_regimes(btc_close: pd.Series, sma_days: int = 200, slope_days: int = 20) -> pd.Series:
    c = btc_close.shift(1)
    sma = c.rolling(sma_days).mean()
    slope = sma.diff(slope_days)
    reg = pd.Series("sideways", index=c.index)
    reg[(c > sma) & (slope > 0)] = "bull"
    reg[(c < sma) & (slope < 0)] = "bear"
    return reg[sma.notna()]


def _sr(x):
    return float(x.mean() / x.std() * np.sqrt(365)) if len(x) > 5 and x.std() > 0 else 0.0


def _dd(x):
    eq = (1 + x).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(x) else 0.0


def regime_stats(daily: pd.Series, regimes: pd.Series) -> dict:
    d = daily.copy()
    d.index = d.index.tz_localize(None) if d.index.tz is not None else d.index
    r = regimes.copy()
    r.index = r.index.tz_localize(None) if r.index.tz is not None else r.index
    r = r.reindex(d.index.normalize(), method="ffill")
    r.index = d.index
    per = {g: {"days": int((r == g).sum()), "sharpe": _sr(d[r == g]), "max_dd": _dd(d[r == g])}
           for g in ("bull", "bear", "sideways")}
    years = {int(y): _sr(x) for y, x in d.groupby(d.index.year) if len(x) >= 60}
    present = [g for g in per if per[g]["days"] >= 30]
    return {"regimes": per, "regimes_positive": sum(per[g]["sharpe"] > 0 for g in present),
            "regimes_present": len(present),
            "worst_regime_dd": min(per[g]["max_dd"] for g in present) if present else 0.0,
            "years": years,
            "years_positive_fraction": float(np.mean([v > 0 for v in years.values()])) if years else 0.0}
