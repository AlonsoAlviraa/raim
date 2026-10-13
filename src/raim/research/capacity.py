"""R14 capacity: traded notional per instrument-day vs trailing 30d median ADV, at target capital,
plus the square-root impact slippage model used in engine (sigma * sqrt(q / ADV))."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def base_symbol(col: str) -> str:
    m = re.match(r"([A-Z0-9]+USDT)", str(col))
    return m.group(1) if m else str(col)


def participation(weights: pd.DataFrame, adv: pd.DataFrame, capital: float) -> dict:
    trade = weights.diff().abs().fillna(weights.abs())
    daily = trade.groupby(trade.index.floor("D")).sum()
    cols = {c: base_symbol(c) for c in daily.columns}
    adv = adv.copy()
    adv.index = pd.to_datetime(adv.index, utc=True).floor("D")
    a30 = adv.rolling(30, min_periods=10).median().shift(1)
    vals, missing = [], []
    for c, s in cols.items():
        if s not in a30.columns:
            missing.append(c)
            continue
        q = daily[c][daily[c] > 0] * capital
        if q.empty:
            continue
        av = a30[s].reindex(q.index)
        vals.append((q / av).replace([np.inf], np.nan).dropna())
    if not vals:
        return {"p95": None, "max": None, "missing": missing}
    v = pd.concat(vals)
    return {"p95": float(v.quantile(0.95)), "max": float(v.max()), "median": float(v.median()),
            "n_obs": int(len(v)), "missing": missing}


def sqrt_impact_bps(sigma_daily: float, q_usd: float, adv_usd: float, k: float = 1.0) -> float:
    return 1e4 * k * sigma_daily * np.sqrt(max(q_usd, 0) / max(adv_usd, 1e-9))
