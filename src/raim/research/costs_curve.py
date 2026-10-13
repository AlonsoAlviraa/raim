"""R10: cost-sensitivity curve. Net = gross - m*cost + funding is linear in the cost multiplier m,
so the break-even multiple is exact: m* = sum(gross + funding) / sum(cost)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily(bar: pd.DataFrame, m: float) -> pd.Series:
    net = bar["gross"] - m * bar["cost"] + bar["funding"]
    return net.groupby(net.index.floor("D")).sum()


def sharpe_at(bar: pd.DataFrame, m: float) -> float:
    d = _daily(bar, m)
    return float(d.mean() / d.std() * np.sqrt(365)) if d.std() > 0 else 0.0


def breakeven_multiple(bar: pd.DataFrame) -> float:
    c = float(bar["cost"].sum())
    edge = float((bar["gross"] + bar["funding"]).sum())
    if c <= 0:
        return np.inf if edge > 0 else 0.0
    return edge / c


def curve(bar: pd.DataFrame, mults=(0, 0.5, 1, 1.5, 2, 3, 4, 5)) -> list[dict]:
    return [{"cost_mult": m, "sharpe": round(sharpe_at(bar, m), 3)} for m in mults]
