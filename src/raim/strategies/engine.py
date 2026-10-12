"""Vectorised, causal portfolio backtester shared by every strategy.

Contract: a strategy returns TARGET weights W[t, i] computed from information available at bar
end t. The engine holds w_held[t] = W[t - 1 - delay] (mandatory delay >= 1 bar beyond the
signal bar, so a signal seen at the close of t is first traded at the close of t+delay and
earns returns from the following bar). Costs are charged on |delta w_held| per instrument;
perp longs pay funding settled at the bar end (shorts receive it)."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from raim.research.registry import deflated_sharpe
from raim.research.robustness import cvar

ANN_DAYS = 365.0


@dataclass
class BTResult:
    bar: pd.DataFrame            # per-bar gross, cost, funding, net, gross exposure
    daily: pd.Series             # daily net simple returns (sum of bar returns)
    weights: pd.DataFrame
    meta: dict = field(default_factory=dict)


def run(weights: pd.DataFrame, returns: pd.DataFrame, side_cost: dict[str, float],
        funding: pd.DataFrame | None = None, delay: int = 1) -> BTResult:
    if delay < 0:
        raise ValueError("delay must be >= 0")
    weights = weights.reindex(returns.index).fillna(0.0)
    held = weights.shift(1 + delay).fillna(0.0)
    gross = (held * returns.fillna(0.0)).sum(axis=1)
    cvec = pd.Series(side_cost).reindex(held.columns).fillna(0.0)
    trade = held.diff().abs().fillna(held.abs())
    cost = (trade * cvec).sum(axis=1)
    fund = pd.Series(0.0, index=held.index)
    if funding is not None:
        f = funding.reindex(held.index).reindex(columns=held.columns).fillna(0.0)
        fund = -(held * f).sum(axis=1)          # long pays positive funding
    net = gross - cost + fund
    bar = pd.DataFrame({"gross": gross, "cost": cost, "funding": fund, "net": net,
                        "exposure": held.abs().sum(axis=1), "turnover": trade.sum(axis=1)})
    daily = net.groupby(net.index.floor("D")).sum()
    return BTResult(bar=bar, daily=daily, weights=held)


def metrics(daily: pd.Series, bar: pd.DataFrame | None = None) -> dict:
    d = daily.dropna()
    if len(d) < 5 or d.std() == 0:
        return {"n_days": int(len(d)), "sharpe": 0.0, "sr_daily": 0.0, "ann_ret": 0.0,
                "ann_vol": 0.0, "max_dd": 0.0, "cvar5": 0.0, "turnover_ann": 0.0,
                "skew": 0.0, "kurt": 3.0, "total_ret": 0.0, "pct_days_active": 0.0}
    eq = (1 + d).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    srd = float(d.mean() / d.std())
    years = len(d) / ANN_DAYS
    out = {"n_days": int(len(d)), "sr_daily": srd, "sharpe": srd * np.sqrt(ANN_DAYS),
           "ann_ret": float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0,
           "ann_vol": float(d.std() * np.sqrt(ANN_DAYS)), "max_dd": dd,
           "cvar5": cvar(d.values, 0.05), "skew": float(d.skew()),
           "kurt": float(d.kurt() + 3), "total_ret": float(eq.iloc[-1] - 1),
           "pct_days_active": float((d != 0).mean())}
    if bar is not None:
        b = bar.loc[(bar.index >= d.index[0]) & (bar.index < d.index[-1] + pd.Timedelta("1D"))]
        out["turnover_ann"] = float(b["turnover"].sum() / max(years, 1e-9))
        out["cost_ann"] = float(b["cost"].sum() / max(years, 1e-9))
        out["funding_ann"] = float(b["funding"].sum() / max(years, 1e-9))
        out["gross_ann"] = float(b["gross"].sum() / max(years, 1e-9))
    return out


def dsr(m: dict, n_trials: int, sr_var: float) -> float:
    return deflated_sharpe(m["sr_daily"], m["n_days"], max(n_trials, 1), max(sr_var, 1e-8),
                           skew=m.get("skew", 0.0), kurt=m.get("kurt", 3.0))


def psr0(m: dict) -> float:
    """Probabilistic Sharpe vs 0 (single trial, no deflation)."""
    sr, n = m["sr_daily"], m["n_days"]
    den = np.sqrt(max(1 - m["skew"] * sr + (m["kurt"] - 1) / 4 * sr * sr, 1e-12))
    return NormalDist().cdf(sr * np.sqrt(max(n - 1, 1)) / den)
