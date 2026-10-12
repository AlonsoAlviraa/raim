"""Trade-level and tail metrics for convex / low-win-rate strategies.

A 'trade' is a maximal run of non-zero, same-sign held weight in one instrument. Its PnL is the
sum of that instrument's net bar PnL over the run PLUS the exit bar (where the exit cost lands).
PnL is in units of portfolio capital."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trades(weights: pd.DataFrame, inst_pnl: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    rows = []
    W = weights if start is None else weights.loc[start:end]
    P = inst_pnl.reindex(W.index).fillna(0.0)
    for c in W.columns:
        s = np.sign(W[c].values)
        p = P[c].values
        i, n = 0, len(s)
        while i < n:
            if s[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < n and s[j + 1] == s[i]:
                j += 1
            end_i = min(j + 1, n - 1)
            rows.append({"inst": c, "entry": W.index[i], "exit": W.index[end_i], "side": s[i],
                         "bars": j - i + 1, "pnl": float(p[i:end_i + 1].sum())})
            i = j + 1
    return pd.DataFrame(rows)


def longest_streak(x: np.ndarray) -> int:
    best = cur = 0
    for v in x:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def recovery_days(daily: pd.Series) -> int:
    eq = (1 + daily.fillna(0)).cumprod()
    under = eq < eq.cummax()
    best = cur = 0
    for u in under.values:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return int(best)


def convex_stats(tr: pd.DataFrame, daily: pd.Series) -> dict:
    out = {"n_trades": int(len(tr)), "time_to_recovery_days": recovery_days(daily),
           "daily_skew": float(daily.skew()) if len(daily) > 3 else 0.0}
    if len(tr) == 0:
        return {**out, "win_rate": 0.0, "payoff": 0.0, "profit_factor": 0.0, "trade_skew": 0.0,
                "top5pct_share": 0.0, "longest_losing_streak": 0}
    p = tr.sort_values("exit")["pnl"].values
    w, l = p[p > 0], p[p <= 0]
    tot = p.sum()
    k = max(1, int(np.ceil(0.05 * len(p))))
    top = np.sort(p)[::-1][:k].sum()
    out.update({"win_rate": float(len(w) / len(p)),
                "payoff": float(w.mean() / abs(l.mean())) if len(w) and len(l) and l.mean() != 0 else float("inf") if len(w) else 0.0,
                "profit_factor": float(w.sum() / abs(l.sum())) if len(l) and l.sum() != 0 else float("inf"),
                "trade_skew": float(pd.Series(p).skew()) if len(p) > 3 else 0.0,
                "top5pct_share": float(top / tot) if tot > 0 else float("nan"),
                "longest_losing_streak": longest_streak(p)})
    return out
