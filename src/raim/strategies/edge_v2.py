"""edge-v2 signals (specs: configs/prereg/edge_v2/*.yaml, committed before data/run).
Info <= t only; the engine adds the +1 bar execution delay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from raim.strategies.edge_v1 import xs_long_short

PREM_ROOT = Path("/workspace/data_hist/premium_1d")
PREM_CACHE = Path("/workspace/data_hist/premium_panel_1d.pkl")


def load_premium(refresh: bool = False) -> pd.DataFrame:
    from raim.strategies.perp_panel import _read, _ts, KCOLS
    if PREM_CACHE.exists() and not refresh:
        return pd.read_pickle(PREM_CACHE)
    out = {}
    for d in sorted(PREM_ROOT.iterdir()):
        parts = []
        for z in sorted(d.glob("*.zip")):
            try:
                k = _read(z, KCOLS)
                k.columns = KCOLS[: len(k.columns)]
                parts.append(k)
            except Exception:
                continue
        if parts:
            k = pd.concat(parts)
            k.index = _ts(k["open_time"]).dt.floor("D")
            k = k[~k.index.duplicated()].sort_index()
            out[d.name] = pd.to_numeric(k["close"], errors="coerce")
    df = pd.DataFrame(out)
    pd.to_pickle(df, PREM_CACHE)
    return df


def resid_flow(P, univ, W=7):
    c, qv, tb = P["close"], P["qv"], P["tbq"]
    flow = (2 * tb - qv).rolling(W).sum() / qv.rolling(W).sum()
    r = (c / c.shift(W) - 1).where(univ)
    f = flow.where(univ & r.notna())
    fm = f.sub(f.mean(axis=1), axis=0)
    rr = r.where(f.notna())
    rm = rr.sub(rr.mean(axis=1), axis=0)
    b = (fm * rm).sum(axis=1) / (rm ** 2).sum(axis=1).replace(0, np.nan)
    return fm - rm.mul(b, axis=0)


def x1_weights(P, prem, univ, sig, W, reb):
    b = prem.reindex(index=P["close"].index, columns=P["close"].columns)
    s = b.rolling(W, min_periods=max(W // 2, 1)).mean() if sig == "level" else b - b.shift(W)
    return xs_long_short(s, univ, reb, long_high=True).where(P["close"].notna(), 0.0)


def x2_weights(P, univ, L, q, btc_filter, max_pos=10):
    c = P["close"]
    hi = c.shift(1).rolling(L, min_periods=L).max()
    lo = c.shift(1).rolling(max(L // 2, 2), min_periods=max(L // 2, 2)).min()
    fr = resid_flow(P, univ).rank(axis=1, pct=True)
    entry = (c >= hi) & (fr >= q) & univ
    if btc_filter:
        b = c["BTCUSDT"]
        ok = b > b.rolling(100, min_periods=100).mean()
        entry = entry & ok.values[:, None]
    ex = (c < lo) | c.isna()
    E, X, F = entry.values, ex.values, fr.fillna(-1).values
    pos = np.zeros(E.shape, dtype=bool)
    cur = np.zeros(E.shape[1], dtype=bool)
    for t in range(E.shape[0]):
        cur &= ~X[t]
        free = max_pos - cur.sum()
        cand = np.where(E[t] & ~cur)[0]
        if free > 0 and len(cand):
            cand = cand[np.argsort(-F[t, cand])][:free]
            cur[cand] = True
        pos[t] = cur
    return pd.DataFrame(pos.astype(float) / max_pos, index=c.index, columns=c.columns)


def x3_weights(P, univ, p, H, mode, per=0.1):
    c, f = P["close"], P["funding"]
    q = f.shift(1).rolling(90, min_periods=60).quantile(p)
    ev = (f <= q) & (f < 0) & univ
    r = c.pct_change(fill_method=None)
    sig = (ev & (r > 0)).astype(float)
    if mode == "both":
        sig = sig - (ev & (r < 0)).astype(float)
    w = sig.rolling(H, min_periods=1).sum() * per
    g = w.abs().sum(axis=1)
    w = w.div(np.maximum(g, 1.0), axis=0)
    return w.where(c.notna(), 0.0).fillna(0.0)


def payoff_stats(daily: pd.Series) -> dict:
    """Right-skew diagnostics on daily net returns (days with |r|>0)."""
    d = daily.dropna()
    d = d[d != 0]
    if len(d) < 10:
        return {}
    win, loss = d[d > 0], d[d < 0]
    tot = d.sum()
    top = d.sort_values(ascending=False).iloc[: max(1, int(0.05 * len(d)))]
    return {"skew": float(d.skew()), "win_rate": float(len(win) / len(d)),
            "payoff_ratio": float(win.mean() / -loss.mean()) if len(loss) else None,
            "top5pct_profit_share": float(top.sum() / tot) if tot > 0 else None,
            "n_active_days": int(len(d))}
