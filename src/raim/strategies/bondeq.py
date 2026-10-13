"""bondeq-v1: bond-vs-equity relative value / timing signals (specs configs/prereg/bondeq_v1/*.yaml,
committed before data). ETF daily data reindexed to CALENDAR days (weekend = 0 return) so the repo's
sqrt(365) annualisation equals the usual sqrt(252) trading-day Sharpe. Info <= t only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace/data_hist/bondeq")
ETFS = ["SPY", "HYG", "IEF", "VCLT", "VCSH", "BIL"]
BORROW = {"IEF": 0.005, "HYG": 0.010}


def _yahoo(sym: str) -> pd.DataFrame:
    r = json.loads((ROOT / f"{sym}.json").read_text())["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    adj = r["indicators"]["adjclose"][0]["adjclose"]
    df = pd.DataFrame({"open": q["open"], "close": q["close"], "adj": adj},
                      index=pd.to_datetime(r["timestamp"], unit="s", utc=True).floor("D"))
    df = df[~df.index.duplicated(keep="last")].dropna()
    f = df["adj"] / df["close"]
    return pd.DataFrame({"open": df["open"] * f, "close": df["adj"]})


def load_panel(end: str | None = None) -> dict:
    cl, op = {}, {}
    for s in ETFS:
        d = _yahoo(s)
        cl[s], op[s] = d["close"], d["open"]
    close, opens = pd.DataFrame(cl), pd.DataFrame(op)
    idx = pd.date_range(close.index.min(), close.index.max(), freq="D", tz="UTC")
    traded = close.reindex(idx).notna()
    close, opens = close.reindex(idx).ffill(), opens.reindex(idx).ffill()
    f = pd.read_csv(ROOT / "BAA10Y.csv")
    f.columns = ["date", "v"]
    baa = pd.Series(pd.to_numeric(f["v"], errors="coerce").values,
                    index=pd.to_datetime(f["date"]).dt.tz_localize("UTC"))
    baa = baa.reindex(idx).ffill().shift(1)                      # published with a lag
    P = {"close": close, "open": opens, "traded": traded, "baa": baa}
    if end:
        P = {k: v.loc[v.index < pd.Timestamp(end, tz="UTC")] for k, v in P.items()}
    return P


def b1_weights(P, k, mode):
    s = P["close"]["SPY"].pct_change(k, fill_method=None)
    sig = np.sign(s) if mode == "both" else (s > 0).astype(float)
    sig = sig.where(P["traded"]["SPY"]).ffill().fillna(0.0)      # no new info on weekends
    W = pd.DataFrame(0.0, index=s.index, columns=["HYG", "IEF", "HYG_S", "IEF_S"])
    W["HYG"] = 0.5 * sig.clip(lower=0)
    W["IEF_S"] = -0.5 * sig.clip(lower=0)
    W["HYG_S"] = 0.5 * sig.clip(upper=0)
    W["IEF"] = -0.5 * sig.clip(upper=0)
    return W


def _switch(on, alt, idx, every=1):
    W = pd.DataFrame(0.0, index=idx, columns=["SPY", "BIL", "IEF"])
    W["SPY"] = on.astype(float)
    W[alt] = 1.0 - on.astype(float)
    if every > 1:
        is_reb = pd.Series(np.arange(len(W)) % every == 0, index=idx)
        W = W.where(is_reb, np.nan).ffill().fillna(0.0)
    return W


def b2_weights(P, L, alt):
    c = P["close"]
    slope = c["VCLT"].pct_change(L * 7 // 5, fill_method=None) - c["VCSH"].pct_change(L * 7 // 5, fill_method=None)
    on = (slope > 0) & slope.notna()
    return _switch(on, alt, c.index, every=7)


def b3_weights(P, N, alt):
    x = P["baa"]
    on = (x <= x.rolling(N * 7 // 5, min_periods=N).mean()) & x.notna()
    return _switch(on, alt, P["close"].index)


def returns_and_costs(P, cols):
    base = [c.replace("_S", "") for c in cols]
    R = P["close"][base].pct_change(fill_method=None).fillna(0.0)
    R.columns = cols
    O = P["open"][base].copy()
    O.columns = cols
    F = pd.DataFrame(0.0, index=R.index, columns=cols)
    for c in cols:
        if c.endswith("_S"):
            F[c] = -BORROW[c[:-2]] / 365                         # short pays borrow
    C = {c: 0.0002 for c in cols}
    return R, O, C, F


def yearly_vs_spy(daily: pd.Series, spy_close: pd.Series) -> dict:
    d = daily.copy()
    b = spy_close.pct_change(fill_method=None).reindex(d.index).fillna(0.0)
    out = {}
    for y in sorted(set(d.index.year)):
        m = d.index.year == y
        if m.sum() < 300:
            continue
        out[int(y)] = {"strat": float((1 + d[m]).prod() - 1), "spy": float((1 + b[m]).prod() - 1)}
        out[int(y)]["excess"] = out[int(y)]["strat"] - out[int(y)]["spy"]
    return out
