"""Markout (adverse-selection) reports and per-regime attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS_MS = (1000, 5000, 30000)


def markouts(fills: list[tuple], mids: list[tuple[int, float]],
             horizons_ms: tuple[int, ...] = HORIZONS_MS) -> pd.DataFrame:
    """Per fill: side * (mid(t+h) - px) / px in bps. Positive = fill was good for us.
    Uses the last mid at or before t+h; fills without a full horizon get NaN (no peeking
    past the end of data)."""
    if not fills or not mids:
        return pd.DataFrame()
    mts = np.array([m[0] for m in mids])
    mpx = np.array([m[1] for m in mids])
    rows = []
    for ts, side, px, qty, fee, regime in fills:
        row = {"ts": ts, "side": side, "px": px, "qty": qty, "fee": fee, "regime": regime}
        for h in horizons_ms:
            if ts + h > mts[-1]:
                row[f"mo_{h // 1000}s_bps"] = np.nan
            else:
                i = np.searchsorted(mts, ts + h, side="right") - 1
                row[f"mo_{h // 1000}s_bps"] = side * (mpx[i] - px) / px * 1e4
        i0 = max(np.searchsorted(mts, ts, side="right") - 1, 0)
        row["edge_vs_mid_bps"] = side * (mpx[i0] - px) / px * 1e4
        rows.append(row)
    return pd.DataFrame(rows)


def regime_attribution(mo: pd.DataFrame) -> pd.DataFrame:
    if mo.empty:
        return mo
    cols = [c for c in mo.columns if c.startswith("mo_")] + ["edge_vs_mid_bps"]
    g = mo.groupby("regime")
    out = g[cols].mean()
    out["n_fills"] = g.size()
    out["volume"] = g["qty"].sum()
    out.index = [f"R{int(i) + 1}" for i in out.index]
    return out
