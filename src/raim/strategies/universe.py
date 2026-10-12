"""Point-in-time (PIT) universe from data.binance.vision daily spot klines of ALL USDT pairs that
exist in the archive (delisted symbols included). At date t the universe is the top-N symbols by
trailing median daily quote volume over [t-lb, t-1] (strictly past) with >= min_hist days of
history. No current market-cap list is used anywhere, so there is no look-ahead selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raim.strategies import data

CACHE = Path("/workspace/data_hist/panel_1d.pkl")


def load_panel(root: Path = data.ROOT, refresh: bool = False) -> dict[str, pd.DataFrame]:
    if CACHE.exists() and not refresh:
        return pd.read_pickle(CACHE)
    syms = sorted(p.name for p in (root / "spot_1d").iterdir() if p.is_dir())
    close, qv, opn, high, low = {}, {}, {}, {}, {}
    for s in syms:
        try:
            k = data.load_klines(s, "spot1d", "1d")
        except (FileNotFoundError, ValueError):
            continue
        close[s], qv[s], opn[s], high[s], low[s] = k.close, k.quote_volume, k.open, k.high, k.low
    panel = {"close": pd.DataFrame(close), "qv": pd.DataFrame(qv), "open": pd.DataFrame(opn),
             "high": pd.DataFrame(high), "low": pd.DataFrame(low)}
    pd.to_pickle(panel, CACHE)
    return panel


def pit_universe(qv: pd.DataFrame, top_n: int, lb: int = 30, min_hist: int = 60) -> pd.DataFrame:
    med = qv.shift(1).rolling(lb, min_periods=lb // 2).median()
    hist = qv.notna().cumsum().shift(1)
    med = med.where((hist >= min_hist) & qv.shift(1).notna())   # must still trade (delisting)
    rank = med.rank(axis=1, ascending=False)
    return (rank <= top_n) & med.notna()


def slip_tier(qv: pd.DataFrame) -> pd.DataFrame:
    """Per-side slippage by trailing (past) median quote volume: >=100M$:3bp, >=10M$:10bp,
    >=1M$: 30bp, else 80bp."""
    med = qv.shift(1).rolling(30, min_periods=5).median()
    s = pd.DataFrame(0.0080, index=qv.index, columns=qv.columns)
    s[med >= 1e6] = 0.0030
    s[med >= 1e7] = 0.0010
    s[med >= 1e8] = 0.0003
    return s
