"""Daily panel of ALL archived Binance USDT-M perpetuals (incl. delisted) from data.binance.vision
monthly 1d klines + fundingRate zips (downloaded for edge-v1, see scripts/data/edge_v1_fetch.md)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace/data_hist")
CACHE = ROOT / "perp_panel_1d.pkl"
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
         "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _read(z: Path, names=None) -> pd.DataFrame:
    with zipfile.ZipFile(z) as f:
        raw = f.read(f.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    has_header = any(c.isalpha() for c in first.split(",")[0])
    return pd.read_csv(io.StringIO(raw), header=0 if has_header else None, names=None if has_header else names)


def _ts(x):
    x = pd.to_numeric(x)
    return pd.to_datetime(x, unit="us" if x.max() > 1e14 else "ms", utc=True)


def load_perp_panel(refresh: bool = False) -> dict[str, pd.DataFrame]:
    if CACHE.exists() and not refresh:
        return pd.read_pickle(CACHE)
    close, qv, tbq, fund = {}, {}, {}, {}
    for d in sorted((ROOT / "perp_1d").iterdir()):
        parts = []
        for z in sorted(d.glob("*.zip")):
            try:
                k = _read(z, KCOLS)
                k.columns = KCOLS[: len(k.columns)]
                parts.append(k)
            except Exception:
                continue
        if not parts:
            continue
        k = pd.concat(parts)
        k.index = _ts(k["open_time"]).dt.floor("D")
        k = k[~k.index.duplicated()].sort_index()
        close[d.name] = k["close"].astype(float)
        qv[d.name] = k["quote_volume"].astype(float)
        tbq[d.name] = k["taker_buy_quote_volume"].astype(float)
        fparts = []
        for z in sorted((ROOT / "funding_all" / d.name).glob("*.zip")):
            try:
                f = _read(z, ["calc_time", "funding_interval_hours", "last_funding_rate"])
                fparts.append(pd.Series(f["last_funding_rate"].astype(float).values, index=_ts(f["calc_time"])))
            except Exception:
                continue
        if fparts:
            fs = pd.concat(fparts).sort_index()
            fs = fs[~fs.index.duplicated()]
            # settlement at 00:00 belongs to the PREVIOUS day's holding; known at that time
            fund[d.name] = fs.groupby((fs.index - pd.Timedelta("1us")).floor("D")).sum()
    panel = {"close": pd.DataFrame(close), "qv": pd.DataFrame(qv), "tbq": pd.DataFrame(tbq),
             "funding": pd.DataFrame(fund)}
    idx = panel["close"].index
    panel["funding"] = panel["funding"].reindex(index=idx, columns=panel["close"].columns)
    pd.to_pickle(panel, CACHE)
    return panel
