"""Loaders for data.binance.vision monthly archives (klines + fundingRate).

Timestamps: Binance switched spot archives to microseconds in 2025; both units are handled.
Bars are indexed by their END time (open_time + interval) so that a bar's close is only
'known' at its index timestamp. This is the convention the leakage tests rely on."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace/data_hist")
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
         "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _to_dt(x: pd.Series) -> pd.DatetimeIndex:
    x = x.astype("int64")
    us = x > 10**14  # microseconds
    ms = np.where(us, x // 1000, x)
    return pd.to_datetime(ms, unit="ms", utc=True)


def _read_zip_csv(path: Path, names=None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:200].decode(errors="ignore").split("\n")[0]
    header = 0 if first[:1].isalpha() else None
    df = pd.read_csv(io.BytesIO(raw), header=header)
    if header is None and names:
        df.columns = names[: df.shape[1]]
    return df


def load_klines(symbol: str, market: str = "spot", interval: str = "1h",
                root: Path = ROOT) -> pd.DataFrame:
    d = {"spot": root / "spot_monthly_klines", "perp": root / "futures_um_monthly",
         "spot1m": root / "spot_1m"}[market] / symbol
    files = sorted(d.glob(f"{symbol}-{interval}-*.zip"))
    if not files:
        raise FileNotFoundError(f"no {market} {interval} klines for {symbol} in {d}")
    df = pd.concat([_read_zip_csv(f, KCOLS) for f in files], ignore_index=True)
    df.columns = KCOLS[: df.shape[1]]
    step = pd.Timedelta(interval.replace("m", "min")) if interval.endswith("m") else pd.Timedelta(interval)
    df["end"] = _to_dt(df["open_time"]) + step
    df = df.drop_duplicates("end").set_index("end").sort_index()
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]:
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]]


def load_funding(symbol: str, root: Path = ROOT) -> pd.Series:
    """Realised funding rate per settlement, indexed by settlement time rounded to the hour."""
    files = sorted((root / "futures_um_monthly" / symbol).glob(f"{symbol}-fundingRate-*.zip"))
    if not files:
        raise FileNotFoundError(f"no funding for {symbol}")
    df = pd.concat([_read_zip_csv(f) for f in files], ignore_index=True)
    t = _to_dt(df["calc_time"]).round("h")
    s = pd.Series(df["last_funding_rate"].astype(float).values, index=t)
    return s[~s.index.duplicated()].sort_index()


def funding_on_grid(funding: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Funding paid AT each bar-end timestamp (0 elsewhere). A position held during the bar
    ending at E (i.e. weight w_E) receives/pays the funding settled at E."""
    return funding.reindex(index).fillna(0.0)
