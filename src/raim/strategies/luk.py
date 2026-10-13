"""luk-v1: Martin Luk daily-bar pullback strategy & diagnostics (spec docs/research/luk_lit_2026-10.md section 3).

SIMULATION / VIRTUAL CAPITAL ONLY. Never live trading. Not financial advice.
All indicators, universe membership, RS rankings, and signals at day t use strictly information <= t.
Fills happen at open_{t+1} (or open_{t+1+extra_delay} for R11)."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from raim.strategies.engine import ANN_DAYS, BTResult, metrics

log = logging.getLogger("luk")

ROOT = Path("/workspace/data_hist/luk")
SP500_CSV_NAME = "S&P 500 Historical Components & Changes (Updated).csv"
SP500_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(08-17-2024).csv"
SP500_REPO_API = "https://api.github.com/repos/fja05680/sp500/contents"


@dataclass(frozen=True)
class LukConfig:
    id: str = "V0"
    trail: str = "ema9"             # "ema9" | "ema21"
    market_filter: str = "qqq"      # "qqq" | "none" | "breadth"
    avwap_confluence: bool = False
    stop: str = "flush_low"         # "flush_low" | "low_minus_half_atr"
    stop_cap_adr: float = 0.5       # max (entry - stop)/entry as multiple of ADR20 (capped at 5% for flush_low)
    rs: str = "composite"           # "composite" | "high52w"
    risk_scaling: str = "fixed"     # "fixed" | "qqq_vol"
    hard_stop_enabled: bool = True  # False for diagnostic (d) "no hard stop"
    delist_mode: str = "pessimistic"  # "pessimistic" (-30%) | "neutral" (0%)
    cost_mult: float = 1.0          # 1.0 base, 2.0 stress
    extra_delay: int = 0            # 0 -> open_{t+1}, 1 -> open_{t+2} (R11)
    target_capital: float = 100_000.0
    base_risk_frac: float = 0.005


def config_from_grid(p: dict, **overrides) -> LukConfig:
    kw = {k: v for k, v in p.items() if k in LukConfig.__dataclass_fields__}
    if p.get("mkt_none"):
        kw["market_filter"] = "none"
    elif p.get("mkt_breadth"):
        kw["market_filter"] = "breadth"
    if kw.get("stop") == "low_minus_half_atr" and "stop_cap_adr" not in p:
        kw["stop_cap_adr"] = 1.0
    kw.update(overrides)
    return LukConfig(**kw)


def normalize_grid(raw_grid: list[dict]) -> list[dict]:
    """Normalize preregistration grid dicts to the 7 single-factor ablation switches from V0 so
    `raim.research.stability.neighbors` detects all 7 1-parameter neighbours of V0."""
    out: list[dict] = []
    for p in raw_grid:
        mf = p.get("market_filter", "qqq")
        out.append({
            "trail": p.get("trail", "ema9"),
            "mkt_none": mf == "none",
            "mkt_breadth": mf == "breadth",
            "avwap_confluence": bool(p.get("avwap_confluence", False)),
            "stop": p.get("stop", "flush_low"),
            "rs": p.get("rs", "composite"),
            "risk_scaling": p.get("risk_scaling", "fixed"),
        })
    return out


# ---------------------------------------------------------------------------
# 3.1 Point-in-time S&P 500 membership & data hygiene
# ---------------------------------------------------------------------------

def parse_sp500_changes(csv_path: str | Path) -> list[tuple[pd.Timestamp, tuple[str, ...]]]:
    """Parse fja05680/sp500 historical components CSV into sorted (date, tickers) snapshots.
    Normalises tickers to Yahoo format (e.g. BRK.B -> BRK-B, BF.B -> BF-B)."""
    df = pd.read_csv(csv_path)
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    tick_col = [c for c in df.columns if "ticker" in c.lower()][0]
    df["dt"] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.floor("D")
    df = df.sort_values("dt")
    out: list[tuple[pd.Timestamp, tuple[str, ...]]] = []
    for dt, raw in zip(df["dt"], df[tick_col]):
        if not isinstance(raw, str):
            continue
        syms = sorted({s.strip().upper().replace(".", "-") for s in raw.split(",") if s.strip()})
        if syms:
            if out and out[-1][0] == dt:
                out[-1] = (dt, tuple(syms))
            else:
                out.append((dt, tuple(syms)))
    return out


def pit_membership_matrix(snapshots: list[tuple[pd.Timestamp, tuple[str, ...]]],
                          dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build boolean DataFrame (index=dates, columns=all tickers ever in snapshots during dates).
    Member on day t = last snapshot with date <= t (strictly causal: never uses a snapshot > t)."""
    naive_dates = pd.DatetimeIndex(dates).tz_localize(None).floor("D")
    snap_dates = [s[0] for s in snapshots]
    all_syms = sorted({sym for _, syms in snapshots for sym in syms})
    mat = np.zeros((len(naive_dates), len(all_syms)), dtype=bool)
    sym_idx = {s: i for i, s in enumerate(all_syms)}
    pos = np.searchsorted(snap_dates, naive_dates, side="right") - 1
    last_p = -2
    row_mask = np.zeros(len(all_syms), dtype=bool)
    for r, p in enumerate(pos):
        if p < 0:
            continue
        if p != last_p:
            row_mask.fill(False)
            for s in snapshots[p][1]:
                row_mask[sym_idx[s]] = True
            last_p = p
        mat[r] = row_mask
    return pd.DataFrame(mat, index=dates, columns=all_syms)


def parse_yahoo_chart(raw_json: dict) -> pd.DataFrame | None:
    """Extract split-adjusted OHLCV and unadjusted close/volume from a Yahoo v8 chart JSON.
    Note: Yahoo v8 chart `quote` block provides split-adjusted OHLCV directly, and `events.splits`
    gives the split history so we reconstruct unadjusted close for the $5 price filter and ADV$."""
    try:
        res = raw_json["chart"]["result"][0]
        ts = res.get("timestamp")
        if not ts:
            return None
        q = res["indicators"]["quote"][0]
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_localize(None).floor("D")
        df = pd.DataFrame({
            "open": pd.to_numeric(q.get("open"), errors="coerce"),
            "high": pd.to_numeric(q.get("high"), errors="coerce"),
            "low": pd.to_numeric(q.get("low"), errors="coerce"),
            "close": pd.to_numeric(q.get("close"), errors="coerce"),
            "volume": pd.to_numeric(q.get("volume"), errors="coerce"),
        }, index=idx)
        df = df[~df.index.duplicated(keep="last")]
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
        df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
        if len(df) == 0:
            return None
        # Reconstruct unadjusted close/volume from split events if present.
        # Yahoo chart quote[] is already split-adjusted to current shares:
        # For a split of ratio (num/den) at date t_s, prices before t_s were divided by (num/den)
        # and volumes were multiplied by (num/den).
        splits = (res.get("events") or {}).get("splits") or {}
        cum_split = pd.Series(1.0, index=df.index)
        for sp in splits.values():
            num = float(sp.get("numerator", 1.0))
            den = float(sp.get("denominator", 1.0))
            if den > 0 and num > 0:
                ratio = num / den
                sdt = pd.to_datetime(int(sp["date"]), unit="s", utc=True).tz_localize(None).floor("D")
                cum_split.loc[cum_split.index < sdt] *= ratio
        df["close_unadj"] = df["close"] * cum_split
        df["volume_unadj"] = df["volume"] / cum_split
        return df.sort_index()
    except Exception:
        return None


def detect_ticker_reuse_and_coverage(
    pit_df: pd.DataFrame,
    stock_dfs: dict[str, pd.DataFrame],
    trading_days: pd.DatetimeIndex,
    start_date: str = "2001-01-02",
) -> dict:
    """Section 3.1 hygiene:
    1. Coverage = member-days with valid price data / total member-days, per year.
    2. Ticker reuse / gap check: mark a series as 'possible reuse' if:
       - its first valid price date in `trading_days` is > first index entry date + 5 sessions
         (for tickers whose first entry is >= start_date, or whose data starts after start_date + 5 sessions
          when already a member on start_date), OR
       - while a member of the index (>= start_date), it has a gap of > 10 trading sessions between valid bars
         followed by resumed data (i.e. not simply delisted at the end of its series).
       Exclude flagged tickers and count them.
    3. Identify series that end while still in the index (delistings while in index)."""
    t_idx = pd.DatetimeIndex(trading_days).tz_localize(None).floor("D")
    t_idx = t_idx[t_idx >= pd.Timestamp(start_date)]
    pit = pit_df.reindex(t_idx).fillna(False)
    pos_map = {d: i for i, d in enumerate(t_idx)}

    flagged_reuse: dict[str, str] = {}
    delisted_in_index: dict[str, pd.Timestamp] = {}

    for sym in pit.columns:
        m_dates = pit.index[pit[sym]]
        if len(m_dates) == 0:
            continue
        df = stock_dfs.get(sym)
        if df is None or len(df) == 0:
            continue
        valid_in_win = df.index.intersection(t_idx)
        if len(valid_in_win) == 0:
            continue

        first_member_dt = m_dates[0]
        first_member_pos = pos_map[first_member_dt]
        first_data_dt = df.index[0]
        # Find position in trading_days of first data bar on or after first_member_dt
        valid_from_entry = df.index[df.index >= first_member_dt].intersection(t_idx)
        if len(valid_from_entry) == 0:
            continue
        first_valid_pos = pos_map[valid_from_entry[0]]

        if first_data_dt > first_member_dt and (first_valid_pos - first_member_pos) > 5:
            flagged_reuse[sym] = f"late_start: first_data={first_data_dt.date()} vs entry={first_member_dt.date()} (+{first_valid_pos - first_member_pos} sessions)"
            continue

        # Check gaps > 10 sessions while in the index
        m_positions = np.array([pos_map[d] for d in m_dates])
        v_positions = np.array([pos_map[d] for d in valid_in_win])
        # Check gaps between consecutive valid bars where the ticker was in the index during the gap
        has_gap = False
        if len(v_positions) >= 2:
            diffs = np.diff(v_positions)
            gap_indices = np.where(diffs > 11)[0]
            for gi in gap_indices:
                p_start, p_end = v_positions[gi], v_positions[gi + 1]
                # Was ticker a member during this gap?
                in_idx_during_gap = pit[sym].iloc[p_start + 1:p_end].sum()
                if in_idx_during_gap > 10:
                    flagged_reuse[sym] = f"gap_in_index: {in_idx_during_gap} missing member sessions between {t_idx[p_start].date()} and {t_idx[p_end].date()}"
                    has_gap = True
                    break
        if has_gap:
            continue

        # Check if series ends while ticker is still in the index (at least 5 sessions before end of t_idx)
        last_valid_dt = valid_in_win[-1]
        last_valid_pos = pos_map[last_valid_dt]
        if last_valid_pos < len(t_idx) - 5:
            # If ticker is still a member on last_valid_dt and on the next trading session
            if pit[sym].iloc[last_valid_pos] and pit[sym].iloc[last_valid_pos + 1]:
                delisted_in_index[sym] = last_valid_dt

    # Compute annual coverage (before and after excluding flagged reuse tickers)
    coverage_by_year = {}
    for yr in sorted(set(t_idx.year)):
        ymask = t_idx.year == yr
        ydates = t_idx[ymask]
        total_member_days = int(pit.loc[ydates].values.sum())
        avail_raw = 0
        avail_clean = 0
        for sym in pit.columns:
            m_col = pit.loc[ydates, sym]
            n_m = int(m_col.sum())
            if n_m == 0:
                continue
            df = stock_dfs.get(sym)
            if df is None:
                continue
            has_d = m_col.index[m_col].isin(df.index).sum()
            avail_raw += int(has_d)
            if sym not in flagged_reuse:
                avail_clean += int(has_d)
        coverage_by_year[int(yr)] = {
            "member_days": total_member_days,
            "avail_raw": avail_raw,
            "avail_clean": avail_clean,
            "coverage_raw": round(avail_raw / max(total_member_days, 1), 4),
            "coverage_clean": round(avail_clean / max(total_member_days, 1), 4),
        }

    tot_m = sum(v["member_days"] for v in coverage_by_year.values())
    tot_r = sum(v["avail_raw"] for v in coverage_by_year.values())
    tot_c = sum(v["avail_clean"] for v in coverage_by_year.values())

    return {
        "flagged_reuse": flagged_reuse,
        "n_flagged_reuse": len(flagged_reuse),
        "delisted_in_index": {k: str(v.date()) for k, v in delisted_in_index.items()},
        "n_delisted_in_index": len(delisted_in_index),
        "coverage_by_year": coverage_by_year,
        "overall_coverage_raw": round(tot_r / max(tot_m, 1), 4),
        "overall_coverage_clean": round(tot_c / max(tot_m, 1), 4),
    }


# ---------------------------------------------------------------------------
# 3.2 Indicators (strictly causal: value at t uses only bars <= t)
# ---------------------------------------------------------------------------

def ema_seeded(arr: np.ndarray, n: int) -> np.ndarray:
    """EMA with alpha = 2/(n+1), seeded with the SMA of the first n valid (non-NaN) bars."""
    out = np.full(len(arr), np.nan, dtype=float)
    valid = np.where(np.isfinite(arr))[0]
    if len(valid) < n:
        return out
    alpha = 2.0 / (n + 1.0)
    # Find first contiguous or valid n bars
    first_n_idx = valid[:n]
    seed_pos = first_n_idx[-1]
    val = float(np.mean(arr[first_n_idx]))
    out[seed_pos] = val
    for i in range(seed_pos + 1, len(arr)):
        x = arr[i]
        if np.isfinite(x):
            val = alpha * x + (1.0 - alpha) * val
            out[i] = val
        else:
            out[i] = val
    return out


def wilder_atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder's ATR(14) seeded with SMA of first n True Range bars."""
    m = len(close)
    out = np.full(m, np.nan, dtype=float)
    if m < n + 1:
        return out
    tr = np.empty(m, dtype=float)
    tr[0] = high[0] - low[0]
    prev_c = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_c), np.abs(low[1:] - prev_c)))
    valid = np.where(np.isfinite(tr))[0]
    if len(valid) < n:
        return out
    seed_pos = valid[n - 1]
    val = float(np.mean(tr[valid[:n]]))
    out[seed_pos] = val
    for i in range(seed_pos + 1, m):
        if np.isfinite(tr[i]):
            val = (val * (n - 1) + tr[i]) / n
        out[i] = val
    return out


def compute_avwap_126(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                      lookback: int = 126, return_anchors: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Section 3.2: Anchor a_t = argmax_{i in [t-126, t-1]} high_i (previous swing high).
    AVWAP_t = sum_{i=a_t..t} tp_i * v_i / sum_{i=a_t..t} v_i, with tp = (h+l+c)/3.
    Recomputed each day using only data <= t."""
    m = len(close)
    out = np.full(m, np.nan, dtype=float)
    anchors = np.full(m, -1, dtype=np.int32)
    if m <= lookback:
        return (out, anchors) if return_anchors else out
    tp = (high + low + close) / 3.0
    v = np.where(np.isfinite(volume) & (volume > 0), volume, 0.0)
    tpv = np.where(np.isfinite(tp), tp * v, 0.0)
    cum_v = np.concatenate(([0.0], np.cumsum(v)))
    cum_tpv = np.concatenate(([0.0], np.cumsum(tpv)))
    for t in range(lookback, m):
        win_h = high[t - lookback:t]
        if not np.all(np.isfinite(win_h)):
            valid_rel = np.where(np.isfinite(win_h))[0]
            if len(valid_rel) == 0:
                continue
            a_t = (t - lookback) + int(valid_rel[np.argmax(win_h[valid_rel])])
        else:
            a_t = (t - lookback) + int(np.argmax(win_h))
        anchors[t] = a_t
        denom = cum_v[t + 1] - cum_v[a_t]
        if denom > 0:
            out[t] = (cum_tpv[t + 1] - cum_tpv[a_t]) / denom
        else:
            out[t] = tp[t]
    return (out, anchors) if return_anchors else out


def cpi_deflator_series(cpi_df: pd.DataFrame, dates: pd.DatetimeIndex,
                        base_month: str = "2025-09") -> pd.Series:
    """Section 3.1 & 3.3: CPI-U (FRED CPIAUCSL) with 1-month publication lag (+35 days from month-start
    observation date so e.g. 2025-08-01 CPI is available from 2025-09-05 onwards, strictly >= 1 month lag).
    Returns ratio CPI_{t-1m} / CPI_{2025-09}."""
    df = cpi_df.copy()
    df["dt"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.floor("D")
    df["val"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["dt", "val"]).sort_values("dt")
    # Base value: observation for 2025-08-01 (available in 2025-09) or 2025-09-01
    base_rows = df[df["dt"] <= pd.Timestamp(f"{base_month}-01")]
    base_val = float(base_rows["val"].iloc[-1]) if len(base_rows) else float(df["val"].iloc[-1])
    # Apply 1-month publication lag: month M (dated M-01 in FRED) is published mid-M+1 -> +45 days from M-01
    df["avail_dt"] = df["dt"] + pd.Timedelta(days=45)
    s = pd.Series(df["val"].values / base_val, index=df["avail_dt"])
    s = s[~s.index.duplicated(keep="last")]
    naive_dates = pd.DatetimeIndex(dates).tz_localize(None).floor("D")
    reindexed = s.reindex(naive_dates.union(s.index)).sort_index().ffill().bfill()
    return pd.Series(reindexed.reindex(naive_dates).values, index=dates)


# ---------------------------------------------------------------------------
# Precomputed Panel for fast multi-variant evaluation
# ---------------------------------------------------------------------------

class LukPanel:
    """Holds aligned daily arrays (T x N_stocks) on NYSE trading sessions, plus QQQ, SPY, and CPI."""

    def __init__(
        self,
        trading_days: pd.DatetimeIndex,
        tickers: list[str],
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
        vol_arr: np.ndarray,
        close_unadj_arr: np.ndarray,
        vol_unadj_arr: np.ndarray,
        valid_arr: np.ndarray,
        pit_arr: np.ndarray,
        delist_bar: np.ndarray,
        qqq_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        cpi_ratio: np.ndarray,
        hygiene: dict,
    ):
        self.dates = trading_days
        self.tickers = tickers
        self.T, self.N = open_arr.shape
        self.open = open_arr
        self.high = high_arr
        self.low = low_arr
        self.close = close_arr
        self.vol = vol_arr
        self.close_unadj = close_unadj_arr
        self.vol_unadj = vol_unadj_arr
        self.valid = valid_arr
        self.pit = pit_arr
        self.delist_bar = delist_bar  # int array (N,): index t of last valid bar if delisted while in index, else -1
        self.qqq = qqq_df
        self.spy = spy_df
        self.cpi_ratio = cpi_ratio
        self.hygiene = hygiene

        self._compute_indicators()

    def _compute_indicators(self) -> None:
        T, N = self.T, self.N
        self.ema9 = np.full((T, N), np.nan)
        self.ema21 = np.full((T, N), np.nan)
        self.ema50 = np.full((T, N), np.nan)
        self.adr20 = np.full((T, N), np.nan)
        self.atr14 = np.full((T, N), np.nan)
        self.adv20 = np.full((T, N), np.nan)
        self.avwap = np.full((T, N), np.nan)
        self.avwap_anchor = np.full((T, N), -1, dtype=np.int32)
        self.bars_count = np.zeros((T, N), dtype=np.int32)
        self.max_close_prev10 = np.full((T, N), np.nan)
        self.high_52w = np.full((T, N), np.nan)

        for j in range(N):
            v_idx = np.where(self.valid[:, j])[0]
            if len(v_idx) < 252:
                continue
            c = self.close[v_idx, j]
            h = self.high[v_idx, j]
            l = self.low[v_idx, j]
            vo = self.vol[v_idx, j]
            cu = self.close_unadj[v_idx, j]
            vu = self.vol_unadj[v_idx, j]

            e9 = ema_seeded(c, 9)
            e21 = ema_seeded(c, 21)
            e50 = ema_seeded(c, 50)
            atr = wilder_atr14(h, l, c, 14)
            avw, anc = compute_avwap_126(h, l, c, vo, 126, return_anchors=True)

            s_hl = pd.Series(h / l - 1.0)
            adr = s_hl.rolling(20, min_periods=20).mean().values
            s_dv = pd.Series(cu * vu)
            adv = s_dv.rolling(20, min_periods=20).mean().values

            s_c = pd.Series(c)
            mc10 = s_c.shift(1).rolling(10, min_periods=10).max().values
            h52 = pd.Series(h).rolling(252, min_periods=252).max().values

            self.ema9[v_idx, j] = e9
            self.ema21[v_idx, j] = e21
            self.ema50[v_idx, j] = e50
            self.adr20[v_idx, j] = adr
            self.atr14[v_idx, j] = atr
            self.adv20[v_idx, j] = adv
            self.avwap[v_idx, j] = avw
            self.avwap_anchor[v_idx, j] = np.where(anc >= 0, v_idx[np.maximum(anc, 0)], -1)
            self.bars_count[v_idx, j] = np.arange(1, len(v_idx) + 1, dtype=np.int32)
            self.max_close_prev10[v_idx, j] = mc10
            self.high_52w[v_idx, j] = h52

        # Returns r_k on valid trading sessions per stock
        self.r21 = np.full((T, N), np.nan)
        self.r63 = np.full((T, N), np.nan)
        self.r126 = np.full((T, N), np.nan)
        self.r189 = np.full((T, N), np.nan)
        self.r252 = np.full((T, N), np.nan)
        self.ema21_prev5 = np.full((T, N), np.nan)
        self.close_prev1 = np.full((T, N), np.nan)
        self.ema50_prev1 = np.full((T, N), np.nan)
        self.adr20_prev1 = np.full((T, N), np.nan)

        for j in range(N):
            v_idx = np.where(self.valid[:, j])[0]
            if len(v_idx) < 253:
                continue
            c = self.close[v_idx, j]
            e21 = self.ema21[v_idx, j]
            e50 = self.ema50[v_idx, j]
            adr = self.adr20[v_idx, j]

            self.r21[v_idx[21:], j] = c[21:] / c[:-21] - 1.0
            self.r63[v_idx[63:], j] = c[63:] / c[:-63] - 1.0
            self.r126[v_idx[126:], j] = c[126:] / c[:-126] - 1.0
            self.r189[v_idx[189:], j] = c[189:] / c[:-189] - 1.0
            self.r252[v_idx[252:], j] = c[252:] / c[:-252] - 1.0
            self.ema21_prev5[v_idx[5:], j] = e21[:-5]
            self.close_prev1[v_idx[1:], j] = c[:-1]
            self.ema50_prev1[v_idx[1:], j] = e50[:-1]
            self.adr20_prev1[v_idx[1:], j] = adr[:-1]

        self.rs_c = 0.4 * self.r63 + 0.2 * self.r126 + 0.2 * self.r189 + 0.2 * self.r252
        self.rs_52w = np.where(self.high_52w > 0, self.close / self.high_52w, np.nan)

        # Base eligibility (Section 3.3 items 1-4 + >=252 bars history)
        adv_thresh = (20_000_000.0 * self.cpi_ratio)[:, None]
        valid_prev = np.zeros((T, N), dtype=bool)
        valid_prev[1:] = self.valid[:-1]
        self.base_eligible = (
            self.pit
            & self.valid
            & valid_prev
            & (self.bars_count >= 252)
            & (self.close_unadj >= 5.0)
            & (self.adv20 >= adv_thresh)
            & (self.adr20 >= 0.035)
        )

        # Trend template (Section 3.3 item 5)
        self.lead_struct = (self.ema9 > self.ema21) & (self.ema21 > self.ema50)
        self.trend_ok = (
            self.lead_struct
            & (self.ema21 > self.ema21_prev5)
            & (self.close > self.ema50)
        )

        # Breadth B_t = fraction of base_eligible universe with EMA9 > EMA21 > EMA50 (Section 3.2)
        n_el = self.base_eligible.sum(axis=1)
        n_lead = (self.base_eligible & self.lead_struct).sum(axis=1)
        self.breadth = np.where(n_el > 0, n_lead / np.maximum(n_el, 1), 0.0)
        self.breadth_med252 = pd.Series(self.breadth).rolling(252, min_periods=63).median().values

        # Cross-sectional RS percentile within eligible universe of the day (Section 3.3 item 6)
        self.rs_c_pct = np.full((T, N), np.nan)
        self.rs_52w_pct = np.full((T, N), np.nan)
        for t in range(T):
            el = self.base_eligible[t]
            if el.sum() < 2:
                continue
            vals_c = self.rs_c[t, el]
            fin_c = np.isfinite(vals_c)
            if fin_c.sum() >= 2:
                el_idx = np.where(el)[0][fin_c]
                v = vals_c[fin_c]
                ranks = pd.Series(v).rank(pct=True).values
                self.rs_c_pct[t, el_idx] = ranks
            vals_h = self.rs_52w[t, el]
            fin_h = np.isfinite(vals_h)
            if fin_h.sum() >= 2:
                el_idx = np.where(el)[0][fin_h]
                v = vals_h[fin_h]
                ranks = pd.Series(v).rank(pct=True).values
                self.rs_52w_pct[t, el_idx] = ranks

        # QQQ indicators (Section 3.4 & V7)
        qc = self.qqq["close"].reindex(self.dates).ffill().values
        self.qqq_close = qc
        self.qqq_ema21 = ema_seeded(qc, 21)
        self.qqq_ema50 = ema_seeded(qc, 50)
        self.qqq_filter_ok = (qc > self.qqq_ema50) & (self.qqq_ema21 > self.qqq_ema50)
        qret = pd.Series(qc).pct_change(fill_method=None)
        qvol21 = qret.rolling(21, min_periods=21).std().values
        qvol_med_exp = pd.Series(qvol21).expanding(min_periods=63).median().values
        self.qqq_vol_scale = np.where(
            np.isfinite(qvol21) & (qvol21 > 0) & np.isfinite(qvol_med_exp),
            np.minimum(1.0, qvol_med_exp / qvol21),
            1.0,
        )

        # Pullback & trigger components at bar t (Section 3.4)
        had_pullback = (self.max_close_prev10 / self.close_prev1 - 1.0) >= self.adr20_prev1
        trend_unbroken = self.close_prev1 > self.ema50_prev1
        touch_ema9 = (self.low <= self.ema9 * (1.0 + 0.25 * self.adr20)) & (self.close >= self.ema9)
        touch_ema21 = (self.low <= self.ema21 * (1.0 + 0.25 * self.adr20)) & (self.close >= self.ema21)
        touch_support = touch_ema9 | touch_ema21
        hl_range = self.high - self.low
        flush_green = (
            (self.low < np.minimum(self.open, self.close_prev1))
            & (self.close > self.close_prev1)
            & (hl_range > 0)
            & ((self.close - self.low) / np.where(hl_range > 0, hl_range, 1.0) >= 0.5)
        )
        self.pullback_trigger = had_pullback & trend_unbroken & touch_support & flush_green
        self.avwap_ok = (self.low <= self.avwap * (1.0 + 0.25 * self.adr20)) & (self.close >= self.avwap)

    def slice_until(self, end_date: str) -> LukPanel:
        """Return a panel strictly before end_date (< pd.Timestamp(end_date))."""
        ts = pd.Timestamp(end_date).tz_localize(None)
        mask = self.dates < ts
        k = int(mask.sum())
        # Recompute delist_bar within sliced window
        delist_sub = np.where(self.delist_bar < k, self.delist_bar, -1)
        return LukPanel(
            trading_days=self.dates[:k],
            tickers=self.tickers,
            open_arr=self.open[:k].copy(),
            high_arr=self.high[:k].copy(),
            low_arr=self.low[:k].copy(),
            close_arr=self.close[:k].copy(),
            vol_arr=self.vol[:k].copy(),
            close_unadj_arr=self.close_unadj[:k].copy(),
            vol_unadj_arr=self.vol_unadj[:k].copy(),
            valid_arr=self.valid[:k].copy(),
            pit_arr=self.pit[:k].copy(),
            delist_bar=delist_sub,
            qqq_df=self.qqq.loc[self.qqq.index < ts].copy(),
            spy_df=self.spy.loc[self.spy.index < ts].copy(),
            cpi_ratio=self.cpi_ratio[:k].copy(),
            hygiene=self.hygiene,
        )


def build_panel_from_dfs(
    snapshots: list[tuple[pd.Timestamp, tuple[str, ...]]],
    stock_dfs: dict[str, pd.DataFrame],
    qqq_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    start_date: str = "1999-06-01",
    end_date: str = "2026-10-01",
) -> LukPanel:
    """Assemble aligned LukPanel from raw DataFrames. Includes warmup from 1999-06-01 so
    252-bar indicators are ready on 2001-01-02."""
    t0, t1 = pd.Timestamp(start_date), pd.Timestamp(end_date)
    trading_days = spy_df.index[(spy_df.index >= t0) & (spy_df.index < t1)].tz_localize(None).floor("D")
    trading_days = trading_days.drop_duplicates().sort_values()

    pit_df = pit_membership_matrix(snapshots, trading_days)
    hygiene = detect_ticker_reuse_and_coverage(pit_df, stock_dfs, trading_days, start_date="2001-01-02")
    flagged = set(hygiene["flagged_reuse"].keys())

    # Keep only tickers that have data and are not flagged as reuse
    active_tickers = [s for s in pit_df.columns if s in stock_dfs and s not in flagged]
    T, N = len(trading_days), len(active_tickers)

    open_arr = np.full((T, N), np.nan)
    high_arr = np.full((T, N), np.nan)
    low_arr = np.full((T, N), np.nan)
    close_arr = np.full((T, N), np.nan)
    vol_arr = np.zeros((T, N))
    close_unadj_arr = np.full((T, N), np.nan)
    vol_unadj_arr = np.zeros((T, N))
    valid_arr = np.zeros((T, N), dtype=bool)
    pit_arr = pit_df[active_tickers].values.copy()
    delist_bar = np.full(N, -1, dtype=np.int32)

    delist_map = hygiene["delisted_in_index"]
    pos_map = {d: i for i, d in enumerate(trading_days)}

    for j, sym in enumerate(active_tickers):
        df = stock_dfs[sym]
        sub = df.reindex(trading_days)
        v = sub["close"].notna().values
        valid_arr[:, j] = v
        # Forward-fill prices only for mark-to-market across single missing days; valid_arr tracks actual bars
        sub_ff = sub.ffill()
        open_arr[:, j] = sub_ff["open"].values
        high_arr[:, j] = sub_ff["high"].values
        low_arr[:, j] = sub_ff["low"].values
        close_arr[:, j] = sub_ff["close"].values
        vol_arr[:, j] = sub["volume"].fillna(0.0).values
        close_unadj_arr[:, j] = sub_ff["close_unadj"].values
        vol_unadj_arr[:, j] = sub["volume_unadj"].fillna(0.0).values
        if sym in delist_map:
            dt = pd.Timestamp(delist_map[sym])
            if dt in pos_map:
                delist_bar[j] = pos_map[dt]

    cpi_ratio = cpi_deflator_series(cpi_df, trading_days).values
    return LukPanel(
        trading_days=trading_days,
        tickers=active_tickers,
        open_arr=open_arr,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
        vol_arr=vol_arr,
        close_unadj_arr=close_unadj_arr,
        vol_unadj_arr=vol_unadj_arr,
        valid_arr=valid_arr,
        pit_arr=pit_arr,
        delist_bar=delist_bar,
        qqq_df=qqq_df,
        spy_df=spy_df,
        cpi_ratio=cpi_ratio,
        hygiene=hygiene,
    )


# ---------------------------------------------------------------------------
# 3.5 - 3.7 Cost model & trade simulation engine
# ---------------------------------------------------------------------------

def one_side_slip_bps(adv_usd: float, price_unadj: float, is_stop: bool) -> float:
    """Section 3.7 tiered slippage:
    >= 500M$: 3 bps; 100-500M$: 5 bps; 20-100M$: 10 bps (<20M$: 10 bps).
    +10 bps if unadjusted price < $10.
    Stop exits pay 2x slippage."""
    if adv_usd >= 500_000_000.0:
        bps = 3.0
    elif adv_usd >= 100_000_000.0:
        bps = 5.0
    else:
        bps = 10.0
    if price_unadj < 10.0:
        bps += 10.0
    if is_stop:
        bps *= 2.0
    return bps * 1e-4


def commission_usd(shares: int) -> float:
    """Section 3.7: $0.005/share with $1.00 minimum per order."""
    if shares <= 0:
        return 0.0
    return max(1.0, 0.005 * shares)


@dataclass
class Position:
    sym_idx: int
    ticker: str
    signal_bar: int
    entry_bar: int
    entry_price: float          # raw open price at entry (before slippage)
    fill_entry_price: float     # actual entry fill price including slippage
    initial_stop: float
    stop_price: float
    r_per_share: float          # entry_price - initial_stop (> 0)
    initial_shares: int
    shares: int
    partial_taken: bool = False
    trailing_active: bool = False
    pending_trail_exit: bool = False
    entry_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    realized_gross_pnl: float = 0.0
    realized_net_pnl: float = 0.0
    exit_reason: str = ""
    adv_at_entry: float = 0.0
    adr_at_entry: float = 0.0
    turnover_at_entry: float = 0.0


def signal_mask_for_config(panel: LukPanel, cfg: LukConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean (T, N) signal matrix and (T, N) ranking score matrix for `cfg`."""
    el = panel.base_eligible & panel.trend_ok
    if cfg.rs == "high52w":
        rs_ok = (panel.rs_52w >= 0.90) & (panel.rs_52w_pct >= 0.80)
        rank_score = panel.rs_52w_pct
    else:
        rs_prev_max = np.maximum(panel.r21, np.maximum(panel.r63, panel.r126))
        rs_ok = (rs_prev_max >= 0.30) & (panel.rs_c_pct >= 0.80)
        rank_score = panel.rs_c_pct

    sig = el & rs_ok & panel.pullback_trigger
    if cfg.avwap_confluence:
        sig = sig & panel.avwap_ok

    if cfg.market_filter == "qqq":
        mkt_ok = panel.qqq_filter_ok
    elif cfg.market_filter == "breadth":
        mkt_ok = np.isfinite(panel.breadth_med252) & (panel.breadth >= panel.breadth_med252)
    else:
        mkt_ok = np.ones(panel.T, dtype=bool)

    sig = sig & mkt_ok[:, None]
    # Only generate signals from 2001-01-02 onwards
    start_mask = (panel.dates >= pd.Timestamp("2001-01-02"))[:, None]
    sig = sig & start_mask
    return sig, rank_score


def simulate_luk(
    panel: LukPanel,
    cfg: LukConfig,
    custom_signals: np.ndarray | None = None,
    custom_rank: np.ndarray | None = None,
) -> BTResult:
    """Event-driven daily bar simulation of Martin Luk pullback strategy (Sections 3.4 - 3.7).
    Strictly causal: signals at t queue orders for t + 1 + cfg.extra_delay."""
    T, N = panel.T, panel.N
    if custom_signals is not None:
        sig_mat = custom_signals
        rank_mat = custom_rank if custom_rank is not None else panel.rs_c_pct
    else:
        sig_mat, rank_mat = signal_mask_for_config(panel, cfg)

    # Precompute stop candidates at signal bar t
    if cfg.stop == "low_minus_half_atr":
        raw_stop_mat = panel.low - 0.5 * panel.atr14
    else:
        raw_stop_mat = panel.low * (1.0 - 0.001)

    cash = float(cfg.target_capital)
    positions: dict[int, Position] = {}
    # Queued signals: map from execution bar_idx -> list of (sym_idx, signal_bar_t)
    queued: dict[int, list[tuple[int, int]]] = {}

    daily_gross = np.zeros(T)
    daily_cost = np.zeros(T)
    daily_net = np.zeros(T)
    daily_exposure = np.zeros(T)
    daily_turnover = np.zeros(T)
    daily_adv_part_max = np.zeros(T)
    daily_overnight_ret = np.zeros(T)
    daily_intraday_ret = np.zeros(T)

    completed_trades: list[dict] = []
    # Track last mark price of each held position (at close of t-1, or entry_price on entry day)
    prev_mark: dict[int, float] = {}

    start_bar = int(np.searchsorted(panel.dates, pd.Timestamp("2001-01-02")))

    for t in range(start_bar, T):
        eq_start = cash + sum(pos.shares * prev_mark[j] for j, pos in positions.items())
        if eq_start <= 0:
            break

        day_gross_usd = 0.0
        day_cost_usd = 0.0
        day_traded_usd = 0.0
        day_overnight_usd = 0.0
        day_intraday_usd = 0.0
        max_adv_part = 0.0

        # ------------------------------------------------------------------
        # 1. Process existing positions (entered on bars < t)
        # ------------------------------------------------------------------
        for j in list(positions.keys()):
            pos = positions[j]
            p_prev = prev_mark[j]
            o_t = panel.open[t, j]
            h_t = panel.high[t, j]
            l_t = panel.low[t, j]
            c_t = panel.close[t, j]
            cu_t = panel.close_unadj[t, j] if np.isfinite(panel.close_unadj[t, j]) else p_prev
            adv_t = panel.adv20[t, j] if np.isfinite(panel.adv20[t, j]) and panel.adv20[t, j] > 0 else pos.adv_at_entry

            # Check delisting on bar t (series ends while in index) — takes priority per Section 3.1.3
            if panel.delist_bar[j] == t:
                sh = pos.shares
                base_px = c_t if np.isfinite(c_t) and c_t > 0 else p_prev
                mult = 0.70 if cfg.delist_mode == "pessimistic" else 1.0
                exit_px = base_px * mult
                o_mark = o_t if np.isfinite(o_t) and o_t > 0 else p_prev
                slip = one_side_slip_bps(adv_t, cu_t, is_stop=(cfg.delist_mode == "pessimistic"))
                cost_u = (sh * exit_px * slip + commission_usd(sh)) * cfg.cost_mult
                gross_u = sh * (exit_px - p_prev)
                day_overnight_usd += sh * (o_mark - p_prev)
                day_intraday_usd += sh * (exit_px - o_mark)
                day_gross_usd += gross_u
                day_cost_usd += cost_u
                day_traded_usd += sh * exit_px
                cash += sh * exit_px - cost_u
                pos.total_cost_usd += cost_u
                pos.realized_gross_pnl += sh * (exit_px - pos.entry_price)
                pos.realized_net_pnl = pos.realized_gross_pnl - pos.total_cost_usd
                completed_trades.append(_trade_dict(pos, panel.dates[t], exit_px, f"delist_{cfg.delist_mode}", cfg))
                del positions[j]
                del prev_mark[j]
                continue

            # Guard against missing bar in Yahoo data while position is open
            if not panel.valid[t, j] or not (np.isfinite(o_t) and np.isfinite(h_t) and np.isfinite(l_t) and np.isfinite(c_t)):
                # If no valid bars remain in the future for this ticker, close position now
                if not np.any(panel.valid[t:, j]):
                    sh = pos.shares
                    mult = 0.70 if cfg.delist_mode == "pessimistic" else 1.0
                    exit_px = p_prev * mult
                    slip = one_side_slip_bps(adv_t, cu_t, is_stop=(cfg.delist_mode == "pessimistic"))
                    cost_u = (sh * exit_px * slip + commission_usd(sh)) * cfg.cost_mult
                    gross_u = sh * (exit_px - p_prev)
                    day_intraday_usd += gross_u
                    day_gross_usd += gross_u
                    day_cost_usd += cost_u
                    day_traded_usd += sh * exit_px
                    cash += sh * exit_px - cost_u
                    pos.total_cost_usd += cost_u
                    pos.realized_gross_pnl += sh * (exit_px - pos.entry_price)
                    pos.realized_net_pnl = pos.realized_gross_pnl - pos.total_cost_usd
                    completed_trades.append(_trade_dict(pos, panel.dates[t], exit_px, f"series_end_{cfg.delist_mode}", cfg))
                    del positions[j]
                    del prev_mark[j]
                continue

            # Check if trailing exit was triggered at close of t-1 -> exit at open_t
            if pos.pending_trail_exit:
                sh = pos.shares
                slip = one_side_slip_bps(adv_t, cu_t, is_stop=False)
                cost_u = (sh * o_t * slip + commission_usd(sh)) * cfg.cost_mult
                gross_u = sh * (o_t - p_prev)
                day_overnight_usd += gross_u
                day_gross_usd += gross_u
                day_cost_usd += cost_u
                day_traded_usd += sh * o_t
                max_adv_part = max(max_adv_part, (sh * o_t) / max(adv_t, 1.0))
                cash += sh * o_t - cost_u
                pos.total_cost_usd += cost_u
                pos.realized_gross_pnl += sh * (o_t - pos.entry_price)
                pos.realized_net_pnl = pos.realized_gross_pnl - pos.total_cost_usd
                completed_trades.append(_trade_dict(pos, panel.dates[t], o_t, "trail_ema", cfg))
                del positions[j]
                del prev_mark[j]
                continue

            # Check hard stop vs partial target on bar t (Section 3.6 items 1 & 2)
            target_px = pos.entry_price + 3.0 * pos.r_per_share
            can_partial = (not pos.partial_taken) and (pos.initial_shares >= 5)
            hit_stop = cfg.hard_stop_enabled and (l_t <= pos.stop_price or o_t <= pos.stop_price)
            hit_target = can_partial and (h_t >= target_px)

            # Section 3.6 rule 2: "If both stop and L are touched in the same bar: stop goes first, unless open >= L."
            if hit_target and o_t >= target_px and not (cfg.hard_stop_enabled and o_t <= pos.stop_price):
                # Partial fills at open_t >= L first
                part_sh = int(math.floor(0.20 * pos.initial_shares))
                if 0 < part_sh < pos.shares:
                    fill_px = max(o_t, target_px)
                    slip = one_side_slip_bps(adv_t, cu_t, is_stop=False)
                    cost_u = (part_sh * fill_px * slip + commission_usd(part_sh)) * cfg.cost_mult
                    gross_u = part_sh * (fill_px - p_prev)
                    day_overnight_usd += part_sh * (o_t - p_prev)
                    day_intraday_usd += part_sh * (fill_px - o_t)
                    day_gross_usd += gross_u
                    day_cost_usd += cost_u
                    day_traded_usd += part_sh * fill_px
                    max_adv_part = max(max_adv_part, (part_sh * fill_px) / max(adv_t, 1.0))
                    cash += part_sh * fill_px - cost_u
                    pos.shares -= part_sh
                    pos.partial_taken = True
                    pos.total_cost_usd += cost_u
                    pos.realized_gross_pnl += part_sh * (fill_px - pos.entry_price)
                hit_target = False  # already handled

            if hit_stop:
                sh = pos.shares
                if o_t <= pos.stop_price:
                    exit_px = o_t
                    reason = "stop_gap_open"
                    day_overnight_usd += sh * (exit_px - p_prev)
                else:
                    exit_px = pos.stop_price
                    reason = "stop_intraday"
                    day_overnight_usd += sh * (o_t - p_prev)
                    day_intraday_usd += sh * (exit_px - o_t)
                slip = one_side_slip_bps(adv_t, cu_t, is_stop=True)
                cost_u = (sh * exit_px * slip + commission_usd(sh)) * cfg.cost_mult
                gross_u = sh * (exit_px - p_prev)
                day_gross_usd += gross_u
                day_cost_usd += cost_u
                day_traded_usd += sh * exit_px
                max_adv_part = max(max_adv_part, (sh * exit_px) / max(adv_t, 1.0))
                cash += sh * exit_px - cost_u
                pos.total_cost_usd += cost_u
                pos.realized_gross_pnl += sh * (exit_px - pos.entry_price)
                pos.realized_net_pnl = pos.realized_gross_pnl - pos.total_cost_usd
                completed_trades.append(_trade_dict(pos, panel.dates[t], exit_px, reason, cfg))
                del positions[j]
                del prev_mark[j]
                continue

            if hit_target:
                part_sh = int(math.floor(0.20 * pos.initial_shares))
                if 0 < part_sh < pos.shares:
                    fill_px = max(o_t, target_px)
                    slip = one_side_slip_bps(adv_t, cu_t, is_stop=False)
                    cost_u = (part_sh * fill_px * slip + commission_usd(part_sh)) * cfg.cost_mult
                    gross_u = part_sh * (fill_px - p_prev)
                    day_overnight_usd += part_sh * (o_t - p_prev)
                    day_intraday_usd += part_sh * (fill_px - o_t)
                    day_gross_usd += gross_u
                    day_cost_usd += cost_u
                    day_traded_usd += part_sh * fill_px
                    max_adv_part = max(max_adv_part, (part_sh * fill_px) / max(adv_t, 1.0))
                    cash += part_sh * fill_px - cost_u
                    pos.shares -= part_sh
                    pos.partial_taken = True
                    pos.total_cost_usd += cost_u
                    pos.realized_gross_pnl += part_sh * (fill_px - pos.entry_price)

            # Still open at close of t: mark to c_t
            sh = pos.shares
            day_overnight_usd += sh * (o_t - p_prev)
            day_intraday_usd += sh * (c_t - o_t)
            day_gross_usd += sh * (c_t - p_prev)
            prev_mark[j] = c_t
            _update_end_of_day_state(pos, panel, t, cfg)

        # ------------------------------------------------------------------
        # 2. Execute queued new entries at open_t (Section 3.5)
        # ------------------------------------------------------------------
        candidates = queued.pop(t, [])
        if candidates and len(positions) < 20:
            # Sort candidates: RS_pct desc, stop% asc, ticker asc
            scored_cands = []
            for j, sig_t in candidates:
                if j in positions:
                    continue
                if not panel.valid[t, j] or not panel.pit[t, j]:
                    continue
                o_t = panel.open[t, j]
                st_px = raw_stop_mat[sig_t, j]
                if not (np.isfinite(o_t) and np.isfinite(st_px) and o_t > st_px):
                    continue
                adr_sig = panel.adr20[sig_t, j]
                stop_pct = (o_t - st_px) / o_t
                # Cap check (Section 3.5 & V5):
                # V0..V4, V6, V7: min(0.5 * ADR20, 5%); V5: min(1.0 * ADR20, 100%)
                cap = min(cfg.stop_cap_adr * adr_sig, 0.05 if cfg.stop == "flush_low" else 1.0)
                if stop_pct > cap or stop_pct <= 1e-5:
                    continue
                rk = float(rank_mat[sig_t, j]) if np.isfinite(rank_mat[sig_t, j]) else 0.0
                scored_cands.append((-rk, stop_pct, panel.tickers[j], j, sig_t, o_t, st_px))

            scored_cands.sort(key=lambda x: (x[0], x[1], x[2]))
            new_entries_today = 0

            # Equity marked at signal bar t_sig (approximated by eq_start at start of bar t)
            for _, stop_pct, sym, j, sig_t, entry_px, st_px in scored_cands:
                if new_entries_today >= 4:
                    break
                r_sh = entry_px - st_px
                risk_frac = cfg.base_risk_frac
                if cfg.risk_scaling == "qqq_vol":
                    risk_frac *= float(panel.qqq_vol_scale[sig_t])

                # Desired shares from risk budget
                raw_shares = int(math.floor((risk_frac * eq_start) / r_sh))
                if raw_shares <= 0:
                    continue

                # Apply caps in exact order of Section 3.5:
                cu_sig = panel.close_unadj[sig_t, j]
                adv_sig = panel.adv20[sig_t, j]
                cpi_sig = panel.cpi_ratio[sig_t]
                # 1. Notional <= 25% of E (or <= 20% if price < $10 or ADV$20 < $50M * CPI ratio)
                max_pos_frac = 0.20 if (cu_sig < 10.0 or adv_sig < 50_000_000.0 * cpi_sig) else 0.25
                cap1_sh = int(math.floor((max_pos_frac * eq_start) / entry_px))

                # 2. Notional <= 1% of ADV$20 (scaled by E / target_capital so $100k reference holds)
                scale_e = eq_start / cfg.target_capital
                cap2_sh = int(math.floor((0.01 * adv_sig / max(scale_e, 1e-9)) / entry_px))

                # 3. Gross exposure <= 100% of E (cash account)
                curr_notional = sum(p.shares * prev_mark[k] for k, p in positions.items())
                rem_cash_notional = max(0.0, min(cash, eq_start - curr_notional))
                # Leave room for entry slippage & commission
                slip_in = one_side_slip_bps(adv_sig, cu_sig, is_stop=False)
                cap3_sh = int(math.floor((rem_cash_notional - 2.0 * cfg.cost_mult) / (entry_px * (1.0 + (slip_in + 0.001) * cfg.cost_mult))))

                # 4. Aggregate open risk sum(max(0, price - stop) * shares) <= 3% of E
                curr_open_risk = sum(max(0.0, prev_mark[k] - p.stop_price) * p.shares for k, p in positions.items())
                rem_risk = max(0.0, 0.03 * eq_start - curr_open_risk)
                cap4_sh = int(math.floor(rem_risk / r_sh))

                sh = min(raw_shares, cap1_sh, cap2_sh, cap3_sh, cap4_sh)
                if sh <= 0:
                    continue

                cost_in = (sh * entry_px * slip_in + commission_usd(sh)) * cfg.cost_mult
                if sh * entry_px + cost_in > cash:
                    sh = int(math.floor((cash - 2.0 * cfg.cost_mult) / (entry_px * (1.0 + slip_in * cfg.cost_mult))))
                    if sh <= 0:
                        continue
                    cost_in = (sh * entry_px * slip_in + commission_usd(sh)) * cfg.cost_mult

                cash -= sh * entry_px + cost_in
                day_cost_usd += cost_in
                day_traded_usd += sh * entry_px
                max_adv_part = max(max_adv_part, (sh * entry_px * (cfg.target_capital / eq_start)) / max(adv_sig, 1.0))

                pos = Position(
                    sym_idx=j,
                    ticker=sym,
                    signal_bar=sig_t,
                    entry_bar=t,
                    entry_price=entry_px,
                    fill_entry_price=entry_px * (1.0 + slip_in * cfg.cost_mult),
                    initial_stop=st_px,
                    stop_price=st_px,
                    r_per_share=r_sh,
                    initial_shares=sh,
                    shares=sh,
                    entry_cost_usd=cost_in,
                    total_cost_usd=cost_in,
                    adv_at_entry=adv_sig,
                    adr_at_entry=panel.adr20[sig_t, j],
                    turnover_at_entry=float(panel.vol[sig_t, j] * panel.close[sig_t, j] / max(adv_sig, 1.0)),
                )
                positions[j] = pos
                prev_mark[j] = entry_px
                new_entries_today += 1

                # Now evaluate bar t intraday on the newly entered position (Section 3.6: hard stop active from entry bar t+1)
                h_t = panel.high[t, j]
                l_t = panel.low[t, j]
                c_t = panel.close[t, j]
                cu_t = panel.close_unadj[t, j]

                target_px = pos.entry_price + 3.0 * pos.r_per_share
                can_partial = (not pos.partial_taken) and (pos.initial_shares >= 5)
                hit_stop = cfg.hard_stop_enabled and (l_t <= pos.stop_price)
                hit_target = can_partial and (h_t >= target_px)

                # Stop goes first if both touched on entry bar (since open == entry_price < target_px)
                if hit_stop:
                    exit_px = pos.stop_price
                    slip_out = one_side_slip_bps(adv_sig, cu_t, is_stop=True)
                    cost_out = (sh * exit_px * slip_out + commission_usd(sh)) * cfg.cost_mult
                    gross_u = sh * (exit_px - entry_px)
                    day_intraday_usd += gross_u
                    day_gross_usd += gross_u
                    day_cost_usd += cost_out
                    day_traded_usd += sh * exit_px
                    cash += sh * exit_px - cost_out
                    pos.total_cost_usd += cost_out
                    pos.realized_gross_pnl += gross_u
                    pos.realized_net_pnl = pos.realized_gross_pnl - pos.total_cost_usd
                    completed_trades.append(_trade_dict(pos, panel.dates[t], exit_px, "stop_entry_bar", cfg))
                    del positions[j]
                    del prev_mark[j]
                    continue

                if hit_target:
                    part_sh = int(math.floor(0.20 * pos.initial_shares))
                    if 0 < part_sh < pos.shares:
                        fill_px = target_px
                        slip_out = one_side_slip_bps(adv_sig, cu_t, is_stop=False)
                        cost_out = (part_sh * fill_px * slip_out + commission_usd(part_sh)) * cfg.cost_mult
                        gross_u = part_sh * (fill_px - entry_px)
                        day_intraday_usd += gross_u
                        day_gross_usd += gross_u
                        day_cost_usd += cost_out
                        day_traded_usd += part_sh * fill_px
                        cash += part_sh * fill_px - cost_out
                        pos.shares -= part_sh
                        pos.partial_taken = True
                        pos.total_cost_usd += cost_out
                        pos.realized_gross_pnl += gross_u

                # Mark remaining shares to close c_t
                rem_sh = pos.shares
                day_intraday_usd += rem_sh * (c_t - entry_px)
                day_gross_usd += rem_sh * (c_t - entry_px)
                prev_mark[j] = c_t
                _update_end_of_day_state(pos, panel, t, cfg)

        # ------------------------------------------------------------------
        # 3. Record end-of-day portfolio metrics & queue signals at close of t
        # ------------------------------------------------------------------
        eod_notional = sum(pos.shares * prev_mark[j] for j, pos in positions.items())
        daily_gross[t] = day_gross_usd / eq_start
        daily_cost[t] = day_cost_usd / eq_start
        daily_net[t] = (day_gross_usd - day_cost_usd) / eq_start
        daily_exposure[t] = eod_notional / eq_start
        daily_turnover[t] = day_traded_usd / eq_start
        daily_adv_part_max[t] = max_adv_part
        daily_overnight_ret[t] = day_overnight_usd / eq_start
        daily_intraday_ret[t] = day_intraday_usd / eq_start

        # Queue signals generated at close of bar t for bar t + 1 + cfg.extra_delay
        exec_bar = t + 1 + cfg.extra_delay
        if exec_bar < T:
            sig_cols = np.where(sig_mat[t])[0]
            if len(sig_cols) > 0:
                queued.setdefault(exec_bar, []).extend((int(j), t) for j in sig_cols)

    # Reindex to calendar days (weekend = 0 return) starting 2001-01-02 so sqrt(365) == sqrt(252)
    win_dates = panel.dates[start_bar:]
    bar_ts = pd.DataFrame({
        "gross": daily_gross[start_bar:],
        "cost": daily_cost[start_bar:],
        "funding": 0.0,
        "net": daily_net[start_bar:],
        "exposure": daily_exposure[start_bar:],
        "turnover": daily_turnover[start_bar:],
        "adv_part": daily_adv_part_max[start_bar:],
        "overnight": daily_overnight_ret[start_bar:],
        "intraday": daily_intraday_ret[start_bar:],
    }, index=pd.DatetimeIndex(win_dates, tz="UTC"))

    cal_idx = pd.date_range(bar_ts.index[0], bar_ts.index[-1], freq="D", tz="UTC")
    bar_cal = bar_ts.reindex(cal_idx).fillna(0.0)
    bar_cal["exposure"] = bar_ts["exposure"].reindex(cal_idx).ffill().fillna(0.0)

    daily_cal = bar_cal["net"].copy()
    tr_df = pd.DataFrame(completed_trades)
    return BTResult(
        bar=bar_cal,
        daily=daily_cal,
        weights=pd.DataFrame(index=cal_idx),
        meta={
            "trades": tr_df,
            "trading_bar": bar_ts,
            "config": cfg,
        },
    )


def _update_end_of_day_state(pos: Position, panel: LukPanel, t: int, cfg: LukConfig) -> None:
    """Section 3.6 rules 3 & 4 at close of bar t (effective from t+1):
    3. Breakeven: when EMA9_t > entry, stop moves to max(stop, entry). Stops never decrease.
    4. Trailing: activates after breakeven (EMA9_t > entry) or a close >= entry + 1R.
       Once active, if close_t < EMA_trail_t -> sell remainder at open_{t+1}."""
    j = pos.sym_idx
    c_t = panel.close[t, j]
    e9_t = panel.ema9[t, j]
    e21_t = panel.ema21[t, j]
    trail_ema = e21_t if cfg.trail == "ema21" else e9_t

    if np.isfinite(e9_t) and e9_t > pos.entry_price:
        pos.stop_price = max(pos.stop_price, pos.entry_price)
        pos.trailing_active = True
    if np.isfinite(c_t) and c_t >= pos.entry_price + pos.r_per_share:
        pos.trailing_active = True

    if pos.trailing_active and np.isfinite(c_t) and np.isfinite(trail_ema) and c_t < trail_ema:
        pos.pending_trail_exit = True


def _trade_dict(pos: Position, exit_dt: pd.Timestamp, exit_px: float, reason: str, cfg: LukConfig) -> dict:
    init_risk_usd = pos.initial_shares * pos.r_per_share
    r_mult_net = pos.realized_net_pnl / init_risk_usd if init_risk_usd > 0 else 0.0
    r_mult_gross = pos.realized_gross_pnl / init_risk_usd if init_risk_usd > 0 else 0.0
    cost_r = pos.total_cost_usd / init_risk_usd if init_risk_usd > 0 else 0.0
    return {
        "ticker": pos.ticker,
        "signal_bar": int(pos.signal_bar),
        "entry_bar": int(pos.entry_bar),
        "exit_date": pd.Timestamp(exit_dt),
        "entry_price": float(pos.entry_price),
        "exit_price": float(exit_px),
        "initial_stop": float(pos.initial_stop),
        "initial_shares": int(pos.initial_shares),
        "gross_pnl_usd": float(pos.realized_gross_pnl),
        "net_pnl_usd": float(pos.realized_net_pnl),
        "cost_usd": float(pos.total_cost_usd),
        "r_net": float(r_mult_net),
        "r_gross": float(r_mult_gross),
        "cost_r": float(cost_r),
        "partial_taken": bool(pos.partial_taken),
        "exit_reason": reason,
        "adv_at_entry": float(pos.adv_at_entry),
        "adr_at_entry": float(pos.adr_at_entry),
        "turnover_at_entry": float(pos.turnover_at_entry),
    }


# ---------------------------------------------------------------------------
# 3.9 Trade-level convex metrics & required diagnostics (Section 3.8 a-f)
# ---------------------------------------------------------------------------

def luk_trade_metrics(tr_df: pd.DataFrame, daily: pd.Series, start_dt=None, end_dt=None) -> dict:
    """Compute Section 3.9 trade metrics on trades exiting in [start_dt, end_dt)."""
    if len(tr_df) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "payoff": 0.0, "profit_factor": 0.0,
            "expectancy_r": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0,
            "worst_trade_r": 0.0, "pct_worse_than_1_5r": 0.0, "mean_cost_r": 0.0,
            "top5pct_profit_share": None, "trade_skew": 0.0, "daily_skew": float(daily.skew()) if len(daily) > 3 else 0.0,
            "longest_losing_streak_trades": 0,
        }
    df = tr_df.copy()
    if start_dt is not None:
        df = df[df["exit_date"] >= pd.Timestamp(start_dt).tz_localize(None)]
    if end_dt is not None:
        df = df[df["exit_date"] < pd.Timestamp(end_dt).tz_localize(None)]
    if len(df) == 0:
        return luk_trade_metrics(pd.DataFrame(), daily)

    df = df.sort_values("exit_date")
    r = df["r_net"].values
    pnl = df["net_pnl_usd"].values
    wins = r[r > 0]
    losses = r[r <= 0]
    w_pnl = pnl[pnl > 0]
    l_pnl = pnl[pnl <= 0]

    k5 = max(1, int(math.ceil(0.05 * len(pnl))))
    top5_sum = float(np.sort(pnl)[::-1][:k5].sum())
    tot_pnl = float(pnl.sum())

    # Longest losing streak in trades
    best_s = cur_s = 0
    for x in r:
        if x <= 0:
            cur_s += 1
            best_s = max(best_s, cur_s)
        else:
            cur_s = 0

    return {
        "n_trades": int(len(df)),
        "win_rate": float(len(wins) / len(r)),
        "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else 0.0,
        "profit_factor": float(w_pnl.sum() / abs(l_pnl.sum())) if len(l_pnl) and l_pnl.sum() != 0 else 0.0,
        "expectancy_r": float(r.mean()),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "worst_trade_r": float(r.min()),
        "pct_worse_than_1_5r": float((r < -1.5).mean()),
        "mean_cost_r": float(df["cost_r"].mean()),
        "top5pct_profit_share": float(top5_sum / tot_pnl) if tot_pnl > 0 else None,
        "trade_skew": float(pd.Series(r).skew()) if len(r) > 3 else 0.0,
        "daily_skew": float(daily.skew()) if len(daily) > 3 else 0.0,
        "longest_losing_streak_trades": int(best_s),
    }


def run_momentum_without_timing(panel: LukPanel, cost_mult: float = 1.0) -> BTResult:
    """Diagnostic (a): Momentum without timing — top decile of C within base_eligible universe,
    monthly rebalance (every 21 trading sessions), equal-weighted, next-open fills (+1 bar delay),
    same tiered slippage + commission costs."""
    T, N = panel.T, panel.N
    start_bar = int(np.searchsorted(panel.dates, pd.Timestamp("2001-01-02")))
    W = np.zeros((T, N))
    last_w = np.zeros(N)
    for t in range(start_bar, T):
        if (t - start_bar) % 21 == 0:
            last_w.fill(0.0)
            el = panel.base_eligible[t] & np.isfinite(panel.rs_c[t])
            el_idx = np.where(el)[0]
            if len(el_idx) >= 10:
                scores = panel.rs_c[t, el_idx]
                k = max(1, len(el_idx) // 10)
                top_idx = el_idx[np.argsort(scores)[-k:]]
                last_w[top_idx] = 1.0 / k
        W[t] = last_w
    return _eval_weight_panel(panel, W, start_bar, cost_mult=cost_mult)


def run_eligible_equal_weight(panel: LukPanel, cost_mult: float = 1.0) -> BTResult:
    """Diagnostic (b): Eligible universe equal-weighted (base_eligible + trend template),
    monthly rebalance, without pullback trigger."""
    T, N = panel.T, panel.N
    start_bar = int(np.searchsorted(panel.dates, pd.Timestamp("2001-01-02")))
    W = np.zeros((T, N))
    last_w = np.zeros(N)
    for t in range(start_bar, T):
        if (t - start_bar) % 21 == 0:
            last_w.fill(0.0)
            el = panel.base_eligible[t] & panel.trend_ok[t]
            el_idx = np.where(el)[0]
            if len(el_idx) > 0:
                last_w[el_idx] = 1.0 / len(el_idx)
        W[t] = last_w
    return _eval_weight_panel(panel, W, start_bar, cost_mult=cost_mult)


def _eval_weight_panel(panel: LukPanel, W: np.ndarray, start_bar: int, cost_mult: float = 1.0) -> BTResult:
    """Evaluate target weights W[t, j] filled at open_{t+1} (open-to-open returns + tiered slippage)."""
    T, N = panel.T, panel.N
    held = np.zeros((T, N))
    held[start_bar + 1:] = W[start_bar:-1]
    # Open-to-open return on bar t (from open_t to open_{t+1})
    o_ret = np.zeros((T, N))
    valid_oo = (
        panel.valid[:-1]
        & panel.valid[1:]
        & np.isfinite(panel.open[:-1])
        & np.isfinite(panel.open[1:])
        & (panel.open[:-1] > 0)
        & (panel.open[1:] > 0)
    )
    ratio = np.ones((T - 1, N))
    np.divide(panel.open[1:], panel.open[:-1], out=ratio, where=valid_oo)
    o_ret[:-1] = ratio - 1.0
    # Apply delisting penalty (-30%) on delist_bar
    for j in range(N):
        db = panel.delist_bar[j]
        if db >= start_bar:
            o_ret[db, j] = -0.30

    gross = (held * o_ret).sum(axis=1)
    trade = np.abs(np.diff(held, axis=0, prepend=0.0))
    # Tiered slippage + approx commission (5 bps on $100k across positions)
    adv_clean = np.nan_to_num(panel.adv20, nan=0.0)
    cu_clean = np.nan_to_num(panel.close_unadj, nan=100.0)
    slip = np.where(adv_clean >= 5e8, 0.0003, np.where(adv_clean >= 1e8, 0.0005, 0.0010))
    slip = np.where(cu_clean < 10.0, slip + 0.0010, slip)
    cost = (trade * (slip + 0.0002) * cost_mult).sum(axis=1)
    net = gross - cost

    win_dates = panel.dates[start_bar:]
    bar_ts = pd.DataFrame({
        "gross": gross[start_bar:],
        "cost": cost[start_bar:],
        "funding": 0.0,
        "net": net[start_bar:],
        "exposure": held[start_bar:].sum(axis=1),
        "turnover": trade[start_bar:].sum(axis=1),
    }, index=pd.DatetimeIndex(win_dates, tz="UTC"))
    cal_idx = pd.date_range(bar_ts.index[0], bar_ts.index[-1], freq="D", tz="UTC")
    bar_cal = bar_ts.reindex(cal_idx).fillna(0.0)
    return BTResult(bar=bar_cal, daily=bar_cal["net"].copy(), weights=pd.DataFrame(index=cal_idx))


def run_random_entry_placebo(panel: LukPanel, n_seeds: int = 1000, oos_start: str | None = None) -> dict:
    """Diagnostic (c): Random entry placebo — same eligible + trend + RS + QQQ filter universe,
    same exits and sizing as V0, but entries chosen at random among eligible trend/RS days
    matching the daily signal count of V0. Runs `n_seeds` Monte Carlo realisations and reports
    empirical p-value of V0 Sharpe vs placebo."""
    cfg0 = LukConfig(id="V0")
    res_v0 = simulate_luk(panel, cfg0)
    d0 = res_v0.daily if oos_start is None else res_v0.daily.loc[res_v0.daily.index >= pd.Timestamp(oos_start, tz="UTC")]
    v0_sr = float(d0.mean() / d0.std() * np.sqrt(ANN_DAYS)) if d0.std() > 0 else 0.0

    sig_v0, rank_mat = signal_mask_for_config(panel, cfg0)
    counts_per_day = sig_v0.sum(axis=1)
    pool_mask = (
        panel.base_eligible
        & panel.trend_ok
        & (np.maximum(panel.r21, np.maximum(panel.r63, panel.r126)) >= 0.30)
        & (panel.rs_c_pct >= 0.80)
        & panel.qqq_filter_ok[:, None]
        & ((panel.dates >= pd.Timestamp("2001-01-02"))[:, None])
    )
    active_days = np.where(counts_per_day > 0)[0]
    pool_indices = {int(t): np.where(pool_mask[t])[0] for t in active_days}

    placebo_srs = np.zeros(n_seeds)
    rng = np.random.default_rng(42)
    T, N = panel.T, panel.N

    for s in range(n_seeds):
        rand_sig = np.zeros((T, N), dtype=bool)
        for t in active_days:
            pool = pool_indices[int(t)]
            k = min(int(counts_per_day[t]), len(pool))
            if k > 0:
                chosen = rng.choice(pool, size=k, replace=False)
                rand_sig[t, chosen] = True
        r_pl = simulate_luk(panel, cfg0, custom_signals=rand_sig, custom_rank=rank_mat)
        dp = r_pl.daily if oos_start is None else r_pl.daily.loc[r_pl.daily.index >= pd.Timestamp(oos_start, tz="UTC")]
        placebo_srs[s] = float(dp.mean() / dp.std() * np.sqrt(ANN_DAYS)) if dp.std() > 0 else 0.0

    p_val = float((np.sum(placebo_srs >= v0_sr) + 1) / (n_seeds + 1))
    return {
        "n_seeds": n_seeds,
        "v0_sharpe": round(v0_sr, 4),
        "placebo_mean_sharpe": round(float(placebo_srs.mean()), 4),
        "placebo_std_sharpe": round(float(placebo_srs.std()), 4),
        "placebo_p5_sharpe": round(float(np.quantile(placebo_srs, 0.05)), 4),
        "placebo_p50_sharpe": round(float(np.quantile(placebo_srs, 0.50)), 4),
        "placebo_p95_sharpe": round(float(np.quantile(placebo_srs, 0.95)), 4),
        "p_value_v0_vs_placebo": round(p_val, 4),
    }


def breakdown_by_turnover_and_volatility(tr_df: pd.DataFrame) -> dict:
    """Diagnostic (e): Results by turnover tercile (Medhat-Schmeling 2022) and ADR20 volatility
    quintile (Han-Yang-Zhou 2013)."""
    if len(tr_df) < 10:
        return {"turnover_terciles": {}, "adr_quintiles": {}}
    df = tr_df.copy()
    out_t, out_v = {}, {}
    try:
        df["turn_terc"] = pd.qcut(df["turnover_at_entry"].rank(method="first"), 3, labels=["T1_low", "T2_mid", "T3_high"])
        for k, grp in df.groupby("turn_terc", observed=False):
            r = grp["r_net"].values
            out_t[str(k)] = {
                "n_trades": int(len(r)),
                "win_rate": round(float((r > 0).mean()), 4),
                "mean_r": round(float(r.mean()), 4),
                "total_net_pnl": round(float(grp["net_pnl_usd"].sum()), 2),
            }
    except Exception:
        pass
    try:
        df["adr_quint"] = pd.qcut(df["adr_at_entry"].rank(method="first"), 5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
        for k, grp in df.groupby("adr_quint", observed=False):
            r = grp["r_net"].values
            out_v[str(k)] = {
                "n_trades": int(len(r)),
                "win_rate": round(float((r > 0).mean()), 4),
                "mean_r": round(float(r.mean()), 4),
                "total_net_pnl": round(float(grp["net_pnl_usd"].sum()), 2),
            }
    except Exception:
        pass
    return {"turnover_terciles": out_t, "adr_quintiles": out_v}


def overnight_vs_intraday_attribution(bar_df: pd.DataFrame, start_dt=None, end_dt=None) -> dict:
    """Diagnostic (f): Overnight vs intraday return attribution (Lou-Polk-Skouras 2019)."""
    b = bar_df.copy()
    if start_dt is not None:
        b = b.loc[b.index >= pd.Timestamp(start_dt, tz=b.index.tz)]
    if end_dt is not None:
        b = b.loc[b.index < pd.Timestamp(end_dt, tz=b.index.tz)]
    ov = b["overnight"]
    id_ = b["intraday"]
    years = max(len(b) / ANN_DAYS, 1e-9)
    return {
        "overnight_ann_gross": round(float(ov.sum() / years), 4),
        "intraday_ann_gross": round(float(id_.sum() / years), 4),
        "overnight_sharpe_gross": round(float(ov.mean() / ov.std() * np.sqrt(ANN_DAYS)) if ov.std() > 0 else 0.0, 3),
        "intraday_sharpe_gross": round(float(id_.mean() / id_.std() * np.sqrt(ANN_DAYS)) if id_.std() > 0 else 0.0, 3),
    }


def subperiod_summary(daily: pd.Series, spy_close: pd.Series, bar_df: pd.DataFrame | None = None) -> dict:
    """Report subperiods: 2001-2019 (pre-popularity), 2020-2025, 2009 crash/rebound, 2020-2022."""
    d = daily.copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None)
    s = spy_close.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    s_ret = s.reindex(d.index).ffill().pct_change(fill_method=None).fillna(0.0)

    windows = {
        "2001_2019_pre_popularity": ("2001-01-01", "2020-01-01"),
        "2020_2025_recent": ("2020-01-01", "2025-10-01"),
        "2009_rebound": ("2009-01-01", "2010-01-01"),
        "2020_2022_covid_bear": ("2020-01-01", "2023-01-01"),
    }
    out = {}
    for name, (t0, t1) in windows.items():
        m = (d.index >= pd.Timestamp(t0)) & (d.index < pd.Timestamp(t1))
        if m.sum() < 30:
            continue
        sub_d = d.loc[m]
        sub_s = s_ret.loc[m]
        md = metrics(sub_d)
        ms = metrics(sub_s)
        out[name] = {
            "strat_sharpe": round(md["sharpe"], 3),
            "strat_ann_ret": round(md["ann_ret"], 4),
            "strat_max_dd": round(md["max_dd"], 4),
            "spy_sharpe": round(ms["sharpe"], 3),
            "spy_ann_ret": round(ms["ann_ret"], 4),
            "spy_max_dd": round(ms["max_dd"], 4),
        }
    return out


def yearly_vs_spy_and_gate(daily: pd.Series, spy_close: pd.Series) -> dict:
    """Compute annual returns vs SPY and evaluate the user gate:
    'beat SPY by >= 3pp in >= 2 of 4 calendar years' (checked on last 4 full calendar years 2021-2024
    and across all full calendar years in OOS)."""
    d = daily.copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None)
    s = spy_close.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    b = s.reindex(d.index).ffill().pct_change(fill_method=None).fillna(0.0)
    years_dict = {}
    for y in sorted(set(d.index.year)):
        m = d.index.year == y
        if m.sum() < 300:
            continue
        sr_y = float((1.0 + d[m]).prod() - 1.0)
        sp_y = float((1.0 + b[m]).prod() - 1.0)
        years_dict[int(y)] = {
            "strat": round(sr_y, 4),
            "spy": round(sp_y, 4),
            "excess": round(sr_y - sp_y, 4),
            "beat_by_3pp": bool((sr_y - sp_y) >= 0.03),
        }
    last4 = [y for y in (2021, 2022, 2023, 2024) if y in years_dict]
    n_beat_last4 = sum(1 for y in last4 if years_dict[y]["beat_by_3pp"])
    n_beat_all = sum(1 for v in years_dict.values() if v["beat_by_3pp"])
    return {
        "years": years_dict,
        "last_4_years": last4,
        "beat_3pp_in_last_4": f"{n_beat_last4}/{len(last4)}",
        "user_gate_pass": bool(len(last4) == 4 and n_beat_last4 >= 2),
        "beat_3pp_all_oos_years": f"{n_beat_all}/{len(years_dict)}",
    }
