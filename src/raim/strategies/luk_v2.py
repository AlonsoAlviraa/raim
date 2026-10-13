"""luk-v2: Martin Luk style small/mid-cap momentum + point-in-time SEC EDGAR fundamentals
(spec configs/prereg/luk_v2/L2_luk_smid_mom_fund.yaml).

SIMULATION / VIRTUAL CAPITAL ONLY. Never live trading. Not financial advice.

Key design guarantees:
1. Point-in-time SEC EDGAR fundamentals (from quarterly Financial Statement Data Sets `sub.txt` + `num.txt`):
   Every filing `f` has an SEC acceptance timestamp (`accepted` / `filed`) and is mapped to `avail_date`,
   which is strictly the first NYSE trading session AFTER the filing date (`filed_date + 1 session`, or
   `filed_date` only if accepted before 09:00 ET, conservatively enforced as ` strictly > filed_date`
   so no filing is ever used on or before its filing date).
2. Point-in-time small/mid-cap universe:
   - US common stocks (NYSE/NASDAQ/AMEX, plus historical delisted/acquired filers from SEC FSDS `instance`
     ticker mappings and historical index snapshots).
   - Strictly `>= 4` valid OHLC price points per stock-day (`open > 0`, `high > 0`, `low > 0`, `close > 0`,
     `high >= max(open, close)`, `low <= min(open, close)`, and `high > low`). Stock-days failing any check
     are flagged and rejected.
   - Unadjusted close `> $5.00`, CPI-deflated `ADV$20 > $5M`, and CPI-deflated PIT market cap
     (`shares_outstanding_pit * close_unadj`) in `[$300M, $20B]`.
3. Conservative intra-bar OHLC ordering:
   - Whenever an intraday stop is active (`V6`), stop is evaluated BEFORE any profit target unless `open >= target`
     (and gap below stop fills at `open`).
4. Clearly separated 60m intraday sub-period test (`run_intraday_subperiod_test`):
   - Evaluates 7 hourly bars/session (`28` OHLC price points/day) over the ~730-day Yahoo 60m window
     (`2023-11-14 .. 2025-09-30` dev), comparing intraday flush-entry + low-of-day stop execution against
     daily-bar execution on the exact same universe and period."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from raim.strategies.engine import BTResult, metrics
from raim.strategies.luk import (
    cpi_deflator_series,
    ema_seeded,
    parse_yahoo_chart,
    wilder_atr14,
)

log = logging.getLogger("luk_v2")

DATA_DIR = Path("/workspace/data_hist/luk_v2")

# Canonical XBRL tags in SEC Financial Statement Data Sets
EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")
REV_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)
OPINC_TAGS = ("OperatingIncomeLoss", "NetIncomeLoss")
SHARES_TAGS = (
    "EntityCommonStockSharesOutstanding",
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
)


@dataclass(frozen=True)
class LukV2Config:
    id: str = "V0"
    turnover_filter: str = "ex_high_tercile"     # "ex_high_tercile" | "low_tercile_only" | "all"
    fund_mode: str = "canonic_eps_rev"           # "canonic_eps_rev" | "accel_and_margin" | "none"
    catalyst_mode: str = "sue_or_gap"            # "sue_or_gap" | "earnings_gap_only" | "none"
    exit_mode: str = "ema21_or_rank"             # "ema21_or_rank" | "ema50_or_rank"
    mkt_filter: str = "iwm_qqq_ema50"            # "iwm_qqq_ema50" | "none"
    stop_mode: str = "none"                      # "none" | "wide_2atr_stop_first"
    reb_days: int = 21                           # 21 (monthly) | 10 (biweekly)
    rank_mode: str = "momentum"                  # "momentum" | "fundamentals_only"
    delist_mode: str = "pessimistic"             # "pessimistic" (-30%) | "neutral" (0%)
    cost_mult: float = 1.0                       # 1.0 base, 2.0 stress
    extra_delay: int = 0                         # 0 -> open_{t+1}, 1 -> open_{t+2} (R11)
    max_positions: int = 20
    target_capital: float = 100_000.0


def config_v2_from_grid(p: dict, **overrides) -> LukV2Config:
    kw = {k: v for k, v in p.items() if k in LukV2Config.__dataclass_fields__}
    kw.update(overrides)
    return LukV2Config(**kw)


def normalize_grid_v2(raw_grid: list[dict]) -> list[dict]:
    """Normalize the 8 preregistered variants V0..V7 to 7 single-factor keys so
    `raim.research.stability.neighbors` links V0 to all 7 single-parameter neighbours V1..V7."""
    defaults = {
        "turnover_filter": "ex_high_tercile",
        "fund_mode": "canonic_eps_rev",
        "catalyst_mode": "sue_or_gap",
        "exit_mode": "ema21_or_rank",
        "mkt_filter": "iwm_qqq_ema50",
        "stop_mode": "none",
        "reb_days": 21,
    }
    out: list[dict] = []
    for p in raw_grid:
        d = dict(defaults)
        for k in defaults:
            if k in p:
                d[k] = p[k]
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# 1. Four-point OHLC quality validation & small-cap slippage model
# ---------------------------------------------------------------------------

def validate_four_price_points_ohlc(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Strictly validate that every stock-day has >= 4 valid OHLC price points:
    1) All 4 of (open, high, low, close) are finite and > 0.
    2) high >= max(open, close) - 1e-9 and low <= min(open, close) + 1e-9.
    3) Non-degenerate range: high > low (at least 4 non-collapsed daily price points).

    Returns:
      - valid_4pt: boolean mask (T, N) of stock-days passing all checks.
      - rejected_incomplete: boolean mask (T, N) of stock-days where at least one price was present
        but the 4-point OHLC check failed."""
    any_present = np.isfinite(open_arr) | np.isfinite(high_arr) | np.isfinite(low_arr) | np.isfinite(close_arr)
    all_four_pos = (
        np.isfinite(open_arr)
        & np.isfinite(high_arr)
        & np.isfinite(low_arr)
        & np.isfinite(close_arr)
        & (open_arr > 0)
        & (high_arr > 0)
        & (low_arr > 0)
        & (close_arr > 0)
    )
    oc_max = np.maximum(open_arr, close_arr)
    oc_min = np.minimum(open_arr, close_arr)
    consistent_bounds = all_four_pos & (high_arr >= oc_max * (1.0 - 1e-7)) & (low_arr <= oc_min * (1.0 + 1e-7))
    non_degenerate = consistent_bounds & (high_arr > low_arr * (1.0 + 1e-7))
    rejected = any_present & (~non_degenerate)
    return non_degenerate, rejected


def smid_one_side_slip_bps(adv_usd: float, price_unadj: float, cpi_ratio: float = 1.0, is_stop: bool = False) -> float:
    """Small/mid-cap tiered one-way slippage (fraction, e.g. 0.0015 = 15 bps):
    - CPI-deflated ADV$20 >= $100M: 8 bps
    - CPI-deflated ADV$20 in [$25M, $100M): 15 bps
    - CPI-deflated ADV$20 in [$5M, $25M): 25 bps
    - +15 bps if unadjusted price < $10
    - 2x slippage on stop executions."""
    real_adv = adv_usd / max(cpi_ratio, 1e-6)
    if real_adv >= 100_000_000.0:
        bps = 8.0
    elif real_adv >= 25_000_000.0:
        bps = 15.0
    else:
        bps = 25.0
    if price_unadj < 10.0:
        bps += 15.0
    if is_stop:
        bps *= 2.0
    return bps * 1e-4


def commission_usd(shares: int) -> float:
    if shares <= 0:
        return 0.0
    return max(1.0, 0.005 * shares)


# ---------------------------------------------------------------------------
# 2. Point-in-time SEC EDGAR fundamental features (strictly > filing date)
# ---------------------------------------------------------------------------

def build_pit_fundamentals_panel(
    filings_df: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    ciks: list[int],
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    valid_arr: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert raw quarterly SEC EDGAR filing records into strictly causal daily (T, N) arrays.

    `filings_df` columns required:
      - `cik` (int)
      - `period` (YYYYMMDD int/str: fiscal period end date)
      - `filed` (YYYYMMDD int/str: SEC filing / acceptance date)
      - `eps` (float, quarterly single-quarter EPS; NaN if missing)
      - `rev` (float, quarterly single-quarter revenue USD; NaN if missing)
      - `opinc` (float, quarterly operating/net income USD; NaN if missing)
      - `shares` (float, point-in-time common shares outstanding; NaN if missing)

    STRICT CAUSALITY RULE (tested in `tests/test_luk_v2_leakage.py`):
      A filing with `filed_date = F` becomes visible ONLY on trading sessions `t > F`
      (i.e. `searchsorted(trading_days, F, side="right")`). It is NEVER visible on `t <= F`.

    Derived point-in-time features per stock j at session t:
      - `shares_pit[t, j]`: latest reported shares outstanding from filings with `filed < t`.
      - `eps_yoy[t, j]`: `(eps_q - eps_{q-4}) / max(|eps_{q-4}|, 0.05)` using only filings with `filed < t`.
      - `eps_qoq[t, j]`: `(eps_q - eps_{q-1}) / max(|eps_{q-1}|, 0.05)` using only filings with `filed < t`.
      - `eps_accel[t, j]`: `eps_yoy_q - eps_yoy_{q-1}` (acceleration of YoY EPS growth).
      - `sue[t, j]`: Standardized Unexpected Earnings `(eps_q - eps_{q-4}) / std(eps_k - eps_{k-4})`
        computed over the up-to-8 most recent seasonal differences available strictly before `t` (min 4 diffs).
      - `rev_yoy[t, j]`: `(rev_q - rev_{q-4}) / rev_{q-4}` (when `rev_{q-4} > 0`).
      - `rev_qoq[t, j]`: `(rev_q - rev_{q-1}) / rev_{q-1}` (when `rev_{q-1} > 0`).
      - `op_margin[t, j]`: `opinc_q / rev_q` (when `rev_q > 0`).
      - `earn_gap[t, j]`: most recent filing-reaction return (`max(open_{t_avail}/close_{t_avail-1}-1,
        close_{t_avail}/close_{t_avail-1}-1)`) observed on the filing's first post-acceptance session `t_avail <= t`."""
    T = len(trading_days)
    N = len(ciks)
    naive_days = pd.DatetimeIndex(trading_days).tz_localize(None).floor("D")

    shares_pit = np.full((T, N), np.nan)
    eps_yoy = np.full((T, N), np.nan)
    eps_qoq = np.full((T, N), np.nan)
    eps_accel = np.full((T, N), np.nan)
    sue = np.full((T, N), np.nan)
    rev_yoy = np.full((T, N), np.nan)
    rev_qoq = np.full((T, N), np.nan)
    op_margin = np.full((T, N), np.nan)
    earn_gap = np.full((T, N), np.nan)
    has_pit_filing = np.zeros((T, N), dtype=bool)

    if filings_df is None or len(filings_df) == 0:
        return {
            "shares_pit": shares_pit,
            "eps_yoy": eps_yoy,
            "eps_qoq": eps_qoq,
            "eps_accel": eps_accel,
            "sue": sue,
            "rev_yoy": rev_yoy,
            "rev_qoq": rev_qoq,
            "op_margin": op_margin,
            "earn_gap": earn_gap,
            "has_pit_filing": has_pit_filing,
        }

    cik_to_j = {int(c): j for j, c in enumerate(ciks)}
    df = filings_df.copy()
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").fillna(-1).astype(int)
    df = df[df["cik"].isin(cik_to_j)]
    if len(df) == 0:
        return {
            "shares_pit": shares_pit,
            "eps_yoy": eps_yoy,
            "eps_qoq": eps_qoq,
            "eps_accel": eps_accel,
            "sue": sue,
            "rev_yoy": rev_yoy,
            "rev_qoq": rev_qoq,
            "op_margin": op_margin,
            "earn_gap": earn_gap,
            "has_pit_filing": has_pit_filing,
        }

    df["filed_dt"] = pd.to_datetime(df["filed"].astype(str), format="mixed", errors="coerce").dt.tz_localize(None).dt.floor("D")
    df["period_dt"] = pd.to_datetime(df["period"].astype(str), format="mixed", errors="coerce").dt.tz_localize(None).dt.floor("D")
    df = df.dropna(subset=["filed_dt", "period_dt"]).sort_values(["cik", "filed_dt", "period_dt"])

    # Strictly causal availability bar: first session strictly > filed_dt
    avail_bars = np.searchsorted(naive_days, df["filed_dt"].values, side="right")
    df["avail_bar"] = avail_bars
    df = df[df["avail_bar"] < T]

    for cik_val, grp in df.groupby("cik", sort=False):
        j = cik_to_j[int(cik_val)]
        # Process filings in chronological order of availability bar
        # Maintain point-in-time dictionary of known period_dt -> (eps, rev, opinc, shares)
        known_periods: dict[pd.Timestamp, tuple[float, float, float, float]] = {}
        last_sh = np.nan

        # Group by avail_bar so multiple rows filed on the same day update state together
        bars_and_states: list[tuple[int, float, float, float, float, float, float, float, float, float]] = []
        for b_avail, b_rows in grp.groupby("avail_bar", sort= True):
            b_idx = int(b_avail)
            for _, r in b_rows.iterrows():
                p_dt = r["period_dt"]
                e_v = float(r["eps"]) if pd.notna(r["eps"]) else np.nan
                rv_v = float(r["rev"]) if pd.notna(r["rev"]) else np.nan
                op_v = float(r["opinc"]) if pd.notna(r["opinc"]) else np.nan
                sh_v = float(r["shares"]) if pd.notna(r["shares"]) and r["shares"] > 0 else np.nan
                if np.isfinite(sh_v):
                    last_sh = sh_v
                prev = known_periods.get(p_dt, (np.nan, np.nan, np.nan, np.nan))
                known_periods[p_dt] = (
                    e_v if np.isfinite(e_v) else prev[0],
                    rv_v if np.isfinite(rv_v) else prev[1],
                    op_v if np.isfinite(op_v) else prev[2],
                    sh_v if np.isfinite(sh_v) else prev[3],
                )

            # Compute point-in-time quarterly series sorted by fiscal period end date (only using periods known at b_idx!)
            p_sorted = sorted(known_periods.keys())
            eps_list = [known_periods[p][0] for p in p_sorted]
            rev_list = [known_periods[p][1] for p in p_sorted]
            op_list = [known_periods[p][2] for p in p_sorted]

            ey = np.nan
            eq = np.nan
            ea = np.nan
            sue_val = np.nan
            ry = np.nan
            rq = np.nan
            om = np.nan

            # Latest period with finite EPS
            eps_valid = [(p, e) for p, e in zip(p_sorted, eps_list) if np.isfinite(e)]
            if len(eps_valid) >= 2:
                p_cur, e_cur = eps_valid[-1]
                p_prev1, e_prev1 = eps_valid[-2]
                if 45 <= (p_cur - p_prev1).days <= 135:
                    eq = (e_cur - e_prev1) / max(abs(e_prev1), 0.10)

                # Match YoY quarter (~365 days earlier: 320..410 days)
                yoy_diffs: list[tuple[pd.Timestamp, float, float]] = []
                for idx_p, (p_k, e_k) in enumerate(eps_valid):
                    for p_m, e_m in eps_valid[:idx_p]:
                        if 320 <= (p_k - p_m).days <= 410:
                            diff_k = e_k - e_m
                            g_k = diff_k / max(abs(e_m), 0.10)
                            yoy_diffs.append((p_k, diff_k, g_k))
                if yoy_diffs and yoy_diffs[-1][0] == p_cur:
                    ey = float(np.clip(yoy_diffs[-1][2], -5.0, 20.0))
                    if len(yoy_diffs) >= 2:
                        ea = float(np.clip(yoy_diffs[-1][2] - yoy_diffs[-2][2], -10.0, 20.0))
                    if len(yoy_diffs) >= 4:
                        recent_d = np.array([x[1] for x in yoy_diffs[-8:]], dtype=float)
                        sd_d = float(np.std(recent_d, ddof=1)) if len(recent_d) > 1 else 0.0
                        if sd_d > 1e-4:
                            sue_val = float(np.clip(yoy_diffs[-1][1] / sd_d, -10.0, 10.0))
                        else:
                            sue_val = 1.0 if yoy_diffs[-1][1] > 0 else (-1.0 if yoy_diffs[-1][1] < 0 else 0.0)

            # Latest period with finite Revenue
            rev_valid = [(p, rv, op) for p, rv, op in zip(p_sorted, rev_list, op_list) if np.isfinite(rv) and rv > 0]
            if rev_valid:
                p_rcur, rv_cur, op_cur = rev_valid[-1]
                if np.isfinite(op_cur):
                    om = float(np.clip(op_cur / rv_cur, -2.0, 1.0))
                if len(rev_valid) >= 2:
                    p_rprev1, rv_prev1, _ = rev_valid[-2]
                    if 45 <= (p_rcur - p_rprev1).days <= 135 and rv_prev1 > 0:
                        rq = float(np.clip((rv_cur - rv_prev1) / rv_prev1, -1.0, 10.0))
                for p_rm, rv_m, _ in reversed(rev_valid[:-1]):
                    if 320 <= (p_rcur - p_rm).days <= 410 and rv_m > 0:
                        ry = float(np.clip((rv_cur - rv_m) / rv_m, -1.0, 20.0))
                        break

            # Earnings report price reaction on the causal availability session b_idx (or b_idx - 1)
            eg = np.nan
            if b_idx >= 1 and valid_arr[b_idx, j] and valid_arr[b_idx - 1, j]:
                c_prev = close_arr[b_idx - 1, j]
                if c_prev > 0:
                    gap_open = open_arr[b_idx, j] / c_prev - 1.0
                    ret_day = close_arr[b_idx, j] / c_prev - 1.0
                    eg = float(max(gap_open, ret_day))

            bars_and_states.append((b_idx, last_sh, ey, eq, ea, sue_val, ry, rq, om, eg))

        # Forward-fill each state across [b_idx, next_b_idx) (up to 100 trading days staleness for quarterly stats,
        # 260 trading days for shares_pit)
        for idx_s, (b0, sh_v, ey, eq, ea, sue_v, ry, rq, om, eg) in enumerate(bars_and_states):
            b1_next = bars_and_states[idx_s + 1][0] if idx_s + 1 < len(bars_and_states) else T
            b_end_q = min(b1_next, b0 + 100, T)
            b_end_sh = min(b1_next, b0 + 260, T)
            if np.isfinite(sh_v):
                shares_pit[b0:b_end_sh, j] = sh_v
            has_pit_filing[b0:b_end_q, j] = True
            if np.isfinite(ey):
                eps_yoy[b0:b_end_q, j] = ey
            if np.isfinite(eq):
                eps_qoq[b0:b_end_q, j] = eq
            if np.isfinite(ea):
                eps_accel[b0:b_end_q, j] = ea
            if np.isfinite(sue_v):
                sue[b0:b_end_q, j] = sue_v
            if np.isfinite(ry):
                rev_yoy[b0:b_end_q, j] = ry
            if np.isfinite(rq):
                rev_qoq[b0:b_end_q, j] = rq
            if np.isfinite(om):
                op_margin[b0:b_end_q, j] = om
            if np.isfinite(eg):
                earn_gap[b0:b_end_q, j] = eg

    return {
        "shares_pit": shares_pit,
        "eps_yoy": eps_yoy,
        "eps_qoq": eps_qoq,
        "eps_accel": eps_accel,
        "sue": sue,
        "rev_yoy": rev_yoy,
        "rev_qoq": rev_qoq,
        "op_margin": op_margin,
        "earn_gap": earn_gap,
        "has_pit_filing": has_pit_filing,
    }


# ---------------------------------------------------------------------------
# 3. LukV2Panel & causal precomputation
# ---------------------------------------------------------------------------

class LukV2Panel:
    """Holds aligned daily arrays (T x N) for the US small/mid-cap universe, PIT fundamentals,
    and SPY / IWM / QQQ / CPI series."""

    def __init__(
        self,
        trading_days: pd.DatetimeIndex,
        tickers: list[str],
        ciks: list[int],
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
        vol_arr: np.ndarray,
        close_unadj_arr: np.ndarray,
        vol_unadj_arr: np.ndarray,
        valid_4pt_arr: np.ndarray,
        pit_active_arr: np.ndarray,
        delist_bar: np.ndarray,
        fund_arrays: dict[str, np.ndarray],
        spy_df: pd.DataFrame,
        iwm_df: pd.DataFrame,
        qqq_df: pd.DataFrame,
        cpi_ratio: np.ndarray,
        hygiene: dict,
        start_eval_date: str = "2009-07-01",
    ):
        self.dates = trading_days
        self.tickers = tickers
        self.ciks = ciks
        self.T, self.N = open_arr.shape
        self.open = open_arr
        self.high = high_arr
        self.low = low_arr
        self.close = close_arr
        self.vol = vol_arr
        self.close_unadj = close_unadj_arr
        self.vol_unadj = vol_unadj_arr
        self.valid = valid_4pt_arr
        self.pit_active = pit_active_arr
        self.delist_bar = delist_bar
        self.fund = fund_arrays
        self.spy = spy_df
        self.iwm = iwm_df
        self.qqq = qqq_df
        self.cpi_ratio = cpi_ratio
        self.hygiene = hygiene
        self.start_eval_date = start_eval_date

        self._compute_indicators()

    def _compute_indicators(self) -> None:
        T, N = self.T, self.N
        self.ema21 = np.full((T, N), np.nan)
        self.ema50 = np.full((T, N), np.nan)
        self.atr14 = np.full((T, N), np.nan)
        self.adr20 = np.full((T, N), np.nan)
        self.adv20 = np.full((T, N), np.nan)
        self.turnover20 = np.full((T, N), np.nan)
        self.bars_count = np.zeros((T, N), dtype=np.int32)

        sh_pit = self.fund["shares_pit"]
        # Point-in-time market cap (USD) = shares_pit * close_unadj
        self.mcap_pit = sh_pit * self.close_unadj
        # Real (2025-09 CPI-deflated) market cap
        self.mcap_real = self.mcap_pit / np.maximum(self.cpi_ratio[:, None], 1e-6)

        for j in range(N):
            v_idx = np.where(self.valid[:, j])[0]
            if len(v_idx) < 252:
                continue
            c = self.close[v_idx, j]
            h = self.high[v_idx, j]
            l = self.low[v_idx, j]
            cu = self.close_unadj[v_idx, j]
            vu = self.vol_unadj[v_idx, j]
            sh_j = sh_pit[v_idx, j]

            e21 = ema_seeded(c, 21)
            e50 = ema_seeded(c, 50)
            atr = wilder_atr14(h, l, c, 14)
            adr = pd.Series(h / l - 1.0).rolling(20, min_periods=20).mean().values
            adv = pd.Series(cu * vu).rolling(20, min_periods=20).mean().values
            # 20-day share turnover = mean_20(volume_unadj) / shares_outstanding_pit
            v20 = pd.Series(vu).rolling(20, min_periods=20).mean().values
            to20 = np.where(np.isfinite(sh_j) & (sh_j > 0), v20 / sh_j, np.nan)

            self.ema21[v_idx, j] = e21
            self.ema50[v_idx, j] = e50
            self.atr14[v_idx, j] = atr
            self.adr20[v_idx, j] = adr
            self.adv20[v_idx, j] = adv
            self.turnover20[v_idx, j] = to20
            self.bars_count[v_idx, j] = np.arange(1, len(v_idx) + 1, dtype=np.int32)

        def _lag_ret(k: int) -> np.ndarray:
            out = np.full((T, N), np.nan)
            valid_pair = (
                self.valid[k:]
                & self.valid[:-k]
                & np.isfinite(self.close[k:])
                & np.isfinite(self.close[:-k])
                & (self.close[:-k] > 0)
            )
            ratio = np.ones((T - k, N))
            np.divide(self.close[k:], self.close[:-k], out=ratio, where=valid_pair)
            out[k:] = np.where(valid_pair, ratio - 1.0, np.nan)
            return out

        self.r21 = _lag_ret(21)
        self.r63 = _lag_ret(63)
        self.r126 = _lag_ret(126)
        self.r189 = _lag_ret(189)
        self.r252 = _lag_ret(252)

        # 6/12-month composite momentum score C_{t,i}
        self.mom_score = 0.4 * self.r63 + 0.2 * self.r126 + 0.2 * self.r189 + 0.2 * self.r252

        # Point-in-time US small/mid-cap eligibility at bar t:
        # 1) valid 4-point OHLC bar and active listing
        # 2) >= 252 valid prior bars
        # 3) unadjusted close > $5.00
        # 4) ADV$20 > $5M * CPI_{t-1m}/CPI_{2025-09}
        # 5) PIT real market cap in [$300M, $20B]
        adv_thresh = 5_000_000.0 * self.cpi_ratio[:, None]
        self.smid_eligible = (
            self.valid
            & self.pit_active
            & (self.bars_count >= 252)
            & (self.close_unadj > 5.0)
            & (self.adv20 >= adv_thresh)
            & np.isfinite(self.mcap_real)
            & (self.mcap_real >= 300_000_000.0)
            & (self.mcap_real <= 20_000_000_000.0)
        )

        # Uptrend alignment (close > EMA50 and EMA21 > EMA50)
        self.trend_ok = (
            np.isfinite(self.ema21)
            & np.isfinite(self.ema50)
            & (self.close > self.ema50)
            & (self.ema21 > self.ema50)
        )

        # Cross-sectional momentum percentile and turnover percentile within smid_eligible at each day t
        self.mom_pct = np.full((T, N), np.nan)
        self.turnover_pct = np.full((T, N), np.nan)
        self.fund_score_pct = np.full((T, N), np.nan)

        ey = self.fund["eps_yoy"]
        ry = self.fund["rev_yoy"]
        sue_arr = self.fund["sue"]
        om = self.fund["op_margin"]
        # Composite fundamental rank score (for diagnostic "fundamentals without momentum")
        raw_fscore = (
            np.clip(np.nan_to_num(ey, nan=-1.0), -1.0, 3.0)
            + np.clip(np.nan_to_num(ry, nan=-1.0), -1.0, 3.0)
            + 0.25 * np.clip(np.nan_to_num(sue_arr, nan=0.0), -3.0, 5.0)
            + np.clip(np.nan_to_num(om, nan=0.0), -0.5, 0.5)
        )

        for t in range(252, T):
            el = self.smid_eligible[t]
            if not np.any(el):
                continue
            # Momentum percentile
            mc = el & np.isfinite(self.mom_score[t])
            n_mc = int(mc.sum())
            if n_mc >= 5:
                ranks = pd.Series(self.mom_score[t, mc]).rank(method="average").values
                self.mom_pct[t, mc] = (ranks - 1.0) / max(n_mc - 1, 1)

            # Turnover percentile
            mt = el & np.isfinite(self.turnover20[t])
            n_mt = int(mt.sum())
            if n_mt >= 5:
                ranks_t = pd.Series(self.turnover20[t, mt]).rank(method="average").values
                self.turnover_pct[t, mt] = (ranks_t - 1.0) / max(n_mt - 1, 1)

            # Fundamental score percentile
            mf = el & self.fund["has_pit_filing"][t] & (np.isfinite(ey[t]) | np.isfinite(ry[t]))
            n_mf = int(mf.sum())
            if n_mf >= 5:
                ranks_f = pd.Series(raw_fscore[t, mf]).rank(method="average").values
                self.fund_score_pct[t, mf] = (ranks_f - 1.0) / max(n_mf - 1, 1)

        # Market filter: IWM close > EMA50(IWM) OR QQQ close > EMA50(QQQ)
        iwm_c = self.iwm["close"].values.astype(float)
        iwm_e50 = ema_seeded(iwm_c, 50)
        qqq_c = self.qqq["close"].values.astype(float)
        qqq_e50 = ema_seeded(qqq_c, 50)
        self.mkt_ok = (np.isfinite(iwm_e50) & (iwm_c > iwm_e50)) | (np.isfinite(qqq_e50) & (qqq_c > qqq_e50))

    def slice_until(self, end_date: str) -> LukV2Panel:
        ts = pd.Timestamp(end_date).tz_localize(None)
        mask = self.dates < ts
        k = int(mask.sum())
        delist_sub = np.where(self.delist_bar < k, self.delist_bar, -1)
        fund_sub = {name: arr[:k].copy() for name, arr in self.fund.items()}
        return LukV2Panel(
            trading_days=self.dates[:k],
            tickers=self.tickers,
            ciks=self.ciks,
            open_arr=self.open[:k].copy(),
            high_arr=self.high[:k].copy(),
            low_arr=self.low[:k].copy(),
            close_arr=self.close[:k].copy(),
            vol_arr=self.vol[:k].copy(),
            close_unadj_arr=self.close_unadj[:k].copy(),
            vol_unadj_arr=self.vol_unadj[:k].copy(),
            valid_4pt_arr=self.valid[:k].copy(),
            pit_active_arr=self.pit_active[:k].copy(),
            delist_bar=delist_sub,
            fund_arrays=fund_sub,
            spy_df=self.spy.loc[self.spy.index < ts].copy(),
            iwm_df=self.iwm.loc[self.iwm.index < ts].copy(),
            qqq_df=self.qqq.loc[self.qqq.index < ts].copy(),
            cpi_ratio=self.cpi_ratio[:k].copy(),
            hygiene=self.hygiene,
            start_eval_date=self.start_eval_date,
        )


# ---------------------------------------------------------------------------
# 4. Candidate selection & daily simulation (with conservative OHLC stop-first)
# ---------------------------------------------------------------------------

def candidate_mask_and_score(panel: LukV2Panel, cfg: LukV2Config) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean (T, N) entry eligibility mask and (T, N) ranking score matrix for `cfg`."""
    el = panel.smid_eligible & panel.trend_ok

    # 1. Turnover filter (per luk-v1 finding on Medhat & Schmeling 2022)
    if cfg.turnover_filter == "low_tercile_only":
        to_ok = np.isfinite(panel.turnover_pct) & (panel.turnover_pct <= (1.0 / 3.0))
    elif cfg.turnover_filter == "ex_high_tercile":
        to_ok = np.isfinite(panel.turnover_pct) & (panel.turnover_pct <= (2.0 / 3.0))
    else:
        to_ok = np.ones_like(el, dtype=bool)

    # 2. Point-in-time fundamental filter
    ey = panel.fund["eps_yoy"]
    eq = panel.fund["eps_qoq"]
    ea = panel.fund["eps_accel"]
    ry = panel.fund["rev_yoy"]
    rq = panel.fund["rev_qoq"]
    om = panel.fund["op_margin"]
    sue = panel.fund["sue"]
    eg = panel.fund["earn_gap"]

    if cfg.fund_mode == "canonic_eps_rev":
        eps_ok = (np.isfinite(ey) & (ey >= 0.25)) | (np.isfinite(eq) & (eq >= 0.25))
        rev_ok = (np.isfinite(ry) & (ry >= 0.20)) | (np.isfinite(rq) & (rq >= 0.15))
        fund_ok = panel.fund["has_pit_filing"] & (eps_ok | rev_ok)
    elif cfg.fund_mode == "accel_and_margin":
        eps_ok = (np.isfinite(ey) & (ey >= 0.20)) & (np.isfinite(ea) & (ea > 0.0))
        rev_ok = np.isfinite(ry) & (ry >= 0.15)
        margin_ok = np.isfinite(om) & (om > 0.0)
        fund_ok = panel.fund["has_pit_filing"] & (eps_ok | (rev_ok & margin_ok))
    else:
        fund_ok = np.ones_like(el, dtype=bool)

    # 3. Catalyst filter (SUE > 0.5 or earnings-report-day reaction >= +2%)
    if cfg.catalyst_mode == "sue_or_gap":
        cat_ok = (np.isfinite(sue) & (sue >= 0.5)) | (np.isfinite(eg) & (eg >= 0.02))
    elif cfg.catalyst_mode == "earnings_gap_only":
        cat_ok = np.isfinite(eg) & (eg >= 0.02)
    else:
        cat_ok = np.ones_like(el, dtype=bool)

    # 4. Ranking mode (top decile 6/12m momentum vs fundamentals-only comparison)
    if cfg.rank_mode == "fundamentals_only":
        rank_ok = np.isfinite(panel.fund_score_pct) & (panel.fund_score_pct >= 0.85)
        score = panel.fund_score_pct
    else:
        rank_ok = np.isfinite(panel.mom_pct) & (panel.mom_pct >= 0.90) & (panel.mom_score > 0.0)
        # Tie-break / composite boost with fundamental percentile when available
        score = panel.mom_pct + 0.05 * np.nan_to_num(panel.fund_score_pct, nan=0.0)

    if cfg.mkt_filter == "iwm_qqq_ema50":
        mkt_mask = panel.mkt_ok[:, None]
    else:
        mkt_mask = np.ones((panel.T, 1), dtype=bool)

    start_mask = (panel.dates >= pd.Timestamp(panel.start_eval_date))[:, None]
    cand = el & to_ok & fund_ok & cat_ok & rank_ok & mkt_mask & start_mask
    return cand, score


@dataclass
class V2Position:
    sym_idx: int
    ticker: str
    signal_bar: int
    entry_bar: int
    entry_price: float
    stop_price: float           # NaN if stop_mode == "none"
    r_unit_price: float         # 2 * ATR14 at signal bar (used as R-unit denominator for trade metrics)
    shares: int
    entry_cost_usd: float
    total_cost_usd: float
    realized_gross_pnl: float = 0.0
    realized_net_pnl: float = 0.0
    adv_at_entry: float = 0.0
    mcap_at_entry: float = 0.0
    pending_exit: bool = False
    pending_reason: str = ""


def simulate_luk_v2(panel: LukV2Panel, cfg: LukV2Config) -> BTResult:
    """Simulate LukV2 small/mid-cap momentum + PIT fundamentals strategy on daily OHLC bars.
    Strictly causal: signals at t execute at open_{t + 1 + cfg.extra_delay}."""
    T, N = panel.T, panel.N
    cand_mat, score_mat = candidate_mask_and_score(panel, cfg)

    start_bar = int(np.searchsorted(panel.dates, pd.Timestamp(panel.start_eval_date)))
    cash = float(cfg.target_capital)
    positions: dict[int, V2Position] = {}
    prev_mark: dict[int, float] = {}
    queued_entries: dict[int, list[tuple[int, int]]] = {}

    daily_gross = np.zeros(T)
    daily_cost = np.zeros(T)
    daily_net = np.zeros(T)
    daily_exposure = np.zeros(T)
    daily_turnover = np.zeros(T)
    daily_adv_part_max = np.zeros(T)

    completed_trades: list[dict] = []

    for t in range(start_bar, T):
        eq_start = cash + sum(pos.shares * prev_mark[j] for j, pos in positions.items())
        if eq_start <= 0:
            break

        day_gross_usd = 0.0
        day_cost_usd = 0.0
        day_traded_usd = 0.0
        max_adv_part = 0.0
        cpi_t = float(panel.cpi_ratio[t])

        # ------------------------------------------------------------------
        # 1. Process existing open positions at bar t
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

            # 1a. Delisting while position is held
            if panel.delist_bar[j] == t or (not panel.valid[t, j] and not np.any(panel.valid[t:, j])):
                sh = pos.shares
                base_px = c_t if (panel.valid[t, j] and np.isfinite(c_t) and c_t > 0) else p_prev
                mult = 0.70 if cfg.delist_mode == "pessimistic" else 1.0
                exit_px = base_px * mult
                slip = smid_one_side_slip_bps(adv_t, cu_t, cpi_t, is_stop=(cfg.delist_mode == "pessimistic"))
                cost_u = (sh * exit_px * slip + commission_usd(sh)) * cfg.cost_mult
                gross_u = sh * (exit_px - p_prev)
                day_gross_usd += gross_u
                day_cost_usd += cost_u
                day_traded_usd += sh * exit_px
                cash += sh * exit_px - cost_u
                pos.total_cost_usd += cost_u
                pos.realized_gross_pnl += sh * (exit_px - pos.entry_price)
                pos.realized_net_pnl = pos.realized_gross_pnl - pos.total_cost_usd
                completed_trades.append(_v2_trade_dict(pos, panel.dates[t], exit_px, f"delist_{cfg.delist_mode}"))
                del positions[j]
                del prev_mark[j]
                continue

            if not panel.valid[t, j]:
                continue

            # 1b. Pending next-open exit (triggered at close of t-1 by EMA trail or rank drop)
            if pos.pending_exit:
                sh = pos.shares
                slip = smid_one_side_slip_bps(adv_t, cu_t, cpi_t, is_stop=False)
                cost_u = (sh * o_t * slip + commission_usd(sh)) * cfg.cost_mult
                gross_u = sh * (o_t - p_prev)
                day_gross_usd += gross_u
                day_cost_usd += cost_u
                day_traded_usd += sh * o_t
                max_adv_part = max(max_adv_part, (sh * o_t * (cfg.target_capital / eq_start)) / max(adv_t, 1.0))
                cash += sh * o_t - cost_u
                pos.total_cost_usd += cost_u
                pos.realized_gross_pnl += sh * (o_t - pos.entry_price)
                pos.realized_net_pnl = pos.realized_gross_pnl - pos.total_cost_usd
                completed_trades.append(_v2_trade_dict(pos, panel.dates[t], o_t, pos.pending_reason))
                del positions[j]
                del prev_mark[j]
                continue

            # 1c. Conservative stop-first check on bar t when stop_mode == "wide_2atr_stop_first" (V6)
            if cfg.stop_mode == "wide_2atr_stop_first" and np.isfinite(pos.stop_price):
                if o_t <= pos.stop_price or l_t <= pos.stop_price:
                    sh = pos.shares
                    exit_px = o_t if o_t <= pos.stop_price else pos.stop_price
                    reason = "stop_gap_open" if o_t <= pos.stop_price else "stop_intraday_first"
                    slip = smid_one_side_slip_bps(adv_t, cu_t, cpi_t, is_stop=True)
                    cost_u = (sh * exit_px * slip + commission_usd(sh)) * cfg.cost_mult
                    gross_u = sh * (exit_px - p_prev)
                    day_gross_usd += gross_u
                    day_cost_usd += cost_u
                    day_traded_usd += sh * exit_px
                    max_adv_part = max(max_adv_part, (sh * exit_px * (cfg.target_capital / eq_start)) / max(adv_t, 1.0))
                    cash += sh * exit_px - cost_u
                    pos.total_cost_usd += cost_u
                    pos.realized_gross_pnl += sh * (exit_px - pos.entry_price)
                    pos.realized_net_pnl = pos.realized_gross_pnl - pos.total_cost_usd
                    completed_trades.append(_v2_trade_dict(pos, panel.dates[t], exit_px, reason))
                    del positions[j]
                    del prev_mark[j]
                    continue

            # Mark to close c_t
            sh = pos.shares
            day_gross_usd += sh * (c_t - p_prev)
            prev_mark[j] = c_t
            _v2_check_eod_exit(pos, panel, t, cfg)

        # ------------------------------------------------------------------
        # 2. Execute queued entries at open_t
        # ------------------------------------------------------------------
        cands = queued_entries.pop(t, [])
        if cands and len(positions) < cfg.max_positions:
            scored = []
            for j, sig_t in cands:
                if j in positions:
                    continue
                if not panel.valid[t, j] or not panel.pit_active[t, j]:
                    continue
                o_t = panel.open[t, j]
                if not (np.isfinite(o_t) and o_t > 0):
                    continue
                sc = float(score_mat[sig_t, j]) if np.isfinite(score_mat[sig_t, j]) else 0.0
                scored.append((-sc, panel.tickers[j], j, sig_t, o_t))
            scored.sort(key=lambda x: (x[0], x[1]))

            slots = cfg.max_positions - len(positions)
            for _, sym, j, sig_t, entry_px in scored[:slots]:
                adv_sig = float(panel.adv20[sig_t, j])
                cu_sig = float(panel.close_unadj[sig_t, j])
                atr_sig = float(panel.atr14[sig_t, j]) if np.isfinite(panel.atr14[sig_t, j]) else entry_px * 0.04
                r_unit = max(2.0 * atr_sig, entry_px * 0.03)

                # Volatility-adjusted / equal-slot sizing:
                # Risk budget 0.75% of equity per 2*ATR unit, capped at 1 / max_positions (5% notional),
                # 1% of ADV$20 at $100k target capital, and available cash.
                sh_risk = int(math.floor((0.0075 * eq_start) / r_unit))
                sh_slot = int(math.floor(((1.0 / cfg.max_positions) * eq_start) / entry_px))
                scale_e = max(eq_start / cfg.target_capital, 1e-9)
                sh_adv = int(math.floor((0.01 * adv_sig / scale_e) / entry_px))
                slip_in = smid_one_side_slip_bps(adv_sig, cu_sig, cpi_t, is_stop=False)
                sh_cash = int(math.floor((cash - 2.0 * cfg.cost_mult) / (entry_px * (1.0 + (slip_in + 0.001) * cfg.cost_mult))))

                sh = min(sh_risk, sh_slot, sh_adv, sh_cash)
                if sh <= 0:
                    continue

                cost_in = (sh * entry_px * slip_in + commission_usd(sh)) * cfg.cost_mult
                if sh * entry_px + cost_in > cash:
                    continue

                cash -= sh * entry_px + cost_in
                day_cost_usd += cost_in
                day_traded_usd += sh * entry_px
                max_adv_part = max(max_adv_part, (sh * entry_px * (cfg.target_capital / eq_start)) / max(adv_sig, 1.0))

                st_px = (entry_px - r_unit) if cfg.stop_mode == "wide_2atr_stop_first" else np.nan
                pos = V2Position(
                    sym_idx=j,
                    ticker=sym,
                    signal_bar=sig_t,
                    entry_bar=t,
                    entry_price=entry_px,
                    stop_price=st_px,
                    r_unit_price=r_unit,
                    shares=sh,
                    entry_cost_usd=cost_in,
                    total_cost_usd=cost_in,
                    adv_at_entry=adv_sig,
                    mcap_at_entry=float(panel.mcap_real[sig_t, j]),
                )
                positions[j] = pos
                prev_mark[j] = entry_px

                # Evaluate entry bar t itself (conservative stop-first if wide_2atr_stop_first is active)
                l_t = panel.low[t, j]
                c_t = panel.close[t, j]
                cu_t = panel.close_unadj[t, j]
                if cfg.stop_mode == "wide_2atr_stop_first" and np.isfinite(pos.stop_price) and l_t <= pos.stop_price:
                    exit_px = pos.stop_price
                    slip_out = smid_one_side_slip_bps(adv_sig, cu_t, cpi_t, is_stop=True)
                    cost_out = (sh * exit_px * slip_out + commission_usd(sh)) * cfg.cost_mult
                    gross_u = sh * (exit_px - entry_px)
                    day_gross_usd += gross_u
                    day_cost_usd += cost_out
                    day_traded_usd += sh * exit_px
                    cash += sh * exit_px - cost_out
                    pos.total_cost_usd += cost_out
                    pos.realized_gross_pnl += gross_u
                    pos.realized_net_pnl = pos.realized_gross_pnl - pos.total_cost_usd
                    completed_trades.append(_v2_trade_dict(pos, panel.dates[t], exit_px, "stop_entry_bar"))
                    del positions[j]
                    del prev_mark[j]
                    continue

                day_gross_usd += sh * (c_t - entry_px)
                prev_mark[j] = c_t
                _v2_check_eod_exit(pos, panel, t, cfg)

        # ------------------------------------------------------------------
        # 3. Record end-of-day portfolio metrics & queue rebalance entries
        # ------------------------------------------------------------------
        eod_notional = sum(pos.shares * prev_mark[j] for j, pos in positions.items())
        daily_gross[t] = day_gross_usd / eq_start
        daily_cost[t] = day_cost_usd / eq_start
        daily_net[t] = (day_gross_usd - day_cost_usd) / eq_start
        daily_exposure[t] = eod_notional / eq_start
        daily_turnover[t] = day_traded_usd / eq_start
        daily_adv_part_max[t] = max_adv_part

        # Rebalance check every `cfg.reb_days` sessions (plus daily exit checks handled in _v2_check_eod_exit)
        if (t - start_bar) % cfg.reb_days == 0:
            # Also mark held positions whose momentum rank fell below top quintile (< 0.75) for next-open exit
            for j, pos in positions.items():
                if not pos.pending_exit:
                    mp = panel.mom_pct[t, j]
                    if not np.isfinite(mp) or mp < 0.75 or not panel.smid_eligible[t, j]:
                        pos.pending_exit = True
                        pos.pending_reason = "rank_drop_rebalance"

            exec_bar = t + 1 + cfg.extra_delay
            if exec_bar < T:
                sig_cols = np.where(cand_mat[t])[0]
                if len(sig_cols) > 0:
                    queued_entries.setdefault(exec_bar, []).extend((int(j), t) for j in sig_cols)

    win_dates = panel.dates[start_bar:]
    bar_ts = pd.DataFrame(
        {
            "gross": daily_gross[start_bar:],
            "cost": daily_cost[start_bar:],
            "funding": 0.0,
            "net": daily_net[start_bar:],
            "exposure": daily_exposure[start_bar:],
            "turnover": daily_turnover[start_bar:],
            "adv_part": daily_adv_part_max[start_bar:],
        },
        index=pd.DatetimeIndex(win_dates, tz="UTC"),
    )
    cal_idx = pd.date_range(bar_ts.index[0], bar_ts.index[-1], freq="D", tz="UTC")
    bar_cal = bar_ts.reindex(cal_idx).fillna(0.0)
    bar_cal["exposure"] = bar_ts["exposure"].reindex(cal_idx).ffill().fillna(0.0)

    return BTResult(
        bar=bar_cal,
        daily=bar_cal["net"].copy(),
        weights=pd.DataFrame(index=cal_idx),
        meta={
            "trades": pd.DataFrame(completed_trades),
            "trading_bar": bar_ts,
            "config": cfg,
        },
    )


def _v2_check_eod_exit(pos: V2Position, panel: LukV2Panel, t: int, cfg: LukV2Config) -> None:
    """At close of bar t, check if daily close broke below trailing EMA21 (or EMA50 for V4)."""
    if pos.pending_exit:
        return
    j = pos.sym_idx
    c_t = panel.close[t, j]
    trail = panel.ema50[t, j] if cfg.exit_mode == "ema50_or_rank" else panel.ema21[t, j]
    if np.isfinite(c_t) and np.isfinite(trail) and c_t < trail:
        pos.pending_exit = True
        pos.pending_reason = "trail_ema50" if cfg.exit_mode == "ema50_or_rank" else "trail_ema21"


def _v2_trade_dict(pos: V2Position, exit_dt: pd.Timestamp, exit_px: float, reason: str) -> dict:
    init_risk_usd = pos.shares * pos.r_unit_price
    r_net = pos.realized_net_pnl / init_risk_usd if init_risk_usd > 0 else 0.0
    r_gross = pos.realized_gross_pnl / init_risk_usd if init_risk_usd > 0 else 0.0
    cost_r = pos.total_cost_usd / init_risk_usd if init_risk_usd > 0 else 0.0
    return {
        "ticker": pos.ticker,
        "signal_bar": int(pos.signal_bar),
        "entry_bar": int(pos.entry_bar),
        "exit_date": pd.Timestamp(exit_dt),
        "entry_price": float(pos.entry_price),
        "exit_price": float(exit_px),
        "initial_stop": float(pos.stop_price) if np.isfinite(pos.stop_price) else None,
        "initial_shares": int(pos.shares),
        "gross_pnl_usd": float(pos.realized_gross_pnl),
        "net_pnl_usd": float(pos.realized_net_pnl),
        "cost_usd": float(pos.total_cost_usd),
        "r_net": float(r_net),
        "r_gross": float(r_gross),
        "cost_r": float(cost_r),
        "exit_reason": reason,
        "adv_at_entry": float(pos.adv_at_entry),
        "mcap_at_entry": float(pos.mcap_at_entry),
    }


# ---------------------------------------------------------------------------
# 5. Separated 60-minute intraday sub-period execution test (~730d Yahoo 60m)
# ---------------------------------------------------------------------------

def parse_yahoo_60m_chart(raw_json: dict) -> pd.DataFrame | None:
    """Parse Yahoo v8 60m chart JSON into a DataFrame indexed by US/Eastern timestamp,
    including trading `date` and hourly bar index within each session (7 regular-session bars/day = 28 price points/day)."""
    try:
        res = raw_json["chart"]["result"][0]
        ts = res.get("timestamp")
        if not ts:
            return None
        q = res["indicators"]["quote"][0]
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York")
        df = pd.DataFrame(
            {
                "open": pd.to_numeric(q.get("open"), errors="coerce"),
                "high": pd.to_numeric(q.get("high"), errors="coerce"),
                "low": pd.to_numeric(q.get("low"), errors="coerce"),
                "close": pd.to_numeric(q.get("close"), errors="coerce"),
                "volume": pd.to_numeric(q.get("volume"), errors="coerce").fillna(0.0),
            },
            index=idx,
        )
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
        # Keep regular trading hours 09:30 .. 16:00 ET
        hm = df.index.hour * 60 + df.index.minute
        df = df[(hm >= 9 * 60 + 30) & (hm < 16 * 60)]
        if len(df) == 0:
            return None
        df["session_date"] = pd.to_datetime(df.index.date)
        return df.sort_index()
    except Exception:
        return None


def run_intraday_subperiod_comparison(
    panel: LukV2Panel,
    hourly_dfs: dict[str, pd.DataFrame],
    start_dt: str = "2023-11-15",
    end_dt: str = "2025-09-30",
) -> dict:
    """Clearly separated intraday sub-period test (`2023-11-15 .. 2025-09-30`):
    Compares on the exact same small/mid-cap candidates and date window:
      1) `intraday_60m_flush_entry_lod_stop`: uses 60-minute intraday bars (7 bars = 28 OHLC price points/day)
         to enter during the session when an hourly bar touches the rising daily EMA21 and turns green
         (`close_h > open_h`), placing the initial stop at the running low-of-day (`LOD * 0.999`) and checking
         the stop hour-by-hour in chronological order (eliminating daily OHLC intra-bar ordering ambiguity).
      2) `daily_bar_same_window_with_stop`: daily OHLC next-open entry with flush-low stop on the same window.
      3) `daily_bar_same_window_no_stop`: `V0` (no hard stop, EMA21 trail) on the same window."""
    t0_ts = pd.Timestamp(start_dt)
    t1_ts = pd.Timestamp(end_dt)

    # Coverage stats of 60m data
    n_syms_with_60m = len(hourly_dfs)
    bars_per_day_counts: list[int] = []
    for df_h in hourly_dfs.values():
        sub = df_h[(df_h["session_date"] >= t0_ts) & (df_h["session_date"] <= t1_ts)]
        if len(sub) > 0:
            bars_per_day_counts.extend(sub.groupby("session_date").size().tolist())

    avg_bars_per_day = float(np.mean(bars_per_day_counts)) if bars_per_day_counts else 0.0
    pct_days_ge_4_bars = float(np.mean(np.array(bars_per_day_counts) >= 4)) if bars_per_day_counts else 0.0

    # Simulate hourly flush entry + chronological hourly stop/trail over [start_dt, end_dt]
    sym_to_j = {s: j for j, s in enumerate(panel.tickers)}
    cand_mat, score_mat = candidate_mask_and_score(panel, LukV2Config(id="V0"))

    w_mask = (panel.dates >= t0_ts) & (panel.dates <= t1_ts)
    w_indices = np.where(w_mask)[0]
    if len(w_indices) == 0 or not hourly_dfs:
        return {"status": "no_intraday_bars"}

    # Build per-date lookup of hourly bars for symbols in hourly_dfs
    hourly_by_sym_date: dict[tuple[int, pd.Timestamp], pd.DataFrame] = {}
    for sym, df_h in hourly_dfs.items():
        if sym not in sym_to_j:
            continue
        j = sym_to_j[sym]
        sub = df_h[(df_h["session_date"] >= t0_ts) & (df_h["session_date"] <= t1_ts)]
        for dt_val, grp in sub.groupby("session_date"):
            hourly_by_sym_date[(j, pd.Timestamp(dt_val))] = grp

    cash = 100_000.0
    positions: dict[int, dict] = {}
    daily_rets = []
    daily_dates = []
    intraday_trades = []

    for t in w_indices:
        dt = panel.dates[t]
        eq_start = cash + sum(p["shares"] * p["mark"] for p in positions.values())
        day_pnl = 0.0
        cpi_t = float(panel.cpi_ratio[t])

        # 1. Step existing positions hour-by-hour on day t
        for j in list(positions.keys()):
            pos = positions[j]
            p_prev = pos["mark"]
            adv_t = float(panel.adv20[t, j]) if np.isfinite(panel.adv20[t, j]) and panel.adv20[t, j] > 0 else pos["adv"]
            cu_t = float(panel.close_unadj[t, j]) if np.isfinite(panel.close_unadj[t, j]) else p_prev
            h_bars = hourly_by_sym_date.get((j, dt))

            exited = False
            if h_bars is not None and len(h_bars) > 0:
                for _, hb in h_bars.iterrows():
                    o_h, l_h, c_h = float(hb["open"]), float(hb["low"]), float(hb["close"])
                    if pos.get("pending_trail"):
                        exit_px = o_h
                        slip = smid_one_side_slip_bps(adv_t, cu_t, cpi_t, is_stop=False)
                        cost_u = pos["shares"] * exit_px * slip + commission_usd(pos["shares"])
                        day_pnl += pos["shares"] * (exit_px - p_prev) - cost_u
                        cash += pos["shares"] * exit_px - cost_u
                        intraday_trades.append({
                            "ticker": panel.tickers[j],
                            "pnl": pos["shares"] * (exit_px - pos["entry_px"]) - pos["cost_in"] - cost_u,
                            "r_net": (pos["shares"] * (exit_px - pos["entry_px"]) - pos["cost_in"] - cost_u) / max(pos["init_risk"], 1.0),
                            "reason": "trail_ema21_60m",
                        })
                        del positions[j]
                        exited = True
                        break
                    if o_h <= pos["stop"] or l_h <= pos["stop"]:
                        exit_px = o_h if o_h <= pos["stop"] else pos["stop"]
                        slip = smid_one_side_slip_bps(adv_t, cu_t, cpi_t, is_stop=True)
                        cost_u = pos["shares"] * exit_px * slip + commission_usd(pos["shares"])
                        day_pnl += pos["shares"] * (exit_px - p_prev) - cost_u
                        cash += pos["shares"] * exit_px - cost_u
                        intraday_trades.append({
                            "ticker": panel.tickers[j],
                            "pnl": pos["shares"] * (exit_px - pos["entry_px"]) - pos["cost_in"] - cost_u,
                            "r_net": (pos["shares"] * (exit_px - pos["entry_px"]) - pos["cost_in"] - cost_u) / max(pos["init_risk"], 1.0),
                            "reason": "lod_stop_60m",
                        })
                        del positions[j]
                        exited = True
                        break
            elif panel.valid[t, j]:
                # Fallback to daily bar if 60m missing on this day
                o_d, l_d = float(panel.open[t, j]), float(panel.low[t, j])
                if o_d <= pos["stop"] or l_d <= pos["stop"]:
                    exit_px = o_d if o_d <= pos["stop"] else pos["stop"]
                    slip = smid_one_side_slip_bps(adv_t, cu_t, cpi_t, is_stop=True)
                    cost_u = pos["shares"] * exit_px * slip + commission_usd(pos["shares"])
                    day_pnl += pos["shares"] * (exit_px - p_prev) - cost_u
                    cash += pos["shares"] * exit_px - cost_u
                    intraday_trades.append({
                        "ticker": panel.tickers[j],
                        "pnl": pos["shares"] * (exit_px - pos["entry_px"]) - pos["cost_in"] - cost_u,
                        "r_net": (pos["shares"] * (exit_px - pos["entry_px"]) - pos["cost_in"] - cost_u) / max(pos["init_risk"], 1.0),
                        "reason": "lod_stop_daily_fb",
                    })
                    del positions[j]
                    exited = True

            if not exited and panel.valid[t, j]:
                c_d = float(panel.close[t, j])
                day_pnl += pos["shares"] * (c_d - p_prev)
                pos["mark"] = c_d
                if np.isfinite(panel.ema21[t, j]) and c_d < float(panel.ema21[t, j]):
                    pos["pending_trail"] = True

        # 2. Check for new intraday 60m flush-and-turn entries among yesterday's eligible leaders (t-1)
        if t - 1 >= 252 and len(positions) < 15 and panel.mkt_ok[t - 1]:
            el_prev = np.where(
                panel.smid_eligible[t - 1]
                & panel.trend_ok[t - 1]
                & (panel.mom_pct[t - 1] >= 0.85)
                & (panel.turnover_pct[t - 1] <= (2.0 / 3.0))
            )[0]
            # Sort by momentum score desc
            el_prev = sorted(el_prev, key=lambda idx: -float(panel.mom_score[t - 1, idx]))
            for j in el_prev:
                if len(positions) >= 15 or j in positions:
                    continue
                h_bars = hourly_by_sym_date.get((j, dt))
                if h_bars is None or len(h_bars) < 4:
                    continue
                ema21_prev = float(panel.ema21[t - 1, j])
                adr_prev = float(panel.adr20[t - 1, j])
                if not (np.isfinite(ema21_prev) and np.isfinite(adr_prev)):
                    continue
                running_lod = float("inf")
                for b_num, (_, hb) in enumerate(h_bars.iterrows()):
                    o_h, h_h, l_h, c_h = float(hb["open"]), float(hb["high"]), float(hb["low"]), float(hb["close"])
                    running_lod = min(running_lod, l_h)
                    # Hourly flush near EMA21 + green hourly turn (enter at close of hourly bar c_h)
                    if b_num >= 1 and running_lod <= ema21_prev * (1.0 + 0.35 * adr_prev) and c_h > o_h and c_h > ema21_prev:
                        stop_px = running_lod * 0.999
                        r_sh = c_h - stop_px
                        if r_sh <= 0 or (r_sh / c_h) > min(0.6 * adr_prev, 0.05):
                            continue
                        sh = min(
                            int(math.floor((0.005 * eq_start) / r_sh)),
                            int(math.floor((0.10 * eq_start) / c_h)),
                            int(math.floor((cash - 5.0) / (c_h * 1.005))),
                        )
                        if sh <= 0:
                            break
                        adv_sig = float(panel.adv20[t - 1, j])
                        cu_sig = float(panel.close_unadj[t - 1, j])
                        slip_in = smid_one_side_slip_bps(adv_sig, cu_sig, cpi_t, is_stop=False)
                        cost_in = sh * c_h * slip_in + commission_usd(sh)
                        cash -= sh * c_h + cost_in
                        # Check remaining hourly bars in the same session for immediate LOD stop!
                        rem_bars = h_bars.iloc[b_num + 1 :]
                        stopped_same_day = False
                        for _, rb in rem_bars.iterrows():
                            if float(rb["low"]) <= stop_px:
                                exit_px = min( float(rb["open"]), stop_px) if float(rb["open"]) <= stop_px else stop_px
                                slip_out = smid_one_side_slip_bps(adv_sig, cu_sig, cpi_t, is_stop=True)
                                cost_out = sh * exit_px * slip_out + commission_usd(sh)
                                day_pnl += sh * (exit_px - c_h) - cost_in - cost_out
                                cash += sh * exit_px - cost_out
                                intraday_trades.append({
                                    "ticker": panel.tickers[j],
                                    "pnl": sh * (exit_px - c_h) - cost_in - cost_out,
                                    "r_net": (sh * (exit_px - c_h) - cost_in - cost_out) / (sh * r_sh),
                                    "reason": "lod_stop_same_day_60m",
                                })
                                stopped_same_day = True
                                break
                        if not stopped_same_day:
                            c_eod = float(h_bars.iloc[-1]["close"])
                            day_pnl += sh * (c_eod - c_h) - cost_in
                            positions[j] = {
                                "shares": sh,
                                "entry_px": c_h,
                                "stop": stop_px,
                                "init_risk": sh * r_sh,
                                "cost_in": cost_in,
                                "mark": c_eod,
                                "adv": adv_sig,
                                "pending_trail": False,
                            }
                        break

        daily_rets.append(day_pnl / max(eq_start, 1.0))
        daily_dates.append(dt)

    s_intra = pd.Series(daily_rets, index=pd.DatetimeIndex(daily_dates, tz="UTC"))
    cal_idx = pd.date_range(s_intra.index[0], s_intra.index[-1], freq="D", tz="UTC")
    s_intra_cal = s_intra.reindex(cal_idx).fillna(0.0)
    m_intra = metrics(s_intra_cal)

    # Compare with V0 (no hard stop) and V6 (daily bar stop-first) on the exact same calendar window
    r_v0 = simulate_luk_v2(panel, LukV2Config(id="V0"))
    r_v6 = simulate_luk_v2(panel, LukV2Config(id="V6", stop_mode="wide_2atr_stop_first"))
    sub_v0 = r_v0.daily.loc[(r_v0.daily.index >= cal_idx[0]) & (r_v0.daily.index <= cal_idx[-1])]
    sub_v6 = r_v6.daily.loc[(r_v6.daily.index >= cal_idx[0]) & (r_v6.daily.index <= cal_idx[-1])]

    wins_i = [x["r_net"] for x in intraday_trades if x["r_net"] > 0]
    loss_i = [x["r_net"] for x in intraday_trades if x["r_net"] <= 0]

    return {
        "window": [start_dt, end_dt],
        "n_symbols_with_60m_bars": n_syms_with_60m,
        "avg_regular_bars_per_day": round(avg_bars_per_day, 2),
        "avg_price_points_per_day": round(avg_bars_per_day * 4.0, 2),
        "pct_stock_days_ge_4_bars": round(pct_days_ge_4_bars, 4),
        "intraday_60m_flush_lod_stop": {
            "sharpe": round(m_intra["sharpe"], 3),
            "ann_ret": round(m_intra["ann_ret"], 4),
            "max_dd": round(m_intra["max_dd"], 4),
            "total_ret": round(m_intra["total_ret"], 4),
            "n_trades": len(intraday_trades),
            "win_rate": round(len(wins_i) / max(len(intraday_trades), 1), 4),
            "payoff": round((np.mean(wins_i) / abs(np.mean(loss_i))) if wins_i and loss_i else 0.0, 3),
            "expectancy_r": round(float(np.mean([x["r_net"] for x in intraday_trades])) if intraday_trades else 0.0, 4),
        },
        "daily_v6_stop_first_same_window": {
            k: round(v, 4) for k, v in metrics(sub_v6).items() if k in ("sharpe", "ann_ret", "max_dd", "total_ret")
        },
        "daily_v0_no_stop_same_window": {
            k: round(v, 4) for k, v in metrics(sub_v0).items() if k in ("sharpe", "ann_ret", "max_dd", "total_ret")
        },
    }
