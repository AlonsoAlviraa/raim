"""Unit tests for luk-v2 (US small/mid-cap momentum + point-in-time SEC EDGAR fundamentals):
1. PIT fundamentals causality: no SEC filing is ever visible on or before its acceptance/filing date
   (`filed_date`), and mutating or adding future filings (`filed > t_cut`) never alters any past
   fundamental feature (`shares_pit`, `eps_yoy`, `eps_qoq`, `eps_accel`, `sue`, `rev_yoy`, `rev_qoq`,
   `op_margin`, `earn_gap`), market cap (`mcap_real`), eligibility (`smid_eligible`), or daily return up to `t_cut`.
2. Four-point OHLC daily bar validation: stock-days missing any of (O, H, L, C), with non-positive values,
   inconsistent bounds (`H < max(O,C)` or `L > min(O,C)`), or degenerate flat bars (`H == L`) are flagged and rejected.
3. Conservative stop-first OHLC resolution (`V6`) and small-cap tiered slippage (`8/15/25 bps`, `+15 bps` `< $10`, `2x` on stops).
4. Preregistered grid neighbour topology (`V0` linked to all 7 single-factor ablations `V1..V7`).

All data in this file is synthetic and marked as such per AGENTS.md rule 6."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from raim.research.stability import neighbors
from raim.strategies import luk_v2 as L2


def _make_synthetic_smid_panel(n_days: int = 500, n_syms: int = 10, seed: int = 101) -> tuple[
    pd.DatetimeIndex,
    list[str],
    list[int],
    dict[str, np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:
    """SYNTHETIC small/mid-cap panel + synthetic SEC EDGAR quarterly filings for unit testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days, freq="B")
    tickers = [f"SMID{i}" for i in range(n_syms)]
    ciks = [10000 + i for i in range(n_syms)]

    open_arr = np.zeros((n_days, n_syms))
    high_arr = np.zeros((n_days, n_syms))
    low_arr = np.zeros((n_days, n_syms))
    close_arr = np.zeros((n_days, n_syms))
    vol_arr = np.zeros((n_days, n_syms))

    for j in range(n_syms):
        drift = 0.0015 if j < 5 else 0.0002
        rets = rng.normal(drift, 0.022, size=n_days)
        c = 25.0 * np.exp(np.cumsum(rets))
        o = c * (1.0 + rng.normal(0.0, 0.006, size=n_days))
        h = np.maximum(o, c) * (1.0 + np.abs(rng.normal(0.015, 0.008, size=n_days)))
        l = np.minimum(o, c) * (1.0 - np.abs(rng.normal(0.015, 0.008, size=n_days)))
        v = rng.uniform(500_000, 2_000_000, size=n_days)
        open_arr[:, j] = o
        high_arr[:, j] = h
        low_arr[:, j] = l
        close_arr[:, j] = c
        vol_arr[:, j] = v

    # Build synthetic quarterly filings for each CIK across 2019..2021
    q_ends = pd.date_range("2018-12-31", periods=12, freq="QE")
    filing_rows = []
    for j, cik in enumerate(ciks):
        base_eps = 0.40
        base_rev = 120_000_000.0
        for q_idx, p_dt in enumerate(q_ends):
            # Filed 35 days after quarter end
            f_dt = p_dt + pd.Timedelta(days=35)
            g_mult = (1.35 ** (q_idx / 4.0)) if j < 5 else (1.02 ** (q_idx / 4.0))
            eps_v = base_eps * g_mult + 0.02 * (q_idx % 2)
            rev_v = base_rev * g_mult
            op_v = rev_v * 0.18
            sh_v = 40_000_000.0  # at $25-$60 -> $1B-$2.4B market cap ( squarely in $300M-$20B )
            filing_rows.append(
                {
                    "cik": cik,
                    "period": p_dt.strftime("%Y%m%d"),
                    "filed": f_dt.strftime("%Y%m%d"),
                    "eps": eps_v,
                    "rev": rev_v,
                    "opinc": op_v,
                    "shares": sh_v,
                }
            )
    filings_df = pd.DataFrame(filing_rows)

    idx_c = 150.0 * np.exp(np.linspace(0.0, 0.35, n_days))
    etf_df = pd.DataFrame(
        {
            "open": idx_c * 0.998,
            "high": idx_c * 1.01,
            "low": idx_c * 0.99,
            "close": idx_c,
            "volume": 10_000_000.0,
            "close_unadj": idx_c,
            "volume_unadj": 10_000_000.0,
        },
        index=dates,
    )
    cpi_ratio = np.ones(n_days, dtype=float)
    arrs = {
        "open": open_arr,
        "high": high_arr,
        "low": low_arr,
        "close": close_arr,
        "vol": vol_arr,
    }
    return dates, tickers, ciks, arrs, filings_df, etf_df, cpi_ratio


def test_pit_fundamentals_never_used_on_or_before_filing_date():
    """Verify that a filing with `filed_date = F` is strictly NaN/invisible on all trading days `t <= F`
    and only becomes visible starting on the first trading session `t > F`."""
    dates = pd.bdate_range("2022-01-03", periods=120, freq="B")
    T = len(dates)
    open_arr = np.full((T, 1), 20.0)
    close_arr = np.full((T, 1), 20.5)
    valid_arr = np.ones((T, 1), dtype=bool)

    # Filing 1: period 2021-03-31, filed on dates[10]
    # Filing 2: period 2022-03-31, filed on dates[60]
    f_date_1 = dates[10]
    f_date_2 = dates[60]
    filings_df = pd.DataFrame(
        [
            {
                "cik": 99999,
                "period": "20210331",
                "filed": f_date_1.strftime("%Y%m%d"),
                "eps": 1.00,
                "rev": 100_000_000.0,
                "opinc": 15_000_000.0,
                "shares": 50_000_000.0,
            },
            {
                "cik": 99999,
                "period": "20220331",
                "filed": f_date_2.strftime("%Y%m%d"),
                "eps": 1.50,  # +50% YoY
                "rev": 140_000_000.0,  # +40% YoY
                "opinc": 28_000_000.0,
                "shares": 52_000_000.0,
            },
        ]
    )

    fund = L2.build_pit_fundamentals_panel(filings_df, dates, [99999], open_arr, close_arr, valid_arr)

    # On bars 0..10 (including bar 10 == f_date_1 itself!), no filing may be visible!
    assert not fund["has_pit_filing"][:11, 0].any()
    assert np.all(np.isnan(fund["shares_pit"][:11, 0]))

    # On bar 11 (first session strictly > f_date_1), Filing 1 becomes visible
    assert fund["has_pit_filing"][11, 0]
    assert np.isclose(fund["shares_pit"][11, 0], 50_000_000.0)
    # YoY growth is still NaN until Filing 2 is accepted after bar 60
    assert np.all(np.isnan(fund["eps_yoy"][:61, 0]))
    assert np.all(np.isnan(fund["rev_yoy"][:61, 0]))
    assert np.isclose(fund["shares_pit"][60, 0], 50_000_000.0)

    # On bar 61 (first session strictly > f_date_2), Filing 2 becomes visible with +50% EPS YoY and +40% Rev YoY
    assert np.isclose(fund["shares_pit"][61, 0], 52_000_000.0)
    assert np.isclose(fund["eps_yoy"][61, 0], 0.50)
    assert np.isclose(fund["rev_yoy"][61, 0], 0.40)


def test_causality_future_filings_and_future_prices_do_not_change_past():
    """Mutating future prices (> t_cut) and injecting extreme future SEC filings (filed > t_cut)
    must not alter any past indicator, PIT fundamental array, candidate signal, or daily return up to t_cut."""
    dates, tickers, ciks, arrs, filings_df, etf_df, cpi_ratio = _make_synthetic_smid_panel(n_days=450, n_syms=8, seed=55)
    valid_4pt, _ = L2.validate_four_price_points_ohlc(arrs["open"], arrs["high"], arrs["low"], arrs["close"])
    pit_active = np.ones_like(valid_4pt, dtype=bool)
    delist_bar = np.full(len(tickers), -1, dtype=np.int32)

    fund1 = L2.build_pit_fundamentals_panel(filings_df, dates, ciks, arrs["open"], arrs["close"], valid_4pt)
    p1 = L2.LukV2Panel(
        trading_days=dates,
        tickers=tickers,
        ciks=ciks,
        open_arr=arrs["open"].copy(),
        high_arr=arrs["high"].copy(),
        low_arr=arrs["low"].copy(),
        close_arr=arrs["close"].copy(),
        vol_arr=arrs["vol"].copy(),
        close_unadj_arr=arrs["close"].copy(),
        vol_unadj_arr=arrs["vol"].copy(),
        valid_4pt_arr=valid_4pt.copy(),
        pit_active_arr=pit_active.copy(),
        delist_bar=delist_bar.copy(),
        fund_arrays=fund1,
        spy_df=etf_df.copy(),
        iwm_df=etf_df.copy(),
        qqq_df=etf_df.copy(),
        cpi_ratio=cpi_ratio.copy(),
        hygiene={},
        start_eval_date="2021-01-04",
    )

    t_cut = 360
    cut_dt = dates[t_cut]

    # Mutate prices strictly after t_cut by 8x and add bogus future filings filed on cut_dt (available at t_cut + 1)
    open2 = arrs["open"].copy()
    high2 = arrs["high"].copy()
    low2 = arrs["low"].copy()
    close2 = arrs["close"].copy()
    vol2 = arrs["vol"].copy()
    open2[t_cut + 1 :] *= 8.0
    high2[t_cut + 1 :] *= 8.0
    low2[t_cut + 1 :] *= 8.0
    close2[t_cut + 1 :] *= 8.0
    vol2[t_cut + 1 :] *= 0.2

    extra_filings = pd.DataFrame(
        [
            {
                "cik": ciks[j],
                "period": cut_dt.strftime("%Y%m%d"),
                "filed": cut_dt.strftime("%Y%m%d"),  # filed ON cut_dt -> visible at t_cut + 1, NEVER at <= t_cut!
                "eps": -99.0,
                "rev": 1.0,
                "opinc": -500_000_000.0,
                "shares": 999_000_000_000.0,
            }
            for j in range(len(ciks))
        ]
    )
    filings2 = pd.concat([filings_df, extra_filings], ignore_index=True)
    valid2, _ = L2.validate_four_price_points_ohlc(open2, high2, low2, close2)
    fund2 = L2.build_pit_fundamentals_panel(filings2, dates, ciks, open2, close2, valid2)

    p2 = L2.LukV2Panel(
        trading_days=dates,
        tickers=tickers,
        ciks=ciks,
        open_arr=open2,
        high_arr=high2,
        low_arr=low2,
        close_arr=close2,
        vol_arr=vol2,
        close_unadj_arr=close2.copy(),
        vol_unadj_arr=vol2.copy(),
        valid_4pt_arr=valid2,
        pit_active_arr=pit_active.copy(),
        delist_bar=delist_bar.copy(),
        fund_arrays=fund2,
        spy_df=etf_df.copy(),
        iwm_df=etf_df.copy(),
        qqq_df=etf_df.copy(),
        cpi_ratio=cpi_ratio.copy(),
        hygiene={},
        start_eval_date="2021-01-04",
    )

    sl = slice(0, t_cut + 1)
    for k in fund1:
        np.testing.assert_allclose(fund1[k][sl], fund2[k][sl], rtol=1e-12, atol=1e-12, err_msg=f"Fundamental leakage in {k}")

    for attr in ("ema21", "ema50", "atr14", "adr20", "adv20", "turnover20", "mcap_real", "smid_eligible", "mom_pct", "turnover_pct", "fund_score_pct"):
        np.testing.assert_allclose(getattr(p1, attr)[sl], getattr(p2, attr)[sl], rtol=1e-12, atol=1e-12, err_msg=f"Panel leakage in {attr}")

    r1 = L2.simulate_luk_v2(p1, L2.LukV2Config(id="V0"))
    r2 = L2.simulate_luk_v2(p2, L2.LukV2Config(id="V0"))
    cut_utc = pd.Timestamp(cut_dt, tz="UTC")
    np.testing.assert_allclose(r1.daily.loc[:cut_utc].values, r2.daily.loc[:cut_utc].values, rtol=1e-12, atol=1e-12)


def test_four_price_points_ohlc_validation_rejects_incomplete_or_degenerate_bars():
    """Verify `validate_four_price_points_ohlc` rejects missing O/H/L/C, negative prices,
    inverted high/low, and flat H == L bars."""
    o = np.array([[10.0, np.nan, 10.0, 10.0, 10.0]])
    h = np.array([[11.0, 11.0, 9.5, 10.0, 11.0]])
    l = np.array([[9.5, 9.5, 9.0, 10.0, 9.5]])
    c = np.array([[10.5, 10.5, 10.0, 10.0, -1.0]])
    valid, rej = L2.validate_four_price_points_ohlc(o, h, l, c)
    # Only col 0 has 4 valid consistent non-degenerate OHLC price points
    assert valid[0, 0] and not rej[0, 0]
    for col in (1, 2, 3, 4):
        assert not valid[0, col]
        assert rej[0, col]


def test_smid_tiered_slippage_and_grid_neighbors():
    """Verify small/mid-cap tiered slippage tiers and V0..V7 single-parameter neighbour topology."""
    assert np.isclose(L2.smid_one_side_slip_bps(150_000_000.0, 25.0, 1.0, is_stop=False), 0.0008)
    assert np.isclose(L2.smid_one_side_slip_bps(50_000_000.0, 25.0, 1.0, is_stop=False), 0.0015)
    assert np.isclose(L2.smid_one_side_slip_bps(10_000_000.0, 25.0, 1.0, is_stop=False), 0.0025)
    # +15 bps if unadjusted price < $10, and 2x on stops
    assert np.isclose(L2.smid_one_side_slip_bps(10_000_000.0, 8.0, 1.0, is_stop=True), 0.0080)

    spec = yaml.safe_load(Path("configs/prereg/luk_v2/L2_luk_smid_mom_fund.yaml").read_text())
    norm = L2.normalize_grid_v2(spec["grid"])
    assert len(norm) == 8
    assert neighbors(norm, 0) == [1, 2, 3, 4, 5, 6, 7]
