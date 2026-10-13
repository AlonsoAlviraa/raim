"""Causality, execution-model, sizing-cap, and point-in-time universe tests for luk-v1
(Martin Luk pullback strategy, spec docs/research/luk_lit_2026-10.md section 3.12).

All data in this file is synthetic and marked as such per AGENTS.md rule 6."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from raim.research.stability import neighbors
from raim.strategies import luk as L


def _make_synthetic_panel(n_days: int = 700, n_syms: int = 8, seed: int = 42) -> tuple[
    list[tuple[pd.Timestamp, tuple[str, ...]]],
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """SYNTHETIC panel generator for unit testing causality, sizing, and execution."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days, freq="B")
    syms = [f"SYN{i}" for i in range(n_syms)]

    # PIT snapshots: SYN0..SYN5 present from start; SYN6 added at bar 400; SYN7 removed at bar 450
    base_members = tuple(syms[:6] + ["SYN7"])
    snap0 = (dates[0] - pd.Timedelta(days=10), base_members)
    snap1 = (dates[400], tuple(syms[:7] + ["SYN7"]))
    snap2 = (dates[450], tuple(syms[:7]))
    snapshots = [snap0, snap1, snap2]

    stock_dfs: dict[str, pd.DataFrame] = {}
    for i, s in enumerate(syms):
        rets = rng.normal(0.0012, 0.025, size=n_days)
        close = 50.0 * np.exp(np.cumsum(rets))
        open_ = close * (1.0 + rng.normal(0.0, 0.008, size=n_days))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.018, 0.01, size=n_days)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.018, 0.01, size=n_days)))
        vol = rng.uniform(1_500_000, 4_000_000, size=n_days)
        stock_dfs[s] = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
                "close_unadj": close,
                "volume_unadj": vol,
            },
            index=dates,
        )

    # Inject a deterministic textbook pullback + flush setup on SYN0 at bar 500
    t_sig = 500
    c_rally = np.linspace(40.0, 90.0, 260)
    idx_slice = dates[t_sig - 259 : t_sig + 1]
    stock_dfs["SYN0"].loc[idx_slice, "close"] = c_rally
    stock_dfs["SYN0"].loc[idx_slice, "close_unadj"] = c_rally
    stock_dfs["SYN0"].loc[idx_slice, "open"] = c_rally * 0.995
    stock_dfs["SYN0"].loc[idx_slice, "high"] = c_rally * 1.025
    stock_dfs["SYN0"].loc[idx_slice, "low"] = c_rally * 0.975
    stock_dfs["SYN0"].loc[idx_slice, "volume"] = 5_000_000.0
    # Pullback at t_sig - 1 and flush+green turn touching EMA9/21 at t_sig
    e9_approx = float(stock_dfs["SYN0"]["close"].iloc[t_sig - 10 : t_sig - 1].mean())
    stock_dfs["SYN0"].iloc[t_sig - 3, stock_dfs["SYN0"].columns.get_loc("close")] = e9_approx * 1.08
    stock_dfs["SYN0"].iloc[t_sig - 3, stock_dfs["SYN0"].columns.get_loc("high")] = e9_approx * 1.10
    stock_dfs["SYN0"].iloc[t_sig - 1, stock_dfs["SYN0"].columns.get_loc("close")] = e9_approx * 1.01
    stock_dfs["SYN0"].iloc[t_sig - 1, stock_dfs["SYN0"].columns.get_loc("low")] = e9_approx * 0.995
    # Flush day t_sig: open above close_{t-1}, low flushes below close_{t-1} and touches EMA9, closes green near high
    stock_dfs["SYN0"].iloc[t_sig, stock_dfs["SYN0"].columns.get_loc("open")] = e9_approx * 1.012
    stock_dfs["SYN0"].iloc[t_sig, stock_dfs["SYN0"].columns.get_loc("low")] = e9_approx * 0.992
    stock_dfs["SYN0"].iloc[t_sig, stock_dfs["SYN0"].columns.get_loc("high")] = e9_approx * 1.030
    stock_dfs["SYN0"].iloc[t_sig, stock_dfs["SYN0"].columns.get_loc("close")] = e9_approx * 1.025
    stock_dfs["SYN0"].iloc[t_sig + 1, stock_dfs["SYN0"].columns.get_loc("open")] = e9_approx * 1.026

    # QQQ & SPY in steady uptrends so market filter is active
    qqq_c = 200.0 * np.exp(np.linspace(0.0, 0.65, n_days))
    qqq_df = pd.DataFrame(
        {
            "open": qqq_c * 0.998,
            "high": qqq_c * 1.01,
            "low": qqq_c * 0.99,
            "close": qqq_c,
            "volume": 20_000_000.0,
            "close_unadj": qqq_c,
            "volume_unadj": 20_000_000.0,
        },
        index=dates,
    )
    spy_df = qqq_df.copy()

    cpi_dates = pd.date_range("2017-01-01", "2026-10-01", freq="MS")
    cpi_df = pd.DataFrame(
        {"date": cpi_dates, "value": np.linspace(245.0, 315.0, len(cpi_dates))}
    )
    return snapshots, stock_dfs, qqq_df, spy_df, cpi_df


def test_causality_indicators_signals_and_trades_unchanged_by_future():
    """Section 3.12 (1): Truncating or mutating future bars (> t_cut) by 10x must not alter
    any indicator, AVWAP anchor, RS rank, breadth, CPI ratio, signal, or trade up to t_cut."""
    snapshots, stock_dfs, qqq_df, spy_df, cpi_df = _make_synthetic_panel(n_days=650, seed=11)
    p_full = L.build_panel_from_dfs(
        snapshots, stock_dfs, qqq_df, spy_df, cpi_df, start_date="2018-01-02", end_date="2026-10-01"
    )

    t_cut = 540
    cut_dt = p_full.dates[t_cut]

    # Create a mutated copy where all prices and volumes strictly after t_cut are multiplied by 10x (or crashed by 90%)
    mut_stocks = {}
    for k, df in stock_dfs.items():
        d2 = df.copy()
        after = d2.index > cut_dt
        d2.loc[after, ["open", "high", "low", "close", "close_unadj"]] *= 10.0
        d2.loc[after, ["volume", "volume_unadj"]] *= 0.1
        mut_stocks[k] = d2

    qqq_mut = qqq_df.copy()
    qqq_mut.loc[qqq_mut.index > cut_dt, ["open", "high", "low", "close"]] *= 0.2
    spy_mut = spy_df.copy()
    spy_mut.loc[spy_mut.index > cut_dt, ["open", "high", "low", "close"]] *= 0.2

    p_mut = L.build_panel_from_dfs(
        snapshots, mut_stocks, qqq_mut, spy_mut, cpi_df, start_date="2018-01-02", end_date="2026-10-01"
    )

    sl = slice(0, t_cut + 1)
    for attr in (
        "ema9", "ema21", "ema50", "adr20", "atr14", "adv20",
        "avwap", "avwap_anchor", "rs_c", "rs_c_pct", "rs_52w", "rs_52w_pct",
        "base_eligible", "trend_ok", "pullback_trigger", "avwap_ok",
        "qqq_filter_ok", "qqq_vol_scale", "breadth", "breadth_med252", "cpi_ratio",
    ):
        a = getattr(p_full, attr)[sl]
        b = getattr(p_mut, attr)[sl]
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, err_msg=f"Leakage in {attr}")

    for variant in ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"):
        cfg = L.LukConfig(id=variant)
        s1, rk1 = L.signal_mask_for_config(p_full, cfg)
        s2, rk2 = L.signal_mask_for_config(p_mut, cfg)
        np.testing.assert_array_equal(s1[sl], s2[sl])
        np.testing.assert_allclose(rk1[sl], rk2[sl], rtol=1e-12, atol=1e-12)

    # Daily returns up to cut_dt must match identically
    r1 = L.simulate_luk(p_full, L.LukConfig(id="V0"))
    r2 = L.simulate_luk(p_mut, L.LukConfig(id="V0"))
    cut_utc = pd.Timestamp(cut_dt, tz="UTC")
    np.testing.assert_allclose(
        r1.daily.loc[:cut_utc].values,
        r2.daily.loc[:cut_utc].values,
        rtol=1e-12,
        atol=1e-12,
    )


def test_execution_gap_below_stop_and_same_bar_stop_vs_target():
    """Section 3.12 (2):
    - Gap below stop at open fills at open (not stop).
    - If both stop and partial target 3R are touched in the same bar: stop goes first unless open >= 3R.
    - Stop on entry bar t+1 itself triggers intraday at stop.
    - Entry cancelled if open_{t+1} <= stop or if (entry - stop)/entry > cap.
    - Stop never decreases after moving to breakeven."""
    snapshots, stock_dfs, qqq_df, spy_df, cpi_df = _make_synthetic_panel(n_days=600, seed=7)
    panel = L.build_panel_from_dfs(
        snapshots, stock_dfs, qqq_df, spy_df, cpi_df, start_date="2018-01-02", end_date="2026-10-01"
    )

    T, N = panel.T, panel.N
    sig = np.zeros((T, N), dtype=bool)
    t_sig = 420
    j = 0  # SYN0
    sig[t_sig, j] = True

    # Setup clean deterministic prices around t_sig..t_sig+5
    panel.adr20[t_sig, j] = 0.06  # 6% ADR -> cap = min(0.5*6%, 5%) = 3%
    panel.adv20[t_sig : t_sig + 10, j] = 600_000_000.0
    panel.close_unadj[t_sig : t_sig + 10, j] = 100.0
    panel.low[t_sig, j] = 98.0 / (1.0 - 0.001)  # raw stop = 98.0

    # Case A: Entry at t_sig+1 open=100.0 (stop=98.0, R=2.0, target=106.0).
    # On t_sig+2, open gaps down to 95.0 (below stop 98.0) -> must fill at 95.0 ("stop_gap_open").
    panel.open[t_sig + 1, j] = 100.0
    panel.high[t_sig + 1, j] = 101.0
    panel.low[t_sig + 1, j] = 99.0
    panel.close[t_sig + 1, j] = 100.5
    panel.ema9[t_sig + 1, j] = 99.0  # below entry -> no breakeven yet

    panel.open[t_sig + 2, j] = 95.0
    panel.high[t_sig + 2, j] = 99.0
    panel.low[t_sig + 2, j] = 94.0
    panel.close[t_sig + 2, j] = 96.0

    res_gap = L.simulate_luk(panel, L.LukConfig(id="V0"), custom_signals=sig)
    tr_gap = res_gap.meta["trades"]
    assert len(tr_gap) == 1
    assert tr_gap.iloc[0]["exit_reason"] == "stop_gap_open"
    assert np.isclose(tr_gap.iloc[0]["exit_price"], 95.0)

    # Case B: Both stop (98.0) and partial target (106.0) touched on bar t_sig+2 with open=101.0 < 106.0
    # -> Stop must go first ("stop_intraday" at 98.0, partial_taken == False).
    panel.open[t_sig + 2, j] = 101.0
    panel.high[t_sig + 2, j] = 107.0
    panel.low[t_sig + 2, j] = 97.5
    panel.close[t_sig + 2, j] = 99.0
    res_both = L.simulate_luk(panel, L.LukConfig(id="V0"), custom_signals=sig)
    tr_both = res_both.meta["trades"]
    assert len(tr_both) == 1
    assert tr_both.iloc[0]["exit_reason"] == "stop_intraday"
    assert np.isclose(tr_both.iloc[0]["exit_price"], 98.0)
    assert not bool(tr_both.iloc[0]["partial_taken"])

    # Case C: Both stop (98.0) and partial target (106.0) touched on bar t_sig+2, but open=106.5 >= target
    # -> Partial 20% fills at open=106.5 first, and remaining 80% stops out at 98.0!
    panel.open[t_sig + 2, j] = 106.5
    panel.high[t_sig + 2, j] = 107.0
    panel.low[t_sig + 2, j] = 97.5
    panel.close[t_sig + 2, j] = 99.0
    res_open_target = L.simulate_luk(panel, L.LukConfig(id="V0"), custom_signals=sig)
    tr_ot = res_open_target.meta["trades"]
    assert len(tr_ot) == 1
    assert bool(tr_ot.iloc[0]["partial_taken"])
    assert tr_ot.iloc[0]["exit_reason"] == "stop_intraday"

    # Case D: Stop hit on the entry bar t_sig+1 itself -> "stop_entry_bar" at 98.0
    panel.open[t_sig + 1, j] = 100.0
    panel.high[t_sig + 1, j] = 100.5
    panel.low[t_sig + 1, j] = 97.8
    panel.close[t_sig + 1, j] = 98.5
    res_eb = L.simulate_luk(panel, L.LukConfig(id="V0"), custom_signals=sig)
    tr_eb = res_eb.meta["trades"]
    assert len(tr_eb) == 1
    assert tr_eb.iloc[0]["exit_reason"] == "stop_entry_bar"
    assert np.isclose(tr_eb.iloc[0]["exit_price"], 98.0)

    # Case E: Entry cancelled if open_{t+1} <= stop OR if stop distance > min(0.5*ADR20, 5%)
    panel.open[t_sig + 1, j] = 97.5  # <= stop 98.0
    panel.low[t_sig + 1, j] = 97.0
    res_cancel1 = L.simulate_luk(panel, L.LukConfig(id="V0"), custom_signals=sig)
    assert len(res_cancel1.meta["trades"]) == 0

    panel.open[t_sig + 1, j] = 105.0  # (105 - 98)/105 = 6.67% > 5% cap
    panel.high[t_sig + 1, j] = 106.0
    panel.low[t_sig + 1, j] = 104.0
    panel.close[t_sig + 1, j] = 105.5
    res_cancel2 = L.simulate_luk(panel, L.LukConfig(id="V0"), custom_signals=sig)
    assert len(res_cancel2.meta["trades"]) == 0

    # Case F: Breakeven stop move when EMA9_{t+1} > entry (100.0), and never decreases later
    panel.open[t_sig + 1, j] = 100.0
    panel.high[t_sig + 1, j] = 102.0
    panel.low[t_sig + 1, j] = 99.5
    panel.close[t_sig + 1, j] = 101.5
    panel.ema9[t_sig + 1, j] = 100.5  # > 100.0 -> stop moves to 100.0 at close of t_sig+1

    # On t_sig+2, EMA9 drops below 100.0, and intraday low touches 99.8 (above original stop 98.0, below breakeven 100.0)
    panel.open[t_sig + 2, j] = 101.0
    panel.high[t_sig + 2, j] = 101.2
    panel.low[t_sig + 2, j] = 99.8
    panel.close[t_sig + 2, j] = 100.2
    panel.ema9[t_sig + 2, j] = 99.0
    res_be = L.simulate_luk(panel, L.LukConfig(id="V0"), custom_signals=sig)
    tr_be = res_be.meta["trades"]
    assert len(tr_be) == 1
    assert tr_be.iloc[0]["exit_reason"] == "stop_intraday"
    assert np.isclose(tr_be.iloc[0]["exit_price"], 100.0)


def test_sizing_caps_and_max_entries_per_day():
    """Section 3.12 (3): Verify ~0.5% risk sizing, 25%/20% notional caps, 1% ADV cap,
    100% gross exposure cap, 3% open risk cap, and <= 4 new entries/day."""
    snapshots, stock_dfs, qqq_df, spy_df, cpi_df = _make_synthetic_panel(n_days=600, n_syms=8, seed=99)
    panel = L.build_panel_from_dfs(
        snapshots, stock_dfs, qqq_df, spy_df, cpi_df, start_date="2018-01-02", end_date="2026-10-01"
    )
    T, N = panel.T, panel.N
    t_sig = 410
    sig = np.zeros((T, N), dtype=bool)
    # Trigger 6 simultaneous signals on t_sig (SYN0..SYN5 are all PIT members)
    for j in range(6):
        sig[t_sig, j] = True
        panel.adr20[t_sig, j] = 0.08
        panel.adv20[t_sig : t_sig + 5, j] = 500_000_000.0
        panel.close_unadj[t_sig : t_sig + 5, j] = 100.0
        panel.low[t_sig, j] = 99.0 / (1.0 - 0.001)  # stop = 99.0
        panel.open[t_sig + 1, j] = 100.0            # R = $1/sh -> raw_shares = 500 ($50k notional > 25% cap!)
        panel.high[t_sig + 1, j] = 101.0
        panel.low[t_sig + 1, j] = 99.5
        panel.close[t_sig + 1, j] = 100.5
        # Exit on t_sig + 2 via stop so we inspect trades
        panel.open[t_sig + 2, j] = 98.5
        panel.high[t_sig + 2, j] = 99.0
        panel.low[t_sig + 2, j] = 98.0
        panel.close[t_sig + 2, j] = 98.5

    res = L.simulate_luk(panel, L.LukConfig(id="V0"), custom_signals=sig)
    tr = res.meta["trades"]
    # Max 4 new entries per day AND each capped at <= 25% of $100k = 250 shares
    assert len(tr) <= 4
    assert (tr["initial_shares"] <= 250).all()
    assert res.bar["exposure"].max() <= 1.0 + 1e-6

    # Verify 20% cap when close_unadj < $10
    sig_lowpx = np.zeros((T, N), dtype=bool)
    sig_lowpx[t_sig, 0] = True
    panel.close_unadj[t_sig, 0] = 8.0
    panel.low[t_sig, 0] = 7.92 / (1.0 - 0.001)
    panel.open[t_sig + 1, 0] = 8.0  # R = $0.08/sh -> raw risk shares = 6,250 ($50k), but 20% cap = $20k = 2,500 sh
    panel.high[t_sig + 1, 0] = 8.1
    panel.low[t_sig + 1, 0] = 7.95
    panel.close[t_sig + 1, 0] = 8.05
    panel.open[t_sig + 2, 0] = 7.8
    res_lowpx = L.simulate_luk(panel, L.LukConfig(id="V0"), custom_signals=sig_lowpx)
    tr_lp = res_lowpx.meta["trades"]
    assert len(tr_lp) == 1
    assert tr_lp.iloc[0]["initial_shares"] == 2500

    # Verify 1% ADV$20 cap when ADV$20 = $1,200,000 -> 1% of ADV = $12,000 -> at $100/sh = 120 shares
    panel.close_unadj[t_sig, 0] = 100.0
    panel.adv20[t_sig, 0] = 1_200_000.0
    panel.low[t_sig, 0] = 98.0 / (1.0 - 0.001)
    panel.open[t_sig + 1, 0] = 100.0  # raw risk shares = 250, 20% cap = 200, 1% ADV cap = 120
    panel.high[t_sig + 1, 0] = 101.0
    panel.low[t_sig + 1, 0] = 99.0
    panel.close[t_sig + 1, 0] = 100.0
    panel.open[t_sig + 2, 0] = 97.0
    res_adv = L.simulate_luk(panel, L.LukConfig(id="V0"), custom_signals=sig_lowpx)
    assert res_adv.meta["trades"].iloc[0]["initial_shares"] == 120


def test_pit_universe_delisting_and_ticker_reuse_hygiene():
    """Section 3.12 (4):
    - Ticker is never eligible before PIT addition or after PIT removal.
    - Series ending while still in index triggers -30% exit in pessimistic mode and 0% in neutral mode.
    - Ticker-reuse / gap check flags series starting >5 sessions after PIT addition or missing >10 member bars."""
    snapshots, stock_dfs, qqq_df, spy_df, cpi_df = _make_synthetic_panel(n_days=600, n_syms=8, seed=21)

    # Truncate SYN5 at bar 480 while still in the index -> delisting while in index!
    stock_dfs["SYN5"] = stock_dfs["SYN5"].iloc[:481].copy()
    # Make SYN4 start 20 sessions late (ticker reuse / missing start)
    stock_dfs["SYN4"] = stock_dfs["SYN4"].iloc[20:].copy()

    panel = L.build_panel_from_dfs(
        snapshots, stock_dfs, qqq_df, spy_df, cpi_df, start_date="2018-01-02", end_date="2026-10-01"
    )

    idx_syn6 = panel.tickers.index("SYN6")
    idx_syn7 = panel.tickers.index("SYN7")
    idx_syn5 = panel.tickers.index("SYN5")

    # SYN6 added at bar 400 -> never PIT or base_eligible before bar 400
    assert not panel.pit[:400, idx_syn6].any()
    assert not panel.base_eligible[:400, idx_syn6].any()
    assert panel.pit[400, idx_syn6]

    # SYN7 removed at bar 450 -> never PIT or base_eligible from bar 450 onwards
    assert panel.pit[449, idx_syn7]
    assert not panel.pit[450:, idx_syn7].any()
    assert not panel.base_eligible[450:, idx_syn7].any()

    # SYN4 started 20 bars late -> flagged in hygiene and completely excluded from panel.tickers
    assert panel.hygiene["n_flagged_reuse"] >= 1
    assert "SYN4" in panel.hygiene["flagged_reuse"]
    assert "SYN4" not in panel.tickers

    # SYN5 ended at bar 480 while still in index -> delist_bar == 480
    assert panel.delist_bar[idx_syn5] == 480
    assert panel.hygiene["n_delisted_in_index"] == 1

    # Enter SYN5 at bar 475 and hold into delist bar 480: compare pessimistic (-30%) vs neutral (0%)
    sig = np.zeros((panel.T, panel.N), dtype=bool)
    sig[474, idx_syn5] = True
    panel.adr20[474, idx_syn5] = 0.06
    panel.adv20[474:481, idx_syn5] = 500_000_000.0
    panel.close_unadj[474:481, idx_syn5] = 100.0
    panel.low[474, idx_syn5] = 98.0 / (1.0 - 0.001)
    for b in range(475, 481):
        panel.open[b, idx_syn5] = 100.0
        panel.high[b, idx_syn5] = 101.0
        panel.low[b, idx_syn5] = 99.0
        panel.close[b, idx_syn5] = 100.0
        panel.ema9[b, idx_syn5] = 99.5

    res_pess = L.simulate_luk(panel, L.LukConfig(id="V0", delist_mode="pessimistic"), custom_signals=sig)
    res_neut = L.simulate_luk(panel, L.LukConfig(id="V0", delist_mode="neutral"), custom_signals=sig)
    tr_p = res_pess.meta["trades"].iloc[0]
    tr_n = res_neut.meta["trades"].iloc[0]
    assert tr_p["exit_reason"] == "delist_pessimistic"
    assert tr_n["exit_reason"] == "delist_neutral"
    assert np.isclose(tr_p["exit_price"], 70.0)
    assert np.isclose(tr_n["exit_price"], 100.0)


def test_prereg_grid_neighbors_topology():
    """Ensure normalized V0..V7 grid links V0 to all 7 single-parameter ablations for R5 plateau stability."""
    spec = yaml.safe_load(Path("configs/prereg/luk_v1/L1_luk_pullback_daily.yaml").read_text())
    norm = L.normalize_grid(spec["grid"])
    assert len(norm) == 8
    assert neighbors(norm, 0) == [1, 2, 3, 4, 5, 6, 7]
    for k in range(1, 8):
        assert 0 in neighbors(norm, k)
