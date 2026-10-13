"""Leakage tests for round-2 convex strategies + PIT universe + trade metrics."""
import numpy as np
import pandas as pd
import pytest

from raim.strategies import convex, universe
from raim.strategies.convex_metrics import convex_stats, longest_streak, recovery_days, trades

CUT = 700


def _walk(n, seed, freq="1h", start="2022-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def _bars(c, seed=0):
    rng = np.random.default_rng(seed)
    q = pd.Series(rng.lognormal(10, 1, len(c)), index=c.index)
    return pd.DataFrame({"open": c.shift(1).fillna(c), "high": c * 1.003, "low": c * 0.997,
                         "close": c, "quote_volume": q})


def _pert(s, cut=CUT, seed=99):
    rng = np.random.default_rng(seed)
    s = s.copy()
    s.iloc[cut:] = s.iloc[cut:] * np.exp(rng.normal(0, 0.05, len(s) - cut))
    return s


def _eq(a, b, cut=CUT):
    pd.testing.assert_frame_equal(pd.DataFrame(a).iloc[:cut], pd.DataFrame(b).iloc[:cut])


def test_intraday_tsmom_causal_and_slot():
    c = _walk(48 * 40, 1, "30min")
    w1 = convex.intraday_tsmom_weights(c, 0, 0.0, False)
    w2 = convex.intraday_tsmom_weights(_pert(c, 48 * 20 + 2), 0, 0.0, False)
    _eq(w1, w2, 48 * 20 + 2)
    nz = w1[w1 != 0].index
    assert len(nz) > 30 and all(t.hour == 23 and t.minute == 0 for t in nz)  # held 23:30-24:00


@pytest.mark.parametrize("fn", [
    lambda b: convex.donchian_breakout_state(b, 48, 3.0),
    lambda b: convex.squeeze_breakout_state(b, 20, 2.0, 1.5, 2.0),
    lambda b: convex.cascade_weights(b, 3.0, 1.5, 24, "cont"),
])
def test_bar_strategies_causal(fn):
    c = _walk(1000, 2)
    b1, b2 = _bars(c), _bars(_pert(c))
    b2["quote_volume"] = b1["quote_volume"]
    _eq(fn(b1), fn(b2))


def test_funding_extremes_causal():
    c = _walk(3000, 3)
    f = pd.Series(np.random.default_rng(4).normal(1e-4, 2e-4, 375), index=c.index[::8])
    w1 = convex.funding_extreme_weights(c, f, 0.05, "conditional")
    f2 = f.copy(); f2[f2.index >= c.index[2000]] = -1.0
    w2 = convex.funding_extreme_weights(_pert(c, 2000), f2, 0.05, "conditional")
    _eq(w1, w2, 2000)


def _panel(n=400, k=6):
    idx = pd.date_range("2021-01-01", periods=n, freq="1D", tz="UTC")
    close = pd.DataFrame({f"C{i}": _walk(n, 10 + i, "1D", "2021-01-01").values for i in range(k)}, index=idx)
    close.iloc[:150, 5] = np.nan                     # listed later
    close.iloc[300:, 4] = np.nan                     # delisted
    qv = close * 0 + np.arange(1, k + 1) * 1e7
    return close, qv


def test_pit_universe_uses_only_past_and_handles_delisting():
    close, qv = _panel()
    u1 = universe.pit_universe(qv, 3)
    qv2 = qv.copy(); qv2.iloc[250:] *= 1000
    u2 = universe.pit_universe(qv2, 3)
    pd.testing.assert_frame_equal(u1.iloc[:251], u2.iloc[:251])
    assert not u1["C5"].iloc[:150 + 60].any()        # needs min history after listing
    assert not u1["C4"].iloc[301:].any()             # gone after delisting


def test_alt_tsmom_and_donchian_pit_causal():
    close, qv = _panel()
    u = universe.pit_universe(qv, 3)
    c2 = close.apply(lambda s: _pert(s, 250))
    _eq(convex.alt_tsmom_weights(close, u, 2, 0.5), convex.alt_tsmom_weights(c2, u, 2, 0.5), 250)
    _eq(convex.donchian_pit_weights(close, u, [5, 20], 0.3), convex.donchian_pit_weights(c2, u, [5, 20], 0.3), 250)
    w = convex.alt_tsmom_weights(close, u, 2, 0.5)
    assert (w["C4"].iloc[300:] == 0).all()


def test_listing_weights_event_known_at_day1_close():
    close, qv = _panel()
    opn = close.shift(1).fillna(close * 0.99)
    P = {"close": close, "open": opn, "qv": qv}
    w = convex.listing_weights(P, "2021-02-01", 7, "all")
    first = close["C5"].first_valid_index()
    assert w["C5"].loc[first] == 0.02 and w["C5"].sum() == pytest.approx(0.14)
    assert (w.drop(columns="C5") == 0).all().all()


def test_trade_metrics():
    idx = pd.date_range("2022-01-01", periods=8, freq="1D", tz="UTC")
    W = pd.DataFrame({"A": [0, 1, 1, 0, -1, 0, 1, 0.0]}, index=idx)
    P = pd.DataFrame({"A": [0, .02, .03, -.001, -.01, -.001, .05, -.001]}, index=idx)
    tr = trades(W, P)
    assert len(tr) == 3 and tr.pnl.round(3).tolist() == [0.049, -0.011, 0.049]
    st = convex_stats(tr, P["A"])
    assert st["win_rate"] == pytest.approx(2 / 3) and st["payoff"] == pytest.approx(0.049 / 0.011)
    assert longest_streak(np.array([1, -1, -1, 1, -1])) == 2
    assert recovery_days(pd.Series([0.1, -0.05, -0.01, 0.2])) == 2
