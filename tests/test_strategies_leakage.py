"""Leakage / look-ahead tests for src/raim/strategies (no network, synthetic data).

Core property: target weights at time t must be invariant to any modification of data AFTER t.
We perturb the future (multiply by noise after a cut) and require identical weights up to the cut."""
import numpy as np
import pandas as pd
import pytest

from raim.strategies import basis, engine, leadlag, pairs, trend
from raim.research.walkforward import assert_no_leakage, purged_walk_forward


def _walk(n, seed, freq="1h"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq=freq, tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def _perturb(s, cut, seed=99):
    rng = np.random.default_rng(seed)
    s = s.copy()
    s.iloc[cut:] = s.iloc[cut:] * np.exp(rng.normal(0, 0.05, len(s) - cut))
    return s


CUT = 700


def _same_until(a, b, cut):
    pd.testing.assert_frame_equal(pd.DataFrame(a).iloc[:cut], pd.DataFrame(b).iloc[:cut])


def test_basis_arb_causal():
    s = _walk(1000, 1)
    f = s * (1 + 0.003 * np.sin(np.arange(1000) / 7))
    w1 = basis.basis_arb_weights(basis.rho_series(s, f), 0.001, allow_short_spot=True)
    w2 = basis.basis_arb_weights(basis.rho_series(_perturb(s, CUT), _perturb(f, CUT, 5)), 0.001,
                                 allow_short_spot=True)
    _same_until(w1, w2, CUT)


def test_funding_carry_causal_and_aligned():
    idx = pd.date_range("2022-01-01", periods=1000, freq="1h", tz="UTC")
    fe = pd.Series(np.random.default_rng(2).normal(1e-4, 2e-4, 125), index=idx[::8])
    w1 = basis.funding_carry_weights(fe, idx, 3, 1e-4)
    fe2 = fe.copy()
    fe2[fe2.index >= idx[CUT]] = 1.0
    w2 = basis.funding_carry_weights(fe2, idx, 3, 1e-4)
    _same_until(w1, w2, CUT)


def test_donchian_causal():
    c = pd.DataFrame({"A": _walk(1000, 3, "1D"), "B": _walk(1000, 4, "1D")})
    w1 = trend.donchian_ensemble_weights(c, [5, 20, 60], 0.3)
    c2 = c.apply(lambda x: _perturb(x, CUT))
    w2 = trend.donchian_ensemble_weights(c2, [5, 20, 60], 0.3)
    _same_until(w1, w2, CUT)


def test_tsmom_atr_causal():
    c = _walk(1000, 5)
    b = pd.DataFrame({"close": c, "high": c * 1.002, "low": c * 0.998})
    c2 = _perturb(c, CUT)
    b2 = pd.DataFrame({"close": c2, "high": c2 * 1.002, "low": c2 * 0.998})
    w1 = trend.tsmom_portfolio_weights({"A": b}, 12, 0.02, 2.5, True)
    w2 = trend.tsmom_portfolio_weights({"A": b2}, 12, 0.02, 2.5, True)
    _same_until(w1, w2, CUT)


def test_pairs_causal():
    y, x = _walk(1000, 6), _walk(1000, 7)
    w1 = pairs.pair_weights(y, x, 100, 2.0)
    w2 = pairs.pair_weights(_perturb(y, CUT), _perturb(x, CUT, 8), 100, 2.0)
    _same_until(w1, w2, CUT)


def test_leadlag_causal():
    rb = np.log(_walk(1000, 9)).diff()
    ra = np.log(_walk(1000, 10)).diff()
    beta = np.array([0.0, 0.3, -0.1])
    w1 = leadlag.leadlag_weights(rb, ra, beta, 0.0, 0.0)
    rb2 = rb.copy(); rb2.iloc[CUT:] *= -3
    w2 = leadlag.leadlag_weights(rb2, ra, beta, 0.0, 0.0)
    _same_until(w1, w2, CUT)


def test_engine_delay_and_costs():
    idx = pd.date_range("2022-01-01", periods=6, freq="1h", tz="UTC")
    r = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01, 0.01, 0.01]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1, 1, 1, 1, 1]}, index=idx)
    res = engine.run(w, r, {"A": 0.001}, delay=1)
    # signal at bar0 -> held from bar2 (1 bar signal + 1 bar delay)
    assert list(res.weights["A"]) == [0, 0, 1, 1, 1, 1]
    assert res.bar["cost"].iloc[2] == pytest.approx(0.001)
    assert res.bar["net"].iloc[2] == pytest.approx(0.01 - 0.001)
    assert res.bar["gross"].iloc[1] == 0.0


def test_engine_rejects_negative_delay():
    idx = pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC")
    with pytest.raises(ValueError):
        engine.run(pd.DataFrame({"A": [1.0] * 3}, index=idx), pd.DataFrame({"A": [0.0] * 3}, index=idx),
                   {"A": 0}, delay=-1)


def test_funding_sign_long_pays():
    idx = pd.date_range("2022-01-01", periods=4, freq="1h", tz="UTC")
    w = pd.DataFrame({"F": [-1.0] * 4}, index=idx)
    r = pd.DataFrame({"F": [0.0] * 4}, index=idx)
    f = pd.DataFrame({"F": [0.0, 0.0, 0.0, 0.001]}, index=idx)
    res = engine.run(w, r, {"F": 0.0}, f, delay=1)
    assert res.bar["funding"].iloc[3] == pytest.approx(0.001)   # short receives positive funding


def test_engine_oracle_detects_lookahead():
    """A deliberately leaky oracle (peeks one bar ahead) must look spectacular when the delay is
    removed (proves the check is sensitive) and lose its edge under the mandatory delay=1."""
    s = _walk(5000, 11)
    r = s.pct_change().to_frame("A")
    peek = np.sign(r["A"].shift(-1)).fillna(0).to_frame("A")    # uses the FUTURE
    leaky = engine.metrics(engine.run(peek, r, {"A": 0.0}, delay=0).daily)["sharpe"]
    safe = engine.metrics(engine.run(peek, r, {"A": 0.0}, delay=1).daily)["sharpe"]
    assert leaky > 20 and abs(safe) < 3.0


def test_walkforward_purge():
    for f in purged_walk_forward(1500, n_folds=5, purge=30, embargo=7, min_train=180):
        assert_no_leakage(f, 30)
