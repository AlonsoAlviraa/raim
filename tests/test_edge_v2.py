import numpy as np
import pandas as pd

from raim.strategies import edge_v2 as X


def _panel(n=300, m=15, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, tz="UTC")
    cols = [f"C{i}USDT" for i in range(m - 1)] + ["BTCUSDT"]
    c = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, .03, (n, m)), 0)), idx, cols)
    qv = pd.DataFrame(rng.uniform(1e7, 2e7, (n, m)), idx, cols)
    tb = qv * rng.uniform(.4, .6, (n, m))
    f = pd.DataFrame(rng.normal(0, 3e-4, (n, m)), idx, cols)
    return {"close": c, "qv": qv, "tbq": tb, "funding": f}, pd.DataFrame(True, idx, cols)


def test_x2_long_only_capped_and_no_lookahead():
    P, u = _panel()
    w = X.x2_weights(P, u, 20, 0.5, False)
    assert (w >= 0).all().all() and (w.sum(axis=1) <= 1 + 1e-9).all()
    P2 = {k: v.copy() for k, v in P.items()}
    P2["close"].iloc[200:] *= 3                     # future change must not alter past weights
    w2 = X.x2_weights(P2, u, 20, 0.5, False)
    assert w.iloc[:200].equals(w2.iloc[:200])


def test_x3_gross_cap_and_sign():
    P, u = _panel()
    w = X.x3_weights(P, u, 0.05, 3, "both")
    assert (w.abs().sum(axis=1) <= 1 + 1e-9).all()
    assert (X.x3_weights(P, u, 0.05, 3, "long_up") >= 0).all().all()


def test_x1_neutral():
    P, u = _panel()
    prem = P["funding"] * 10
    w = X.x1_weights(P, prem, u, "level", 5, 5)
    assert np.allclose(w.sum(axis=1), 0, atol=1e-9)


def test_payoff_stats_right_skew():
    d = pd.Series([-0.01] * 90 + [0.2] * 10)
    s = X.payoff_stats(d)
    assert abs(s["win_rate"] - 0.1) < 1e-12 and s["skew"] > 0 and abs(s["payoff_ratio"] - 20) < 1e-9
