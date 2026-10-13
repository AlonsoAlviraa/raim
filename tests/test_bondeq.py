import numpy as np
import pandas as pd

from raim.strategies import bondeq as B


def _P(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC")
    c = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, .01, (n, 6)), 0)), idx, B.ETFS)
    return {"close": c, "open": c.shift(1).bfill(), "traded": pd.DataFrame(True, idx, B.ETFS),
            "baa": pd.Series(2 + np.cumsum(rng.normal(0, .02, n)), idx)}


def test_b1_dollar_neutral_and_causal():
    P = _P()
    w = B.b1_weights(P, 3, "both")
    assert np.allclose(w.sum(axis=1), 0) and (w.abs().sum(axis=1) <= 1 + 1e-12).all()
    P2 = {k: v.copy() for k, v in P.items()}
    P2["close"].iloc[300:] *= 2
    assert w.iloc[:300].equals(B.b1_weights(P2, 3, "both").iloc[:300])


def test_switch_fully_invested_long_only():
    P = _P()
    for w in (B.b2_weights(P, 21, "BIL"), B.b3_weights(P, 50, "IEF")):
        assert (w >= 0).all().all() and np.allclose(w.sum(axis=1), 1)


def test_short_leg_pays_borrow():
    P = _P()
    R, O, C, F = B.returns_and_costs(P, ["HYG", "IEF_S"])
    assert (F["IEF_S"] < 0).all() and (F["HYG"] == 0).all()
