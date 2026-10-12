"""C++ vs Python reference parity (skipped if raim_core is not built)."""
import math

import numpy as np
import pytest

from conftest import HAS_CPP
from raim.core import features_ref, governor_ref, hmm_ref, quoter_ref, simexchange_ref
from raim.data.synthetic import SyntheticCfg, generate
from raim.types import BookSnapshot

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="raim_core (C++) not built")


def _events(n=4000, seed=3):
    b, t, _ = generate(SyntheticCfg(n_steps=n, seed=seed))
    return sorted(b + t, key=lambda e: (e.ts_ms, isinstance(e, BookSnapshot)))


def test_features_parity():
    import raim_core
    py, cc = features_ref.FeatureEngine(), raim_core.FeatureEngine()
    for ev in _events():
        for fe in (py, cc):
            if isinstance(ev, BookSnapshot):
                fe.on_book(ev.ts_ms, ev.best_bid, ev.bids[0][1], ev.best_ask, ev.asks[0][1], 3.0)
            else:
                fe.on_trade(ev.ts_ms, ev.price, ev.qty, ev.sign)
        if isinstance(ev, BookSnapshot):
            np.testing.assert_allclose(cc.values(), py.values(), rtol=1e-7, atol=1e-10)


def test_hmm_parity():
    import raim_core
    rng = np.random.default_rng(0)
    args = ([0.25] * 4, (np.eye(4) * 0.9 + 0.025).tolist(), rng.normal(size=(4, 6)).tolist(),
            (rng.random((4, 6)) + 0.5).tolist())
    a, b = hmm_ref.HMMFilter(*args), raim_core.HMMFilter(*args)
    for _ in range(500):
        x = rng.normal(size=6).tolist()
        if rng.random() < 0.05:
            x[2] = math.nan
        np.testing.assert_allclose(b.step(x), a.step(x), rtol=1e-9, atol=1e-12)


def test_quote_parity():
    import raim_core
    rng = np.random.default_rng(1)
    for _ in range(2000):
        args = (60000 * (1 + rng.normal() * 0.01), abs(rng.normal()) * 3, rng.normal() * 5, 0.1,
                1.5, 10.0, 0.5, 1 + rng.random() * 3, 0.01)
        np.testing.assert_allclose(raim_core.quote(*args), quoter_ref.quote(*args), rtol=1e-12)


def test_governor_parity():
    import raim_core
    kw = (50, 2.33, 30, 0.1, 500, 25, 2000, 0.2, 10, 3, 6, 60)
    a, b = governor_ref.RiskGovernor(*kw), raim_core.RiskGovernor(*kw)
    rng = np.random.default_rng(2)
    eq = 0.0
    for i in range(3000):
        now = i * 100
        if rng.random() < 0.1:
            pnl = rng.normal()
            a.on_trade_result(pnl, now); b.on_trade_result(pnl, now)
        m1, m2 = rng.random() * 1.4 - 0.2, rng.random()
        a.set_external(m1, m2); b.set_external(m1, m2)
        eq += rng.normal() * 0.3
        args = (now, now - int(rng.integers(0, 2500)), eq, rng.normal() * 0.01, 60000.0,
                abs(rng.normal()) * 1e-4, 5.0, rng.random() * 0.3, rng.random() * 12)
        da, db = a.evaluate(*args), b.evaluate(*args)
        for f in ("allow_bid", "allow_ask", "cancel_all", "halted", "reason"):
            assert getattr(da, f) == getattr(db, f), (i, f)
        for f in ("size_mult", "spread_mult", "qmax"):
            assert getattr(da, f) == pytest.approx(getattr(db, f), rel=1e-12)


def test_simexchange_parity():
    import raim_core
    a = simexchange_ref.SimExchange(1.0, 0.0, 30, 10, 30, 2.0, 99)
    b = raim_core.SimExchange(1.0, 0.0, 30, 10, 30, 2.0, 99)
    rng = np.random.default_rng(5)
    fa, fb = [], []
    for ev in _events(3000, 11):
        now = ev.ts_ms
        if isinstance(ev, BookSnapshot):
            fa += a.on_book(now, ev.best_bid, ev.bids[0][1], ev.best_ask, ev.asks[0][1])
            fb += b.on_book(now, ev.best_bid, ev.bids[0][1], ev.best_ask, ev.asks[0][1])
            if rng.random() < 0.2:
                side = 1 if rng.random() < 0.5 else -1
                px = ev.best_bid if side == 1 else ev.best_ask
                px = round(px - side * 0.01 * int(rng.integers(0, 3)), 2)
                ia, ib = a.place(now, side, px, 0.01), b.place(now, side, px, 0.01)
                assert ia == ib
            if rng.random() < 0.1 and a.open_orders():
                oid = a.open_orders()[0][0]
                a.cancel(now, oid); b.cancel(now, oid)
        else:
            fa += a.on_trade(now, ev.price, ev.qty, ev.sign)
            fb += b.on_trade(now, ev.price, ev.qty, ev.sign)
    assert len(fa) == len(fb) and len(fa) > 0
    for x, y in zip(fa, fb):
        assert x[0] == y[0] and x[1] == y[1] and x[5] == y[5]
        np.testing.assert_allclose(x[2:5], y[2:5], rtol=1e-12)
    assert a.n_rejects == b.n_rejects and a.n_msgs == b.n_msgs
