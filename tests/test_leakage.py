"""Look-ahead / leakage guards."""
import numpy as np

from conftest import mod
from raim.config import RaimConfig
from raim.data.synthetic import SyntheticCfg, generate
from raim.research.markout import markouts
from raim.research.walkforward import assert_no_leakage, purged_walk_forward
from raim.sim.engine import feature_rows
from raim.types import BookSnapshot


def _events(n=3000, seed=1):
    b, t, _ = generate(SyntheticCfg(n_steps=n, seed=seed))
    return sorted(b + t, key=lambda e: (e.ts_ms, isinstance(e, BookSnapshot)))


def test_features_do_not_depend_on_future():
    ev = _events()
    cfg = RaimConfig()
    _, full = feature_rows(ev, cfg)
    _, part = feature_rows(ev[: len(ev) // 2], cfg)
    np.testing.assert_array_equal(full[: len(part)], part)


def test_feature_perturbing_future_leaves_past(backend):
    ev = _events(1500)
    FE = mod("features", backend).FeatureEngine
    def run(events):
        fe, out = FE(), []
        for e in events:
            if isinstance(e, BookSnapshot):
                fe.on_book(e.ts_ms, e.best_bid, e.bids[0][1], e.best_ask, e.asks[0][1], 1.0)
                out.append(fe.values())
            else:
                fe.on_trade(e.ts_ms, e.price, e.qty, e.sign)
        return np.array(out)
    cut = len(ev) // 2
    shocked = ev[:cut] + [BookSnapshot(e.ts_ms, ((e.best_bid * 0.9, 9.0),), ((e.best_ask * 0.9, 9.0),), e.recv_ms)
                          if isinstance(e, BookSnapshot) else e for e in ev[cut:]]
    a, b = run(ev), run(shocked)
    n_past = sum(isinstance(e, BookSnapshot) for e in ev[:cut])
    np.testing.assert_array_equal(a[:n_past], b[:n_past])


def test_walk_forward_purged_embargoed():
    folds = purged_walk_forward(10_000, 5, purge=200, embargo=100, min_train=500)
    assert folds
    for f in folds:
        assert_no_leakage(f, 200)
        assert f.train.max() < f.test.min()


def test_markout_never_peeks_past_end():
    mids = [(i * 1000, 100.0 + i) for i in range(10)]
    fills = [(8000, 1, 108.0, 1.0, 0.0, 0)]
    mo = markouts(fills, mids)
    assert np.isnan(mo.loc[0, "mo_5s_bps"]) and not np.isnan(mo.loc[0, "mo_1s_bps"])
