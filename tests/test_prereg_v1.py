"""prereg-v1 signals are causal: changing the future never changes past weights; specs are valid."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from raim.research import prereg
from raim.research.verdict import load_rules
from raim.strategies import prereg_v1 as P

ROOT = Path(__file__).resolve().parents[1]


def _panel(T=700, n=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=T, freq="D", tz="UTC")
    cols = ["BTCUSDT", "ETHUSDT"] + [f"A{i}USDT" for i in range(n - 2)]
    close = pd.DataFrame(np.exp(np.cumsum(0.03 * rng.standard_normal((T, n)), 0)) * 100, index=idx, columns=cols)
    univ = pd.DataFrame(True, index=idx, columns=cols)
    return close, univ


@pytest.mark.parametrize("fn", [
    lambda c, u: P.p1_weights(c, u, 4, 100),
    lambda c, u: P.p2_weights(c, u, 0.5, "btc200"),
    lambda c, u: P.p3_weights(c, u, 3, 30, 7),
    lambda c, u: P.p4_weights(c, "BTC_ETH", "donchian"),
    lambda c, u: P.p4_weights(c, "BTC", "sma200"),
])
def test_causal(fn):
    c, u = _panel()
    w = fn(c, u)
    c2 = c.copy()
    c2.iloc[500:] *= 3.0
    w2 = fn(c2, u)
    pd.testing.assert_frame_equal(w.iloc[:500], w2.iloc[:500])
    assert (w >= -1e-12).all().all() and (w.sum(axis=1) <= 1 + 1e-9).all()


def test_specs_valid_and_small():
    rules, _ = load_rules()
    for p in sorted((ROOT / "configs/prereg/v1").glob("P*.yaml")):
        spec = yaml.safe_load(p.read_text())
        prereg.validate(spec, rules)
        assert prereg.grid_size(spec["grid"]) <= 10 and not spec["retroactive"]
