"""edge-v1 signals: causal and dollar-neutral; specs valid (<=10 variants, market-neutral)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from raim.research import prereg
from raim.research.verdict import load_rules
from raim.strategies import edge_v1 as E

ROOT = Path(__file__).resolve().parents[1]


def _P(T=300, n=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=T, freq="D", tz="UTC")
    cols = [f"C{i}USDT" for i in range(n)]
    c = pd.DataFrame(np.exp(np.cumsum(0.04 * rng.standard_normal((T, n)), 0)), index=idx, columns=cols)
    qv = pd.DataFrame(np.exp(rng.normal(15, 1, (T, n))), index=idx, columns=cols)
    tb = qv * rng.uniform(0.4, 0.6, (T, n))
    f = pd.DataFrame(rng.normal(1e-4, 3e-4, (T, n)), index=idx, columns=cols)
    return {"close": c, "qv": qv, "tbq": tb, "funding": f}, pd.DataFrame(True, index=idx, columns=cols)


FNS = [lambda P, u: E.e1_weights(P, u, 7), lambda P, u: E.e2_weights(P, u, 1, "low_vshock"),
       lambda P, u: E.e2_weights(P, u, 3, "all"), lambda P, u: E.e3_weights(P, u, 1, True),
       lambda P, u: E.e3_weights(P, u, 7, False)]


@pytest.mark.parametrize("fn", FNS)
def test_causal_and_neutral(fn):
    P, u = _P()
    w = fn(P, u)
    P2 = {k: v.copy() for k, v in P.items()}
    for v in P2.values():
        v.iloc[200:] *= 1.7
    pd.testing.assert_frame_equal(w.iloc[:200], fn(P2, u).iloc[:200])
    act = w.abs().sum(axis=1) > 0
    assert act.any()
    assert np.allclose(w.sum(axis=1)[act], 0, atol=1e-9)          # dollar-neutral
    assert np.allclose(w.abs().sum(axis=1)[act], 1.0)


def test_specs():
    rules, _ = load_rules()
    for p in sorted((ROOT / "configs/prereg/edge_v1").glob("E*.yaml")):
        s = yaml.safe_load(p.read_text())
        prereg.validate(s, rules)
        assert prereg.grid_size(s["grid"]) <= 10 and s["market_neutral"] and not s["retroactive"]
