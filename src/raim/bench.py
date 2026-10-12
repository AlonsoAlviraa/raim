"""Micro-benchmark: ns per update, Python reference vs C++ (if built). Honest numbers include
Python->C++ call overhead (pybind11), which dominates for small per-event work."""
from __future__ import annotations

import time

import numpy as np

from raim.core import features_ref, governor_ref, hmm_ref


def _time(fn, n: int) -> float:
    t0 = time.perf_counter_ns()
    fn(n)
    return (time.perf_counter_ns() - t0) / n


def run_bench(n: int = 200_000) -> dict[str, float]:
    rng = np.random.default_rng(0)
    mids = 60000 * np.exp(np.cumsum(rng.normal(0, 1e-5, n)))
    impls = {"python": (features_ref, hmm_ref, governor_ref)}
    try:
        import raim_core
        impls["cpp"] = (raim_core, raim_core, raim_core)
    except ImportError:
        pass
    out: dict[str, float] = {}
    for name, (fm, hm, gm) in impls.items():
        fe = fm.FeatureEngine()

        def feat(k: int, fe=fe) -> None:
            for i in range(k):
                m = mids[i]
                fe.on_book(i * 100, m - 0.5, 1.0, m + 0.5, 1.2, 10.0)
                fe.values()
        out[f"features_{name}"] = _time(feat, n)
        k = 4
        trans = (np.eye(k) * 0.96 + 0.01).tolist()
        h = hm.HMMFilter([0.25] * 4, trans, rng.normal(size=(4, 6)).tolist(), np.ones((4, 6)).tolist())
        x = rng.normal(size=6).tolist()
        out[f"hmm_step_{name}"] = _time(lambda kk, h=h: [h.step(x) for _ in range(kk)], n // 4)
        g = gm.RiskGovernor(50, 2.33, 30, 0.1, 500, 25, 2000, 0.2, 10, 3, 6, 60)
        out[f"governor_eval_{name}"] = _time(
            lambda kk, g=g: [g.evaluate(i, i, 0.0, 0.0, 60000.0, 1e-4, 10.0, 0.0, 1.0) for i in range(kk)], n // 4)
    return out
