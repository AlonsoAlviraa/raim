"""Phase-2 adversarial review findings, written BEFORE the fixes (tests-first).
IDs map to docs/reviews/*.md."""
import math

import numpy as np
import pytest

from conftest import mod
from raim.config import load_config
from raim.data.synthetic import SyntheticCfg, generate
from raim.jev import adapter as jev
from raim.research.pipeline import fit_regimes, split_by_books
from raim.sim.engine import run_sim
from raim.types import BookSnapshot, Trade

KW = (50, 2.33, 30, 0.1, 500, 25, 2000, 0.2, 10, 3, 6, 60)


def _events(n=6000, seed=9):
    b, t, _ = generate(SyntheticCfg(n_steps=n, seed=seed))
    return sorted(b + t, key=lambda e: (e.ts_ms, isinstance(e, BookSnapshot)))


# ---- RISK-1: non-finite inputs must not bypass limits ---------------------------------
@pytest.mark.parametrize("field", ["equity", "position"])
def test_risk1_nonfinite_inputs_cancel(backend, field):
    g = mod("governor", backend).RiskGovernor(*KW)
    eq, pos = (math.nan, 0.0) if field == "equity" else (0.0, math.nan)
    d = g.evaluate(0, 0, eq, pos, 60000.0, 1e-4, 10.0, 0.0, 1.0)
    assert d.cancel_all and not d.allow_bid and not d.allow_ask and d.reason == "bad_input"


def test_risk1_nan_equity_after_peak_does_not_unlatch(backend):
    g = mod("governor", backend).RiskGovernor(*KW)
    g.evaluate(0, 0, 10.0, 0.0, 6e4, 1e-4, 10.0, 0.0, 1.0)
    g.evaluate(1, 1, -20.0, 0.0, 6e4, 1e-4, 10.0, 0.0, 1.0)
    assert g.evaluate(2, 2, math.nan, 0.0, 6e4, 1e-4, 10.0, 0.0, 1.0).halted


# ---- QUOTE-1: degenerate params must yield NO quote, never a NaN/inf price ------------
@pytest.mark.parametrize("gamma,kappa,mid", [(0.0, 1.5, 6e4), (0.1, 0.0, 6e4), (-0.1, 1.5, 6e4),
                                             (0.1, 1.5, math.nan), (0.1, 1.5, 0.0)])
def test_quote1_degenerate_returns_nan(backend, gamma, kappa, mid):
    b, a, _, _ = mod("quoter", backend).quote(mid, 1.0, 0.0, gamma, kappa, 10, 0.5, 1.0, 0.01)
    assert math.isnan(b) and math.isnan(a)


# ---- SIM-1: queue position behind the touch must use visible L2 depth -----------------
def test_sim1_behind_touch_uses_l2_queue(backend):
    e = mod("sim", backend).SimExchange(1.0, 0.0, 30.0, 0.0, 30.0, 1.0, 1)
    bids, asks = [(100.0, 5.0), (99.9, 7.0)], [(100.1, 5.0), (100.2, 5.0)]
    e.on_book_l2(0, [p for p, _ in bids], [q for _, q in bids], [p for p, _ in asks], [q for _, q in asks])
    e.place(0, 1, 99.9, 1.0)
    e.on_book_l2(40, [100.0, 99.9], [5.0, 7.0], [100.1, 100.2], [5.0, 5.0])
    assert e.on_trade(50, 99.9, 7.0, -1) == []             # 7 ahead of us at 99.9
    f = e.on_trade(60, 99.9, 1.0, -1)
    assert len(f) == 1 and f[0][3] == pytest.approx(1.0)


def test_sim1_beyond_visible_depth_is_conservative(backend):
    e = mod("sim", backend).SimExchange(1.0, 0.0, 30.0, 0.0, 30.0, 1.0, 1)
    e.on_book_l2(0, [100.0, 99.9], [5.0, 7.0], [100.1, 100.2], [5.0, 5.0])
    e.place(0, 1, 99.5, 1.0)                               # below visible depth: unknown queue
    e.on_book_l2(40, [100.0, 99.9], [5.0, 7.0], [100.1, 100.2], [5.0, 5.0])
    assert e.on_trade(50, 99.5, 100.0, -1) == []           # at-price trade: assume queue ahead
    assert e.on_trade(60, 99.4, 1.0, -1)                   # trade-through still fills


# ---- SIM-2: one clock (receive time) for fills, mids, markouts ------------------------
def test_sim2_uses_receive_clock():
    cfg = load_config(None)
    ev = _events()
    shifted = [Trade(e.ts_ms - 10_037, e.price, e.qty, e.is_buyer_maker, e.ts_ms) if isinstance(e, Trade)
               else e for e in ev]
    tr, te = split_by_books(shifted, 1800)
    params, scaler = fit_regimes(tr, cfg)
    r = run_sim(te, cfg, params, scaler)
    recv_times = {e.recv_ms for e in te}
    assert r.fills and all(f[0] in recv_times for f in r.fills)
    assert min(f[0] for f in r.fills) >= r.mids[0][0]


# ---- REGIME-1: slow context must actually tilt the HMM (toward stress only) ------------
def test_regime1_slow_context_tilts_toward_stress():
    cfg = load_config(None)
    tr, te = split_by_books(_events(), 1800)
    params, scaler = fit_regimes(tr, cfg)
    base = run_sim(te, cfg, params, scaler)
    calm = run_sim(te, cfg, params, scaler, slow_context=lambda now: (-3.0, 0.0))
    hot = run_sim(te, cfg, params, scaler, slow_context=lambda now: (5.0, 1.0))
    stress = lambda r: np.mean(np.asarray(r.regimes) >= 2)
    assert stress(calm) == pytest.approx(stress(base))     # negative context never loosens
    assert stress(hot) >= stress(base)


# ---- JEV-1 / BRAIN-1: external models must see real features, not {} -----------------
def test_jev1_features_callback_receives_named_features():
    cfg = load_config(None)
    tr, te = split_by_books(_events(3000), 900)
    params, scaler = fit_regimes(tr, cfg)
    seen = []
    run_sim(te, cfg, params, scaler, on_features=seen.append)
    assert seen and {"vpin", "spread_bps", "bv_vol", "p_regime"} <= set(seen[-1])


def test_brain1_context_forwarded():
    from raim.runtime import ExternalContext
    from raim.slowbrain.providers import MockSlowProvider

    class Spy(MockSlowProvider):
        ctx = None

        async def infer(self, context):
            Spy.ctx = context
            return await super().infer(context)
    from raim.slowbrain.brain import SlowBrain
    import asyncio
    cfg = load_config(None)
    ctx = ExternalContext(cfg, brain=SlowBrain(Spy(), 60, 600), jev=jev.MockJev())
    ctx.update_features({"vpin": 0.4, "bv_vol": 1e-4})
    asyncio.run(ctx.brain.poll_once())
    assert Spy.ctx and Spy.ctx["fast_features"]["vpin"] == 0.4


# ---- JEV-2: malformed / out-of-range answers -> clamp or conservative fallback --------
@pytest.mark.parametrize("answers,expect_src", [
    ({"toxic_flow": {"type": "noul", "noul": 1.7},
      "regime": {"type": "choice", "choice": "R1", "confidence": 0.9, "probabilities": {"R1": 1.0}}}, "jev"),
    ({"toxic_flow": {"type": "noul", "noul": 0.2}}, "fallback"),
    ({"toxic_flow": {"type": "noul", "noul": "nan"},
      "regime": {"type": "choice", "choice": "R1", "confidence": 0.9, "probabilities": {"R1": 1.0}}}, "fallback"),
])
def test_jev2_malformed_answers(monkeypatch, answers, expect_src):
    import httpx
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-not-a-real-key")
    c = jev.HttpJev()
    c._cli = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers})))
    d = c.ask({})
    assert d.source == expect_src and 0.0 <= d.p_toxic <= 1.0
    assert set(d.regime_probs) == {"R1", "R2", "R3", "R4"}
