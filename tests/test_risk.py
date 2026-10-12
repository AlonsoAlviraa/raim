"""Risk monotonicity: external inputs (LLM / Jev) can only tighten. Runs on every backend
available, including the C++ governor when built."""
import itertools
import math

import pytest

from conftest import mod

KW = dict(budget_usd=50, z=2.33, h_unwind_s=30, depth_frac=0.1, max_gross_usd=500,
          max_drawdown_usd=25, stale_feed_ms=2000, max_reject_rate=0.2, max_msgs_per_s=10,
          loss_streak_widen=3, loss_streak_pause=6, pause_s=60)
EXT = [-5.0, -0.1, 0.0, 0.3, 0.7, 1.0, 1.5, 10.0, math.inf, math.nan]
STATES = [  # (pos, equity, rej, msg, sigma, depth)
    (0.0, 0.0, 0.0, 1.0, 1e-4, 10.0), (0.005, -1.0, 0.0, 1.0, 1e-4, 10.0),
    (-0.02, 0.0, 0.0, 1.0, 3e-4, 1.0), (0.0, 0.0, 0.5, 1.0, 1e-4, 10.0),
    (0.0, 0.0, 0.0, 50.0, 1e-4, 10.0), (0.01, 0.0, 0.0, 1.0, 1e-4, 0.0)]


def gov(backend):
    return mod("governor", backend).RiskGovernor(*KW.values())


def no_looser(d_ext, d_base):
    assert d_ext.size_mult <= d_base.size_mult + 1e-15
    assert d_ext.spread_mult >= d_base.spread_mult - 1e-15
    assert d_ext.halted >= d_base.halted and d_ext.cancel_all >= d_base.cancel_all
    assert d_ext.qmax == pytest.approx(d_base.qmax)  # qmax itself is rules-only


def test_external_never_loosens(backend):
    for (pos, eq, rej, msg, sig, dep), m1, m2 in itertools.product(STATES, EXT, EXT):
        base, ext = gov(backend), gov(backend)
        ext.set_external(m1, m2)
        a = (1000, 900, eq, pos, 60000.0, sig, dep, rej, msg)
        db, de = base.evaluate(*a), ext.evaluate(*a)
        no_looser(de, db)
        assert 0.0 <= de.size_mult <= 1.0
        # opening-side permissions can only shrink (reducing side always allowed if over limit)
        if pos >= 0:
            assert de.allow_bid <= db.allow_bid
        if pos <= 0:
            assert de.allow_ask <= db.allow_ask


def test_applied_multiplier_is_min(backend):
    g = gov(backend)
    g.set_external(0.6, 0.3)
    d = g.evaluate(1000, 1000, 0.0, 0.0, 60000.0, 1e-4, 10.0, 0.0, 1.0)
    assert d.size_mult == pytest.approx(0.3) and d.reason == "external_tighten"
    g.set_external(0.9, 0.95)
    for _ in range(3):
        g.on_trade_result(-1.0, 2000)
    d = g.evaluate(3000, 3000, 0.0, 0.0, 60000.0, 1e-4, 10.0, 0.0, 1.0)
    assert d.size_mult == pytest.approx(0.5) and d.spread_mult == 2.0  # m_rules wins


def test_nan_external_is_zero(backend):
    g = gov(backend)
    g.set_external(math.nan, 1.0)
    assert g.evaluate(0, 0, 0.0, 0.0, 60000.0, 1e-4, 10.0, 0.0, 1.0).size_mult == 0.0


def test_drawdown_halt_latched(backend):
    g = gov(backend)
    g.evaluate(0, 0, 10.0, 0.0, 60000.0, 1e-4, 10.0, 0.0, 1.0)
    d = g.evaluate(100, 100, -15.1, 0.0, 60000.0, 1e-4, 10.0, 0.0, 1.0)
    assert d.halted and d.cancel_all
    g.set_external(1.0, 1.0)
    d = g.evaluate(200, 200, 100.0, 0.0, 60000.0, 1e-4, 10.0, 0.0, 1.0)
    assert d.halted and not d.allow_bid and not d.allow_ask


def test_stale_feed_cancels(backend):
    d = gov(backend).evaluate(10_000, 7_000, 0.0, 0.0, 60000.0, 1e-4, 10.0, 0.0, 1.0)
    assert d.cancel_all and d.reason == "stale_feed"


def test_rate_backoff_and_loss_pause(backend):
    g = gov(backend)
    assert g.evaluate(0, 0, 0.0, 0.0, 6e4, 1e-4, 10.0, 0.5, 1.0).size_mult == 0.0
    for _ in range(6):
        g.on_trade_result(-1.0, 1000)
    d = g.evaluate(2000, 2000, 0.0, 0.0, 6e4, 1e-4, 10.0, 0.0, 1.0)
    assert d.reason == "loss_streak_pause" and d.cancel_all
    assert g.evaluate(62_000, 62_000, 0.0, 0.0, 6e4, 1e-4, 10.0, 0.0, 1.0).reason == "ok"


def test_qmax_formula(backend):
    g = gov(backend)
    sigma, mid, depth = 1e-4, 60000.0, 100.0
    expect = min(50 / (2.33 * sigma * mid * math.sqrt(30)), 0.1 * depth)
    assert g.qmax(sigma, mid, depth) == pytest.approx(expect)
    assert g.qmax(sigma, mid, 0.01) == pytest.approx(0.001)
    assert g.qmax(0.0, mid, depth) == 0.0


def test_gross_limit_blocks_increasing_side(backend):
    g = gov(backend)
    d = g.evaluate(0, 0, 0.0, 0.009, 60000.0, 1e-5, 1000.0, 0.0, 1.0)  # gross 540 > 500
    assert not d.allow_bid and d.allow_ask
