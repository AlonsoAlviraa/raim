import pytest

from conftest import mod
from raim.config import load_config
from raim.data.synthetic import SyntheticCfg, generate
from raim.execution.guard import LiveTradingDisabled, assert_not_live
from raim.research.pipeline import train_test_run, walk_forward
from raim.types import BookSnapshot


def ex(backend, stress=1.0):
    return mod("sim", backend).SimExchange(1.0, 0.0, 30.0, 0.0, 30.0, stress, 1)


def test_queue_position_and_partial_fill(backend):
    e = ex(backend)
    e.on_book(0, 100.0, 5.0, 100.1, 5.0)
    oid = e.place(0, 1, 100.0, 2.0)
    assert e.on_book(10, 100.0, 5.0, 100.1, 5.0) == []      # not yet acked
    e.on_book(40, 100.0, 5.0, 100.1, 5.0)                     # active, 5 ahead
    assert e.on_trade(50, 100.0, 4.0, -1) == []               # eats queue only
    f = e.on_trade(60, 100.0, 2.0, -1)                        # 1 ahead, 1 for us
    assert len(f) == 1 and f[0][3] == pytest.approx(1.0) and f[0][0] == oid
    f = e.on_trade(70, 100.0, 5.0, -1)
    assert f[0][3] == pytest.approx(1.0)                      # remainder, order done
    assert e.open_orders() == []


def test_fee_and_trade_through(backend):
    e = ex(backend)
    e.on_book(0, 100.0, 5.0, 100.1, 5.0)
    e.place(0, -1, 100.2, 1.0)
    e.on_book(40, 100.0, 5.0, 100.1, 5.0)
    f = e.on_book(50, 100.25, 1.0, 100.3, 1.0)                # bid moved through our ask
    assert f and f[0][4] == pytest.approx(100.2 * 1.0 * 1e-4)


def test_post_only_reject_and_cancel_latency(backend):
    e = ex(backend)
    e.on_book(0, 100.0, 5.0, 100.1, 5.0)
    e.place(0, 1, 100.1, 1.0)
    e.on_book(40, 100.0, 5.0, 100.1, 5.0)
    assert e.n_rejects == 1
    oid = e.place(40, 1, 100.0, 1.0)
    e.on_book(80, 100.0, 0.0, 100.1, 5.0)
    e.cancel(80, oid)
    assert e.on_trade(90, 100.0, 1.0, -1)                     # still fillable before cancel ack
    oid2 = e.place(200, 1, 99.9, 1.0)
    e.on_book(240, 100.0, 1.0, 100.1, 1.0)
    e.cancel(240, oid2)
    e.on_book(300, 100.0, 1.0, 100.1, 1.0)
    assert e.open_orders() == []


def test_latency_stress_delays_ack(backend):
    e = ex(backend, stress=2.0)
    e.on_book(0, 100.0, 1.0, 100.1, 1.0)
    e.place(0, 1, 100.0, 1.0)
    e.on_book(40, 100.0, 1.0, 100.1, 1.0)
    assert e.open_orders()[0][4] == 0                         # still pending at 2x latency
    e.on_book(61, 100.0, 1.0, 100.1, 1.0)
    assert e.open_orders()[0][4] == 1


def _events(n, seed=4):
    b, t, _ = generate(SyntheticCfg(n_steps=n, seed=seed))
    return sorted(b + t, key=lambda e: (e.ts_ms, isinstance(e, BookSnapshot)))


def test_end_to_end_sim_is_deterministic_and_bounded():
    cfg = load_config(None)
    ev = _events(8000)
    r1, r2 = train_test_run(ev, cfg).result, train_test_run(ev, cfg).result
    assert r1.final_equity == r2.final_equity and len(r1.fills) == len(r2.fills)
    qmax_cap = cfg.risk.max_gross_usd / 50_000 + cfg.quoting.order_size * 2
    assert abs(r1.final_position) <= qmax_cap
    assert r1.max_drawdown <= cfg.risk.max_drawdown_usd + 5


def test_walk_forward_runs():
    df = walk_forward(_events(8000), load_config(None), n_folds=3)
    assert len(df) >= 2 and (df["train_books"] > 0).all()


def test_live_mode_hard_fails(monkeypatch):
    monkeypatch.setenv("RAIM_LIVE_CONFIRM", "YES")
    with pytest.raises(LiveTradingDisabled):
        assert_not_live("live")
    cfg = load_config(None, {"mode": "live"})
    with pytest.raises(LiveTradingDisabled):
        train_test_run(_events(2000), cfg)
