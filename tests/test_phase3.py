"""Phase 3 (deep research 2026-10) — written BEFORE implementation (tests-first)."""
import math

import numpy as np
import pytest

from conftest import HAS_CPP, mod

# ---------------------------------------------------------------- (1) flow-direction CPD
def _flowcpd(backend):
    m = mod("flowcpd", backend)
    return m.FlowCPD(0.01, 1.0, 1.0, 200, 30)


def test_flowcpd_detects_flow_sign_regime_change(backend):
    rng = np.random.default_rng(0)
    signs = np.r_[np.where(rng.random(400) < 0.5, 1, -1), np.where(rng.random(200) < 0.9, 1, -1)]
    f = _flowcpd(backend)
    out = np.array([f.update(int(s)) for s in signs])
    pc, bias = out[:, 0], out[:, 1]
    assert abs(bias[350:400].mean()) < 0.3
    assert bias[-50:].mean() > 0.6
    assert pc[400:440].max() > pc[100:390].mean() + 0.2


def test_flowcpd_is_causal(backend):
    f1, f2 = _flowcpd(backend), _flowcpd(backend)
    s = [1, -1, 1, 1, 1, -1, 1, 1, 1, 1] * 20
    a = [f1.update(x) for x in s[:100]]
    b = [f2.update(x) for x in s][:100]
    np.testing.assert_allclose(a, b)


@pytest.mark.skipif(not HAS_CPP, reason="raim_core not built")
def test_flowcpd_parity():
    rng = np.random.default_rng(1)
    a, b = _flowcpd("python"), _flowcpd("cpp")
    for s in np.where(rng.random(3000) < 0.6, 1, -1):
        np.testing.assert_allclose(b.update(int(s)), a.update(int(s)), rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------- (2) exposure asymmetry
def _adj(backend, *a):
    return mod("quoter", backend).exposure_adjust(*a)


def test_exposure_only_widens_and_shrinks(backend):
    rng = np.random.default_rng(2)
    for _ in range(2000):
        qb, qa = (rng.choice([0.0, 1e300, rng.random() * 10]) for _ in range(2))
        be, ae, bm, am = _adj(backend, qb, qa, rng.uniform(-1, 1), rng.random(), 2.0, 1.0, 5.0)
        assert 0.0 <= be <= 5.0 and 0.0 <= ae <= 5.0
        assert 0.0 <= bm <= 1.0 and 0.0 <= am <= 1.0


def test_exposure_buy_flow_protects_ask(backend):
    be, ae, bm, am = _adj(backend, 5.0, 5.0, 0.8, 0.9, 2.0, 1.0, 5.0)
    assert ae > be and am < bm


def test_exposure_short_queue_side_is_more_exposed(backend):
    be, ae, bm, am = _adj(backend, 0.1, 9.0, 0.0, 0.0, 2.0, 1.0, 5.0)
    assert be > ae


def test_exposure_nan_inputs_conservative(backend):
    be, ae, bm, am = _adj(backend, math.nan, 1.0, math.nan, math.nan, 2.0, 1.0, 5.0)
    assert be == 5.0 and ae == 5.0 and bm == 0.0 and am == 0.0


@pytest.mark.skipif(not HAS_CPP, reason="raim_core not built")
def test_exposure_parity():
    rng = np.random.default_rng(3)
    for _ in range(3000):
        args = (rng.random() * 10, rng.random() * 10, rng.uniform(-1, 1), rng.random(), 2.0, 1.0, 5.0)
        np.testing.assert_allclose(_adj("cpp", *args), _adj("python", *args), rtol=1e-12)


# ---------------------------------------------------------------- RISK-2 taker unwind in sim
def test_sim_take_walks_l2_with_limit_and_taker_fee(backend):
    e = mod("sim", backend).SimExchange(1.0, 0.0, 30.0, 0.0, 30.0, 1.0, 1, 5.0)
    e.on_book_l2(0, [100.0, 99.9], [1.0, 2.0], [100.1, 100.2], [1.0, 2.0])
    f = e.take(0, -1, 2.5, 99.9)                     # sell 2.5 into bids, limit 99.9
    assert [round(x[3], 9) for x in f] == [1.0, 1.5] and [x[2] for x in f] == [100.0, 99.9]
    assert f[0][4] == pytest.approx(100.0 * 1.0 * 5e-4)
    assert e.take(0, -1, 1.0, 100.5) == []           # limit not reachable -> nothing


def test_sim_queue_ahead_accessor(backend):
    e = mod("sim", backend).SimExchange(1.0, 0.0, 30.0, 0.0, 30.0, 1.0, 1, 5.0)
    e.on_book_l2(0, [100.0], [4.0], [100.1], [3.0])
    oid = e.place(0, 1, 100.0, 1.0)
    assert math.isnan(e.queue_ahead(oid))           # pending
    e.on_book_l2(40, [100.0], [4.0], [100.1], [3.0])
    assert e.queue_ahead(oid) == 4.0


def test_unwinder_passive_then_aggressive():
    from raim.execution.unwind import Unwinder
    u = Unwinder(passive_s=5.0, clip=0.01, max_slip_bps=10.0)
    a = u.plan(0, 0.025, 100.0, 100.1)
    assert a.kind == "passive" and a.side == -1 and a.price == 100.1 and a.qty == pytest.approx(0.01)
    b = u.plan(6000, 0.025, 100.0, 100.1)
    assert b.kind == "aggressive" and b.side == -1 and b.limit == pytest.approx(100.0 * (1 - 10e-4))
    assert u.plan(7000, 0.0, 100.0, 100.1).kind == "done"
    assert u.plan(8000, -0.004, 100.0, 100.1).qty == pytest.approx(0.004)   # never overshoots


def test_halt_triggers_unwind_to_flat_in_sim():
    from raim.config import load_config
    from raim.research.pipeline import fit_regimes, split_by_books
    from raim.sim.engine import run_sim
    from raim.sim.scenarios import Scenario, scenario_events
    cfg = load_config(None, {"risk": {"max_drawdown_usd": 0.05},
                             "quoting": {"min_half_spread_bps": 0.01, "gamma": 0.01, "kappa": 100.0}})
    ev = scenario_events(Scenario("drift_up", drift_bps=0.3), n_steps=6000, seed=3)
    tr, te = split_by_books(ev, 1500)
    p, s = fit_regimes(tr, cfg)
    r = run_sim(te, cfg, p, s)
    assert r.halted and abs(r.final_position) < 1e-9 and r.unwind_fills > 0


# ---------------------------------------------------------------- RISK-3 side-aware depth
def test_unwind_depth_side_aware():
    from raim.sim.engine import unwind_depth
    assert unwind_depth(0.5, 1.0, 9.0) == 1.0     # long: sells into bids
    assert unwind_depth(-0.5, 1.0, 9.0) == 9.0    # short: buys from asks
    assert unwind_depth(0.0, 1.0, 9.0) == 1.0     # flat: worst side


# ---------------------------------------------------------------- (3) stress index baseline
def test_stress_index_orientation_and_bands():
    from raim.research.stress_index import StressIndex
    rng = np.random.default_rng(0)
    calm = rng.normal(-1, 0.3, (500, 6)); hot = rng.normal(1.5, 0.3, (500, 6))
    x = np.r_[calm, hot]; x[:, 1] += np.r_[np.zeros(500), np.ones(500)]
    si = StressIndex.fit(x, n_bands=4, vol_col=1)
    assert si.score(hot[0]) > si.score(calm[0])
    p = si.posterior(x[500:].mean(0) + 0.5)   # clearly above the 75th train percentile
    assert len(p) == 4 and p[3] == 1.0
    assert si.n_bands <= 4


def test_compare_regime_models_report():
    from raim.config import load_config
    from raim.research.pipeline import compare_regime_models
    from raim.sim.scenarios import Scenario, scenario_events
    df = compare_regime_models(scenario_events(Scenario("base"), 8000, 5), load_config(None), n_folds=3)
    assert set(df["model"]) == {"hmm", "stress_index"} and (df.groupby("model").size() >= 2).all()


# ---------------------------------------------------------------- (4) robustness protocol
def test_cvar_and_inventory_bias():
    from raim.research.robustness import cvar, inventory_bias_test
    x = np.arange(100, dtype=float) - 50
    assert cvar(x, 0.10) == pytest.approx(np.mean(x[:10]))
    assert inventory_bias_test(np.random.default_rng(0).normal(0, 1, 200), seed=1)["p_value"] > 0.05
    assert inventory_bias_test(np.random.default_rng(0).normal(2, 1, 200), seed=1)["p_value"] < 0.01


def test_stress_battery_runs():
    from raim.config import load_config
    from raim.research.robustness import stress_battery
    df = stress_battery(load_config(None), seeds=(1, 2), n_steps=3000,
                        scenarios=["base", "drift_up", "thin", "hawkes_high"])
    assert {"scenario", "cvar10_usd", "cvar30_usd", "inv_bias_p", "mean_pnl_usd"} <= set(df.columns)
    assert len(df) == 4


# ---------------------------------------------------------------- (5) ablation
def test_ablation_cost_aware():
    from raim.config import load_config
    from raim.research.ablation import ablation
    from raim.sim.scenarios import Scenario, scenario_events
    ev = scenario_events(Scenario("base"), 5000, 7)
    df = ablation(ev, load_config(None), m_llm=lambda t: 0.5, m_jev=lambda t: 0.8,
                  cost_llm_usd=0.01, cost_jev_usd=0.0001, llm_calls=10, jev_calls=100)
    assert list(df["arm"]) == ["rules_only", "+llm", "+jev", "+llm+jev"]
    row = df.set_index("arm")
    assert row.loc["rules_only", "api_cost_usd"] == 0.0
    assert row.loc["+llm", "api_cost_usd"] == pytest.approx(0.1)
    assert (df["net_usd"] == df["pnl_usd"] - df["api_cost_usd"]).all()


# ---------------------------------------------------------------- (6) registry + DSR
def test_deflated_sharpe_penalizes_trials():
    from raim.research.registry import deflated_sharpe
    one = deflated_sharpe(sr=0.1, n_obs=1000, n_trials=1, sr_var=0.002)
    many = deflated_sharpe(sr=0.1, n_obs=1000, n_trials=200, sr_var=0.002)
    assert 0 <= many < one <= 1


def test_registry_append_only_and_frozen_validator(tmp_path):
    from raim.research.registry import FrozenValidator, TrialRegistry
    r = TrialRegistry(tmp_path / "trials.jsonl")
    r.record({"gamma": 0.1}, {"sharpe": 0.2}); r.record({"gamma": 0.2}, {"sharpe": 0.3})
    assert r.n_trials() == 2
    data = tmp_path / "test.jsonl"; data.write_text("x\n", encoding="utf-8")
    lock = tmp_path / "validator.lock"
    FrozenValidator.freeze(lock, data)
    FrozenValidator(lock).verify(data)
    data.write_text("y\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        FrozenValidator(lock).verify(data)


# ---------------------------------------------------------------- (7) Jev calibration
def test_calibration_metrics_and_gate():
    from raim.research.calibration import brier, calibration_gate, ece
    rng = np.random.default_rng(0)
    p = rng.random(5000); y = (rng.random(5000) < p).astype(float)
    assert brier(p, y) < 0.2 and ece(p, y) < 0.03
    assert calibration_gate(p, y, max_ece=0.05, min_n=500)["passed"]
    bad = np.clip(p * 0.2 + 0.8, 0, 1)
    assert not calibration_gate(bad, y, max_ece=0.05, min_n=500)["passed"]
    assert not calibration_gate(p[:50], y[:50], max_ece=0.05, min_n=500)["passed"]


def test_uncalibrated_jev_is_not_used():
    from raim.config import load_config
    from raim.jev.adapter import MockJev
    from raim.runtime import ExternalContext
    ctx = ExternalContext(load_config(None), jev=MockJev())
    assert ctx.jev_enabled is False
    ctx._jev_tick({"vpin": 0.9})
    assert ctx.multipliers()[1] == 1.0                 # ignored, not trusted
    ctx.set_calibration({"passed": True})
    ctx._jev_tick({"vpin": 0.9})
    assert ctx.multipliers()[1] < 1.0


# ---------------------------------------------------------------- (8) supply chain
def test_mcp_command_pinned_and_npm_refused():
    from raim.config import load_config
    from raim.slowbrain.mcp_client import validate_mcp_command
    cmd = load_config("configs/default.yaml").slow_brain.mcp_command
    assert "tradingview-mcp-server==0.8.1" in cmd
    validate_mcp_command(cmd)
    for bad in (["npx", "tradingview-mcp"], ["npm", "exec", "tradingview-mcp"], ["node", "x.js"],
                ["uvx", "--from", "tradingview-mcp-server", "tradingview-mcp"],
                ["uvx", "--from", "tradingview-mcp-clone==1.0", "tradingview-mcp"]):
        with pytest.raises(ValueError):
            validate_mcp_command(bad)


def test_repo_has_no_npm_tradingview_install():
    import pathlib
    import re
    pat = re.compile(r"(npm\s+(i|install)|npx)\s+\S*tradingview", re.I)
    root = pathlib.Path(__file__).resolve().parents[1]
    hits = [p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".md", ".yaml", ".yml", ".ps1", ".sh", ".toml"}
            and "tests" not in p.parts and ".venv" not in p.parts and pat.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert hits == []
