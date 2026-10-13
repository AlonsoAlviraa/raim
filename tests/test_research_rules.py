import json

import numpy as np
import pandas as pd
import pytest

from raim.paper.forward_gate import paper_summary
from raim.paper.kill_switch import KillSwitch
from raim.research import costs_curve, prereg
from raim.research.bootstrap import block_bootstrap
from raim.research.capacity import participation
from raim.research.mintrl import min_trl
from raim.research.pbo import cscv_pbo
from raim.research.regime_split import btc_regimes, regime_stats
from raim.research.spa import hansen_spa
from raim.research.stability import neighbors, plateau_select, stability
from raim.research.vault import Vault, VaultError
from raim.research.verdict import load_rules
from raim.strategies import engine


def test_pbo_noise_near_half_and_signal_low():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((1600, 20)) * 0.01
    assert 0.3 < cscv_pbo(noise, 16)["pbo"] < 0.7
    sig = noise + np.linspace(0, 0.003, 20)
    assert cscv_pbo(sig, 16)["pbo"] < 0.1


def test_spa_noise_vs_signal():
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((1000, 5)) * 0.01
    assert hansen_spa(noise, 500)["spa_p"] > 0.05
    assert hansen_spa(noise + 0.002, 500)["spa_p"] < 0.05


def test_mintrl():
    assert min_trl(0.0, 0, 3) == float("inf")
    assert min_trl(0.1, 0, 3) > min_trl(0.2, 0, 3) > 1
    assert min_trl(0.1, -1, 6) > min_trl(0.1, 0, 3)     # negative skew / fat tails need more data


def test_plateau_beats_spike():
    grid = [{"a": a, "b": b} for a in (1, 2, 3, 4, 5) for b in ("x", "y")]
    sc = [0.1] * 10
    sc[0] = 2.0                                  # isolated spike at a=1,b=x
    for k in (6, 7, 8, 9):                       # plateau at a=4..5
        sc[k] = 1.0
    assert set(neighbors(grid, 0)) == {1, 2}
    k = plateau_select(sc, grid)
    assert grid[k]["a"] in (4, 5)
    assert stability(sc, grid, 0) == 0.0 and stability(sc, grid, k) >= 0.5


def test_breakeven_cost_is_exact():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(2)
    bar = pd.DataFrame({"gross": 0.003 + 0.01 * rng.standard_normal(100), "cost": 0.001, "funding": 0.0}, index=idx)
    m = costs_curve.breakeven_multiple(bar)
    assert abs(costs_curve.sharpe_at(bar, m)) < 1e-9
    assert costs_curve.sharpe_at(bar, 2) < costs_curve.sharpe_at(bar, 1)


def test_bootstrap_quantiles_sane():
    rng = np.random.default_rng(3)
    b = block_bootstrap(0.002 + 0.01 * rng.standard_normal(800), n=500, mean_block=10)
    assert b["sharpe_p5"] < b["sharpe_p50"] < b["sharpe_p95"]
    assert b["dd_p99"] <= b["dd_p95"] <= 0 and b["losing_streak_p95_days"] >= 1


def test_regimes_are_causal():
    idx = pd.date_range("2020-01-01", periods=600, freq="D")
    c = pd.Series(np.r_[np.linspace(1, 3, 300), np.linspace(3, 1, 300)], index=idx)
    r = btc_regimes(c)
    c2 = c.copy()
    c2.iloc[450:] *= 5                           # change the future
    assert (btc_regimes(c2).loc[:idx[450]] == r.loc[:idx[450]]).all()
    st = regime_stats(pd.Series(0.001, index=idx) + np.random.default_rng(0).normal(0, 0.01, 600), r)
    assert set(st["regimes"]) == {"bull", "bear", "sideways"}


def test_vault_open_once_and_guard(tmp_path):
    rules = tmp_path / "rules.yaml"
    rules.write_text("version: 1\n")
    v = Vault(tmp_path / "v.json", tmp_path / "log.jsonl", rules)
    v.seal("s1", "2026-01-01", "2027-01-01")
    with pytest.raises(VaultError):
        v.guard("FAM", pd.date_range("2026-06-01", periods=3))
    v.guard("FAM", pd.date_range("2025-06-01", periods=3))          # before the seal: fine
    v.open_once("FAM", "s1")
    v.guard("FAM", pd.date_range("2026-06-01", periods=3))          # opened: allowed
    with pytest.raises(VaultError):
        v.open_once("FAM", "s1")                                     # second access refused
    rules.write_text("version: 2\n")
    with pytest.raises(VaultError):
        Vault(tmp_path / "v.json", tmp_path / "log.jsonl", rules).open_once("OTHER", "s1")
    log = [json.loads(x)["event"] for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert log == ["seal", "open"]


def test_vault_contamination_recorded(tmp_path):
    v = Vault(tmp_path / "v.json", tmp_path / "log.jsonl", tmp_path / "none.yaml")
    v.mark_contaminated("S3", ("2025-10-01", "2026-10-01"), "seen in OOS")
    assert v.status("S3")["status"] == "spent_contaminated"


def test_prereg_limits_and_retroactive(tmp_path):
    rules, _ = load_rules()
    log = tmp_path / "p.jsonl"
    ok = tmp_path / "ok.yaml"
    ok.write_text("family: F\nhypothesis: h\nmechanism: m\nuniverse: u\nmetric: sr\ngrid: {a: [1,2,3], b: [1,2]}\n")
    sha = prereg.register(ok, rules, log, now=100.0)
    assert prereg.check(sha, first_trial_ts=200.0, log=log)["ok"]
    assert not prereg.check(sha, first_trial_ts=50.0, log=log)["ok"]        # after the data
    assert not prereg.check(None, 200.0, log)["ok"]
    big = tmp_path / "big.yaml"
    big.write_text("family: G\nhypothesis: h\nmechanism: m\nuniverse: u\nmetric: sr\n"
                   "grid: {a: [1,2,3,4,5,6,7,8], b: [1,2,3,4,5,6,7]}\n")
    with pytest.raises(ValueError):
        prereg.register(big, rules, log)
    with pytest.raises(ValueError):
        prereg.validate({"family": "x"}, rules)


def test_engine_next_open_and_impact():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    opens = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41, 161.051]}, index=idx)
    closes = opens * 1.0
    W = pd.DataFrame({"A": [1, 1, 1, 1, 1, 1.0]}, index=idx)
    r = engine.run(W, closes.pct_change(), {"A": 0.0}, delay=0, fill="next_open", opens=opens)
    assert np.allclose(r.bar["gross"].iloc[1:-1], 0.1)
    with pytest.raises(ValueError):
        engine.run(W, closes.pct_change(), {"A": 0.0}, fill="next_open")
    imp = {"sigma": pd.DataFrame(0.02, index=idx, columns=["A"]),
           "adv": pd.DataFrame(1e6, index=idx, columns=["A"]), "capital": 1e4}
    r2 = engine.run(W, closes.pct_change(), {"A": 0.001}, impact=imp)
    r1 = engine.run(W, closes.pct_change(), {"A": 0.001})
    assert r2.bar["cost"].sum() > r1.bar["cost"].sum()


def test_capacity_and_paper_gate_and_kill_switch():
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    W = pd.DataFrame({"BTCUSDT_F": np.tile([0.0, 1.0], 30)}, index=idx)
    adv = pd.DataFrame({"BTCUSDT": 1e9}, index=idx)
    p = participation(W, adv, 1e5)
    assert p["p95"] == pytest.approx(1e-4)
    rng = np.random.default_rng(0)
    oos = pd.Series(0.001 + 0.01 * rng.standard_normal(500))
    s = paper_summary(oos.iloc[:100], 120, oos)
    assert s["days"] == 100 and s["within_ci"]
    ks = KillSwitch.from_backtest(oos.values, dd_p99=-0.2)
    assert not any(ks.update(x) for x in oos.iloc[:50])
    ks2 = KillSwitch.from_backtest(oos.values, dd_p99=-0.2)
    assert any(ks2.update(-0.01) for _ in range(40)) and ks2.reason


def test_aggressive_profile_only_loosens_drawdown(tmp_path):
    from raim.research.verdict import AGGRESSIVE_PATH
    base, s0 = load_rules()
    agg, s1 = load_rules(AGGRESSIVE_PATH)
    assert s0 != s1 and agg["profile"] == "aggressive_user_dd70"
    assert agg["r12_regime"]["worst_regime_min_dd"] == -0.70
    assert agg["r13_bootstrap"]["min_p95_dd"] == -0.70
    for sec in base:
        for k, v in (base[sec].items() if isinstance(base[sec], dict) else []):
            if (sec, k) not in {("r12_regime", "worst_regime_min_dd"), ("r13_bootstrap", "min_p95_dd")}:
                assert agg[sec][k] == v, (sec, k)
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"extends: {load_rules.__globals__['RULES_PATH']}\noverrides:\n  r5_dsr:\n    min_dsr: 0.1\n")
    with pytest.raises(ValueError):
        load_rules(bad)
