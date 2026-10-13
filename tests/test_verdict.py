"""Verdict CI: pure noise -> KILL; planted signal -> passes every statistical rule; reports
without prereg / manifest are rejected; rules file cannot change alongside results."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raim.research.evaluate import evidence
from raim.research.verdict import STATISTICAL, ReportError, load_rules, validate_report, verdict
from raim.strategies.engine import BTResult, metrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))


def _fake(mu_per_variant, seed, T=1460):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=T, freq="D")
    grid = [{"p": k} for k in range(len(mu_per_variant))]
    results = {}
    for k, mu in enumerate(mu_per_variant):
        d = pd.Series(mu + 0.01 * rng.standard_normal(T), index=idx)
        bar = pd.DataFrame({"gross": d + 2e-5, "cost": 2e-5, "funding": 0.0, "net": d,
                            "turnover": 0.01, "exposure": 1.0}, index=idx)
        results[k] = BTResult(bar=bar, daily=d, weights=pd.DataFrame(index=idx))
    best = int(np.argmax(mu_per_variant))
    oos_idx = idx[365:]
    oos = results[best].daily.loc[oos_idx]
    m = metrics(oos, results[best].bar)
    folds = np.array_split(oos, 5)
    m["n_folds"], m["folds_positive"] = 5, int(sum(f.sum() > 0 for f in folds))
    res = {"oos_daily": oos, "oos_bar": results[best].bar.loc[oos_idx], "metrics": m, "folds": [],
           "results": results, "grid": grid, "selector": "plateau", "chosen_idx": [best] * 5,
           "purge_days": 10, "embargo_days": 20}
    btc = pd.Series(np.exp(np.cumsum(0.02 * rng.standard_normal(T) + np.sin(np.arange(T) / 150) * 0.004)) * 1e4,
                    index=idx)
    return res, btc


def _report(res, btc, tmp_path, n_trials=10, sr_sd=0.02):
    rules, _ = load_rules()
    reg = tmp_path / "trials.jsonl"
    rng = np.random.default_rng(0)
    with reg.open("w") as fh:
        for i in range(n_trials):
            fh.write(json.dumps({"trial": f"t{i}", "params": {"strategy": "FAM", "p": i, "stage": "grid"},
                                 "metrics": {"sr_daily": float(rng.normal(0, sr_sd)), "n_days": 1000,
                                             "pct_days_active": 1.0}}) + "\n")
    ev = evidence(res, rules, btc, res["oos_daily"] * 0.9)
    return {"name": "FAM", "family": "FAM", "prereg_sha": "abc", "data_manifest_sha": "def",
            "registry_path": str(reg), **ev, "n_trades": 500, "max_horizon_days": 5,
            "capacity_p95": 0.001, "pit_universe": True,
            "prereg_check": {"ok": True, "n_variants": 10}, "vault": {"status": "sealed"}}


def test_pure_noise_is_killed(tmp_path):
    res, btc = _fake([0.0] * 10, seed=1)
    v = verdict(_report(res, btc, tmp_path))
    assert v["verdict"] == "KILL"
    assert not v["checks"]["R5"]["pass"] and not v["checks"]["R7"]["pass"]


def test_planted_signal_passes_statistical_rules(tmp_path):
    res, btc = _fake([0.0003 * k for k in range(10)], seed=2)
    v = verdict(_report(res, btc, tmp_path))
    failed = [k for k in STATISTICAL if not v["checks"][k]["pass"]]
    assert failed == [], v["checks"]
    # vault never opened -> cannot be better than WATCH; never PROMOTE without paper
    assert v["verdict"] == "WATCH"


def test_planted_signal_with_clean_vault_is_only_candidate_for_paper(tmp_path):
    res, btc = _fake([0.0003 * k for k in range(10)], seed=2)
    rep = _report(res, btc, tmp_path)
    rep["vault"] = {"status": "opened", "holdout_sharpe": 1.2}
    assert verdict(rep)["verdict"] == "CANDIDATE_FOR_PAPER"
    rep["paper"] = {"days": 95, "trades": 120, "sharpe": rep["oos"]["sharpe"], "within_ci": True}
    assert verdict(rep)["verdict"] == "PROMOTE"


@pytest.mark.parametrize("missing", ["prereg_sha", "data_manifest_sha"])
def test_report_without_prereg_or_manifest_is_rejected(missing, tmp_path):
    rep = {"name": "x", "family": "x", "prereg_sha": "a", "data_manifest_sha": "b",
           "registry_path": "r", "oos": {}}
    rep[missing] = None
    with pytest.raises(ReportError):
        validate_report(rep)
    import check_reports
    (tmp_path / "r.json").write_text(json.dumps(rep))
    assert check_reports.check_dir(tmp_path, tmp_path / "none.jsonl", None)


def test_check_reports_rejects_unknown_prereg(tmp_path):
    import check_reports
    rep = {"name": "x", "family": "x", "prereg_sha": "zzz", "data_manifest_sha": "b",
           "registry_path": "r", "oos": {"a": 1}}
    (tmp_path / "r.json").write_text(json.dumps(rep))
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"sha": "other"}) + "\n")
    errs = check_reports.check_dir(tmp_path, log, "b")
    assert errs and "prereg_sha" in errs[0]


def test_rules_file_cannot_change_with_results():
    import check_rules_separation as c
    assert c.violations(["configs/research_rules.yaml", "docs/research/x.md"])
    assert c.violations(["configs/research_rules.yaml", "reports/research_rules/S1.json"])
    assert not c.violations(["configs/research_rules.yaml", "src/raim/research/verdict.py"])
    assert not c.violations(["docs/research/x.md"])


def test_verdict_reads_thresholds_only_from_yaml():
    import re
    src = (ROOT / "src/raim/research/verdict.py").read_text()
    body = src.split("def verdict(", 1)[1]
    # no numeric threshold literal (other than 0) is compared inside verdict(): all come from YAML
    code = re.sub(r'f?"[^"]*"', '""', body)
    lits = re.findall(r"[<>]=?\s*(-?\d*\.?\d+)", code)
    assert all(float(x) == 0 for x in lits), lits
    for f in ("scripts/run_strategies.py", "scripts/run_convex.py"):
        s = (ROOT / f).read_text()
        assert '"PROMOTE"' not in s and '"WATCH" if' not in s, f"duplicated verdict code in {f}"


def test_market_neutral_benchmark_is_cash_only(tmp_path):
    res, btc = _fake([0.0003 * k for k in range(10)], seed=2)
    rep = _report(res, btc, tmp_path)
    rep["spa"] = {"btc_buy_hold": 0.9, "cash": 0.01}
    assert not verdict(rep)["checks"]["R7"]["pass"]
    rep["market_neutral"] = True
    c = verdict(rep)["checks"]["R7"]
    assert c["pass"] and list(c["value"]) == ["cash"]
