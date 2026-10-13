"""Run the edge-v2 families (aggressive DD profile) (BACKTEST ONLY): development < 2025-10-01 with purged walk-forward,
verdict via raim.research.verdict, then ONE-SHOT holdout 2025-10-01..2026-10-01 (Vault.open_once,
scored designer-contaminated per configs/prereg/v1/HOLDOUT_POLICY.yaml). Never reads >= 2026-10-13."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import rescore_rules as RR  # noqa: E402  (patches engine.run with the R11 extra-delay hook)
from raim.research import prereg  # noqa: E402
from raim.research.capacity import participation  # noqa: E402
from raim.research.evaluate import evidence  # noqa: E402
from raim.research.registry import TrialRegistry  # noqa: E402
from raim.research.stability import plateau_select  # noqa: E402
from raim.research.vault import Vault  # noqa: E402
from raim.research.verdict import AGGRESSIVE_PATH, load_rules, verdict  # noqa: E402
from raim.strategies import edge_v2 as X, engine, universe  # noqa: E402
from raim.strategies.perp_panel import load_perp_panel  # noqa: E402
from raim.strategies.costs import VIP0  # noqa: E402
from raim.strategies.wf import walk_forward  # noqa: E402

log = logging.getLogger("edge_v2")
OUT = ROOT / "reports/edge_v2"
REG = ROOT / "reports/strategies/trials.jsonl"   # copy of the shared registry (N=all trials)
SPECS = ROOT / "configs/prereg/edge_v2"
DEV_END, HO_END = "2025-10-01", "2026-10-01"


def costs(qv):
    return universe.slip_tier(qv) + VIP0.perp_fee


def make_ev(fam, Pn, prem):
    close, qv = Pn["close"], Pn["qv"]
    R = close.pct_change(fill_method=None)
    C = costs(qv)
    F = Pn["funding"].fillna(0.0)
    univ = {n: universe.pit_universe(qv, n) for n in (30, 60)}
    cache = {}

    def ev(p):
        key = tuple(sorted(p.items()))
        if key not in cache:
            if fam.startswith("X1"):
                W = X.x1_weights(Pn, prem, univ[p["N"]], p["sig"], p["W"], p["reb"])
            elif fam.startswith("X2"):
                W = X.x2_weights(Pn, univ[60], p["L"], p["q"], p["btc_filter"])
            else:
                W = X.x3_weights(Pn, univ[60], p["p"], p["H"], p["mode"])
            cache[key] = W
        return engine.run(cache[key], R, C, F, delay=1)
    return ev


def cut(panel, end):
    return {k: v.loc[v.index < pd.Timestamp(end, tz=v.index.tz)] for k, v in panel.items()}


def main():
    global MANIFEST_SHA
    from raim.data import manifest
    mf = manifest.build(["/workspace/data_hist/perp_1d", "/workspace/data_hist/funding_all", "/workspace/data_hist/premium_1d"],
                        Path("/workspace/data_hist/.manifest_cache.json"))
    MANIFEST_SHA = mf["manifest_sha"]
    (ROOT / "docs/research/edge_v2_2026-10").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/research/edge_v2_2026-10/data_manifest_sha.json").write_text(json.dumps({"manifest_sha": MANIFEST_SHA, "n_files": mf["n_files"]}))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    rules, _ = load_rules(AGGRESSIVE_PATH)
    prem = X.load_premium()
    vault = Vault()
    reg = TrialRegistry(REG)
    full = load_perp_panel()
    dev = cut(full, DEV_END)
    ho_panel = cut(full, HO_END)
    btc_dev = dev["close"]["BTCUSDT"]
    btc_ho = ho_panel["close"]["BTCUSDT"]
    policy = yaml.safe_load((SPECS / "HOLDOUT_POLICY.yaml").read_text())
    for spec_path in sorted(SPECS.glob("X*.yaml")):
        spec = yaml.safe_load(spec_path.read_text())
        pol = policy["families"][spec["family"]]
        fam, grid = spec["family"], spec["grid"]
        if (OUT / f"{fam}.json").exists():
            continue
        sha = prereg.file_sha(spec_path)
        commit_ts = prereg.git_commit_ts(spec_path)
        t0 = time.time()
        vault.guard(fam, dev["close"].index)                       # dev must not touch holdouts
        ev = make_ev(fam, dev, prem.loc[prem.index < pd.Timestamp(DEV_END, tz=prem.index.tz)])
        RR.EXTRA[0] = 0
        res = walk_forward(fam, grid, ev, reg, purge_days=90, embargo_days=7, min_train_days=365,
                           min_embargo_frac=0.01, selector="plateau")
        RR.EXTRA[0] = 1
        res_d = walk_forward(fam + "_delay2", grid, ev, reg, purge_days=90, embargo_days=7,
                             min_train_days=365, min_embargo_frac=0.01, selector="plateau")
        RR.EXTRA[0] = 0
        first_ts = min(x["ts"] for x in reg.all() if x["params"].get("strategy") == fam)
        pc = prereg.check(sha, first_ts)
        pc["git_commit_ts"], pc["commit_before_first_trial"] = commit_ts, bool(commit_ts and commit_ts < first_ts)
        pc["ok"] = pc["ok"] and pc["commit_before_first_trial"]
        evd = evidence(res, rules, btc_dev, res_d["oos_daily"])
        n_tr, hold_days, W = RR.trade_stats(res)
        cap = participation(W, dev["qv"], rules["r14_data_capacity"]["target_capital_usd"])
        # ---- one-shot holdout: freeze the plateau centre on the full dev sample, open ONCE
        M = pd.DataFrame({k: r.daily for k, r in res["results"].items()}).fillna(0.0)
        sc = [float(M[k].mean() / M[k].std()) if M[k].std() > 0 else -np.inf for k in M.columns]
        k_star = plateau_select(sc, grid)
        vault.open_once(fam, "holdout_2025-10_edge2")
        r_ho = make_ev(fam, ho_panel, prem.loc[prem.index < pd.Timestamp(HO_END, tz=prem.index.tz)])(grid[k_star])
        d = r_ho.daily
        d = d[(d.index >= pd.Timestamp(DEV_END, tz=d.index.tz))]
        hm = engine.metrics(d)
        bh = ho_panel["close"]["BTCUSDT"].pct_change().loc[d.index]
        ho = {"param": grid[k_star], "sharpe": hm["sharpe"], "ann_ret": hm["ann_ret"], "max_dd": hm["max_dd"],
              "total_ret": hm["total_ret"], "btc_bh_total_ret": float((1 + bh.fillna(0)).prod() - 1),
              "btc_bh_sharpe": engine.metrics(bh.fillna(0))["sharpe"], "days": hm["n_days"]}
        rep = {"name": fam, "family": fam, "prereg_sha": sha, "prereg_check": pc,
               "data_manifest_sha": MANIFEST_SHA,
               "registry_path": str(REG), **evd, "n_trades": n_tr, "max_horizon_days": hold_days or 0,
               "capacity": cap, "capacity_p95": cap.get("p95"), "pit_universe": True,
               "market_neutral": bool(spec.get("market_neutral")), "rules_profile": "aggressive_user_dd70",
               "payoff_dev_oos": X.payoff_stats(res["oos_daily"]), "payoff_holdout": X.payoff_stats(d),
               "vault": {"status": pol["r3_status"], "reason": pol["reason"].strip(),
                         "holdout_sharpe": hm["sharpe"], "holdout": ho},
               "paper": None, "folds": res["folds"],
               "oos_window": [str(res["oos_daily"].index[0].date()), str(res["oos_daily"].index[-1].date())],
               "dev_end": DEV_END, "elapsed_s": round(time.time() - t0, 1)}
        rep["verdict"] = verdict(rep, AGGRESSIVE_PATH)
        (OUT / f"{fam}.json").write_text(json.dumps(rep, indent=2, default=str))
        log.info("%s -> %s failed=%s holdout SR=%.2f", fam, rep["verdict"]["verdict"],
                 rep["verdict"]["failed"], hm["sharpe"])


if __name__ == "__main__":
    main()
