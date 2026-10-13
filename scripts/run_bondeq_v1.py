"""bondeq-v1 (BACKTEST ONLY, virtual capital): dev 2010-01-01..2025-10-01 purged walk-forward, verdict with
the BASE rules (aggressive DD profile also reported), one-shot designer-contaminated holdout 2025-10..2026-10.
Trial registry = the GLOBAL shared file (env RAIM_TRIAL_REGISTRY), so N counts every trial of the program."""
from __future__ import annotations

import json, logging, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import rescore_rules as RR  # noqa: E402
from raim.research import prereg  # noqa: E402
from raim.research.evaluate import evidence  # noqa: E402
from raim.research.registry import TrialRegistry  # noqa: E402
from raim.research.stability import plateau_select  # noqa: E402
from raim.research.vault import Vault  # noqa: E402
from raim.research.verdict import AGGRESSIVE_PATH, load_rules, verdict  # noqa: E402
from raim.strategies import bondeq as B, engine  # noqa: E402
from raim.strategies.edge_v2 import payoff_stats  # noqa: E402
from raim.strategies.wf import walk_forward  # noqa: E402

log = logging.getLogger("bondeq")
OUT = ROOT / "reports/bondeq_v1"
DOC = ROOT / "docs/research/bondeq_v1_2026-10"
REG = Path(os.environ.get("RAIM_TRIAL_REGISTRY", "/workspace/raim-strategies/reports/strategies/trials.jsonl"))
SPECS = ROOT / "configs/prereg/bondeq_v1"
START, DEV_END, HO_END = "2010-01-01", "2025-10-01", "2026-10-01"


def make_ev(fam, P):
    def ev(p):
        if fam.startswith("B1"):
            W = B.b1_weights(P, p["k"], p["mode"])
        elif fam.startswith("B2"):
            W = B.b2_weights(P, p["L"], p["alt"])
        else:
            W = B.b3_weights(P, p["N"], p["alt"])
        R, O, C, F = B.returns_and_costs(P, list(W.columns))
        W = W.loc[W.index >= pd.Timestamp(START, tz="UTC")]
        R, O, F = (x.loc[x.index >= pd.Timestamp(START, tz="UTC")] for x in (R, O, F))
        return engine.run(W, R, C, F, delay=1, fill="next_open", opens=O)
    return ev


def main():
    from raim.data import manifest
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mf = manifest.build([str(B.ROOT)], B.ROOT / ".manifest_cache.json")
    OUT.mkdir(parents=True, exist_ok=True); DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "data_manifest_sha.json").write_text(json.dumps({"manifest_sha": mf["manifest_sha"], "n_files": mf["n_files"]}))
    rules, _ = load_rules()
    vault, reg = Vault(), TrialRegistry(REG)
    dev, ho = B.load_panel(DEV_END), B.load_panel(HO_END)
    spy_dev, spy_ho = dev["close"]["SPY"], ho["close"]["SPY"]
    policy = yaml.safe_load((SPECS / "HOLDOUT_POLICY.yaml").read_text())
    for sp in sorted(SPECS.glob("B*.yaml")):
        spec = yaml.safe_load(sp.read_text()); fam, grid = spec["family"], spec["grid"]
        if (OUT / f"{fam}.json").exists():
            continue
        pol = policy["families"][fam]
        t0 = time.time()
        vault.guard(fam, dev["close"].index)
        ev = make_ev(fam, dev)
        RR.EXTRA[0] = 0
        res = walk_forward(fam, grid, ev, reg, purge_days=90, embargo_days=7, min_train_days=730,
                           min_embargo_frac=0.01, selector="plateau")
        RR.EXTRA[0] = 1
        res_d = walk_forward(fam + "_delay2", grid, ev, reg, purge_days=90, embargo_days=7,
                             min_train_days=730, min_embargo_frac=0.01, selector="plateau")
        RR.EXTRA[0] = 0
        sha = prereg.file_sha(sp); cts = prereg.git_commit_ts(sp)
        first = min(x["ts"] for x in reg.all() if x["params"].get("strategy") == fam)
        pc = prereg.check(sha, first)
        pc["git_commit_ts"], pc["commit_before_first_trial"] = cts, bool(cts and cts < first)
        pc["ok"] = pc["ok"] and pc["commit_before_first_trial"]
        evd = evidence(res, rules, spy_dev, res_d["oos_daily"])
        n_tr, hold, _ = RR.trade_stats(res)
        M = pd.DataFrame({k: r.daily for k, r in res["results"].items()}).fillna(0.0)
        sc = [float(M[k].mean() / M[k].std()) if M[k].std() > 0 else -np.inf for k in M.columns]
        k_star = plateau_select(sc, grid)
        vault.open_once(fam, "holdout_2025-10_bondeq")
        d = make_ev(fam, ho)(grid[k_star]).daily
        d = d[d.index >= pd.Timestamp(DEV_END, tz="UTC")]
        hm = engine.metrics(d)
        bh = spy_ho.pct_change(fill_method=None).reindex(d.index).fillna(0.0)
        oos = res["oos_daily"]
        rep = {"name": fam, "family": fam, "prereg_sha": sha, "prereg_check": pc,
               "data_manifest_sha": mf["manifest_sha"], "registry_path": str(REG), **evd,
               "n_trades": n_tr, "max_horizon_days": hold or 0, "capacity": {"note": "liquid US ETFs"},
               "capacity_p95": 0.0, "pit_universe": True, "market_neutral": bool(spec.get("market_neutral")),
               "benchmark_note": "R7 'btc_buy_hold' slot = SPY buy&hold for this family",
               "payoff_dev_oos": payoff_stats(oos), "payoff_holdout": payoff_stats(d),
               "years_vs_spy_dev": B.yearly_vs_spy(oos, spy_dev),
               "spy_oos_metrics": engine.metrics(spy_dev.pct_change(fill_method=None).reindex(oos.index).fillna(0.0)),
               "vault": {"status": pol["r3_status"], "reason": pol["reason"], "holdout_sharpe": hm["sharpe"],
                         "holdout": {"param": grid[k_star], **{k: hm[k] for k in ("sharpe", "ann_ret", "max_dd", "total_ret")},
                                     "spy_total_ret": float((1 + bh).prod() - 1)}},
               "paper": None, "folds": res["folds"],
               "oos_window": [str(oos.index[0].date()), str(oos.index[-1].date())], "dev_end": DEV_END,
               "elapsed_s": round(time.time() - t0, 1)}
        rep["verdict"] = verdict(rep)
        rep["verdict_aggressive"] = verdict(rep, AGGRESSIVE_PATH)
        (OUT / f"{fam}.json").write_text(json.dumps(rep, indent=2, default=str))
        log.info("%s -> %s failed=%s", fam, rep["verdict"]["verdict"], rep["verdict"]["failed"])


if __name__ == "__main__":
    main()
