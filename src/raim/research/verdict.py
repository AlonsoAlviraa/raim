"""SINGLE source of truth for research verdicts (super rules R1-R15).

verdict(report) -> {"verdict": PROMOTE | CANDIDATE_FOR_PAPER | WATCH | KILL, "checks": {...}}
Thresholds come ONLY from configs/research_rules.yaml (hashed into the result). N for the
Deflated Sharpe is read from the trial registry file named in the report, never from a parameter.

Ladder:
- KILL: any statistical / robustness / data rule fails (R2, R4-R14).
- WATCH: all of those pass but the process rules fail (R1 prereg or R3 clean vault) -> the
  result cannot be trusted until re-run with a prereg and a fresh vault.
- CANDIDATE_FOR_PAPER: R1-R14 pass; best possible verdict without forward paper evidence.
- PROMOTE: additionally R15 (>=90 days or >=100 trades of paper, consistent with backtest)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from raim.research.mintrl import min_trl
from raim.research.registry import deflated_sharpe

RULES_PATH = Path(__file__).resolve().parents[3] / "configs" / "research_rules.yaml"
STATISTICAL = ["R2", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", "R14"]
PROCESS = ["R1", "R3"]
# trials with too little data are excluded from the SR VARIANCE only (still counted in N)
DEGENERATE_MIN_DAYS, DEGENERATE_MIN_ACTIVE, SR_CLIP = 60, 0.05, 0.5
REQUIRED = ["name", "family", "prereg_sha", "data_manifest_sha", "registry_path", "oos"]


class ReportError(ValueError):
    pass


def load_rules(path: str | Path | None = None) -> tuple[dict, str]:
    b = Path(path or RULES_PATH).read_bytes()
    return yaml.safe_load(b), hashlib.sha256(b).hexdigest()


def validate_report(rep: dict) -> None:
    miss = [k for k in REQUIRED if k not in rep or rep[k] in (None, "")]
    if miss:
        raise ReportError(f"report rejected, missing: {miss}")


def registry_stats(path: str | Path, family: str | None = None) -> dict:
    """Unique trials (dedupe by trial hash: an identical re-run is not a new variant) and the
    robust cross-trial variance of daily SR (degenerate trials excluded from the variance)."""
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x]
    uniq = {}
    for r in rows:
        uniq[r["trial"]] = r
    trials = list(uniq.values())
    fam = [t for t in trials if family and t["params"].get("strategy", "").split("_stress2x")[0]
           .split("_delay")[0] == family]
    srs = [t["metrics"]["sr_daily"] for t in trials
           if t["params"].get("stage") != "walk_forward_oos" and t["metrics"].get("n_days", 0) >= DEGENERATE_MIN_DAYS
           and t["metrics"].get("pct_days_active", 0) >= DEGENERATE_MIN_ACTIVE]
    var = float(np.var(np.clip(srs, -SR_CLIP, SR_CLIP))) if len(srs) > 1 else 1e-4
    return {"n_trials": len(trials), "n_family": len(fam), "sr_var": var, "n_rows": len(rows)}


def _c(ok, value=None, thr=None, note=""):
    return {"pass": bool(ok) if ok is not None else False, "value": value, "threshold": thr,
            "note": note if ok is not None else (note or "not computable -> fail")}


def _get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def verdict(rep: dict, rules_path: str | Path | None = None) -> dict:
    validate_report(rep)
    R, rsha = load_rules(rules_path)
    o = rep["oos"]
    ch = {}
    # R1 prereg
    pr = rep.get("prereg_check") or {}
    nv = pr.get("n_variants")
    ch["R1"] = _c(pr.get("ok", False) and nv is not None and nv <= R["r1_prereg"]["max_variants_per_family"],
                  nv, f"<= {R['r1_prereg']['max_variants_per_family']} variants, before data",
                  pr.get("reason", ""))
    # R2 registry (N read from the file)
    rs = registry_stats(rep["registry_path"], rep["family"])
    ok2 = rs["n_family"] >= (rep.get("n_variants") or 1)
    ch["R2"] = _c(ok2, {"N_global": rs["n_trials"], "N_family": rs["n_family"]},
                  "every variant of the family is in the registry")
    # R3 vault
    v = rep.get("vault") or {}
    ok3 = v.get("status") == "opened" and v.get("holdout_sharpe") is not None and \
        v["holdout_sharpe"] > R["r3_vault"]["min_holdout_sharpe"]
    ch["R3"] = _c(ok3, v.get("status"), "opened once, clean, holdout SR > 0", v.get("reason", ""))
    # R4 walk-forward
    nf, fp = o.get("n_folds", 0), o.get("folds_positive", 0)
    emb_ok = (rep.get("embargo_fraction") or 0) >= R["r4_walk_forward"]["min_embargo_fraction"] or \
        rep.get("embargo_fraction") is None
    purge_ok = rep.get("purge_days", 0) >= rep.get("max_horizon_days", 0)
    ok4 = nf >= R["r4_walk_forward"]["min_folds"] and nf > 0 and \
        fp / nf >= R["r4_walk_forward"]["min_positive_fraction"] and purge_ok and emb_ok
    ch["R4"] = _c(ok4, f"{fp}/{nf}", f">= {R['r4_walk_forward']['min_folds']} folds, "
                  f">= {R['r4_walk_forward']['min_positive_fraction']:.0%} positive, purge>=horizon")
    # R5 DSR
    d = deflated_sharpe(o["sr_daily"], o["n_days"], max(rs["n_trials"], 1), max(rs["sr_var"], 1e-8),
                        skew=o.get("skew", 0.0), kurt=o.get("kurt", 3.0))
    ch["R5"] = _c(d >= R["r5_dsr"]["min_dsr"], round(d, 3), f">= {R['r5_dsr']['min_dsr']}")
    # R6 PBO
    pbo = rep.get("pbo")
    ch["R6"] = _c(None if pbo is None else pbo <= R["r6_pbo"]["max_pbo"], pbo, f"<= {R['r6_pbo']['max_pbo']}")
    # R7 SPA vs each benchmark
    spa = rep.get("spa") or {}
    ps = [spa.get(b) for b in R["r7_spa"]["benchmarks"]]
    ok7 = None if any(p is None for p in ps) else all(p <= R["r7_spa"]["max_p"] for p in ps)
    ch["R7"] = _c(ok7, dict(zip(R["r7_spa"]["benchmarks"], ps)), f"p <= {R['r7_spa']['max_p']} vs all")
    # R8 MinTRL + trades
    mt = min_trl(o["sr_daily"], o.get("skew", 0.0), o.get("kurt", 3.0), R["r8_mintrl"]["sr_star"],
                 R["r8_mintrl"]["confidence"])
    ntr = rep.get("n_trades")
    ok8 = None if ntr is None else (o["n_days"] >= mt and ntr >= R["r8_mintrl"]["min_trades"])
    ch["R8"] = _c(ok8, {"n_days": o["n_days"], "min_trl_days": None if math.isinf(mt) else round(mt),
                        "n_trades": ntr}, f"n_days >= MinTRL and trades >= {R['r8_mintrl']['min_trades']}")
    # R9 plateau
    st = rep.get("plateau_stability")
    ok9 = None if st is None else (st >= R["r9_plateau"]["min_stability"]
                                   and rep.get("selector") == R["r9_plateau"]["selector"])
    ch["R9"] = _c(ok9, st, f">= {R['r9_plateau']['min_stability']} of neighbours keep "
                  f"{R['r9_plateau']['neighbor_keep']:.0%} SR; plateau-centre selection")
    # R10 costs
    cs = rep.get("costs") or {}
    be, s2 = cs.get("breakeven_multiple"), cs.get("sharpe_at_stress")
    ok10 = None if be is None or s2 is None else (be >= R["r10_costs"]["min_breakeven_multiple"]
                                                  and s2 > R["r10_costs"]["min_sharpe_at_stress"])
    ch["R10"] = _c(ok10, {"breakeven_x": None if be is None else round(be, 2),
                          "sharpe_2x": None if s2 is None else round(s2, 2)},
                   f"breakeven >= {R['r10_costs']['min_breakeven_multiple']}x, SR@2x > "
                   f"{R['r10_costs']['min_sharpe_at_stress']}")
    # R11 execution delay
    sd = rep.get("sharpe_extra_delay")
    base = o.get("sharpe", 0.0)
    ret = None if sd is None else (sd / base if base > 0 else 0.0)
    ch["R11"] = _c(None if ret is None else ret >= R["r11_execution"]["min_sharpe_retention"],
                   None if ret is None else round(ret, 2),
                   f">= {R['r11_execution']['min_sharpe_retention']:.0%} SR kept with +1 bar delay")
    # R12 regimes
    rg = rep.get("regime") or {}
    ok12 = None if not rg else (rg["regimes_positive"] >= R["r12_regime"]["min_regimes_positive"]
                                and rg["years_positive_fraction"] >= R["r12_regime"]["min_years_positive_fraction"]
                                and rg["worst_regime_dd"] >= R["r12_regime"]["worst_regime_min_dd"])
    ch["R12"] = _c(ok12, None if not rg else {"regimes_pos": f"{rg['regimes_positive']}/{rg['regimes_present']}",
                                              "years_pos": round(rg["years_positive_fraction"], 2),
                                              "worst_dd": round(rg["worst_regime_dd"], 3)},
                   ">=2/3 regimes SR>0, >=75% years, worst regime DD >= -30%")
    # R13 bootstrap
    bs = rep.get("bootstrap") or {}
    ok13 = None if not bs else (bs["sharpe_p5"] > R["r13_bootstrap"]["min_p5_sharpe"]
                                and bs["dd_p95"] >= R["r13_bootstrap"]["min_p95_dd"]
                                and bs["losing_streak_p95_days"] <= R["r13_bootstrap"]["max_p95_losing_streak_days"])
    ch["R13"] = _c(ok13, None if not bs else {"sr_p5": round(bs["sharpe_p5"], 2), "dd_p95": round(bs["dd_p95"], 3),
                                              "lose_streak_p95_d": bs["losing_streak_p95_days"]},
                   "p5 SR > 0, p95 DD >= -35%, p95 losing streak tolerable")
    # R14 data + capacity
    cap = rep.get("capacity_p95")
    ok14 = None if cap is None else (bool(rep.get("data_manifest_sha")) and bool(rep.get("pit_universe"))
                                     and cap <= R["r14_data_capacity"]["max_adv_participation"])
    ch["R14"] = _c(ok14, {"manifest": bool(rep.get("data_manifest_sha")), "pit": rep.get("pit_universe"),
                          "adv_p95": None if cap is None else round(cap, 5)},
                   f"manifest + PIT universe + <= {R['r14_data_capacity']['max_adv_participation']:.0%} ADV")
    # R15 paper gate
    pp = rep.get("paper") or {}
    ok15 = bool(pp) and (pp.get("days", 0) >= R["r15_paper"]["min_days"] or
                         pp.get("trades", 0) >= R["r15_paper"]["min_trades"]) and \
        pp.get("sharpe", -9) >= R["r15_paper"]["min_sharpe_ratio_vs_oos"] * base and pp.get("within_ci", False)
    ch["R15"] = _c(ok15, pp or "no paper track", ">= 90d or 100 trades paper, SR >= 50% OOS, inside CI")

    stat_ok = all(ch[k]["pass"] for k in STATISTICAL)
    proc_ok = all(ch[k]["pass"] for k in PROCESS)
    if not stat_ok:
        v_ = "KILL"
    elif not proc_ok:
        v_ = "WATCH"
    elif not ch["R15"]["pass"]:
        v_ = "CANDIDATE_FOR_PAPER"
    else:
        v_ = "PROMOTE"
    return {"verdict": v_, "checks": ch, "rules_sha": rsha, "rules_version": R.get("version"),
            "failed": [k for k in ch if not ch[k]["pass"]]}
