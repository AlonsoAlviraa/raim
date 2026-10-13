"""Rescore every prior strategy (S1-S6, C1-C8) under the 15 super rules (BACKTEST ONLY).

Re-runs each family with the NEW protocol (plateau selection, >=5 valid folds, embargo >=1% of
the sample) at VIP0 costs, plus a +1-bar extra-delay pass (R11). Evidence goes to
raim.research.evaluate, the verdict ONLY to raim.research.verdict. One JSON per strategy is
written to reports/research_rules/ as soon as it is done (resumable with --skip-done)."""
from __future__ import annotations

import argparse
import functools
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
import run_convex as RC  # noqa: E402
import run_strategies as RS  # noqa: E402
from raim.data import manifest  # noqa: E402
from raim.research import prereg  # noqa: E402
from raim.research.capacity import participation  # noqa: E402
from raim.research.evaluate import evidence  # noqa: E402
from raim.research.registry import TrialRegistry  # noqa: E402
from raim.research.vault import Vault  # noqa: E402
from raim.research.verdict import load_rules, verdict  # noqa: E402
from raim.strategies import engine, universe  # noqa: E402
from raim.strategies.costs import VIP0  # noqa: E402
from raim.strategies.wf import walk_forward  # noqa: E402

log = logging.getLogger("rescore")
OUT = ROOT / "reports/research_rules"
REG = ROOT / "reports/strategies/trials.jsonl"
DATA_ROOTS = ["/workspace/data_hist"]
PIT = {"C6_alt_tsmom_weekly_pit", "C7_S3_donchian_pit20", "C8_new_listings"}
LAST_DATA = "2026-10-01"
CONTAMINATED = ("2025-10-01", LAST_DATA)
FRESH_SEAL = ("vault_2026-10-13", "2026-10-13", "2027-10-13")

EXTRA = [0]
_orig_run = engine.run


def _run(*a, delay=1, **k):                       # R11: extra execution delay when EXTRA[0] > 0
    return _orig_run(*a, delay=delay + EXTRA[0], **k)


engine.run = _run
_wf = functools.partial(walk_forward, min_embargo_frac=0.01, selector="plateau")
RS.walk_forward = _wf
RC.walk_forward = _wf
ALL = {**RS.STRATS, **RC.STRATS}


def call(name, fn, reg, tag):
    out = fn(reg, VIP0, tag)
    return out[0]


def chosen_weights(res):
    parts = []
    for f, k in zip(res["folds"], res["chosen_idx"]):
        r = res["results"][k]
        t0 = pd.Timestamp(f["test_start"], tz="UTC")
        t1 = pd.Timestamp(f["test_end"], tz="UTC") + pd.Timedelta("1D")
        parts.append((r.weights.loc[(r.weights.index >= t0) & (r.weights.index < t1)], r.meta["inst"]))
    return parts


def trade_stats(res):
    if "results" not in res:
        b = res["oos_bar"]
        return int((b["turnover"] > 0).sum() // 2), None, None   # S5: approx. entries+exits
    tr = pd.concat([RC.trades(w, i) for w, i in chosen_weights(res)])
    W = pd.concat([w for w, _ in chosen_weights(res)])
    step = pd.Series(W.index).diff().median()
    hold_days = float(tr["bars"].quantile(0.95) * step / pd.Timedelta("1D")) if len(tr) else 0.0
    return int(len(tr)), hold_days, W


def write_prereg(name, res, first_ts):
    grid = res.get("grid")
    n = len(grid) if grid else 12 * 5        # S5: 12 train variants x 5 alts per fold
    spec = {"family": name, "retroactive": True,
            "hypothesis": "see docs/research/strategies_2026-10.md / strategies_convex_2026-10.md",
            "mechanism": "as documented in the round 1/2 reports",
            "universe": "PIT top-N incl. delisted" if name in PIT else "fixed list chosen in 2026 (survivorship)",
            "grid": grid if grid else {"entry": [0, 1, 2], "hold": [0, -1e-4], "delay": [0, 1], "alt": 5},
            "metric": "daily net Sharpe, walk-forward OOS",
            "note": f"written retroactively on {time.strftime('%Y-%m-%d')} AFTER data and trials "
                    f"(first trial ts {first_ts}); kept for audit, cannot satisfy R1"}
    p = ROOT / "configs/prereg" / f"{name}.yaml"
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    R, _ = load_rules()
    try:
        sha = prereg.register(p, R, ROOT / "registry/prereg_log.jsonl")
    except ValueError as e:                           # e.g. > 50 variants: log it as a fail
        sha = prereg.file_sha(p)
        with (ROOT / "registry/prereg_log.jsonl").open("a") as fh:
            fh.write(json.dumps({"sha": sha, "family": name, "path": str(p), "registered_ts": time.time(),
                                 "n_variants": n, "retroactive": True, "invalid": str(e)}) + "\n")
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    rules, rsha = load_rules()
    vault = Vault(ROOT / "registry/vault.json", ROOT / "registry/vault_log.jsonl",
                  ROOT / "configs/research_rules.yaml")
    if FRESH_SEAL[0] not in vault.state["seals"]:
        vault.seal(*FRESH_SEAL, families="*", data_sha=None)
    mf_path = OUT / "data_manifest.json"
    if not mf_path.exists():
        mf = manifest.build(DATA_ROOTS, Path("/workspace/data_hist/.manifest_cache.json"))
        mf_path.write_text(json.dumps(mf))
    mf = json.loads(mf_path.read_text())
    reg = TrialRegistry(REG)
    trials0 = reg.all()
    btc = RS.daily_closes(["BTCUSDT"])["BTCUSDT"]
    adv = universe.load_panel()["qv"]
    for name, fn in ALL.items():
        if a.only and name not in a.only:
            continue
        if a.skip_done and (OUT / f"{name}.json").exists():
            continue
        t = time.time()
        try:
            EXTRA[0] = 0
            res = call(name, fn, reg, "")
            EXTRA[0] = 1
            res_d = call(name, fn, reg, "_delay2")
            EXTRA[0] = 0
        except Exception as e:
            log.exception("%s failed: %s", name, e)
            EXTRA[0] = 0
            continue
        ev = evidence(res, rules, btc, res_d["oos_daily"])
        n_tr, hold_days, W = trade_stats(res)
        cap = participation(W, adv, rules["r14_data_capacity"]["target_capital_usd"]) if W is not None else {}
        fam_ts = [x["ts"] for x in trials0 if x["params"].get("strategy", "").startswith(name)]
        first_ts = min(fam_ts) if fam_ts else None
        sha = write_prereg(name, res, first_ts)
        pc = prereg.check(sha, first_ts, ROOT / "registry/prereg_log.jsonl")
        oos_end = str(res["oos_daily"].index[-1].date())
        if oos_end >= CONTAMINATED[0] and name not in vault.state.get("contaminated", {}):
            vault.mark_contaminated(name, CONTAMINATED, "last 12 months of available data were already "
                                    "used as walk-forward OOS in rounds 1-2 -> no clean hold-out")
        vs = vault.status(name)
        rep = {"name": name, "family": name, "prereg_sha": sha, "prereg_check": pc,
               "data_manifest_sha": mf["manifest_sha"], "registry_path": str(REG),
               **ev, "n_trades": n_tr, "max_horizon_days": hold_days or 0,
               "capacity": cap, "capacity_p95": cap.get("p95") if cap else None,
               "pit_universe": name in PIT,
               "vault": {"status": vs.get("status"), "reason": vs.get("reason", ""),
                         "window": vs.get("window"), "holdout_sharpe": None,
                         "fresh_seal": {"name": FRESH_SEAL[0], "window": FRESH_SEAL[1:]}},
               "paper": None, "folds": res["folds"], "oos_window": [str(res["oos_daily"].index[0].date()), oos_end],
               "selector_note": "plateau-centre selection (R9); rounds 1-2 used argmax",
               "elapsed_s": round(time.time() - t, 1)}
        rep["verdict"] = verdict(rep)
        (OUT / f"{name}.json").write_text(json.dumps(rep, indent=2, default=str))
        log.info("%s -> %s failed=%s (%.0fs)", name, rep["verdict"]["verdict"],
                 rep["verdict"]["failed"], time.time() - t)


if __name__ == "__main__":
    main()
