"""Round 2: convex / trend / event strategies across horizons (BACKTEST ONLY).

Shares the trial registry with round 1 (reports/strategies/trials.jsonl) so the GLOBAL Deflated
Sharpe counts every variant ever tried. Writes reports/strategies_convex/<name>.json after each
strategy (resumable with --skip-done) and summary.csv/json."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from raim.research.registry import TrialRegistry  # noqa: E402
from raim.strategies import convex, data, engine, universe  # noqa: E402
from raim.strategies.convex_metrics import convex_stats, trades  # noqa: E402
from raim.strategies.costs import STRESS2X, VIP0  # noqa: E402
from raim.strategies.wf import walk_forward  # noqa: E402
from run_strategies import RF_ANNUAL, _sr_var, btc_bh  # noqa: E402

log = logging.getLogger("convex")
OUT = Path("reports/strategies_convex")
REG = Path("reports/strategies/trials.jsonl")
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
          "LTCUSDT", "LINKUSDT", "AVAXUSDT"]
_PANEL = {}


def panel():
    if not _PANEL:
        _PANEL.update(universe.load_panel())
    return _PANEL


def oos_trades(res, grid):
    parts = []
    for f in res["folds"]:
        r = res["results"][grid.index(f["param"])]
        t0 = pd.Timestamp(f["test_start"], tz="UTC")
        t1 = pd.Timestamp(f["test_end"], tz="UTC") + pd.Timedelta("1D")
        tr = trades(r.weights.loc[(r.weights.index >= t0) & (r.weights.index < t1)], r.meta["inst"])
        parts.append(tr)
    return pd.concat(parts) if parts else pd.DataFrame(columns=["pnl", "exit"])


def save(name, res, grid, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    tr = oos_trades(res, grid) if "trades" not in res else res["trades"]
    cs = convex_stats(tr, res["oos_daily"])
    m = {**res["metrics"], **cs}
    payload = {"name": name, "metrics": m, "folds": res["folds"], **(extra or {})}
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
    res["oos_daily"].to_csv(OUT / f"{name}_oos_daily.csv", header=["net"])
    log.info("%s OOS sharpe=%.2f dd=%.1f%% win=%.2f payoff=%.2f skew=%.2f trades=%d", name,
             m["sharpe"], 100 * m["max_dd"], cs["win_rate"], cs["payoff"], cs["daily_skew"],
             cs["n_trades"])


def perp_bars(sym, rule):
    h = data.load_klines(sym, "perp", "1h")
    return h.resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "quote_volume": "sum"}).dropna()


def fund_grid(sym, idx, rule=None):
    f = data.load_funding(sym)
    if rule:
        f = f.groupby(f.index.ceil(rule)).sum()
    return f.reindex(idx).fillna(0.0)


# ---------------- C1 intraday TSMOM, 30m
def c1(reg, costs, tag):
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    c30 = {s: data.load_klines(s, "spot30m", "30m").close for s in syms}
    idx = c30["BTCUSDT"].index
    R = pd.DataFrame({s: c30[s].reindex(idx).pct_change() for s in syms})
    F = pd.DataFrame({s: fund_grid(s, idx) for s in syms})
    C = {s: costs.perp_side() for s in syms}

    def ev(p):
        W = pd.DataFrame({s: convex.intraday_tsmom_weights(c30[s].reindex(idx), p["h"], p["k"],
                                                           p["lo"]) / len(syms) for s in syms})
        return engine.run(W, R, C, F, delay=1)
    grid = [{"h": h, "k": k, "lo": lo} for h in (0, 13) for k in (0.0, 1.0) for lo in (False, True)]
    return walk_forward("C1_intraday_tsmom_30m" + tag, grid, ev, reg, purge_days=2, embargo_days=1,
                        min_train_days=365), grid, {"symbols": syms}


# ---------------- C2 15m Donchian breakout + ATR trail
def c2(reg, costs, tag):
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    bars = {}
    for s in syms:
        k = data.load_klines(s, "perp15m", "15m")
        bars[s] = k
    idx = bars["BTCUSDT"].index
    R = pd.DataFrame({s: bars[s].close.reindex(idx).pct_change() for s in syms})
    F = pd.DataFrame({s: fund_grid(s, idx) for s in syms})
    C = {s: costs.perp_side() for s in syms}

    def ev(p):
        W = pd.DataFrame({s: convex.donchian_breakout_state(bars[s].reindex(idx).ffill(), p["n"],
                                                             p["a"], p["lo"]) / len(syms) for s in syms})
        return engine.run(W, R, C, F, delay=1)
    grid = [{"n": n, "a": a, "lo": lo} for n in (96, 384) for a in (3.0, 6.0) for lo in (False, True)]
    return walk_forward("C2_donchian_15m" + tag, grid, ev, reg, purge_days=5, embargo_days=1,
                        min_train_days=365), grid, {"symbols": syms}


# ---------------- C3 4h squeeze breakout
def c3(reg, costs, tag):
    bars = {s: perp_bars(s, "4h") for s in MAJORS}
    idx = bars["BTCUSDT"].index
    for s in MAJORS:
        idx = idx.union(bars[s].index)
    R = pd.DataFrame({s: bars[s].close.reindex(idx).pct_change() for s in MAJORS})
    F = pd.DataFrame({s: fund_grid(s, idx, "4h") for s in MAJORS})
    C = {s: costs.perp_side() for s in MAJORS}

    def ev(p):
        W = pd.DataFrame({s: convex.squeeze_breakout_state(bars[s], 20, 2.0, p["kc"], p["a"])
                          .reindex(idx).fillna(0.0) / len(MAJORS) for s in MAJORS})
        return engine.run(W, R, C, F, delay=1)
    grid = [{"kc": kc, "a": a} for kc in (1.5, 2.0) for a in (2.0, 4.0)]
    return walk_forward("C3_squeeze_4h" + tag, grid, ev, reg, purge_days=10, embargo_days=2,
                        min_train_days=365), grid, {"symbols": MAJORS}


# ---------------- C4 funding extremes
def c4(reg, costs, tag):
    px = {s: data.load_klines(s, "perp", "1h") for s in MAJORS}
    idx = px["BTCUSDT"].index
    for s in MAJORS:
        idx = idx.union(px[s].index)
    fu = {s: data.load_funding(s) for s in MAJORS}
    R = pd.DataFrame({s: px[s].close.reindex(idx).pct_change() for s in MAJORS})
    F = pd.DataFrame({s: fu[s].reindex(idx).fillna(0.0) for s in MAJORS})
    C = {s: costs.perp_side() for s in MAJORS}

    def ev(p):
        W = pd.DataFrame({s: convex.funding_extreme_weights(px[s].close.reindex(idx), fu[s], p["q"],
                                                            p["mode"]) / len(MAJORS) * 3 for s in MAJORS})
        return engine.run(W, R, C, F, delay=1)
    grid = [{"q": q, "mode": m} for q in (0.02, 0.05) for m in ("conditional", "squeeze", "fade_pos")]
    return walk_forward("C4_funding_extremes" + tag, grid, ev, reg, purge_days=3, embargo_days=1,
                        min_train_days=365), grid, {"symbols": MAJORS, "per_event_weight": 0.3}


# ---------------- C5 liquidation-cascade proxy
def c5(reg, costs, tag):
    px = {s: data.load_klines(s, "perp", "1h") for s in MAJORS}
    idx = px["BTCUSDT"].index
    for s in MAJORS:
        idx = idx.union(px[s].index)
    R = pd.DataFrame({s: px[s].close.reindex(idx).pct_change() for s in MAJORS})
    F = pd.DataFrame({s: fund_grid(s, idx) for s in MAJORS})
    C = {s: costs.perp_side() for s in MAJORS}

    def ev(p):
        W = pd.DataFrame({s: convex.cascade_weights(px[s], p["k"], 3.0, p["H"], p["dir"])
                          .reindex(idx).fillna(0.0) / len(MAJORS) * 3 for s in MAJORS})
        return engine.run(W, R, C, F, delay=1)
    grid = [{"k": k, "H": H, "dir": d} for k in (4.0, 6.0) for H in (4, 24) for d in ("cont", "rev")]
    return walk_forward("C5_cascade_1h" + tag, grid, ev, reg, purge_days=3, embargo_days=1,
                        min_train_days=365), grid, {"symbols": MAJORS, "per_event_weight": 0.3}


def _spot_costs(costs):
    P = panel()
    sl = universe.slip_tier(P["qv"])
    if costs is STRESS2X:
        sl = sl * 2
    return sl + costs.spot_fee


# ---------------- C6 alt TSMOM weekly PIT
def c6(reg, costs, tag):
    P = panel()
    close = P["close"]
    R = close.pct_change(fill_method=None)
    C = _spot_costs(costs)
    univs = {n: universe.pit_universe(P["qv"], n) for n in (20, 50)}

    def ev(p):
        W = convex.alt_tsmom_weights(close, univs[p["N"]], p["L"], p["tv"])
        return engine.run(W, R, C, None, delay=1)
    grid = [{"N": N, "L": L, "tv": tv} for N in (20, 50) for L in (1, 4, 12) for tv in (0.5,)]
    return walk_forward("C6_alt_tsmom_weekly_pit" + tag, grid, ev, reg, purge_days=90,
                        embargo_days=7, min_train_days=365), grid, {"universe": "PIT top-N trailing 30d median quote volume, all archived USDT spot pairs incl. delisted"}


# ---------------- C7 = S3 rerun on PIT universe
def c7(reg, costs, tag):
    P = panel()
    close = P["close"]
    R = close.pct_change(fill_method=None)
    C = _spot_costs(costs)
    univ = universe.pit_universe(P["qv"], 20)
    sets = {"all": [5, 10, 20, 30, 60, 90, 150, 250, 360], "short": [5, 10, 20, 30, 60],
            "long": [60, 90, 150, 250, 360]}
    cache = {}

    def ev(p):
        W = convex.donchian_pit_weights(close, univ, sets[p["lb"]], p["tv"])
        return engine.run(W, R, C, None, delay=1)
    grid = [{"lb": lb, "tv": tv} for lb in sets for tv in (0.25, 0.5)]
    return walk_forward("C7_S3_donchian_pit20" + tag, grid, ev, reg, purge_days=60, embargo_days=7,
                        min_train_days=365), grid, {"universe": "PIT top-20"}


# ---------------- C8 new listings
def c8(reg, costs, tag):
    P = panel()
    close = P["close"]
    R = close.pct_change(fill_method=None)
    C = _spot_costs(costs)

    def ev(p):
        W = convex.listing_weights(P, "2021-03-01", p["H"], p["cond"])
        return engine.run(W, R, C, None, delay=1)
    grid = [{"H": H, "cond": c} for H in (3, 7, 30) for c in ("mom", "all", "fade_skip")]
    return walk_forward("C8_new_listings" + tag, grid, ev, reg, purge_days=30, embargo_days=3,
                        min_train_days=365), grid, {"note": "listing = first daily bar in archive after 2021-03-01; long-only spot"}


STRATS = {"C1_intraday_tsmom_30m": c1, "C2_donchian_15m": c2, "C3_squeeze_4h": c3,
          "C4_funding_extremes": c4, "C5_cascade_1h": c5, "C6_alt_tsmom_weekly_pit": c6,
          "C7_S3_donchian_pit20": c7, "C8_new_listings": c8}


def summarize(reg):
    trials = reg.all()
    n, var = len(trials), _sr_var(trials)
    rows = []
    for f in sorted(OUT.glob("C*.json")):
        j = json.loads(f.read_text())
        m = j["metrics"]
        fam = j["name"].replace("_stress2x", "")
        ft = [t for t in trials if t["params"].get("strategy", "").replace("_stress2x", "") == fam]
        oos = pd.read_csv(OUT / f"{j['name']}_oos_daily.csv", index_col=0, parse_dates=True)["net"]
        bh = btc_bh(oos.index)
        mx = engine.metrics(oos - RF_ANNUAL / 365)
        d = engine.dsr(m, n, var)
        dfam = engine.dsr(m, len(ft), _sr_var(ft) if len(ft) > 2 else var)
        p0 = engine.psr0(m)
        fp, nf = m.get("folds_positive", 0), max(m.get("n_folds", 1), 1)
        convex_ok = m["daily_skew"] > 0 and m["payoff"] >= 2
        verdict = ("PROMOTE" if mx["sharpe"] > 1.0 and d > 0.95 and m["max_dd"] > -0.20
                   and fp >= 2 / 3 * nf and m["sharpe"] > bh["sharpe"] and convex_ok else
                   "WATCH" if mx["sharpe"] > 0.5 and p0 > 0.9 and convex_ok else "KILL")
        rows.append({"strategy": j["name"], "oos": f"{oos.index[0].date()}→{oos.index[-1].date()}",
                     "sharpe": round(m["sharpe"], 2), "sharpe_xs_rf4": round(mx["sharpe"], 2),
                     "psr0": round(p0, 3), "dsr_global": round(d, 3), "dsr_family": round(dfam, 3),
                     "max_dd": round(m["max_dd"], 3), "cvar5": round(m["cvar5"], 4),
                     "ann_ret": round(m["ann_ret"], 3), "turnover": round(m.get("turnover_ann", 0), 1),
                     "cost_ann": round(m.get("cost_ann", 0), 4), "fund_ann": round(m.get("funding_ann", 0), 4),
                     "n_trades": m["n_trades"], "win_rate": round(m["win_rate"], 3),
                     "payoff": round(m["payoff"], 2), "pf": round(m["profit_factor"], 2),
                     "skew_d": round(m["daily_skew"], 2), "skew_tr": round(m["trade_skew"], 2),
                     "top5_share": round(m["top5pct_share"], 2) if m["top5pct_share"] == m["top5pct_share"] else None,
                     "lose_streak": m["longest_losing_streak"], "ttr_days": m["time_to_recovery_days"],
                     "folds_pos": f"{fp}/{nf}", "btc_sharpe": round(bh["sharpe"], 2),
                     "btc_dd": round(bh["max_dd"], 3), "verdict": verdict})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"n_trials_global": n, "sr_var_daily": var,
                                                 "rows": rows}, indent=2, default=str))
    pd.set_option("display.width", 300)
    print(df.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reg = TrialRegistry(REG)
    if not a.summary_only:
        for name, fn in STRATS.items():
            if a.only and name not in a.only:
                continue
            tag = "_stress2x" if a.stress else ""
            if a.skip_done and (OUT / f"{name}{tag}.json").exists():
                continue
            try:
                res, grid, extra = fn(reg, STRESS2X if a.stress else VIP0, tag)
                save(name + tag, res, grid, extra)
            except Exception as e:  # keep going; record the failure honestly
                log.exception("%s failed: %s", name, e)
    summarize(reg)


if __name__ == "__main__":
    main()
