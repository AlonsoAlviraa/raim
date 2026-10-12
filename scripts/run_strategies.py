"""Run the 2026-10 alternative-strategy study end to end (BACKTEST ONLY).

Writes reports/strategies/<name>.json after EACH strategy (resumable: --skip-done), the trial
registry reports/strategies/trials.jsonl, and summary.json/summary.csv with DSR computed using
the final honest trial count."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from raim.research.registry import TrialRegistry  # noqa: E402
from raim.strategies import basis, data, engine, leadlag, pairs, trend  # noqa: E402
from raim.strategies.costs import STRESS2X, VIP0  # noqa: E402
from raim.strategies.wf import walk_forward  # noqa: E402

log = logging.getLogger("strategies")
OUT = Path("reports/strategies")
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
            "LTCUSDT", "LINKUSDT", "AVAXUSDT"]


def avail(market):
    out = []
    for s in UNIVERSE:
        try:
            data.load_klines(s, market, "1h")
            out.append(s)
        except FileNotFoundError:
            pass
    return out


def save(name, res, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    m = res["metrics"]
    payload = {"name": name, "metrics": m, "folds": res["folds"], **(extra or {})}
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
    res["oos_daily"].to_csv(OUT / f"{name}_oos_daily.csv", header=["net"])
    log.info("%s OOS sharpe=%.2f maxdd=%.1f%% total=%.1f%%", name, m["sharpe"],
             100 * m["max_dd"], 100 * m["total_ret"])


def hourly_pair(sym):
    s = data.load_klines(sym, "spot", "1h")
    f = data.load_klines(sym, "perp", "1h")
    idx = s.index.intersection(f.index)
    return s.loc[idx], f.loc[idx], data.load_funding(sym)


# ---------------------------------------------------------------- S1 basis arbitrage
def s1(reg, costs, tag=""):
    syms = [s for s in avail("perp") if s in avail("spot")][:6]
    rt = 2 * (costs.spot_side() + costs.perp_side())          # round trip both legs
    prepared = {s: hourly_pair(s) for s in syms}

    def ev(p):
        W, R, C, Fd = [], [], {}, []
        for s, (sp, pf, fu) in prepared.items():
            rho = basis.rho_series(sp.close, pf.close)
            w = basis.basis_arb_weights(rho, open_thr=p["k"] * rt,
                                        allow_short_spot=p["short_spot"]) / len(prepared)
            w.columns = [f"{s}_S", f"{s}_F"]
            W.append(w)
            R.append(pd.DataFrame({f"{s}_S": sp.close.pct_change(), f"{s}_F": pf.close.pct_change()}))
            C[f"{s}_S"], C[f"{s}_F"] = costs.spot_side(), costs.perp_side()
            Fd.append(pd.DataFrame({f"{s}_F": data.funding_on_grid(fu, sp.index)}))
        W, R, Fd = (pd.concat(x, axis=1).fillna(0.0) for x in (W, R, Fd))
        return engine.run(W, R, C, Fd, delay=1)

    grid = [{"k": k, "short_spot": ss} for k in (1.0, 1.5, 2.0) for ss in (False, True)]
    res = walk_forward("S1_basis_arb" + tag, grid, ev, reg, purge_days=7, embargo_days=2)
    return res, {"symbols": syms, "round_trip_cost": rt}


# ---------------------------------------------------------------- S2 funding carry
def s2(reg, costs, tag=""):
    syms = [s for s in avail("perp") if s in avail("spot")][:6]
    prepared = {s: hourly_pair(s) for s in syms}

    def ev(p):
        W, R, C, Fd = [], [], {}, []
        for s, (sp, pf, fu) in prepared.items():
            if p["lookback"] == 0:
                w = pd.DataFrame({"S": 0.5, "F": -0.5}, index=sp.index)
            else:
                w = basis.funding_carry_weights(fu, sp.index, p["lookback"], p["thr"])
            w = w / len(prepared)
            w.columns = [f"{s}_S", f"{s}_F"]
            W.append(w)
            R.append(pd.DataFrame({f"{s}_S": sp.close.pct_change(), f"{s}_F": pf.close.pct_change()}))
            C[f"{s}_S"], C[f"{s}_F"] = costs.spot_side(), costs.perp_side()
            Fd.append(pd.DataFrame({f"{s}_F": data.funding_on_grid(fu, sp.index)}))
        W, R, Fd = (pd.concat(x, axis=1).fillna(0.0) for x in (W, R, Fd))
        return engine.run(W, R, C, Fd, delay=1)

    grid = [{"lookback": 0, "thr": 0.0}] + [{"lookback": L, "thr": t}
                                             for L in (3, 9, 21) for t in (0.0001, 0.0002, 0.0004)]
    return walk_forward("S2_funding_carry" + tag, grid, ev, reg, purge_days=7, embargo_days=2), \
        {"symbols": syms}


# ---------------------------------------------------------------- S3 Donchian ensemble
def daily_closes(syms):
    return pd.DataFrame({s: data.load_klines(s, "spot", "1h").close.resample("1D", label="right",
                                                                           closed="right").last()
                         for s in syms})


def s3(reg, costs, tag=""):
    syms = avail("spot")
    closes = daily_closes(syms)
    rets = closes.pct_change()
    C = {s: costs.spot_side() for s in syms}
    sets = {"all": [5, 10, 20, 30, 60, 90, 150, 250, 360], "short": [5, 10, 20, 30, 60],
            "long": [60, 90, 150, 250, 360]}

    def ev(p):
        w = trend.donchian_ensemble_weights(closes, sets[p["lb"]], p["tv"])
        return engine.run(w, rets, C, None, delay=1)

    grid = [{"lb": lb, "tv": tv} for lb in sets for tv in (0.25, 0.5)]
    return walk_forward("S3_donchian_ensemble" + tag, grid, ev, reg, purge_days=60,
                        embargo_days=7, min_train_days=365), {"symbols": syms}


# ---------------------------------------------------------------- S4 TSMOM 6h + ATR stop
def s4(reg, costs, tag=""):
    syms = avail("perp")
    bars, R, Fd = {}, {}, {}
    for s in syms:
        h = data.load_klines(s, "perp", "1h")
        b = h.resample("6h", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        bars[s] = b
        R[s] = b.close.pct_change()
        f = data.load_funding(s)
        Fd[s] = f.groupby(f.index.ceil("6h")).sum()          # funding within each 6h bar
    R = pd.DataFrame(R)
    Fd = pd.DataFrame(Fd).reindex(R.index).fillna(0.0)
    C = {s: costs.perp_side() for s in syms}

    def ev(p):
        w = trend.tsmom_portfolio_weights(bars, p["L"], p["theta"], p["alpha"], p["short"])
        return engine.run(w, R, C, Fd, delay=1)

    grid = [{"L": L, "theta": th, "alpha": a, "short": sh} for L in (12, 28) for th in (0.02, 0.05)
            for a in (2.5,) for sh in (False, True)]
    return walk_forward("S4_tsmom_atr_6h" + tag, grid, ev, reg, purge_days=14, embargo_days=3,
                        min_train_days=365), {"symbols": syms}


# ---------------------------------------------------------------- S5 lead-lag 1m
def s5(reg, costs, tag=""):
    btc = data.load_klines("BTCUSDT", "spot1m", "1m").close
    out_parts, folds = [], []
    alts = ["ETHUSDT", "LTCUSDT", "PIVXUSDT", "QKCUSDT", "GNOUSDT"]
    per_alt_daily = {}
    for alt in alts:
        small = alt in ("PIVXUSDT", "QKCUSDT", "GNOUSDT")
        a = data.load_klines(alt, "spot1m", "1m").close
        idx = btc.index.intersection(a.index)
        rb, ra = np.log(btc.loc[idx]).diff(), np.log(a.loc[idx]).diff()
        months = sorted(set(idx.to_period("M").astype(str)))
        side = costs.spot_side(small)
        parts = []
        for i in range(3, len(months)):            # expanding train >= 3 months, test 1 month
            train_mask = idx.to_period("M").astype(str) < months[i]
            test_mask = idx.to_period("M").astype(str) == months[i]
            tr_end = idx[train_mask][-1] - pd.Timedelta("1D")       # 1-day purge
            trm = idx <= tr_end
            beta = leadlag.fit_ols(rb[trm], ra[trm])
            best, best_sr = None, -np.inf
            for entry in (0.0, side, 2 * side):
                for hold in (0.0, -0.0001):
                    for delay in (0, 1):
                        w = leadlag.leadlag_weights(rb[trm], ra[trm], beta, entry, hold)
                        r = engine.run(w.to_frame("A"), ra[trm].apply(np.expm1).to_frame("A"),
                                       {"A": side}, None, delay=delay)
                        m = engine.metrics(r.daily, r.bar)
                        reg.record({"strategy": "S5_leadlag_1m" + tag, "alt": alt, "entry": entry,
                                    "hold": hold, "delay": delay, "fold": months[i],
                                    "stage": "train_grid"}, m)
                        if delay == 1 and m["sr_daily"] > best_sr:   # select on realistic delay
                            best, best_sr = (entry, hold), m["sr_daily"]
            sel = idx[test_mask]
            w = leadlag.leadlag_weights(rb.loc[sel], ra.loc[sel], beta, *best)
            r = engine.run(w.to_frame("A"), ra.loc[sel].apply(np.expm1).to_frame("A"),
                           {"A": side}, None, delay=1)
            parts.append(r)
            folds.append({"alt": alt, "test_month": months[i], "entry": best[0], "hold": best[1],
                          "beta_btc": float(beta[1]), "oos_sharpe": engine.metrics(r.daily)["sharpe"],
                          "oos_total_ret": float((1 + r.daily).prod() - 1),
                          "turnover": float(r.bar.turnover.sum())})
        d = pd.concat([p.daily for p in parts])
        per_alt_daily[alt] = d
        out_parts.append(pd.concat([p.bar for p in parts]))
    daily = pd.DataFrame(per_alt_daily).fillna(0.0).mean(axis=1)    # equal-weight sleeves
    bar = pd.concat(out_parts)
    bar = bar.assign(turnover=bar.turnover / len(alts), cost=bar.cost / len(alts),
                     gross=bar.gross / len(alts), funding=0.0)
    m = engine.metrics(daily, bar)
    m["folds_positive"] = int(sum(f["oos_total_ret"] > 0 for f in folds))
    m["n_folds"] = len(folds)
    reg.record({"strategy": "S5_leadlag_1m" + tag, "stage": "walk_forward_oos"}, m)
    per_alt = {a: engine.metrics(d) for a, d in per_alt_daily.items()}
    return {"oos_daily": daily, "oos_bar": bar, "metrics": m, "folds": folds}, {"per_alt": per_alt}


# ---------------------------------------------------------------- S6 pairs
PAIRS = [("ETHUSDT", "BTCUSDT"), ("SOLUSDT", "ETHUSDT"), ("BNBUSDT", "BTCUSDT"),
         ("LTCUSDT", "BTCUSDT"), ("LINKUSDT", "ETHUSDT"), ("ADAUSDT", "XRPUSDT"),
         ("AVAXUSDT", "SOLUSDT"), ("DOGEUSDT", "XRPUSDT")]


def s6(reg, costs, tag=""):
    have = set(avail("perp"))
    prs = [p for p in PAIRS if p[0] in have and p[1] in have]
    px = {s: data.load_klines(s, "perp", "1h").close for s in {x for p in prs for x in p}}
    fu = {s: data.load_funding(s) for s in px}
    idx = None
    for v in px.values():
        idx = v.index if idx is None else idx.union(v.index)

    def ev(p):
        W, R, Fd, C = [], [], [], {}
        for y, x in prs:
            pyy, pxx = px[y].reindex(idx), px[x].reindex(idx)
            w = pairs.pair_weights(pyy, pxx, p["W"], p["z"]) / len(prs)
            k = f"{y}/{x}"
            w.columns = [f"{k}_Y", f"{k}_X"]
            W.append(w)
            R.append(pd.DataFrame({f"{k}_Y": pyy.pct_change(), f"{k}_X": pxx.pct_change()}))
            Fd.append(pd.DataFrame({f"{k}_Y": data.funding_on_grid(fu[y], idx),
                                    f"{k}_X": data.funding_on_grid(fu[x], idx)}))
            C[f"{k}_Y"] = C[f"{k}_X"] = costs.perp_side()
        W, R, Fd = (pd.concat(x, axis=1).fillna(0.0) for x in (W, R, Fd))
        return engine.run(W, R, C, Fd, delay=1)

    grid = [{"W": W, "z": z} for W in (168, 720) for z in (2.0, 3.0)]
    return walk_forward("S6_pairs_coint" + tag, grid, ev, reg, purge_days=30, embargo_days=3,
                        min_train_days=365), {"pairs": prs}


def btc_bh(index_days):
    c = daily_closes(["BTCUSDT"])["BTCUSDT"]
    r = c.pct_change().reindex(index_days).fillna(0.0)
    r.iloc[0] -= VIP0.spot_side()
    return engine.metrics(r)


STRATS = {"S1_basis_arb": s1, "S2_funding_carry": s2, "S3_donchian_ensemble": s3,
          "S4_tsmom_atr_6h": s4, "S5_leadlag_1m": s5, "S6_pairs_coint": s6}


RF_ANNUAL = 0.04   # USD stablecoin/T-bill proxy: the opportunity cost of idle capital


def _sr_var(trials):
    """Cross-trial variance of daily SR. Degenerate trials (few days / almost never active) give
    meaningless SR estimates, so they are excluded from the VARIANCE (but still count in N)."""
    srs = [t["metrics"]["sr_daily"] for t in trials
           if t["params"].get("stage") != "walk_forward_oos" and t["metrics"].get("n_days", 0) >= 60
           and t["metrics"].get("pct_days_active", 0) >= 0.05]
    srs = np.clip(srs, -0.5, 0.5)
    return float(np.var(srs)) if len(srs) > 1 else 1e-4


def summarize(reg):
    trials = reg.all()
    n, var = len(trials), _sr_var(trials)
    rows = []
    for f in sorted(OUT.glob("S*.json")):
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
        verdict = ("PROMOTE" if mx["sharpe"] > 1.0 and d > 0.95 and m["max_dd"] > -0.20
                   and fp >= 2 / 3 * nf and m["sharpe"] > bh["sharpe"] else
                   "WATCH" if mx["sharpe"] > 0.5 and p0 > 0.9 else "KILL")
        rows.append({"strategy": j["name"], "oos_start": str(oos.index[0].date()),
                     "oos_end": str(oos.index[-1].date()), "oos_sharpe": round(m["sharpe"], 2),
                     "sharpe_xs_rf4": round(mx["sharpe"], 2),
                     "psr_vs0": round(p0, 3), "dsr_global": round(d, 3), "dsr_family": round(dfam, 3),
                     "n_family": len(ft), "max_dd": round(m["max_dd"], 3),
                     "cvar5_daily": round(m["cvar5"], 4), "ann_ret_net": round(m["ann_ret"], 3),
                     "turnover_ann": round(m.get("turnover_ann", 0), 1),
                     "cost_ann": round(m.get("cost_ann", 0), 4),
                     "funding_ann": round(m.get("funding_ann", 0), 4),
                     "days_active": round(m.get("pct_days_active", 0), 2),
                     "folds_pos": f"{fp}/{nf}",
                     "btc_bh_sharpe": round(bh["sharpe"], 2), "btc_bh_maxdd": round(bh["max_dd"], 3),
                     "btc_bh_ann_ret": round(bh["ann_ret"], 3), "verdict": verdict})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"n_trials": n, "sr_var_daily": var,
                                                 "rf_annual": RF_ANNUAL, "rows": rows}, indent=2))
    print(df.to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--stress", action="store_true", help="2x slippage cost robustness pass")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reg = TrialRegistry(OUT / "trials.jsonl")
    if not a.summary_only:
        for name, fn in STRATS.items():
            if a.only and name not in a.only:
                continue
            tag = "_stress2x" if a.stress else ""
            if a.skip_done and (OUT / f"{name}{tag}.json").exists():
                continue
            res, extra = fn(reg, STRESS2X if a.stress else VIP0, tag)
            save(name + tag, res, extra)
    summarize(reg)


if __name__ == "__main__":
    main()
