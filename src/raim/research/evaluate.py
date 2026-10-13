"""Turn a walk-forward result into the evidence dict consumed by raim.research.verdict.
Pure computation; no thresholds live here."""
from __future__ import annotations

import numpy as np
import pandas as pd

from raim.research import costs_curve
from raim.research.bootstrap import block_bootstrap
from raim.research.pbo import cscv_pbo
from raim.research.regime_split import btc_regimes, regime_stats
from raim.research.spa import hansen_spa
from raim.research.stability import stability


def _naive(ix):
    ix = pd.DatetimeIndex(ix)
    return ix.tz_localize(None) if ix.tz is not None else ix


def variant_matrix(res: dict) -> pd.DataFrame | None:
    if "results" not in res:
        return None
    cols = {k: r.daily for k, r in res["results"].items()}
    M = pd.DataFrame(cols).fillna(0.0)
    M.index = _naive(M.index)
    return M


def evidence(res: dict, rules: dict, btc_close: pd.Series, oos_delay: pd.Series | None) -> dict:
    o = dict(res["metrics"])
    oos = res["oos_daily"].copy()
    oos.index = _naive(oos.index)
    out = {"oos": o}
    M = variant_matrix(res)
    grid = res.get("grid")
    if M is not None and M.shape[1] >= 2:
        out["pbo"] = cscv_pbo(M.values, rules["r6_pbo"]["blocks"])["pbo"]
        Mo = M.reindex(oos.index).fillna(0.0)
        Mo["wf"] = oos
        bh = btc_close.copy()
        bh.index = _naive(bh.index)
        bret = bh.pct_change().reindex(oos.index).fillna(0.0)
        rf = rules["rf_annual"] / 365
        sp = rules["r7_spa"]
        out["spa"] = {"btc_buy_hold": hansen_spa(Mo.sub(bret, axis=0).values, sp["bootstrap"], sp["mean_block"])["spa_p"],
                      "cash": hansen_spa((Mo - rf).values, sp["bootstrap"], sp["mean_block"])["spa_p"]}
        # plateau stability of the parameter chosen most often out of sample, on full-sample SR
        sc = [float(M[k].mean() / M[k].std()) if M[k].std() > 0 else -np.inf for k in M.columns]
        ch = pd.Series(res["chosen_idx"]).mode().iloc[0]
        out["plateau_stability"] = stability(sc, grid, int(ch), rules["r9_plateau"]["neighbor_keep"])
        out["selector"] = res.get("selector")
        out["n_variants"] = len(grid)
    bar = res["oos_bar"]
    out["costs"] = {"breakeven_multiple": costs_curve.breakeven_multiple(bar),
                    "sharpe_at_stress": costs_curve.sharpe_at(bar, rules["r10_costs"]["stress_multiplier"]),
                    "curve": costs_curve.curve(bar)}
    if oos_delay is not None:
        d = oos_delay.dropna()
        out["sharpe_extra_delay"] = float(d.mean() / d.std() * np.sqrt(365)) if d.std() > 0 else 0.0
    rg = rules["r12_regime"]
    bc = btc_close.copy()
    bc.index = _naive(bc.index)
    out["regime"] = regime_stats(oos, btc_regimes(bc, rg["sma_days"], rg["slope_days"]))
    b = rules["r13_bootstrap"]
    out["bootstrap"] = block_bootstrap(oos.values, b["n_resamples"], b["mean_block"])
    out["purge_days"] = res.get("purge_days", 0)
    n = len(M) if M is not None else len(oos)
    out["embargo_fraction"] = res.get("embargo_days", 0) / max(n, 1) if "embargo_days" in res else None
    return out
