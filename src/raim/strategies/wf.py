"""Purged + embargoed walk-forward parameter selection over DAILY net returns.

Signals are causal by construction (tested), so each parameter's full-history net series is
computed once; for every fold the parameter is chosen using ONLY the purged training window
(expanding), and the out-of-sample test block uses that choice. Every parameter evaluated is
recorded in the trial registry (honest N for the Deflated Sharpe Ratio)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from raim.research.registry import TrialRegistry
from raim.research.stability import plateau_select
from raim.research.walkforward import assert_no_leakage, purged_walk_forward
from raim.strategies.engine import BTResult, metrics


def walk_forward(strategy: str, grid: list[dict], evaluate: Callable[[dict], BTResult],
                 registry: TrialRegistry, n_folds: int = 5, purge_days: int = 30,
                 embargo_days: int = 7, min_train_days: int = 180,
                 selector: str = "plateau", min_embargo_frac: float = 0.0) -> dict:
    """selector='plateau' (R9, default): centre of the best parameter plateau on the TRAIN
    window; 'argmax' reproduces the legacy round-1/2 behaviour."""
    results: dict[int, BTResult] = {}
    for k, p in enumerate(grid):
        r = evaluate(p)
        results[k] = r
        registry.record({"strategy": strategy, **p, "stage": "grid_full_sample"},
                        metrics(r.daily, r.bar))
    idx = results[0].daily.index
    for r in results.values():
        idx = idx.union(r.daily.index)
    daily = {k: r.daily.reindex(idx).fillna(0.0) for k, r in results.items()}
    embargo_days = max(embargo_days, int(np.ceil(min_embargo_frac * len(idx))))
    # R4: guarantee >= n_folds VALID test folds (min_train can drop the first blocks)
    nf = n_folds
    while True:
        folds = purged_walk_forward(len(idx), n_folds=nf, purge=purge_days,
                                    embargo=embargo_days, min_train=min_train_days)
        if len(folds) >= n_folds or nf >= n_folds + 10:
            break
        nf += 1
    oos_parts, chosen, fold_rows = [], [], []
    for f in folds:
        assert_no_leakage(f, purge_days)
        tr = {k: d.iloc[f.train] for k, d in daily.items()}
        score = {k: (v.mean() / v.std() if v.std() > 0 else -np.inf) for k, v in tr.items()}
        if selector == "plateau":
            best = plateau_select([score[k] for k in range(len(grid))], grid)
        else:
            best = max(score, key=score.get)
        test = daily[best].iloc[f.test]
        oos_parts.append(test)
        chosen.append(grid[best])
        fm = metrics(test)
        fold_rows.append({"test_start": str(test.index[0].date()), "test_end": str(test.index[-1].date()),
                          "param": grid[best], "train_sr_daily": float(score[best]),
                          "oos_sharpe": fm["sharpe"], "oos_total_ret": fm["total_ret"],
                          "selector": selector})
    oos = pd.concat(oos_parts).sort_index()
    # bar-level frame for turnover/cost attribution on the OOS dates actually traded
    bars = []
    for f, p in zip(folds, chosen):
        r = results[grid.index(p)]
        t0, t1 = idx[f.test[0]], idx[f.test[-1]] + pd.Timedelta("1D")
        bars.append(r.bar.loc[(r.bar.index >= t0) & (r.bar.index < t1)])
    oos_bar = pd.concat(bars).sort_index()
    m = metrics(oos, oos_bar)
    m["folds_positive"] = int(sum(fr["oos_total_ret"] > 0 for fr in fold_rows))
    m["n_folds"] = len(fold_rows)
    registry.record({"strategy": strategy, "stage": "walk_forward_oos"}, m)
    return {"oos_daily": oos, "oos_bar": oos_bar, "metrics": m, "folds": fold_rows,
            "results": results, "grid": grid, "selector": selector, "purge_days": purge_days,
            "embargo_days": embargo_days, "chosen_idx": [grid.index(p) for p in chosen]}
