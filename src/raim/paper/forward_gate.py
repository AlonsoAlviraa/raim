"""R15 forward gate: a CANDIDATE_FOR_PAPER can be PROMOTEd only after >= 90 days or >= 100 trades
of PAPER trading whose Sharpe is >= 50% of the OOS Sharpe and whose cumulative return lies inside
the block-bootstrap confidence interval implied by the backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paper_summary(paper_daily: pd.Series, n_trades: int, oos_daily: pd.Series,
                  q: tuple[float, float] = (0.05, 0.95), n_boot: int = 2000, seed: int = 3) -> dict:
    d = paper_daily.dropna()
    sr = float(d.mean() / d.std() * np.sqrt(365)) if len(d) > 5 and d.std() > 0 else 0.0
    rng = np.random.default_rng(seed)
    x = oos_daily.dropna().values
    sims = [(1 + rng.choice(x, len(d))).prod() - 1 for _ in range(n_boot)] if len(d) else [0]
    lo, hi = np.quantile(sims, q)
    cum = float((1 + d).prod() - 1)
    return {"days": int(len(d)), "trades": int(n_trades), "sharpe": sr, "cum_ret": cum,
            "ci": [float(lo), float(hi)], "within_ci": bool(lo <= cum <= hi)}
