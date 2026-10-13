"""Agentic-viability ablation (TradeLens, arXiv:2607.10286): does the slow brain / Jev add net
value over rules alone, AFTER API costs? Arms: rules_only, +llm, +jev, +llm+jev, run on the
SAME events with the same fitted regime model. m_llm / m_jev are step functions of time — in
paper mode, feed the multipliers logged by the audit trail (see `step_fn`).
If no arm beats rules_only on net value and markouts, disable the external models."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from raim.config import RaimConfig
from raim.research.markout import markouts
from raim.research.pipeline import fit_regimes, split_by_books
from raim.sim.engine import run_sim
from raim.types import BookSnapshot


def step_fn(times_ms, values, default: float = 1.0) -> Callable[[int], float]:
    t = np.asarray(times_ms)
    v = np.asarray(values, dtype=float)

    def f(now: int) -> float:
        i = int(np.searchsorted(t, now, side="right")) - 1
        return float(v[i]) if i >= 0 else default
    return f


def ablation(events: list, cfg: RaimConfig, m_llm: Callable[[int], float],
             m_jev: Callable[[int], float], cost_llm_usd: float, cost_jev_usd: float,
             llm_calls: int, jev_calls: int, train_frac: float = 0.3) -> pd.DataFrame:
    n_books = sum(isinstance(e, BookSnapshot) for e in events)
    train, test = split_by_books(events, int(n_books * train_frac))
    params, scaler = fit_regimes(train, cfg)
    one = lambda t: 1.0  # noqa: E731
    arms = {"rules_only": (one, one, 0.0), "+llm": (m_llm, one, llm_calls * cost_llm_usd),
            "+jev": (one, m_jev, jev_calls * cost_jev_usd),
            "+llm+jev": (m_llm, m_jev, llm_calls * cost_llm_usd + jev_calls * cost_jev_usd)}
    rows = []
    for arm, (fl, fj, cost) in arms.items():
        r = run_sim(test, cfg, params, scaler, ext_multipliers=lambda t, fl=fl, fj=fj: (fl(t), fj(t)))
        mo = markouts(r.fills, r.mids)
        pnl = float(r.final_equity)
        rows.append({"arm": arm, "fills": len(r.fills), "pnl_usd": pnl, "api_cost_usd": float(cost),
                     "net_usd": pnl - float(cost), "max_dd_usd": r.max_drawdown,
                     "mo_5s_bps": float(mo["mo_5s_bps"].mean()) if not mo.empty else np.nan})
    return pd.DataFrame(rows)
