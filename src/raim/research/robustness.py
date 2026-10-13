"""Tail-risk robustness protocol (Yang & Xu, arXiv:2609.22785): CVaR10/30 of episode PnL plus a
bootstrap test for terminal-inventory bias, over a battery of stress scenarios.
Acceptance rule (documented, not enforced): a configuration is preferred only if it improves
CVaR10 without increasing inventory bias. Synthetic scenarios != real markets."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from raim.config import RaimConfig
from raim.research.pipeline import train_test_run
from raim.sim.scenarios import PRESETS, scenario_events

log = logging.getLogger(__name__)


def cvar(x, alpha: float) -> float:
    """Mean of the worst alpha-fraction of outcomes (lower tail of PnL)."""
    v = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    if len(v) == 0:
        return float("nan")
    k = max(1, int(round(alpha * len(v))))
    return float(v[:k].mean())


def inventory_bias_test(terminal_inv, n_boot: int = 2000, seed: int = 0) -> dict:
    """Two-sided bootstrap test of H0: E[terminal inventory] = 0."""
    x = np.asarray(terminal_inv, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"mean": float("nan"), "p_value": float("nan"), "n": len(x)}
    rng = np.random.default_rng(seed)
    obs = x.mean()
    centered = x - obs
    boots = centered[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    p = float((np.abs(boots) >= abs(obs)).mean())
    return {"mean": float(obs), "p_value": p, "n": len(x)}


def episodes(curve: list[tuple[int, float, float]], episode_books: int = 600):
    """Split an (ts, equity, position) curve into non-overlapping episodes."""
    pnl, inv = [], []
    for i in range(0, len(curve) - episode_books + 1, episode_books):
        a, b = curve[i], curve[i + episode_books - 1]
        pnl.append(b[1] - a[1])
        inv.append(b[2])
    return np.asarray(pnl), np.asarray(inv)


def stress_battery(cfg: RaimConfig, seeds=(1, 2, 3), n_steps: int = 8000,
                   scenarios: list[str] | None = None, episode_books: int = 300) -> pd.DataFrame:
    rows = []
    for name in scenarios or list(PRESETS):
        pnl, inv, mo = [], [], []
        halts = 0
        for sd in seeds:
            rep = train_test_run(scenario_events(name, n_steps, sd), cfg)
            p, q = episodes(rep.result.curve, episode_books)
            pnl += list(p)
            inv += list(q)
            halts += int(rep.result.halted)
            if not rep.markouts.empty:
                mo += list(rep.markouts["mo_5s_bps"].dropna())
        ib = inventory_bias_test(inv, seed=0)
        rows.append({"scenario": name, "episodes": len(pnl), "mean_pnl_usd": float(np.mean(pnl)) if pnl else np.nan,
                     "cvar10_usd": cvar(pnl, 0.10), "cvar30_usd": cvar(pnl, 0.30),
                     "inv_bias_mean": ib["mean"], "inv_bias_p": ib["p_value"],
                     "mo_5s_bps": float(np.mean(mo)) if mo else np.nan, "halts": halts})
    return pd.DataFrame(rows)
