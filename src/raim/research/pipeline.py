"""Fit-on-train, simulate-on-test pipeline shared by CLI and tests."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from raim.config import RaimConfig
from raim.regime.hmm_fit import HMMParams, fit_hmm
from raim.research.markout import markouts, regime_attribution
from raim.research.obs import Standardizer, obs_from_features
from raim.research.walkforward import assert_no_leakage, purged_walk_forward
from raim.sim.engine import SimResult, feature_rows, run_sim
from raim.types import BookSnapshot, Trade

log = logging.getLogger(__name__)


def fit_regimes(events: list, cfg: RaimConfig, max_rows: int = 4000) -> tuple[HMMParams, Standardizer]:
    _, feats = feature_rows(events, cfg)
    obs = np.array([obs_from_features(list(r)) for r in feats])
    obs = obs[np.isfinite(obs).all(1)]
    if len(obs) < 50:
        raise ValueError(f"not enough rows to fit HMM ({len(obs)})")
    scaler = Standardizer.fit(obs)
    z = (obs - scaler.mean) / scaler.std
    step = max(1, len(z) // max_rows)
    params = fit_hmm(z[::step], cfg.regime.n_states, seed=cfg.seed)
    return params, scaler


def split_by_books(events: list, book_idx_cut: int) -> tuple[list, list]:
    n = 0
    for i, ev in enumerate(events):
        if isinstance(ev, BookSnapshot):
            if n == book_idx_cut:
                return events[:i], events[i:]
            n += 1
    return events, []


@dataclass
class RunReport:
    result: SimResult
    markouts: pd.DataFrame
    attribution: pd.DataFrame


def train_test_run(events: list, cfg: RaimConfig, train_frac: float = 0.3, **kw) -> RunReport:
    n_books = sum(isinstance(e, BookSnapshot) for e in events)
    train, test = split_by_books(events, int(n_books * train_frac))
    params, scaler = fit_regimes(train, cfg)
    res = run_sim(test, cfg, params, scaler, **kw)
    mo = markouts(res.fills, res.mids)
    return RunReport(res, mo, regime_attribution(mo))


def walk_forward(events: list, cfg: RaimConfig, n_folds: int = 4, purge: int = 200,
                 embargo: int = 100) -> pd.DataFrame:
    books_at = [i for i, e in enumerate(events) if isinstance(e, BookSnapshot)]
    folds = purged_walk_forward(len(books_at), n_folds, purge, embargo,
                                min_train=min(1000, len(books_at) // 5))
    rows = []
    for k, f in enumerate(folds):
        assert_no_leakage(f, purge)
        tr = events[: books_at[f.train[-1]] + 1]
        te = events[books_at[f.test[0]]: books_at[f.test[-1]] + 1]
        params, scaler = fit_regimes(tr, cfg)
        res = run_sim(te, cfg, params, scaler)
        mo = markouts(res.fills, res.mids)
        rows.append({"fold": k, "train_books": len(f.train), "test_books": len(f.test),
                     "fills": len(res.fills), "equity_usd": res.final_equity,
                     "max_dd_usd": res.max_drawdown, "fees_usd": res.fees_paid,
                     "end_pos": res.final_position, "halted": res.halted,
                     "mo_5s_bps": float(mo["mo_5s_bps"].mean()) if not mo.empty else np.nan})
    return pd.DataFrame(rows)


_ = Trade
