"""Synthetic regime-switching L1 book + trade stream. FOR TESTS/RESEARCH ONLY — never
present results on synthetic data as evidence of real-market edge."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from raim.types import BookSnapshot, Trade

# per regime: (vol bps per step, spread ticks, trade rate per step, informed drift prob, depth)
REGIME_SPECS = {
    0: (0.3, 1, 0.3, 0.5, 5.0),    # R1 calm
    1: (0.6, 2, 0.6, 0.75, 3.0),   # R2 trending/informed
    2: (2.0, 5, 0.9, 0.55, 1.5),   # R3 high-vol stress
    3: (4.0, 12, 1.5, 0.9, 0.5),   # R4 toxic/break
}


@dataclass(frozen=True)
class SyntheticCfg:
    n_steps: int = 20_000
    dt_ms: int = 100
    start_price: float = 60_000.0
    tick: float = 0.01
    seed: int = 7
    stay_prob: float = 0.998


def generate(cfg: SyntheticCfg) -> tuple[list[BookSnapshot], list[Trade], np.ndarray]:
    """Returns (books, trades, true_regime_per_book)."""
    rng = np.random.default_rng(cfg.seed)
    regime = 0
    mid = cfg.start_price
    books: list[BookSnapshot] = []
    trades: list[Trade] = []
    regimes = np.zeros(cfg.n_steps, dtype=int)
    drift_dir = 1
    for i in range(cfg.n_steps):
        if rng.random() > cfg.stay_prob:
            regime = int(rng.integers(0, 4))
            drift_dir = 1 if rng.random() < 0.5 else -1
        vol, spr, rate, p_inf, depth = REGIME_SPECS[regime]
        ts = i * cfg.dt_ms
        n_tr = rng.poisson(rate * (1.0 + (2.0 if regime == 3 and rng.random() < 0.3 else 0.0)))
        flow = 0
        for _ in range(n_tr):
            sign = drift_dir if rng.random() < p_inf else -drift_dir
            qty = float(rng.exponential(0.05 * (2 if regime >= 2 else 1)))
            flow += sign
            half = spr * cfg.tick / 2
            sweep = mid * 1e-4 * vol * 3.0 * float(rng.exponential()) if rng.random() < 0.3 else 0.0
            px = round((mid + sign * (half + sweep)) / cfg.tick) * cfg.tick
            trades.append(Trade(ts, px, qty, sign < 0, ts))
        mid *= 1.0 + 1e-4 * (vol * rng.standard_normal() + 0.03 * vol * flow)
        half = max(spr, 1) * cfg.tick / 2
        bid = round((mid - half) / cfg.tick) * cfg.tick
        ask = max(round((mid + half) / cfg.tick) * cfg.tick, bid + cfg.tick)
        bq = float(rng.exponential(depth)) + 0.01
        aq = float(rng.exponential(depth)) + 0.01
        books.append(BookSnapshot(ts, ((bid, bq),), ((ask, aq),), ts))
        regimes[i] = regime
    return books, trades, regimes
