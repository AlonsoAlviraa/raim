"""Stress-scenario generator with 5-level L2 books (SYNTHETIC, for robustness testing only).

Scenario matrix inspired by Yang & Xu (arXiv:2609.22785): directional drift +/-, thin/thick
liquidity, high self-excitation (Hawkes-like clustering), high price impact, and combinations.
Results on these scenarios say nothing about real-market edge."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from raim.types import BookSnapshot, Trade


@dataclass(frozen=True)
class Scenario:
    name: str
    drift_bps: float = 0.0       # per-step drift; also biases trade signs (informed flow)
    vol_bps: float = 0.5
    depth_mult: float = 1.0
    hawkes: float = 0.0          # self-excitation of trade arrivals in [0, 0.9)
    impact: float = 0.03         # mid impact per unit signed trade count (in vol units)
    spread_ticks: int = 2
    base_rate: float = 0.4


PRESETS: dict[str, Scenario] = {
    "base": Scenario("base"),
    "drift_up": Scenario("drift_up", drift_bps=0.3),
    "drift_down": Scenario("drift_down", drift_bps=-0.3),
    "thin": Scenario("thin", depth_mult=0.15),
    "thick": Scenario("thick", depth_mult=5.0),
    "hawkes_high": Scenario("hawkes_high", hawkes=0.8),
    "impact_high": Scenario("impact_high", impact=0.3),
    "drift_up+thin": Scenario("drift_up+thin", drift_bps=0.3, depth_mult=0.15),
    "hawkes_high+impact_high": Scenario("hawkes_high+impact_high", hawkes=0.8, impact=0.3),
    "vol_spike": Scenario("vol_spike", vol_bps=3.0, spread_ticks=6),
}


def scenario_events(scn: Scenario | str, n_steps: int = 10_000, seed: int = 0, dt_ms: int = 100,
                    start: float = 60_000.0, tick: float = 0.01, levels: int = 5) -> list:
    s = PRESETS[scn] if isinstance(scn, str) else scn
    rng = np.random.default_rng(seed)
    p_buy = 0.5 + 0.45 * math.tanh(s.drift_bps / 0.3)
    mid, excite = start, 0.0
    events: list = []
    for i in range(n_steps):
        ts = i * dt_ms
        lam = s.base_rate * (1.0 + s.hawkes * excite / max(1e-9, 1 - s.hawkes))
        n_tr = int(rng.poisson(min(lam, 50.0)))
        excite = s.hawkes * excite + (1 - s.hawkes) * n_tr if s.hawkes > 0 else 0.0
        flow = 0
        half = s.spread_ticks * tick / 2
        for _ in range(n_tr):
            sign = 1 if rng.random() < p_buy else -1
            flow += sign
            sweep = mid * 1e-4 * s.vol_bps * 3.0 * float(rng.exponential()) if rng.random() < 0.3 else 0.0
            px = round((mid + sign * (half + sweep)) / tick) * tick
            events.append(Trade(ts, px, float(rng.exponential(0.05)), sign < 0, ts))
        mid *= 1.0 + 1e-4 * (s.vol_bps * rng.standard_normal() + s.drift_bps + s.impact * s.vol_bps * flow)
        bb = round((mid - half) / tick) * tick
        ba = max(round((mid + half) / tick) * tick, bb + tick)
        bids = tuple((round(bb - k * tick, 2), float(rng.exponential(3.0 * s.depth_mult)) + 0.001)
                     for k in range(levels))
        asks = tuple((round(ba + k * tick, 2), float(rng.exponential(3.0 * s.depth_mult)) + 0.001)
                     for k in range(levels))
        events.append(BookSnapshot(ts, bids, asks, ts))
    events.sort(key=lambda e: (e.ts_ms, isinstance(e, BookSnapshot)))
    return events


_ = replace
