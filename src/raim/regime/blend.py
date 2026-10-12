"""Parameter blending theta = sum_k p_k theta_k with conservative bias + hysteresis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def conservative_posterior(p: np.ndarray, bias: float, cpd_alarm: bool) -> np.ndarray:
    """Move a fraction `bias` of R1/R2 mass into R3/R4 (more if a change point fired)."""
    p = np.asarray(p, dtype=float).copy()
    b = min(1.0, bias + (0.5 if cpd_alarm else 0.0))
    moved = b * p[:2].sum()
    p[:2] *= 1.0 - b
    p[2:] += moved / 2.0
    return p / p.sum()


def blend_params(p: np.ndarray, regime_params: dict[str, list[float]]) -> tuple[float, float, float]:
    """Returns (gamma_mult, spread_mult, size_mult)."""
    theta = np.array([regime_params[k] for k in ("R1", "R2", "R3", "R4")], dtype=float)
    g, s, z = (p[:, None] * theta).sum(0)
    return float(g), float(s), float(z)


@dataclass
class Hysteresis:
    on: float = 0.7
    off: float = 0.4
    pulled: bool = False

    def update(self, p_r4: float) -> bool:
        if not self.pulled and p_r4 > self.on:
            self.pulled = True
        elif self.pulled and p_r4 < self.off:
            self.pulled = False
        return self.pulled
