"""R8: Minimum Track Record Length (Bailey & López de Prado 2012). Per-period SR, kurt non-excess."""
from __future__ import annotations

import math
from statistics import NormalDist


def min_trl(sr: float, skew: float, kurt: float, sr_star: float = 0.0, conf: float = 0.95) -> float:
    if sr <= sr_star:
        return math.inf
    z = NormalDist().inv_cdf(conf)
    return 1 + (1 - skew * sr + (kurt - 1) / 4 * sr * sr) * (z / (sr - sr_star)) ** 2
