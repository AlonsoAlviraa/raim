"""Pure-Python reference: Avellaneda-Stoikov (2008) / GLFT (2013) quoter in bps space.

reservation shift (bps)  = -q * gamma * sigma_bps^2 * T
half spread (bps)        = 0.5*gamma*sigma_bps^2*T + (1/gamma)*ln(1 + gamma/kappa)
q is inventory in units of order size; sigma_bps is per sqrt(second).
"""
from __future__ import annotations

import math


def quote(mid: float, sigma_bps: float, q: float, gamma: float, kappa: float,
          horizon_s: float, min_half_spread_bps: float, spread_mult: float,
          tick: float) -> tuple[float, float, float, float]:
    """Returns (bid, ask, reservation_price, half_spread_bps)."""
    var_t = sigma_bps * sigma_bps * horizon_s
    shift = -q * gamma * var_t
    half = 0.5 * gamma * var_t + math.log(1.0 + gamma / kappa) / gamma
    half = max(half, min_half_spread_bps) * max(spread_mult, 1.0)
    res = mid * (1.0 + shift * 1e-4)
    bid = math.floor(res * (1.0 - half * 1e-4) / tick + 1e-9) * tick
    ask = math.ceil(res * (1.0 + half * 1e-4) / tick - 1e-9) * tick
    return bid, ask, res, half
