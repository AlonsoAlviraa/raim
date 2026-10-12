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
    """Returns (bid, ask, reservation_price, half_spread_bps); all NaN = no quote."""
    ok = (gamma > 0.0 and kappa > 0.0 and mid > 0.0 and tick > 0.0 and horizon_s >= 0.0
          and all(math.isfinite(x) for x in (mid, sigma_bps, q, gamma, kappa, horizon_s,
                                              min_half_spread_bps, spread_mult)))
    if not ok:
        return math.nan, math.nan, math.nan, math.nan
    var_t = sigma_bps * sigma_bps * horizon_s
    shift = -q * gamma * var_t
    half = 0.5 * gamma * var_t + math.log(1.0 + gamma / kappa) / gamma
    half = max(half, min_half_spread_bps) * max(spread_mult, 1.0)
    res = mid * (1.0 + shift * 1e-4)
    bid = math.floor(res * (1.0 - half * 1e-4) / tick + 1e-9) * tick
    ask = math.ceil(res * (1.0 + half * 1e-4) / tick - 1e-9) * tick
    return bid, ask, res, half


def exposure_adjust(qa_bid: float, qa_ask: float, flow_bias: float, p_change: float,
                    k_flow_bps: float, k_queue_bps: float, max_extra_bps: float
                    ) -> tuple[float, float, float, float]:
    """Queue-adjusted exposure asymmetry (Moret & Lillo, arXiv:2609.11614).

    Returns (bid_extra_bps, ask_extra_bps, bid_size_mult, ask_size_mult). It can only WIDEN
    a side (extra >= 0) and SHRINK its size (mult <= 1):
    - persistent buy flow (flow_bias > 0) exposes our ASK, amplified by a recent change point;
    - the side with the SHORTER queue ahead of us fills first, so it is the more exposed side.
    Any NaN input -> maximal widening and zero size on both sides (conservative)."""
    vals = (qa_bid, qa_ask, flow_bias, p_change)
    if any(v != v for v in vals) or max_extra_bps <= 0.0:
        return max_extra_bps, max_extra_bps, 0.0, 0.0
    cap = 1e12
    qb = min(max(qa_bid, 0.0), cap)
    qa = min(max(qa_ask, 0.0), cap)
    tot = qb + qa
    imb = (qb - qa) / tot if tot > 0.0 else 0.0
    fb = min(max(flow_bias, -1.0), 1.0)
    pc = min(max(p_change, 0.0), 1.0)
    amp = 1.0 + pc
    ask_x = k_flow_bps * max(fb, 0.0) * amp + k_queue_bps * max(imb, 0.0)
    bid_x = k_flow_bps * max(-fb, 0.0) * amp + k_queue_bps * max(-imb, 0.0)
    bid_x = min(max(bid_x, 0.0), max_extra_bps)
    ask_x = min(max(ask_x, 0.0), max_extra_bps)
    return bid_x, ask_x, 1.0 - bid_x / max_extra_bps, 1.0 - ask_x / max_extra_bps
