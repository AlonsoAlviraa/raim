"""Pure-Python reference: online Bayesian change-point detection on the SIGN of order flow
(Adams & MacKay 2007, Beta-Bernoulli model), used as a fast-body feature following
Moret & Lillo (arXiv:2609.11614): a change-point filter on directional flow bias restores
market-maker profitability under regime-switching (metaorder) flow.

update(sign) -> [p_change, flow_bias]
  p_change  = posterior mass on run lengths < short_run (a recent change in flow direction)
  flow_bias = 2*E[P(buy)] - 1 in [-1, 1], posterior-averaged over run lengths.
Causal: depends only on signs seen so far. Loops are explicit to match the C++ summation order.
"""
from __future__ import annotations


class FlowCPD:
    def __init__(self, hazard: float = 0.01, a0: float = 1.0, b0: float = 1.0,
                 max_run: int = 200, short_run: int = 30) -> None:
        self.h = hazard
        self.a0 = a0
        self.b0 = b0
        self.max_run = max_run
        self.short_run = short_run
        self.r = [1.0]
        self.n1 = [0.0]
        self.n = [0.0]

    def update(self, sign: int) -> list[float]:
        x = 1.0 if sign > 0 else 0.0
        a0, b0, h = self.a0, self.b0, self.h
        k = len(self.r)
        growth = [0.0] * k
        cp = 0.0
        for i in range(k):
            denom = a0 + b0 + self.n[i]
            pb = (a0 + self.n1[i]) / denom
            pred = pb if x == 1.0 else 1.0 - pb
            w = self.r[i] * pred
            growth[i] = w * (1.0 - h)
            cp += w * h
        r = [cp] + growth
        n1 = [0.0] + [v + x for v in self.n1]
        n = [0.0] + [v + 1.0 for v in self.n]
        if len(r) > self.max_run:
            r, n1, n = r[: self.max_run], n1[: self.max_run], n[: self.max_run]
        s = 0.0
        for v in r:
            s += v
        if s > 0.0:
            r = [v / s for v in r]
        else:
            r = [1.0] + [0.0] * (len(r) - 1)
        self.r, self.n1, self.n = r, n1, n
        pc = 0.0
        eb = 0.0
        for i in range(len(r)):
            if i < self.short_run:
                pc += r[i]
            eb += r[i] * (a0 + n1[i]) / (a0 + b0 + n[i])
        return [pc, 2.0 * eb - 1.0]
