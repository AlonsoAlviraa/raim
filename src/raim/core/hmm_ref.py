"""Pure-Python reference: diagonal-Gaussian HMM forward filter (causal)."""
from __future__ import annotations

import math


class HMMFilter:
    def __init__(self, pi0: list[float], trans: list[list[float]],
                 means: list[list[float]], variances: list[list[float]]) -> None:
        self.k = len(pi0)
        self.d = len(means[0])
        self.trans = [list(r) for r in trans]
        self.means = [list(r) for r in means]
        self.vars = [[max(v, 1e-12) for v in r] for r in variances]
        self.post = list(pi0)

    def set_transition(self, trans: list[list[float]]) -> None:
        self.trans = [list(r) for r in trans]

    def step(self, x: list[float]) -> list[float]:
        """p_t(j) ∝ f_j(x_t) * sum_i p_{t-1}(i) A_ij. NaN obs dims are skipped."""
        k, d = self.k, self.d
        prior = [sum(self.post[i] * self.trans[i][j] for i in range(k)) for j in range(k)]
        logl = [0.0] * k
        for j in range(k):
            acc = 0.0
            for t in range(d):
                xv = x[t]
                if xv != xv:  # NaN
                    continue
                v = self.vars[j][t]
                diff = xv - self.means[j][t]
                acc += -0.5 * (math.log(2.0 * math.pi * v) + diff * diff / v)
            logl[j] = acc
        mx = max(logl)
        w = [prior[j] * math.exp(logl[j] - mx) for j in range(k)]
        s = sum(w)
        if not (s > 0.0) or s != s:
            self.post = prior  # uninformative obs: keep prediction
        else:
            self.post = [v / s for v in w]
        return list(self.post)
