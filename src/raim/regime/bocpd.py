"""Adams & MacKay (2007) Bayesian online change-point detection, Normal-Gamma (unknown
mean & variance) with constant hazard. Truncated run-length for O(R) per step."""
from __future__ import annotations

import math

import numpy as np


class BOCPD:
    def __init__(self, hazard: float = 0.004, mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0, max_run: int = 300) -> None:
        self.h = hazard
        self.max_run = max_run
        self.prior = (mu0, kappa0, alpha0, beta0)
        self.r = np.array([1.0])
        self.mu = np.array([mu0])
        self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])
        idx = np.arange(max_run + 2)
        df = 2 * (alpha0 + 0.5 * idx)
        self._lg = np.array([math.lgamma((d + 1) / 2) - math.lgamma(d / 2) for d in df])

    def _pred(self, x: float) -> np.ndarray:
        df = 2 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        z = (x - self.mu) / scale
        logp = self._lg[: len(df)] - 0.5 * np.log(df * np.pi) - np.log(scale) \
            - (df + 1) / 2 * np.log1p(z * z / df)
        return np.exp(logp)

    def update(self, x: float) -> float:
        """Returns P(change point at this step) = P(run length == 0 | data)."""
        if x != x:
            return 0.0
        p = self._pred(x)
        growth = self.r * p * (1 - self.h)
        cp = float((self.r * p * self.h).sum())
        r = np.concatenate([[cp], growth])
        s = r.sum()
        r = r / s if s > 0 else np.r_[1.0, np.zeros(len(growth))]
        mu0, k0, a0, b0 = self.prior
        mu = np.concatenate([[mu0], (self.kappa * self.mu + x) / (self.kappa + 1)])
        beta = np.concatenate([[b0], self.beta + self.kappa * (x - self.mu) ** 2 / (2 * (self.kappa + 1))])
        kappa = np.concatenate([[k0], self.kappa + 1])
        alpha = np.concatenate([[a0], self.alpha + 0.5])
        if len(r) > self.max_run:
            r, mu, kappa, alpha, beta = (v[: self.max_run] for v in (r, mu, kappa, alpha, beta))
            r = r / r.sum()
        self.r, self.mu, self.kappa, self.alpha, self.beta = r, mu, kappa, alpha, beta
        # alarm: mass on short run lengths (recent change), robust vs. r0 which equals hazard
        return float(r[: min(5, len(r))].sum())
