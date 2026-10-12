"""Offline Baum-Welch fit for a diagonal-Gaussian HMM (numpy, scaled forward-backward).
Fit only on TRAIN folds; the fitted params are then used by the causal forward filter."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)
REGIME_LABELS = ("R1", "R2", "R3", "R4")


@dataclass(frozen=True)
class HMMParams:
    pi0: np.ndarray
    trans: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    def as_lists(self) -> tuple[list, list, list, list]:
        return (self.pi0.tolist(), self.trans.tolist(), self.means.tolist(), self.variances.tolist())


def _loglik(x: np.ndarray, means: np.ndarray, var: np.ndarray) -> np.ndarray:
    d = x[:, None, :] - means[None]
    return -0.5 * (np.log(2 * np.pi * var)[None] + d * d / var[None]).sum(-1)


def fit_hmm(x: np.ndarray, n_states: int = 4, n_iter: int = 50, seed: int = 0,
            tol: float = 1e-4, var_floor: float = 1e-4) -> HMMParams:
    """x: (T, D) standardized observations. States are re-ordered by a stress score
    (mean of observations) so that R1 is the calmest and R4 the most extreme."""
    rng = np.random.default_rng(seed)
    t_len, d = x.shape
    q = np.quantile(x.mean(1), np.linspace(0.1, 0.9, n_states))
    order = np.argsort(x.mean(1))
    chunks = np.array_split(order, n_states)
    means = np.stack([x[c].mean(0) for c in chunks]) + 1e-3 * rng.standard_normal((n_states, d))
    var = np.stack([x[c].var(0) + var_floor for c in chunks])
    trans = np.full((n_states, n_states), 0.02 / (n_states - 1))
    np.fill_diagonal(trans, 0.98)
    pi0 = np.full(n_states, 1.0 / n_states)
    prev = -np.inf
    _ = q
    for it in range(n_iter):
        ll = _loglik(x, means, var)
        mx = ll.max(1, keepdims=True)
        b = np.exp(ll - mx)
        alpha = np.zeros((t_len, n_states))
        c = np.zeros(t_len)
        a = pi0 * b[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, t_len):
            a = (alpha[t - 1] @ trans) * b[t]
            c[t] = a.sum() + 1e-300
            alpha[t] = a / c[t]
        beta = np.ones((t_len, n_states))
        for t in range(t_len - 2, -1, -1):
            beta[t] = trans @ (b[t + 1] * beta[t + 1]) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        xi_sum = np.zeros((n_states, n_states))
        for t in range(t_len - 1):
            xi = alpha[t][:, None] * trans * (b[t + 1] * beta[t + 1])[None] / c[t + 1]
            xi_sum += xi / max(xi.sum(), 1e-300)
        pi0 = gamma[0] + 1e-6
        pi0 /= pi0.sum()
        trans = xi_sum + 1e-6
        trans /= trans.sum(1, keepdims=True)
        w = gamma.sum(0)[:, None]
        means = (gamma.T @ x) / w
        var = (gamma.T @ (x * x)) / w - means ** 2 + var_floor
        var = np.maximum(var, var_floor)
        total = float(np.log(c).sum() + mx.sum())
        if abs(total - prev) < tol * max(1.0, abs(prev)):
            log.info("Baum-Welch converged at iter %d, loglik=%.3f", it, total)
            break
        prev = total
    stress = means.mean(1)
    perm = np.argsort(stress)
    return HMMParams(pi0[perm], trans[np.ix_(perm, perm)], means[perm], var[perm])


def tilt_transition(trans: np.ndarray, vix_z: float, event_risk: float,
                    max_tilt: float = 0.5) -> np.ndarray:
    """Slow-state context tilts the transition matrix TOWARD stress states only
    (conservative): extra mass moved into columns R3/R4, never out of them."""
    s = float(np.clip(0.1 * max(vix_z, 0.0) + event_risk, 0.0, 1.0)) * max_tilt
    if s <= 0.0:
        return trans.copy()
    out = trans.copy()
    k = out.shape[0]
    for i in range(k):
        moved = s * out[i, : k - 2].sum()
        out[i, : k - 2] *= 1.0 - s
        out[i, k - 2:] += moved / 2.0
    return out / out.sum(1, keepdims=True)
