"""Calibration checks for external probabilistic judges (Jev) BEFORE they are trusted.
Jev is early-access and proprietary; no independent evidence of calibration on market data
exists (deep research 2026-10, §5). RAIM measures Brier score and ECE against realized
outcomes on recorded data; an uncalibrated Jev is ignored (it could only tighten anyway)."""
from __future__ import annotations

import numpy as np


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def reliability(p, y, n_bins: int = 10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((b, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return rows   # (bin, n, mean_pred, observed_freq)


def ece(p, y, n_bins: int = 10) -> float:
    n = len(p)
    return float(sum(c / n * abs(mp - fy) for _, c, mp, fy in reliability(p, y, n_bins)))


def calibration_gate(p, y, max_ece: float = 0.05, min_n: int = 500) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    e, b = (ece(p, y), brier(p, y)) if len(p) else (float("nan"), float("nan"))
    base = brier(np.full_like(y, y.mean()), y) if len(y) else float("nan")
    passed = bool(len(p) >= min_n and e <= max_ece and b <= base)
    return {"passed": passed, "n": int(len(p)), "ece": e, "brier": b, "brier_climatology": base}


def toxic_labels(mid_ts, mids, query_ts, horizon_ms: int = 5000, threshold_bps: float = 2.0):
    """Outcome for 'is flow toxic now?': |mid move| over the next horizon exceeds threshold.
    Labels use FUTURE mids by construction — use only for offline evaluation, never as features."""
    mt, mv = np.asarray(mid_ts), np.asarray(mids, float)
    out = []
    for t in query_ts:
        i0 = np.searchsorted(mt, t, side="right") - 1
        i1 = np.searchsorted(mt, t + horizon_ms, side="right") - 1
        out.append(np.nan if i0 < 0 or t + horizon_ms > mt[-1] else
                   float(abs(mv[i1] - mv[i0]) / mv[i0] * 1e4 > threshold_bps))
    return np.asarray(out)
