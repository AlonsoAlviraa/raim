"""Observation vector for the HMM from causal features + train-only standardization."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

OBS_NAMES = ("log_spread", "log_vol", "vpin", "abs_ofi", "log_vr", "hawkes_n")


def obs_from_features(v: list[float]) -> list[float]:
    """v follows FEATURE_NAMES: mid, micro, spread_bps, depth, ofi, bv_vol, vpin, vr, hawkes_n."""
    if v[0] != v[0]:
        return [math.nan] * len(OBS_NAMES)
    return [math.log(max(v[2], 1e-6)), math.log(max(v[5], 1e-9)), v[6], abs(v[4]),
            math.log(max(v[7], 1e-3)), v[8]]


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardizer:
        """Fit on TRAIN rows only (leakage guard)."""
        return cls(np.nanmean(x, 0), np.nanstd(x, 0) + 1e-9)

    def apply(self, row: list[float]) -> list[float]:
        return [float((a - m) / s) for a, m, s in zip(row, self.mean, self.std)]
