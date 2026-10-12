"""S1 random-maturity basis arbitrage (He, Manela, Ross, von Wachter, arXiv:2212.06888) and
S2 funding-rate carry (Christin et al. carry used as baseline in that paper; risk framing from
Pindza 2026, Digital Finance, doi:10.1007/s42521-026-00213-3).

Instruments: spot 'S' and USDT-M perpetual 'F' of the same coin, hourly bars.
Deviation (paper Eq. 17, per 8h funding units, Binance clamp kappa=1, gamma=5bp, iota=1bp):
    rho = (F - S)/F + sgn(iota - r) * gamma - r_8h
With r (stablecoin rate) < iota the clamp-adjusted benchmark sits at rho = 0.
Open short-perp/long-spot when rho > c (round-trip cost, paper Eq. 19 rho ~ kappa*c), the
reverse (needs spot borrow) when rho < -c; close when rho crosses 0. Legs are +-0.5 of capital."""
from __future__ import annotations

import numpy as np
import pandas as pd

GAMMA, IOTA = 0.0005, 0.0001


def rho_series(spot_close: pd.Series, perp_close: pd.Series, r_annual: float = 0.05) -> pd.Series:
    r8 = r_annual / (3 * 365)
    p = (perp_close - spot_close) / perp_close
    return p + np.sign(IOTA - r8) * GAMMA - r8


def basis_arb_weights(rho: pd.Series, open_thr: float, close_thr: float = 0.0,
                      allow_short_spot: bool = False) -> pd.DataFrame:
    """Causal state machine on rho_t (known at bar end t)."""
    v = rho.values
    st = np.zeros(len(v))
    s = 0
    for i, x in enumerate(v):
        if np.isnan(x):
            st[i] = s
            continue
        if s == 0:
            if x > open_thr:
                s = 1                       # short perp, long spot
            elif allow_short_spot and x < -open_thr:
                s = -1
        elif s == 1 and x <= close_thr:
            s = 0
        elif s == -1 and x >= -close_thr:
            s = 0
        st[i] = s
    st = pd.Series(st, index=rho.index)
    return pd.DataFrame({"S": 0.5 * st, "F": -0.5 * st})


def funding_carry_weights(funding_events: pd.Series, index: pd.DatetimeIndex, lookback: int,
                          thr: float) -> pd.DataFrame:
    """Hold long spot / short perp while the trailing mean of the last `lookback` realised
    funding prints exceeds thr. Decision updates only at settlement times (known then)."""
    sig = (funding_events.rolling(lookback).mean() > thr).astype(float)
    sig = sig.reindex(index).ffill().fillna(0.0)
    return pd.DataFrame({"S": 0.5 * sig, "F": -0.5 * sig})
