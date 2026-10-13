"""Round-2 convex / trend / event strategies (C1-C8). All causal: target weights at t use data
<= t only; the engine adds the mandatory execution delay. See docs/research/strategies_convex_2026-10.md."""
from __future__ import annotations

import numpy as np
import pandas as pd

from raim.strategies.trend import atr


# ---------------------------------------------------------------- C1 intraday TSMOM (30m)
def intraday_tsmom_weights(c30: pd.Series, start_hour: int, k_sigma: float, long_only: bool,
                           delay: int = 1) -> pd.Series:
    """Shen-Urquhart-Wang: sign of the first half-hour of a 24h 'session' starting at start_hour
    UTC predicts the LAST half-hour. Target weight is set (delay+1) bars before the last bar so the
    engine holds it exactly during the last half-hour."""
    r = c30.pct_change()
    sess = (c30.index - pd.Timedelta(hours=start_hour) - pd.Timedelta("1ns")).floor("D")
    first_end = sess + pd.Timedelta(hours=start_hour, minutes=30)
    is_first = c30.index == first_end
    first_r = pd.Series(np.where(is_first, r, np.nan), index=c30.index)
    first_r = first_r.groupby(sess).transform("max")         # only one non-NaN per session
    sig_vol = r.where(is_first).dropna().shift(1).rolling(60, min_periods=20).std()
    sig_vol = sig_vol.reindex(c30.index).groupby(sess).transform("max")
    known = c30.index >= first_end                              # first half-hour finished
    mask = (first_r.abs() > k_sigma * sig_vol.fillna(np.inf)) if k_sigma > 0 else pd.Series(True, index=c30.index)
    s = np.sign(first_r).where(mask, 0.0)
    if long_only:
        s = s.clip(lower=0)
    last_end = sess + pd.Timedelta(hours=start_hour + 24)
    target_slot = last_end - pd.Timedelta(minutes=30 * (delay + 1))
    w = pd.Series(np.where((c30.index == target_slot) & known, s, 0.0), index=c30.index)
    return w.fillna(0.0)


# ---------------------------------------------------------------- C2/C3 breakouts with ATR trail
def breakout_atr_state(close, high, low, entry_long, entry_short, alpha, k=14):
    a = atr(high, low, close, k).values
    c = close.values
    el, es = entry_long.values, entry_short.values
    out = np.zeros(len(c))
    s, stop = 0, np.nan
    for i in range(len(c)):
        if np.isnan(a[i]):
            continue
        if s == 1:
            stop = max(stop, c[i] - alpha * a[i])
            if c[i] < stop:
                s = 0
        elif s == -1:
            stop = min(stop, c[i] + alpha * a[i])
            if c[i] > stop:
                s = 0
        if s == 0:
            if el[i]:
                s, stop = 1, c[i] - alpha * a[i]
            elif es[i]:
                s, stop = -1, c[i] + alpha * a[i]
        out[i] = s
    return pd.Series(out, index=close.index)


def donchian_breakout_state(b: pd.DataFrame, n: int, alpha: float, long_only=False):
    hi = b.high.shift(1).rolling(n).max()
    lo = b.low.shift(1).rolling(n).min()
    el = (b.close > hi).fillna(False)
    es = ((b.close < lo).fillna(False)) & (not long_only)
    return breakout_atr_state(b.close, b.high, b.low, el, es, alpha)


def squeeze_breakout_state(b: pd.DataFrame, n: int, bb_k: float, kc_k: float, alpha: float):
    """Bollinger-inside-Keltner squeeze; enter when the squeeze has been on within the last 3
    bars and close breaks the Bollinger band; ATR trailing exit."""
    m = b.close.rolling(n).mean()
    sd = b.close.rolling(n).std()
    a = atr(b.high, b.low, b.close, n)
    squeeze = ((m + bb_k * sd) < (m + kc_k * a)) & ((m - bb_k * sd) > (m - kc_k * a))
    recent = squeeze.rolling(3).max().fillna(0).astype(bool)
    el = recent & (b.close > m + bb_k * sd)
    es = recent & (b.close < m - bb_k * sd)
    return breakout_atr_state(b.close, b.high, b.low, el.fillna(False), es.fillna(False), alpha)


# ---------------------------------------------------------------- C4 funding extremes
def funding_extreme_weights(close_1h: pd.Series, funding: pd.Series, q: float, mode: str,
                            hold: int = 24, lookback: int = 270) -> pd.Series:
    """Event at a funding settlement whose rate is below the rolling q-quantile of the previous
    `lookback` settlements (~90 days). mode='conditional' (Shek 2026): long if the prior 24h return
    is up (reversal), short if down (continuation); 'squeeze': always long; 'fade_pos': short when
    funding is above the (1-q) quantile."""
    f = funding.sort_index()
    lo = f.shift(1).rolling(lookback, min_periods=90).quantile(q)
    hi = f.shift(1).rolling(lookback, min_periods=90).quantile(1 - q)
    r24 = close_1h / close_1h.shift(24) - 1
    sig = pd.Series(0.0, index=close_1h.index)
    if mode == "fade_pos":
        ev = f[f > hi].index
    else:
        ev = f[f < lo].index
    for t in ev:
        if t not in sig.index:
            continue
        if mode == "conditional":
            d = 1.0 if r24.get(t, np.nan) > 0 else -1.0 if r24.get(t, np.nan) < 0 else 0.0
        elif mode == "squeeze":
            d = 1.0
        else:
            d = -1.0
        i = sig.index.get_loc(t)
        sig.iloc[i:i + hold] = d
    return sig


# ---------------------------------------------------------------- C5 cascade continuation/reversal
def cascade_weights(b: pd.DataFrame, k: float, vol_mult: float, hold: int, direction: str) -> pd.Series:
    r = b.close.pct_change()
    sd = r.shift(1).rolling(168, min_periods=100).std()
    med_v = b.quote_volume.shift(1).rolling(168, min_periods=100).median()
    ev = (r.abs() > k * sd) & (b.quote_volume > vol_mult * med_v)
    d = np.sign(r) * (1 if direction == "cont" else -1)
    sig = np.zeros(len(r))
    ev_v, d_v = ev.values, d.values
    for i in np.where(ev_v)[0]:
        sig[i:i + hold] = d_v[i]
    return pd.Series(sig, index=b.index)


# ---------------------------------------------------------------- C6 altcoin TSMOM weekly (PIT)
def alt_tsmom_weights(close: pd.DataFrame, univ: pd.DataFrame, L_weeks: int, target_vol: float,
                      max_gross: float = 1.0, weekday: int = 6) -> pd.DataFrame:
    mom = close / close.shift(7 * L_weeks) - 1
    vol = np.log(close).diff().rolling(30, min_periods=20).std() * np.sqrt(365)
    reb = close.index.weekday == weekday
    raw = ((mom > 0) & univ).astype(float) * (target_vol / vol).clip(upper=2.0)
    n = univ.sum(axis=1).clip(lower=1)
    raw = raw.div(n, axis=0).fillna(0.0)
    g = raw.sum(axis=1)
    raw = raw.mul((max_gross / g).where(g > max_gross, 1.0), axis=0)
    w = raw.where(pd.Series(reb, index=close.index), np.nan).ffill().fillna(0.0)
    alive = close.notna()                                     # never hold after the data ends
    return w.where(alive, 0.0)


# ---------------------------------------------------------------- C7 Donchian ensemble on PIT
def donchian_pit_weights(close: pd.DataFrame, univ: pd.DataFrame, lookbacks, target_vol,
                         max_gross: float = 1.0) -> pd.DataFrame:
    from raim.strategies.trend import donchian_state
    sig = {}
    for c in close.columns:
        s = close[c].dropna()
        if len(s) < max(lookbacks) // 3:
            continue
        e = sum(donchian_state(s, n) for n in lookbacks) / len(lookbacks)
        sig[c] = e.reindex(close.index).fillna(0.0)
    sig = pd.DataFrame(sig).reindex(columns=close.columns).fillna(0.0) * univ.astype(float)
    vol = np.log(close).diff().rolling(90, min_periods=30).std() * np.sqrt(365)
    n = univ.sum(axis=1).clip(lower=1)
    w = sig.div(n, axis=0) * (target_vol / vol).clip(upper=3.0)
    w = w.fillna(0.0)
    g = w.abs().sum(axis=1)
    w = w.mul((max_gross / g).where(g > max_gross, 1.0), axis=0)
    return w.where(close.notna(), 0.0)


# ---------------------------------------------------------------- C8 new listings
def listing_weights(panel: dict, first_allowed: str, hold: int, cond: str, size: float = 0.02,
                    min_qv: float = 1e6) -> pd.DataFrame:
    """Event = first daily bar of a USDT spot symbol in the archive after `first_allowed` (a
    listing). Decision at the close of day 1 (known then): cond='mom' buys only if day-1
    close > open, 'all' buys every listing, 'fade_skip' buys only if day-1 close < open.
    Each event gets a fixed `size` of capital for `hold` days."""
    close, opn, qv = panel["close"], panel["open"], panel["qv"]
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for c in close.columns:
        s = close[c].dropna()
        if s.empty or s.index[0] <= pd.Timestamp(first_allowed, tz="UTC"):
            continue
        t0 = s.index[0]
        r1 = s.iloc[0] / opn[c].loc[t0] - 1
        if qv[c].loc[t0] < min_qv:
            continue
        if (cond == "mom" and r1 <= 0) or (cond == "fade_skip" and r1 >= 0):
            continue
        i = close.index.get_loc(t0)
        w.iloc[i:i + hold, close.columns.get_loc(c)] = size
    return w.where(close.notna(), 0.0)
