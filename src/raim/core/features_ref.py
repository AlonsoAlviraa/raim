"""Pure-Python reference: incremental, causal microstructure features.

Every value at time t uses only events with timestamp <= t (no look-ahead).
Order of outputs is FEATURE_NAMES; the C++ engine must match it exactly.
"""
from __future__ import annotations

import math
from collections import deque

FEATURE_NAMES = ("mid", "microprice", "spread_bps", "depth", "ofi", "bv_vol",
                 "vpin", "vr", "hawkes_n")


class _Rolling:
    """Rolling sum / sum of squares over a fixed window."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.buf: deque[float] = deque()
        self.s = 0.0
        self.s2 = 0.0

    def push(self, x: float) -> None:
        self.buf.append(x)
        self.s += x
        self.s2 += x * x
        if len(self.buf) > self.n:
            y = self.buf.popleft()
            self.s -= y
            self.s2 -= y * y

    def __len__(self) -> int:
        return len(self.buf)

    def mean(self) -> float:
        return self.s / len(self.buf) if self.buf else 0.0

    def var(self) -> float:
        k = len(self.buf)
        if k < 2:
            return 0.0
        m = self.s / k
        return max(self.s2 / k - m * m, 0.0)


class FeatureEngine:
    def __init__(self, vol_window: int = 200, ofi_window: int = 50,
                 vpin_bucket_volume: float = 5.0, vpin_buckets: int = 50,
                 vr_lag: int = 10, hawkes_window: int = 500) -> None:
        self.vol_window = vol_window
        self.vr_lag = vr_lag
        self.bucket_v = vpin_bucket_volume
        self._bp = math.nan
        self._bq = 0.0
        self._ap = math.nan
        self._aq = 0.0
        self._depth = 0.0
        self._last_logm = math.nan
        self._last_absr = math.nan
        self._bv = _Rolling(vol_window)
        self._ofi = _Rolling(ofi_window)
        self._ofi_abs = _Rolling(ofi_window)
        self._r1 = _Rolling(vol_window)
        self._rk = _Rolling(vol_window)
        self._logm_hist: deque[float] = deque(maxlen=vr_lag + 1)
        self._vpin = _Rolling(vpin_buckets)
        self._b_buy = 0.0
        self._b_sell = 0.0
        self._hk = _Rolling(hawkes_window)
        self._hk_bin = -1
        self._hk_count = 0.0

    # ---- events -------------------------------------------------------
    def on_book(self, ts_ms: int, bid: float, bid_qty: float, ask: float,
                ask_qty: float, depth: float) -> None:
        if not (bid > 0 and ask > 0 and ask > bid):
            return  # contradictory book: ignore conservatively
        if not math.isnan(self._bp):
            e = 0.0
            if bid >= self._bp:
                e += bid_qty
            if bid <= self._bp:
                e -= self._bq
            if ask <= self._ap:
                e -= ask_qty
            if ask >= self._ap:
                e += self._aq
            self._ofi.push(e)
            self._ofi_abs.push(abs(e))
        logm = math.log(0.5 * (bid + ask))
        if not math.isnan(self._last_logm):
            r = logm - self._last_logm
            absr = abs(r)
            if not math.isnan(self._last_absr):
                self._bv.push(absr * self._last_absr)
            self._last_absr = absr
            self._r1.push(r)
        self._logm_hist.append(logm)
        if len(self._logm_hist) == self.vr_lag + 1:
            self._rk.push(self._logm_hist[-1] - self._logm_hist[0])
        self._last_logm = logm
        self._bp, self._bq, self._ap, self._aq, self._depth = bid, bid_qty, ask, ask_qty, depth
        self._hawkes_advance(ts_ms)

    def on_trade(self, ts_ms: int, price: float, qty: float, sign: int) -> None:
        self._hawkes_advance(ts_ms)
        self._hk_count += 1.0
        remaining = qty
        while remaining > 0.0:
            room = self.bucket_v - (self._b_buy + self._b_sell)
            take = remaining if remaining < room else room
            if sign > 0:
                self._b_buy += take
            else:
                self._b_sell += take
            remaining -= take
            if self._b_buy + self._b_sell >= self.bucket_v - 1e-12:
                self._vpin.push(abs(self._b_buy - self._b_sell) / self.bucket_v)
                self._b_buy = 0.0
                self._b_sell = 0.0

    def _hawkes_advance(self, ts_ms: int) -> None:
        b = ts_ms // 1000
        if self._hk_bin < 0:
            self._hk_bin = b
            return
        if b <= self._hk_bin:
            return
        gap = min(b - self._hk_bin, self._hk.n + 1)
        self._hk.push(self._hk_count)
        for _ in range(gap - 1):
            self._hk.push(0.0)
        self._hk_count = 0.0
        self._hk_bin = b

    # ---- outputs ------------------------------------------------------
    def values(self) -> list[float]:
        if math.isnan(self._bp):
            return [math.nan] * len(FEATURE_NAMES)
        mid = 0.5 * (self._bp + self._ap)
        tot = self._bq + self._aq
        micro = (self._bp * self._aq + self._ap * self._bq) / tot if tot > 0 else mid
        spread_bps = (self._ap - self._bp) / mid * 1e4
        ofi = self._ofi.s / self._ofi_abs.s if self._ofi_abs.s > 1e-12 else 0.0
        bv_vol = math.sqrt(max(math.pi / 2.0 * self._bv.mean(), 0.0))
        vpin = self._vpin.mean()
        v1 = self._r1.var()
        vr = self._rk.var() / (self.vr_lag * v1) if v1 > 1e-18 and len(self._rk) > 1 else 1.0
        hk = 0.0
        m = self._hk.mean()
        if len(self._hk) >= 10 and m > 1e-12:
            fano = self._hk.var() / m
            if fano > 1.0:
                hk = min(1.0 - 1.0 / math.sqrt(fano), 0.99)
        return [mid, micro, spread_bps, self._depth, ofi, bv_vol, vpin, vr, hk]
