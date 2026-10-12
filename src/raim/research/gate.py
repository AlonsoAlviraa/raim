"""First gate: maker_fee + E[adverse_selection] < 0.5 * typical spread.
Reports PASS/FAIL with the numbers. A PASS is a necessary condition, NOT evidence of profit."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateResult:
    maker_fee_bps: float
    adverse_selection_bps: float
    typical_spread_bps: float
    threshold_bps: float
    n_samples: int
    passed: bool
    note: str

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (f"[{verdict}] maker_fee={self.maker_fee_bps:.3f}bps + E[AS]={self.adverse_selection_bps:.3f}bps "
                f"vs 0.5*median_spread={self.threshold_bps:.3f}bps (n={self.n_samples}). {self.note}")


def adverse_selection_from_trades(mids_ts: np.ndarray, mids: np.ndarray, trade_ts: np.ndarray,
                                  trade_px: np.ndarray, trade_sign: np.ndarray,
                                  horizon_ms: int = 5000) -> np.ndarray:
    """Venue-level proxy: for each aggressive trade, the move of the mid in the aggressor's
    direction over `horizon_ms`, measured from the mid at trade time (bps). A passive maker
    on the other side loses this on average (before capturing half-spread)."""
    out = []
    for t, s in zip(trade_ts, trade_sign):
        if t + horizon_ms > mids_ts[-1]:
            continue
        i0 = np.searchsorted(mids_ts, t, side="right") - 1
        i1 = np.searchsorted(mids_ts, t + horizon_ms, side="right") - 1
        if i0 < 0:
            continue
        out.append(s * (mids[i1] - mids[i0]) / mids[i0] * 1e4)
    _ = trade_px
    return np.asarray(out)


def first_gate(spread_bps: np.ndarray, adverse_bps: np.ndarray, maker_fee_bps: float,
               min_samples: int = 50) -> GateResult:
    spread_bps = spread_bps[np.isfinite(spread_bps)]
    adverse_bps = adverse_bps[np.isfinite(adverse_bps)]
    n = int(min(len(spread_bps), len(adverse_bps)))
    typ = float(np.median(spread_bps)) if len(spread_bps) else float("nan")
    eas = float(np.mean(adverse_bps)) if len(adverse_bps) else float("nan")
    thr = 0.5 * typ
    ok = n >= min_samples and np.isfinite(eas) and (maker_fee_bps + max(eas, 0.0)) < thr
    note = "insufficient samples -> FAIL (conservative)" if n < min_samples else \
        "necessary condition only; not a profitability claim"
    return GateResult(maker_fee_bps, eas, typ, thr, n, bool(ok), note)


def gate_from_markouts(mo: pd.DataFrame, spread_bps: np.ndarray, maker_fee_bps: float) -> GateResult:
    col = "mo_5s_bps"
    adv = -mo[col].to_numpy() if not mo.empty and col in mo else np.array([])
    return first_gate(spread_bps, adv, maker_fee_bps)
