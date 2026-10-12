"""Deterministic post-halt inventory unwind (SIMULATION ONLY).

RISK-2: a drawdown halt used to cancel quotes but leave inventory open. After a halt the
engine now flattens deterministically:
  1) passive phase (passive_s): one reduce-only maker order at the touch on the reducing side;
  2) aggressive phase: reduce-only taker clips of at most `clip`, never beyond a price limit of
     `max_slip_bps` from the touch; never larger than |position| (cannot flip or add exposure).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnwindAction:
    kind: str            # "passive" | "aggressive" | "done"
    side: int = 0        # +1 buy, -1 sell
    qty: float = 0.0
    price: float = 0.0   # passive limit price
    limit: float = 0.0   # aggressive worst acceptable price


class Unwinder:
    def __init__(self, passive_s: float = 5.0, clip: float = 0.01, max_slip_bps: float = 10.0,
                 eps: float = 1e-12) -> None:
        self.passive_ms = int(passive_s * 1000)
        self.clip = clip
        self.slip = max_slip_bps * 1e-4
        self.eps = eps
        self.start_ms: int | None = None

    def plan(self, now_ms: int, position: float, best_bid: float, best_ask: float) -> UnwindAction:
        if self.start_ms is None:
            self.start_ms = now_ms
        if abs(position) <= self.eps:
            return UnwindAction("done")
        side = -1 if position > 0 else 1
        qty = min(self.clip, abs(position))
        if now_ms - self.start_ms < self.passive_ms:
            return UnwindAction("passive", side, qty, price=best_ask if side == -1 else best_bid)
        limit = best_bid * (1 - self.slip) if side == -1 else best_ask * (1 + self.slip)
        return UnwindAction("aggressive", side, qty, limit=limit)
