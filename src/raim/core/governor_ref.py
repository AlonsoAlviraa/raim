"""Pure-Python reference: deterministic risk governor with hard limits.

External multipliers (LLM, Jev) can only TIGHTEN: applied m = min(m_llm, m_rules, m_jev),
clamped to [0,1]; NaN -> 0. Drawdown halt is latched and cannot be cleared by any input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    allow_bid: bool
    allow_ask: bool
    cancel_all: bool
    halted: bool
    size_mult: float
    spread_mult: float
    qmax: float
    reason: str


def _clamp01(x: float) -> float:
    if x != x:
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class RiskGovernor:
    def __init__(self, budget_usd: float, z: float, h_unwind_s: float, depth_frac: float,
                 max_gross_usd: float, max_drawdown_usd: float, stale_feed_ms: int,
                 max_reject_rate: float, max_msgs_per_s: float, loss_streak_widen: int,
                 loss_streak_pause: int, pause_s: float) -> None:
        self.budget = budget_usd
        self.z = z
        self.h = h_unwind_s
        self.c = depth_frac
        self.max_gross = max_gross_usd
        self.max_dd = max_drawdown_usd
        self.stale_ms = stale_feed_ms
        self.max_rej = max_reject_rate
        self.max_msg = max_msgs_per_s
        self.widen_n = loss_streak_widen
        self.pause_n = loss_streak_pause
        self.pause_ms = int(pause_s * 1000)
        self.m_llm = 1.0
        self.m_jev = 1.0
        self.halted = False
        self.peak = -math.inf
        self.loss_streak = 0
        self.pause_until = -1

    def set_external(self, m_llm: float, m_jev: float) -> None:
        self.m_llm = _clamp01(m_llm)
        self.m_jev = _clamp01(m_jev)

    def on_trade_result(self, pnl: float, now_ms: int) -> None:
        if pnl < 0.0:
            self.loss_streak += 1
            if self.loss_streak >= self.pause_n:
                self.pause_until = now_ms + self.pause_ms
                self.loss_streak = 0
        elif pnl > 0.0:
            self.loss_streak = 0

    def qmax(self, sigma_frac_s: float, mid: float, depth_qty: float) -> float:
        if not (sigma_frac_s > 0.0 and mid > 0.0):
            return 0.0
        q_var = self.budget / (self.z * sigma_frac_s * mid * math.sqrt(self.h))
        return max(min(q_var, self.c * max(depth_qty, 0.0)), 0.0)

    def evaluate(self, now_ms: int, last_feed_ms: int, equity: float, position: float,
                 mid: float, sigma_frac_s: float, depth_qty: float, reject_rate: float,
                 msg_rate: float) -> Decision:
        if equity > self.peak:
            self.peak = equity
        if self.peak - equity >= self.max_dd:
            self.halted = True
        qm = self.qmax(sigma_frac_s, mid, depth_qty)
        if self.halted:
            return Decision(False, False, True, True, 0.0, 1.0, qm, "drawdown_halt")
        if now_ms - last_feed_ms > self.stale_ms or not (mid > 0.0):
            return Decision(False, False, True, False, 0.0, 1.0, qm, "stale_feed")
        if now_ms < self.pause_until:
            return Decision(False, False, True, False, 0.0, 1.0, qm, "loss_streak_pause")
        m_rules = 1.0
        spread_mult = 1.0
        reason = "ok"
        if reject_rate > self.max_rej or msg_rate > self.max_msg:
            m_rules = 0.0
            reason = "rate_backoff"
        elif self.loss_streak >= self.widen_n:
            m_rules = 0.5
            spread_mult = 2.0
            reason = "loss_streak_widen"
        m = min(self.m_llm, m_rules, self.m_jev)
        if m < 1.0 and reason == "ok":
            reason = "external_tighten"
        lim = qm * m
        gross = abs(position) * mid
        allow_bid = m > 0.0 and position < lim and (gross < self.max_gross or position < 0.0)
        allow_ask = m > 0.0 and -position < lim and (gross < self.max_gross or position > 0.0)
        # if over the limit, allow only the reducing side even when m==0 (de-risking)
        if position > lim and position > 0.0:
            allow_ask = True
        if -position > lim and position < 0.0:
            allow_bid = True
        return Decision(allow_bid, allow_ask, False, False, m, spread_mult, qm, reason)
