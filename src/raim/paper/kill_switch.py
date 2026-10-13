"""R15 live kill-switch SPEC (for paper now; live would need explicit human approval).

Kill when EITHER
- drawdown exceeds the backtest block-bootstrap p99 max drawdown, OR
- a one-sided CUSUM of (realised - expected) daily PnL, in units of backtest daily sigma, drops
  below -h (default k=0.5 sigma drift allowance, h=5 sigma)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KillSwitch:
    expected_daily: float
    sigma_daily: float
    dd_limit: float                     # negative, e.g. bootstrap dd_p99
    k: float = 0.5
    h: float = 5.0
    equity: float = 1.0
    peak: float = 1.0
    cusum: float = 0.0
    killed: bool = False
    reason: str = ""
    history: list = field(default_factory=list)

    def update(self, ret: float) -> bool:
        if self.killed:
            return True
        self.equity *= 1 + ret
        self.peak = max(self.peak, self.equity)
        dd = self.equity / self.peak - 1
        z = (ret - self.expected_daily) / max(self.sigma_daily, 1e-12)
        self.cusum = min(0.0, self.cusum + z + self.k)
        self.history.append((ret, dd, self.cusum))
        if dd < self.dd_limit:
            self.killed, self.reason = True, f"drawdown {dd:.1%} < bootstrap p99 {self.dd_limit:.1%}"
        elif self.cusum < -self.h:
            self.killed, self.reason = True, f"CUSUM {self.cusum:.2f} < -{self.h}"
        return self.killed

    @classmethod
    def from_backtest(cls, oos_daily, dd_p99: float, k: float = 0.5, h: float = 5.0) -> "KillSwitch":
        x = np.asarray(oos_daily, float)
        return cls(float(np.nanmean(x)), float(np.nanstd(x)), dd_p99, k, h)
