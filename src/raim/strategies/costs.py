"""Retail (VIP0) cost model. Fees as fractions of notional per side.
Binance VIP0 (public fee schedule, re-check before any deployment): spot 0.10% maker/taker
(0.075% with BNB, not assumed); USDT-M futures 0.02% maker / 0.05% taker.
Slippage is an extra half-spread+impact allowance per side. We assume TAKER fills everywhere:
maker fills on slow strategies are not guaranteed and modelling them would flatter results."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    spot_fee: float = 0.0010
    perp_fee: float = 0.0005
    slip: float = 0.0002          # per side, large caps
    slip_small: float = 0.0010    # per side, small caps (PIVX/QKC/GNO)

    def spot_side(self, small: bool = False) -> float:
        return self.spot_fee + (self.slip_small if small else self.slip)

    def perp_side(self) -> float:
        return self.perp_fee + self.slip


VIP0 = Costs()
STRESS2X = Costs(spot_fee=0.0010, perp_fee=0.0005, slip=0.0004, slip_small=0.0020)
