"""Core market-data event types (venue-agnostic)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BookSnapshot:
    ts_ms: int                      # exchange event time (ms)
    bids: tuple[tuple[float, float], ...]  # (price, qty), best first
    asks: tuple[tuple[float, float], ...]
    recv_ms: int = 0                # local receive time (ms)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0]

    @property
    def best_ask(self) -> float:
        return self.asks[0][0]

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)


@dataclass(frozen=True)
class Trade:
    ts_ms: int
    price: float
    qty: float
    is_buyer_maker: bool            # True => aggressor was SELL
    recv_ms: int = 0

    @property
    def sign(self) -> int:
        return -1 if self.is_buyer_maker else 1


@dataclass
class Quote:
    bid_px: float | None
    ask_px: float | None
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    reason: str = ""
    meta: dict = field(default_factory=dict)
