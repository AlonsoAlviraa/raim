"""Pure-Python reference: tick-level paper exchange with queue position, partial fills,
ack/cancel latency with jitter (latency stress multiplier), post-only rejects, fees/rebates.

Conservative assumptions (documented):
- On activation we join the BACK of the visible L2 queue at our price. A price level inside
  the visible range but absent from the book is empty (queue 0). A price BEYOND the deepest
  visible level has unknown queue and is treated as infinite (only trade-throughs fill it).
- Visible-queue decreases are assumed to come from orders BEHIND us unless the level
  shrinks below our queue_ahead (queue_ahead = min(queue_ahead, level_qty)).
- A fill only happens from an aggressive trade at/through our price, or the opposite
  best moving through our price (trade-through).
- Orders remain fillable until the cancel is acknowledged.
"""
from __future__ import annotations

from dataclasses import dataclass

from raim.core.rng import SplitMix64

BID, ASK = 1, -1
PENDING, ACTIVE, DONE = 0, 1, 2
UNKNOWN_QUEUE = 1e300


@dataclass
class _Order:
    oid: int
    side: int
    price: float
    qty: float
    active_at: int
    state: int = PENDING
    queue_ahead: float = 0.0
    cancel_at: int = -1


class SimExchange:
    def __init__(self, maker_fee_bps: float, rebate_bps: float, ack_ms: float,
                 jitter_ms: float, cancel_ms: float, stress: float, seed: int,
                 taker_fee_bps: float = 5.0) -> None:
        self.fee = (maker_fee_bps - rebate_bps) * 1e-4
        self.taker_fee = taker_fee_bps * 1e-4
        self.ack = ack_ms
        self.jit = jitter_ms
        self.cxl = cancel_ms
        self.stress = stress
        self.rng = SplitMix64(seed)
        self.orders: dict[int, _Order] = {}
        self.next_id = 1
        self.bid = 0.0
        self.bq = 0.0
        self.ask = 0.0
        self.aq = 0.0
        self.n_msgs = 0
        self.n_rejects = 0
        self.bpx: list[float] = []
        self.bqs: list[float] = []
        self.apx: list[float] = []
        self.aqs: list[float] = []

    def level_qty(self, side: int, price: float) -> float:
        px, qs = (self.bpx, self.bqs) if side == BID else (self.apx, self.aqs)
        if not px:
            return UNKNOWN_QUEUE
        for p, q in zip(px, qs):
            if p == price:
                return q
        better = price > px[0] if side == BID else price < px[0]
        within = price > px[-1] if side == BID else price < px[-1]
        return 0.0 if (better or within) else UNKNOWN_QUEUE

    def _lat(self, base: float) -> int:
        return int(round(self.stress * (base + self.jit * self.rng.uniform())))

    def place(self, now_ms: int, side: int, price: float, qty: float) -> int:
        oid = self.next_id
        self.next_id += 1
        self.orders[oid] = _Order(oid, side, price, qty, now_ms + self._lat(self.ack))
        self.n_msgs += 1
        return oid

    def cancel(self, now_ms: int, oid: int) -> None:
        o = self.orders.get(oid)
        if o is not None and o.state != DONE and o.cancel_at < 0:
            o.cancel_at = now_ms + self._lat(self.cxl)
            self.n_msgs += 1

    def queue_ahead(self, oid: int) -> float:
        o = self.orders.get(oid)
        return o.queue_ahead if o is not None and o.state == ACTIVE else float("nan")

    def take(self, now_ms: int, side: int, qty: float, limit_px: float) -> list:
        """Immediate taker execution against the LAST visible L2 snapshot (sim only, used by
        the deterministic unwinder). Walks levels up to `limit_px`; consumes visible size so a
        repeated take in the same snapshot cannot reuse liquidity. Fill id = 0."""
        px, qs = (self.apx, self.aqs) if side == BID else (self.bpx, self.bqs)
        out: list = []
        rem = qty
        self.n_msgs += 1
        for i in range(len(px)):
            if rem <= 1e-12:
                break
            if (side == BID and px[i] > limit_px) or (side == ASK and px[i] < limit_px):
                break
            q = min(rem, qs[i])
            if q <= 0.0:
                continue
            qs[i] -= q
            rem -= q
            out.append((0, side, px[i], q, px[i] * q * self.taker_fee, now_ms))
        return out

    def open_orders(self) -> list[tuple[int, int, float, float, int]]:
        return [(o.oid, o.side, o.price, o.qty, o.state) for o in self.orders.values()
                if o.state != DONE]

    def _housekeep(self, now_ms: int) -> None:
        for o in list(self.orders.values()):
            if o.state == DONE:
                del self.orders[o.oid]
                continue
            if 0 <= o.cancel_at <= now_ms:
                o.state = DONE
                del self.orders[o.oid]
                continue
            if o.state == PENDING and o.active_at <= now_ms:
                crosses = (o.side == BID and self.ask > 0 and o.price >= self.ask) or \
                          (o.side == ASK and self.bid > 0 and o.price <= self.bid)
                if crosses:
                    o.state = DONE  # post-only reject
                    self.n_rejects += 1
                    del self.orders[o.oid]
                    continue
                o.state = ACTIVE
                o.queue_ahead = self.level_qty(o.side, o.price)

    def _fill(self, o: _Order, qty: float, ts: int, out: list) -> float:
        q = qty if qty < o.qty else o.qty
        if q <= 0.0:
            return 0.0
        o.qty -= q
        out.append((o.oid, o.side, o.price, q, o.price * q * self.fee, ts))
        if o.qty <= 1e-12:
            o.state = DONE
        return q

    def on_book(self, now_ms: int, bid: float, bq: float, ask: float, aq: float) -> list:
        """Top-of-book only (levels behind the touch are unknown -> conservative)."""
        return self.on_book_l2(now_ms, [bid], [bq], [ask], [aq])

    def on_book_l2(self, now_ms: int, bid_px: list[float], bid_qty: list[float],
                   ask_px: list[float], ask_qty: list[float]) -> list:
        self.bpx, self.bqs, self.apx, self.aqs = list(bid_px), list(bid_qty), list(ask_px), list(ask_qty)
        bid, bq, ask, aq = bid_px[0], bid_qty[0], ask_px[0], ask_qty[0]
        self.bid, self.bq, self.ask, self.aq = bid, bq, ask, aq
        self._housekeep(now_ms)
        out: list = []
        for o in self.orders.values():
            if o.state != ACTIVE:
                continue
            through = (o.side == BID and ask <= o.price) or (o.side == ASK and bid >= o.price)
            if through:                       # opposite side moved through us
                self._fill(o, o.qty, now_ms, out)
            else:
                o.queue_ahead = min(o.queue_ahead, self.level_qty(o.side, o.price))
        return out

    def on_trade(self, now_ms: int, price: float, qty: float, sign: int) -> list:
        self._housekeep(now_ms)
        out: list = []
        remaining = qty
        for o in sorted(self.orders.values(), key=lambda x: (-x.side * x.price, x.oid)):
            if o.state != ACTIVE or remaining <= 0.0:
                continue
            hit = (sign < 0 and o.side == BID and price <= o.price) or \
                  (sign > 0 and o.side == ASK and price >= o.price)
            if not hit:
                continue
            if price != o.price:
                remaining -= self._fill(o, remaining, now_ms, out)
                continue
            eat = min(o.queue_ahead, remaining)
            o.queue_ahead -= eat
            remaining -= eat
            if remaining > 0.0:
                remaining -= self._fill(o, remaining, now_ms, out)
        return out
