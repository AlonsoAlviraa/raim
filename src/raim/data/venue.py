"""Pluggable venue interface + Binance public market-data adapter (no keys, no orders)."""
from __future__ import annotations

import abc
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from raim.types import BookSnapshot, Trade

log = logging.getLogger(__name__)


class VenueAdapter(abc.ABC):
    name: str

    @abc.abstractmethod
    def stream(self) -> AsyncIterator[BookSnapshot | Trade]:
        """Yield normalized market events. Must never place orders."""


class BinancePublicAdapter(VenueAdapter):
    """Binance spot partial-depth (@depthN@100ms) + @trade streams.
    Default URL is data-stream.binance.vision (market-data only). Check the venue's terms
    for your jurisdiction (binance.com is geo-restricted in some regions)."""

    name = "binance"

    def __init__(self, symbol: str, ws_url: str, levels: int = 20, speed_ms: int = 100,
                 reconnect_s: float = 2.0) -> None:
        self.symbol = symbol.lower()
        self.base = ws_url.rstrip("/").removesuffix("/ws")
        self.levels = levels
        self.speed = speed_ms
        self.reconnect_s = reconnect_s

    def url(self) -> str:
        s = self.symbol
        return f"{self.base}/stream?streams={s}@depth{self.levels}@{self.speed}ms/{s}@trade"

    @staticmethod
    def parse(msg: dict, recv_ms: int) -> BookSnapshot | Trade | None:
        data = msg.get("data", msg)
        stream = msg.get("stream", "")
        if "bids" in data and "asks" in data:
            bids = tuple((float(p), float(q)) for p, q in data["bids"] if float(q) > 0)
            asks = tuple((float(p), float(q)) for p, q in data["asks"] if float(q) > 0)
            if not bids or not asks:
                return None
            # partial depth payload has no event time; use receive time (documented)
            ts = int(data.get("E", recv_ms))
            return BookSnapshot(ts, bids, asks, recv_ms)
        if data.get("e") == "trade" or stream.endswith("@trade"):
            return Trade(int(data["T"]), float(data["p"]), float(data["q"]), bool(data["m"]), recv_ms)
        return None

    async def stream(self) -> AsyncIterator[BookSnapshot | Trade]:  # type: ignore[override]
        import websockets

        while True:
            try:
                async with websockets.connect(self.url(), ping_interval=20, max_size=2**22) as ws:
                    log.info("connected %s", self.url())
                    async for raw in ws:
                        ev = self.parse(json.loads(raw), int(time.time() * 1000))
                        if ev is not None:
                            yield ev
            except (OSError, asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                if isinstance(exc, asyncio.CancelledError):
                    raise
                log.warning("ws error %r; reconnecting in %.1fs", exc, self.reconnect_s)
                await asyncio.sleep(self.reconnect_s)
