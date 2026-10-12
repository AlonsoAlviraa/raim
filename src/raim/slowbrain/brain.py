"""Async slow-brain loop. Polled, rate-limited, never in the tick loop. On any error it
keeps the last good state; the fast body reads `multiplier()` which decays conservatively."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from typing import Any

from raim.slowbrain.mcp_client import MCPError
from raim.slowbrain.providers import SlowProvider
from raim.slowbrain.schema import SlowState, effective_multiplier

log = logging.getLogger(__name__)


class SlowBrain:
    def __init__(self, provider: SlowProvider, poll_s: float, tau_s: float,
                 context_fn: Callable[[], dict[str, Any]] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.provider = provider
        self.poll_s = max(poll_s, 1.0)  # hard cap <= 1 call/s
        self.tau = tau_s
        self.context_fn = context_fn or (lambda: {})
        self.clock = clock
        self.state: SlowState | None = None
        self.updated_at: float | None = None
        self.errors = 0
        self._backoff = 1.0

    def multiplier(self) -> float:
        age = 1e12 if self.updated_at is None else self.clock() - self.updated_at
        return effective_multiplier(self.state, age, self.tau)

    async def poll_once(self) -> bool:
        try:
            st = await self.provider.infer(self.context_fn())
        except MCPError as exc:
            self.errors += 1
            log.warning("slow brain MCP error %s (retryable=%s); keeping last good", exc.code, exc.retryable)
            if exc.retryable:
                self._backoff = min(self._backoff * 2, 300)
            return False
        except Exception as exc:  # noqa: BLE001 - any failure keeps last good state
            self.errors += 1
            self._backoff = min(self._backoff * 2, 300)
            log.warning("slow brain error %r; keeping last good", exc)
            return False
        self.state, self.updated_at, self._backoff = st, self.clock(), 1.0
        return True

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            ok = await self.poll_once()
            delay = self.poll_s if ok else self.poll_s * self._backoff * (0.5 + random.random())
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
