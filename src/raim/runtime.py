"""Background (non-tick) context: slow brain + Jev polled in their own thread; the tick loop
only reads cached multipliers (lock-free floats). Outage => cached values decay conservatively."""
from __future__ import annotations

import asyncio
import logging
import threading
import time

from raim.config import RaimConfig
from raim.jev.adapter import CONSERVATIVE, JevClient, jev_multiplier, make_jev
from raim.slowbrain.brain import SlowBrain
from raim.slowbrain.providers import AnthropicSlowProvider, MockSlowProvider, SlowProvider

log = logging.getLogger(__name__)


def make_slow_provider(cfg: RaimConfig) -> SlowProvider:
    if cfg.slow_brain.provider == "anthropic":
        return AnthropicSlowProvider(cfg.slow_brain.model)
    return MockSlowProvider()


class ExternalContext:
    def __init__(self, cfg: RaimConfig, brain: SlowBrain | None = None, jev: JevClient | None = None) -> None:
        self.cfg = cfg
        self.brain = brain or SlowBrain(make_slow_provider(cfg), cfg.slow_brain.poll_interval_s,
                                        cfg.slow_brain.tau_s)
        self.jev = jev or make_jev(cfg.jev.provider, cfg.jev.model, cfg.jev.timeout_ms)
        self.features: dict = {}
        self.brain.context_fn = lambda: {"fast_features": dict(self.features),
                                         "symbol": cfg.venue.symbol}
        self._m_jev = 0.5
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update_features(self, features: dict) -> None:
        """Called from the tick loop (cheap: dict swap). Consumers read a snapshot."""
        self.features = dict(features)

    def slow_context(self, _now_ms: int = 0) -> tuple[float, float]:
        st = self.brain.state
        return (0.0, 0.0) if st is None else (st.vix_z, st.event_risk)

    def multipliers(self, _now_ms: int = 0) -> tuple[float, float]:
        return self.brain.multiplier(), self._m_jev

    def _jev_loop(self) -> None:
        while not self._stop.is_set():
            try:
                d = self.jev.ask(dict(self.features))
            except Exception:  # noqa: BLE001
                d = CONSERVATIVE
            self._m_jev = jev_multiplier(d, self.cfg.jev.min_confidence)
            self._stop.wait(1.0)

    def start(self) -> None:
        def _run() -> None:
            stop = asyncio.Event()

            async def main() -> None:
                task = asyncio.create_task(self.brain.run(stop))
                while not self._stop.is_set():
                    await asyncio.sleep(0.2)
                stop.set()
                await task
            asyncio.run(main())
        threading.Thread(target=self._jev_loop, daemon=True).start()
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        time.sleep(0)
