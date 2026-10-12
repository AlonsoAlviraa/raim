"""Live trading is OUT OF SCOPE. There is no live executor in this codebase."""
from __future__ import annotations

import os


class LiveTradingDisabled(RuntimeError):
    pass


def assert_not_live(mode: str) -> None:
    """Hard-fail on live mode. Even with mode=live AND RAIM_LIVE_CONFIRM set, we still refuse,
    because no live order path is implemented until OOS + stress gates pass and the user
    explicitly approves a separately reviewed change."""
    if mode == "live":
        confirm = os.environ.get("RAIM_LIVE_CONFIRM", "")
        raise LiveTradingDisabled(
            "Live mode requested (RAIM_LIVE_CONFIRM=%s) but no live executor exists. "
            "RAIM is SIMULATION/PAPER only." % ("set" if confirm else "unset"))
