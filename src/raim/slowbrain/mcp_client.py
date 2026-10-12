"""Client for the OFFICIAL tradingview-mcp server (PyPI: tradingview-mcp-server,
repo github.com/atilaahmettaner/tradingview-mcp). Launched via
`uvx --from tradingview-mcp-server tradingview-mcp` over stdio. Never install npm clones.

Respects: >=0.8s spacing between calls (configurable >=1s), max 4 concurrent, structured
error envelope {"error": {"code", "message", "retryable"}}.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

log = logging.getLogger(__name__)


class MCPError(Exception):
    def __init__(self, code: str, retryable: bool, message: str = "") -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.retryable = retryable


def check_envelope(result: Any) -> Any:
    if isinstance(result, dict) and isinstance(result.get("error"), dict):
        e = result["error"]
        raise MCPError(str(e.get("code", "UNKNOWN")), bool(e.get("retryable", False)),
                       str(e.get("message", "")))
    if isinstance(result, dict) and isinstance(result.get("error"), str):
        raise MCPError("LEGACY_ERROR", False, result["error"])
    return result


class TradingViewMCP:
    def __init__(self, command: list[str], min_spacing_s: float = 1.0) -> None:
        self.command = command
        self.min_spacing = max(min_spacing_s, 0.8)
        self._sem = asyncio.Semaphore(4)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def call(self, tool: str, args: dict[str, Any]) -> Any:
        """Requires the optional `mcp` extra (uv sync --extra mcp)."""
        from mcp import ClientSession, StdioServerParameters  # type: ignore
        from mcp.client.stdio import stdio_client  # type: ignore

        async with self._sem:
            async with self._lock:
                wait = self._last + self.min_spacing - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last = time.monotonic()
            params = StdioServerParameters(command=self.command[0], args=self.command[1:])
            async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool(tool, args)
                text = "".join(getattr(c, "text", "") for c in res.content)
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {"text": text}
                return check_envelope(payload)
