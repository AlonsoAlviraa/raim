"""Slow-state providers. Mock is default. Anthropic uses the Messages API over httpx;
model string is configurable (default claude-opus-5-5, verified on Anthropic docs 2026-10-12)."""
from __future__ import annotations

import abc
import json
import logging
import os
from typing import Any

import httpx

from raim.slowbrain.schema import SlowState

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the SLOW BRAIN of a market-making research system. You never price, place or "
    "cancel orders. Given market context JSON, return ONLY a JSON object with keys: vix_z, "
    "trend_align[-1,1], vol_expansion[0,1], event_risk[0,1], universe_score[0,1], "
    "risk_multiplier[0,1], confidence[0,1], rationale. When unsure, lower risk_multiplier "
    "and confidence. Never exceed 1.0.")


class SlowProvider(abc.ABC):
    @abc.abstractmethod
    async def infer(self, context: dict[str, Any]) -> SlowState: ...


class MockSlowProvider(SlowProvider):
    def __init__(self, fixed: SlowState | None = None, fail: bool = False) -> None:
        self.fixed = fixed
        self.fail = fail

    async def infer(self, context: dict[str, Any]) -> SlowState:
        if self.fail:
            raise ConnectionError("mock outage")
        if self.fixed is not None:
            return self.fixed
        vol = float(context.get("vol_expansion", 0.3))
        return SlowState(vix_z=0.0, trend_align=0.0, vol_expansion=min(max(vol, 0), 1),
                         event_risk=0.1, universe_score=0.5,
                         risk_multiplier=round(1.0 - 0.5 * min(max(vol, 0), 1), 3),
                         confidence=0.6, rationale="mock")


class AnthropicSlowProvider(SlowProvider):
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "claude-opus-5-5", timeout_s: float = 30.0) -> None:
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (see .env.example / README)")
        self.model = model
        self.timeout = timeout_s

    async def infer(self, context: dict[str, Any]) -> SlowState:
        body = {"model": self.model, "max_tokens": 600, "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": json.dumps(context, default=str)}]}
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout) as cli:
            r = await cli.post(self.URL, json=body, headers=headers)
            r.raise_for_status()
            text = "".join(b.get("text", "") for b in r.json().get("content", []))
        start, end = text.find("{"), text.rfind("}")
        return SlowState.model_validate_json(text[start:end + 1])  # raises on schema violation
