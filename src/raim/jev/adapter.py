"""Jev (TypeSafe AI, System One) adapter: typed questions -> calibrated probabilities.

Real API (docs.typesafe.ai, verified 2026-10-12): POST https://api.typesafe.ai/v1/systemone,
Bearer TYPESAFE_API_KEY, body {state, model: "jev-latest", questions: {id: {type: noul|choice|score,
instructions, criteria}}}. Answers: noul -> {"noul": p}; choice -> {"choice", "probabilities",
"confidence"}. Jev only TIGHTENS: its output becomes m_jev in min(m_llm, m_rules, m_jev).
Runs outside the tick loop with a hard timeout; on timeout/error -> conservative fallback.
"""
from __future__ import annotations

import abc
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

API_URL = "https://api.typesafe.ai/v1/systemone"

QUESTIONS: dict[str, dict[str, Any]] = {
    "toxic_flow": {"type": "noul",
                   "instructions": "Given `features`, is recent order flow likely toxic/informed "
                                   "(adverse selection risk for a passive market maker)?"},
    "regime": {"type": "choice", "instructions": "Which microstructure regime best matches `features`?",
               "criteria": {"R1": "calm, mean-reverting", "R2": "trending / informed flow",
                            "R3": "high volatility / stress", "R4": "toxic flow or structural break"}},
}


@dataclass(frozen=True)
class JevDecision:
    p_toxic: float
    regime_probs: dict[str, float]
    confidence: float
    latency_ms: float
    source: str  # "jev" | "mock" | "fallback"


CONSERVATIVE = JevDecision(0.5, {"R1": 0.0, "R2": 0.0, "R3": 0.5, "R4": 0.5}, 0.0, 0.0, "fallback")


def jev_multiplier(d: JevDecision, min_confidence: float) -> float:
    """Map a Jev decision to m_jev in [0,1]. Low confidence -> conservative 0.5 cap."""
    if d.confidence < min_confidence:
        return 0.5
    m = 1.0 - d.p_toxic - 0.5 * d.regime_probs.get("R4", 0.0)
    return max(0.0, min(1.0, m))


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        raise ValueError(f"non-finite probability {x!r}")
    return min(max(x, 0.0), 1.0)


def parse_answers(ans: dict[str, Any], latency_ms: float) -> JevDecision:
    """Validate a /v1/systemone answers map. Missing questions or non-finite values raise
    (caller falls back conservatively); out-of-range values are clamped; regime probs are
    completed to R1..R4 and renormalized."""
    p = _clamp01(float(ans["toxic_flow"]["noul"]))
    raw = ans["regime"]["probabilities"]
    probs = {k: _clamp01(float(raw.get(k, 0.0))) for k in ("R1", "R2", "R3", "R4")}
    s = sum(probs.values())
    if s <= 0.0:
        raise ValueError("empty regime distribution")
    probs = {k: v / s for k, v in probs.items()}
    conf = _clamp01(float(ans["regime"].get("confidence", 0.0)))
    return JevDecision(p, probs, conf, latency_ms, "jev")


class JevClient(abc.ABC):
    @abc.abstractmethod
    def ask(self, state: dict[str, Any]) -> JevDecision: ...


class MockJev(JevClient):
    """Deterministic heuristic stand-in (NOT Jev): VPIN/spread-based."""

    def ask(self, state: dict[str, Any]) -> JevDecision:
        vpin = float(state.get("vpin", 0.0) or 0.0)
        p = max(0.0, min(1.0, (vpin - 0.3) / 0.5))
        return JevDecision(p, {"R1": 1 - p, "R2": 0.0, "R3": 0.0, "R4": p}, 0.8, 0.0, "mock")


class HttpJev(JevClient):
    def __init__(self, model: str = "jev-latest", timeout_ms: int = 500) -> None:
        key = os.environ.get("TYPESAFE_API_KEY", "")
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY not set: get one at https://console.typesafe.ai/keys")
        self.model = model
        self.timeout = max(min(timeout_ms, 500), 70) / 1000.0
        self._cli = httpx.Client(timeout=self.timeout,
                                 headers={"Authorization": f"Bearer {key}"})

    def ask(self, state: dict[str, Any]) -> JevDecision:
        t0 = time.perf_counter()
        try:
            r = self._cli.post(API_URL, json={"state": {"features": state}, "model": self.model,
                                              "questions": QUESTIONS})
            r.raise_for_status()
            ans = r.json()["answers"]
            return parse_answers(ans, (time.perf_counter() - t0) * 1e3)
        except Exception as exc:  # noqa: BLE001
            log.warning("Jev call failed (%r); conservative fallback", exc)
            return CONSERVATIVE


def make_jev(provider: str, model: str, timeout_ms: int) -> JevClient:
    return HttpJev(model, timeout_ms) if provider == "http" else MockJev()
