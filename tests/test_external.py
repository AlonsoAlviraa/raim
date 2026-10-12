"""Slow brain / Jev isolation, outage handling, schema validation."""
import asyncio
import json
import math

import httpx
import pydantic
import pytest

from raim.config import load_config
from raim.data.synthetic import SyntheticCfg, generate
from raim.jev import adapter as jev
from raim.research.gate import first_gate
from raim.research.pipeline import train_test_run
from raim.slowbrain.brain import SlowBrain
from raim.slowbrain.mcp_client import MCPError, check_envelope
from raim.slowbrain.providers import MockSlowProvider
from raim.slowbrain.schema import SlowState, effective_multiplier
from raim.types import BookSnapshot

GOOD = SlowState(vix_z=0, trend_align=0, vol_expansion=0.1, event_risk=0.1, universe_score=0.5,
                 risk_multiplier=0.9, confidence=1.0)


def test_schema_rejects_out_of_range():
    with pytest.raises(pydantic.ValidationError):
        SlowState(vix_z=0, trend_align=2, vol_expansion=0, event_risk=0, universe_score=0,
                  risk_multiplier=1.5, confidence=1)
    with pytest.raises(pydantic.ValidationError):
        SlowState.model_validate({**GOOD.model_dump(), "place_order": True})


def test_confidence_decay_is_conservative():
    assert effective_multiplier(None, 0, 600) == 0.5
    assert effective_multiplier(GOOD, 0, 600) == pytest.approx(0.9)
    old = effective_multiplier(GOOD, 6000, 600)
    assert old <= 0.9 and old == pytest.approx(0.5, abs=1e-3)
    low = GOOD.model_copy(update={"risk_multiplier": 0.2})
    assert effective_multiplier(low, 1e9, 600) <= 0.2      # decay never loosens


def test_mcp_error_envelope():
    with pytest.raises(MCPError) as e:
        check_envelope({"error": {"code": "ALL_BATCHES_FAILED", "retryable": True, "message": "x"}})
    assert e.value.retryable and e.value.code == "ALL_BATCHES_FAILED"
    assert check_envelope({"ok": 1}) == {"ok": 1}


class _Flaky(MockSlowProvider):
    def __init__(self):
        super().__init__(GOOD)
        self.n = 0

    async def infer(self, context):
        self.n += 1
        if self.n > 1:
            raise MCPError("RATE_LIMIT", True)
        return await super().infer(context)


def test_brain_keeps_last_good_on_error():
    t = [0.0]
    b = SlowBrain(_Flaky(), 60, 600, clock=lambda: t[0])
    assert asyncio.run(b.poll_once())
    t[0] = 10
    assert not asyncio.run(b.poll_once())
    assert b.state == GOOD and b.errors == 1 and b._backoff == 2


def _events(n=6000):
    b, tr, _ = generate(SyntheticCfg(n_steps=n, seed=9))
    return sorted(b + tr, key=lambda e: (e.ts_ms, isinstance(e, BookSnapshot)))


def test_fast_body_survives_slow_brain_outage():
    cfg = load_config(None)
    brain = SlowBrain(MockSlowProvider(fail=True), 60, 600)
    asyncio.run(brain.poll_once())
    calls = []

    def ext(now):
        calls.append(now)
        return brain.multiplier(), jev.jev_multiplier(jev.CONSERVATIVE, 0.6)
    rep = train_test_run(_events(), cfg, ext_multipliers=ext)
    assert rep.result.n_books > 1000 and len(calls) == rep.result.n_books
    assert rep.result.reasons.get("external_tighten", 0) > 0     # outage => tighter, still running


def test_stale_feed_injection_cancels():
    cfg = load_config(None)
    rep = train_test_run(_events(), cfg, stale_inject=lambda now: 300_000 <= now <= 320_000)
    assert rep.result.reasons.get("stale_feed", 0) > 0


def test_jev_http_matches_documented_schema(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-not-a-real-key")
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
            "toxic_flow": {"type": "noul", "noul": 0.8},
            "regime": {"type": "choice", "choice": "R4", "confidence": 0.9,
                       "probabilities": {"R1": 0.05, "R2": 0.05, "R3": 0.1, "R4": 0.8}}},
            "usage": {"input_tokens": 10, "output_tokens": 5}})
    c = jev.HttpJev()
    c._cli = httpx.Client(transport=httpx.MockTransport(handler), headers=c._cli.headers)
    d = c.ask({"vpin": 0.7})
    assert seen["body"]["model"] == "jev-latest" and seen["auth"].startswith("Bearer ")
    assert seen["body"]["questions"]["toxic_flow"]["type"] == "noul"
    assert d.source == "jev" and d.p_toxic == 0.8
    assert jev.jev_multiplier(d, 0.6) == 0.0


def test_jev_timeout_falls_back(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-not-a-real-key")
    c = jev.HttpJev(timeout_ms=10_000)
    assert c.timeout == 0.5                                    # clamped to 70-500ms
    def boom(req):
        raise httpx.ReadTimeout("t")
    c._cli = httpx.Client(transport=httpx.MockTransport(boom))
    d = c.ask({})
    assert d.source == "fallback" and jev.jev_multiplier(d, 0.6) == 0.5


def test_jev_requires_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        jev.HttpJev()


def test_gate_pass_fail_and_min_samples():
    import numpy as np
    assert first_gate(np.full(100, 4.0), np.full(100, 0.5), 1.0).passed
    assert not first_gate(np.full(100, 2.0), np.full(100, 0.5), 1.0).passed
    assert not first_gate(np.full(10, 40.0), np.full(10, 0.0), 0.0).passed
    assert not math.isnan(first_gate(np.full(100, 4.0), np.full(100, 0.5), 1.0).threshold_bps)
