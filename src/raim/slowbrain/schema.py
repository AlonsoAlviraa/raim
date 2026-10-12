"""Schema-validated slow state produced by the slow brain (LLM)."""
from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field


class SlowState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vix_z: float = Field(ge=-10, le=10)
    trend_align: float = Field(ge=-1, le=1)
    vol_expansion: float = Field(ge=0, le=1)
    event_risk: float = Field(ge=0, le=1)
    universe_score: float = Field(ge=0, le=1)
    risk_multiplier: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(default="", max_length=2000)


NEUTRAL_CONSERVATIVE = SlowState(vix_z=0, trend_align=0, vol_expansion=0.5, event_risk=0.5,
                                 universe_score=0.5, risk_multiplier=0.5, confidence=0.0)


def effective_multiplier(state: SlowState | None, age_s: float, tau_s: float) -> float:
    """c_eff = c*exp(-age/tau). As confidence decays, m moves toward the conservative 0.5
    cap — NEVER toward 1. No state -> 0.5. Result is only ever used through min()."""
    if state is None:
        return 0.5
    c = state.confidence * math.exp(-max(age_s, 0.0) / tau_s)
    return min(state.risk_multiplier, c * state.risk_multiplier + (1 - c) * 0.5)
