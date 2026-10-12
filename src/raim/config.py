"""Dataclass configs loaded from YAML. Pure, typed, hashable for audit."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VenueCfg:
    name: str = "binance"
    symbol: str = "BTCUSDT"
    ws_url: str = "wss://data-stream.binance.vision/ws"
    depth_levels: int = 20
    depth_speed_ms: int = 100


@dataclass(frozen=True)
class FeesCfg:
    maker_bps: float = 1.0
    taker_bps: float = 1.0
    rebate_bps: float = 0.0


@dataclass(frozen=True)
class FeaturesCfg:
    vol_window: int = 200
    ofi_window: int = 50
    vpin_bucket_volume: float = 5.0
    vpin_buckets: int = 50
    vr_lag: int = 10
    hawkes_window: int = 500


@dataclass(frozen=True)
class RegimeCfg:
    n_states: int = 4
    pull_on: float = 0.7
    pull_off: float = 0.4
    conservative_bias: float = 0.15
    cpd_hazard: float = 0.004
    cpd_alarm: float = 0.5


@dataclass(frozen=True)
class QuotingCfg:
    gamma: float = 0.1
    kappa: float = 1.5
    horizon_s: float = 10.0
    min_half_spread_bps: float = 0.5
    order_size: float = 0.001
    tick_size: float = 0.01
    flow_hazard: float = 0.01        # FlowCPD (Moret & Lillo 2026)
    flow_short_run: int = 30
    k_flow_bps: float = 2.0          # exposure asymmetry: flow-bias widening
    k_queue_bps: float = 1.0         # exposure asymmetry: queue-imbalance widening
    max_extra_bps: float = 5.0
    regime_params: dict = field(default_factory=lambda: {
        "R1": [1.0, 1.0, 1.0], "R2": [2.0, 1.6, 0.6],
        "R3": [3.0, 2.5, 0.4], "R4": [6.0, 5.0, 0.0]})


@dataclass(frozen=True)
class RiskCfg:
    budget_usd: float = 50.0
    z: float = 2.33
    h_unwind_s: float = 30.0
    depth_frac: float = 0.1
    max_gross_usd: float = 500.0
    max_drawdown_usd: float = 25.0
    stale_feed_ms: int = 2000
    max_reject_rate: float = 0.2
    max_msgs_per_s: float = 10.0
    loss_streak_widen: int = 3
    loss_streak_pause: int = 6
    pause_s: float = 60.0
    unwind_passive_s: float = 5.0    # RISK-2 post-halt unwind
    unwind_clip: float = 0.01
    unwind_max_slip_bps: float = 10.0


@dataclass(frozen=True)
class SimCfg:
    ack_latency_ms: float = 30.0
    ack_jitter_ms: float = 10.0
    cancel_latency_ms: float = 30.0
    latency_stress: float = 1.0
    slippage_bps: float = 0.5


@dataclass(frozen=True)
class SlowBrainCfg:
    provider: str = "mock"
    model: str = "claude-opus-5-5"
    poll_interval_s: float = 60.0
    min_call_spacing_s: float = 1.0
    tau_s: float = 600.0
    mcp_command: list = field(default_factory=lambda: [
        "uvx", "--from", "tradingview-mcp-server==0.8.1", "tradingview-mcp"])


@dataclass(frozen=True)
class JevCfg:
    provider: str = "mock"
    model: str = "jev-latest"
    timeout_ms: int = 500
    min_confidence: float = 0.6
    max_ece: float = 0.05            # calibration gate before Jev is used
    min_calibration_n: int = 500


@dataclass(frozen=True)
class AuditCfg:
    dir: str = "logs"


@dataclass(frozen=True)
class RaimConfig:
    mode: str = "simulation"
    seed: int = 42
    venue: VenueCfg = field(default_factory=VenueCfg)
    fees: FeesCfg = field(default_factory=FeesCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    regime: RegimeCfg = field(default_factory=RegimeCfg)
    quoting: QuotingCfg = field(default_factory=QuotingCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    slow_brain: SlowBrainCfg = field(default_factory=SlowBrainCfg)
    jev: JevCfg = field(default_factory=JevCfg)
    audit: AuditCfg = field(default_factory=AuditCfg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    data = data or {}
    kwargs: dict[str, Any] = {}
    names = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for name, f in names.items():
        if name not in data:
            continue
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        kwargs[name] = _build(type(default), data[name]) if is_dataclass(default) else data[name]
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> RaimConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for k, v in (overrides or {}).items():
        if isinstance(v, dict):
            raw.setdefault(k, {}).update(v)
        else:
            raw[k] = v
    cfg = _build(RaimConfig, raw)
    if cfg.mode not in {"simulation", "paper", "live"}:
        raise ValueError(f"invalid mode {cfg.mode}")
    return cfg
