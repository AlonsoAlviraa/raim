"""Event-driven paper/sim runner: features -> regime -> quotes -> paper orders -> risk.
Deterministic given (events, config, seed). The slow brain and Jev only feed multipliers
in via `ext_multipliers` (read-only, polled), never inside pricing."""
from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from raim import core
from raim.config import RaimConfig
from raim.execution.guard import assert_not_live
from raim.regime.blend import Hysteresis, blend_params, conservative_posterior
from raim.regime.bocpd import BOCPD
from raim.regime.hmm_fit import HMMParams, tilt_transition
from raim.core import FEATURE_NAMES
from raim.research.obs import Standardizer, obs_from_features
from raim.types import BookSnapshot, Trade

log = logging.getLogger(__name__)


@dataclass
class SimResult:
    fills: list[tuple] = field(default_factory=list)          # (ts, side, px, qty, fee, regime)
    mids: list[tuple[int, float]] = field(default_factory=list)
    regimes: list[int] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)
    final_position: float = 0.0
    final_equity: float = 0.0
    fees_paid: float = 0.0
    max_drawdown: float = 0.0
    halted: bool = False
    n_msgs: int = 0
    n_rejects: int = 0
    n_books: int = 0


def make_governor(cfg: RaimConfig):
    r = cfg.risk
    return core.RiskGovernor(r.budget_usd, r.z, r.h_unwind_s, r.depth_frac, r.max_gross_usd,
                             r.max_drawdown_usd, r.stale_feed_ms, r.max_reject_rate,
                             r.max_msgs_per_s, r.loss_streak_widen, r.loss_streak_pause, r.pause_s)


def make_features(cfg: RaimConfig):
    f = cfg.features
    return core.FeatureEngine(f.vol_window, f.ofi_window, f.vpin_bucket_volume, f.vpin_buckets,
                              f.vr_lag, f.hawkes_window)


def feature_rows(events: Iterable[BookSnapshot | Trade], cfg: RaimConfig) -> tuple[np.ndarray, np.ndarray]:
    """Causal feature row per book event. Returns (ts array, raw feature matrix)."""
    fe = make_features(cfg)
    ts, rows = [], []
    for ev in events:
        if isinstance(ev, Trade):
            fe.on_trade(ev.ts_ms, ev.price, ev.qty, ev.sign)
        else:
            depth = sum(q for _, q in ev.bids[:5]) + sum(q for _, q in ev.asks[:5])
            fe.on_book(ev.ts_ms, ev.best_bid, ev.bids[0][1], ev.best_ask, ev.asks[0][1], depth)
            ts.append(ev.ts_ms)
            rows.append(list(fe.values()))
    return np.asarray(ts), np.asarray(rows, dtype=float)


def run_sim(events: Iterable[BookSnapshot | Trade], cfg: RaimConfig, params: HMMParams,
            scaler: Standardizer,
            ext_multipliers: Callable[[int], tuple[float, float]] | None = None,
            stale_inject: Callable[[int], bool] | None = None,
            slow_context: Callable[[int], tuple[float, float]] | None = None,
            on_features: Callable[[dict], None] | None = None,
            features_every: int = 10) -> SimResult:
    """All timing uses the RECEIVE clock (recv_ms, falling back to ts_ms): that is what the
    strategy can know. slow_context -> (vix_z, event_risk) tilts the HMM toward stress only.
    on_features receives a named feature dict every `features_every` books (for Jev/brain)."""
    assert_not_live(cfg.mode)
    q, s = cfg.quoting, cfg.sim
    fe = make_features(cfg)
    hmm = core.HMMFilter(*params.as_lists())
    gov = make_governor(cfg)
    ex = core.SimExchange(cfg.fees.maker_bps, cfg.fees.rebate_bps, s.ack_latency_ms,
                          s.ack_jitter_ms, s.cancel_latency_ms, s.latency_stress, cfg.seed)
    cpd = BOCPD(hazard=cfg.regime.cpd_hazard, max_run=200)
    hyst = Hysteresis(cfg.regime.pull_on, cfg.regime.pull_off)
    res = SimResult()
    pos = cash = fees = 0.0
    peak = 0.0
    live: dict[int, int] = {}            # side -> order id
    live_px: dict[int, float] = {}
    last_feed = 0
    dt_ewma = 100.0
    last_ts = None
    msg_times: deque[int] = deque()
    pending_marks: deque[tuple[int, int, float]] = deque()  # (mature_ts, side, px)
    slip = s.slippage_bps * 1e-4
    regime = 0
    rej_mark = [0, 0, 0]
    requote_tol = 0.5e-4  # do not chase the book for < 0.5 bps changes (message discipline)

    def apply_fills(fills: list, mid: float) -> None:
        nonlocal pos, cash, fees
        for (_oid, side, px, qty, fee, ts) in fills:
            eff = px * (1 + slip) if side == 1 else px * (1 - slip)  # conservative haircut
            pos += side * qty
            cash -= side * qty * eff
            fees += fee
            res.fills.append((ts, side, px, qty, fee, regime))
            pending_marks.append((ts + 5000, side, px))
            if live.get(side) == _oid and not any(o[0] == _oid for o in ex.open_orders()):
                live.pop(side, None)

    last_ctx: tuple[float, float] | None = None
    for ev in events:
        now = ev.recv_ms or ev.ts_ms
        if isinstance(ev, Trade):
            fe.on_trade(now, ev.price, ev.qty, ev.sign)
            apply_fills(ex.on_trade(now, ev.price, ev.qty, ev.sign), 0.0)
            continue
        bid, bq, ask, aq = ev.best_bid, ev.bids[0][1], ev.best_ask, ev.asks[0][1]
        depth = sum(x for _, x in ev.bids[:5]) + sum(x for _, x in ev.asks[:5])
        if stale_inject is None or not stale_inject(now):
            last_feed = now
            apply_fills(ex.on_book_l2(now, [p_ for p_, _ in ev.bids], [x for _, x in ev.bids],
                                      [p_ for p_, _ in ev.asks], [x for _, x in ev.asks]), ev.mid)
            fe.on_book(now, bid, bq, ask, aq, depth)
        if last_ts is not None and now > last_ts:
            dt_ewma = 0.98 * dt_ewma + 0.02 * (now - last_ts)
        last_ts = now
        v = fe.values()
        mid = v[0]
        if mid != mid:
            continue
        res.n_books += 1
        res.mids.append((now, mid))
        if slow_context is not None:
            ctx = slow_context(now)
            if ctx != last_ctx:
                hmm.set_transition(tilt_transition(params.trans, ctx[0], ctx[1]).tolist())
                last_ctx = ctx
        post = np.asarray(hmm.step(scaler.apply(obs_from_features(v))))
        r_alarm = cpd.update(scaler.apply(obs_from_features(v))[1]) > cfg.regime.cpd_alarm
        p = conservative_posterior(post, cfg.regime.conservative_bias, r_alarm)
        regime = int(np.argmax(post))
        res.regimes.append(regime)
        pulled = hyst.update(float(p[3]))
        if on_features is not None and res.n_books % features_every == 0:
            fd = dict(zip(FEATURE_NAMES, v))
            fd["p_regime"] = [float(x) for x in post]
            fd["position"] = pos
            on_features(fd)
        g_mult, s_mult, z_mult = blend_params(p, q.regime_params)
        # matured markouts -> loss-streak logic (causal: only past fills, current mid)
        while pending_marks and pending_marks[0][0] <= now:
            _, side, px = pending_marks.popleft()
            gov.on_trade_result(side * (mid - px), now)
        m_llm, m_jev = ext_multipliers(now) if ext_multipliers else (1.0, 1.0)
        gov.set_external(m_llm, m_jev)
        equity = cash + pos * mid - fees
        peak = max(peak, equity)
        res.max_drawdown = max(res.max_drawdown, peak - equity)
        while msg_times and msg_times[0] < now - 1000:
            msg_times.popleft()
        if now - rej_mark[0] >= 10_000:  # rolling 10s reject-rate window
            rej_mark[:] = [now, ex.n_rejects, ex.n_msgs]
        d_msgs = ex.n_msgs - rej_mark[2]
        rej_rate = (ex.n_rejects - rej_mark[1]) / d_msgs if d_msgs >= 10 else 0.0
        sigma_frac_s = max(v[5], 1e-7) * math.sqrt(1000.0 / max(dt_ewma, 1.0))
        d = gov.evaluate(now, last_feed, equity, pos, mid, sigma_frac_s, depth, rej_rate,
                         float(len(msg_times)))
        res.reasons[d.reason] = res.reasons.get(d.reason, 0) + 1
        want: dict[int, float] = {}
        size = 0.0
        if not d.cancel_all:
            size = q.order_size * d.size_mult * z_mult
            qnorm = pos / q.order_size
            b, a, _, _ = core.quote(v[1], sigma_frac_s * 1e4, qnorm, q.gamma * g_mult, q.kappa,
                                    q.horizon_s, q.min_half_spread_bps, s_mult * d.spread_mult,
                                    q.tick_size)
            if not (math.isfinite(b) and math.isfinite(a)):
                b = a = math.nan
                want_ok = False
            else:
                want_ok = True
            # post-only discipline: never cross the current touch
            b = min(b, ask - q.tick_size)
            a = max(a, bid + q.tick_size)
            reduce_size = min(q.order_size, abs(pos))
            if not want_ok:
                pass
            elif pulled:  # R4: pull quotes, allow only inventory-reducing passive order
                if pos > 0 and d.allow_ask:
                    want[-1] = a
                elif pos < 0 and d.allow_bid:
                    want[1] = b
                size = reduce_size
            else:
                if d.allow_bid and (size > 0 or pos < 0):
                    want[1] = b
                if d.allow_ask and (size > 0 or pos > 0):
                    want[-1] = a
            if size <= 0:
                size = reduce_size
        for side in (1, -1):
            oid = live.get(side)
            target = want.get(side)
            if oid is not None and (target is None or abs(live_px[side] - target) >= max(q.tick_size / 2, mid * requote_tol)):
                ex.cancel(now, oid)
                msg_times.append(now)
                live.pop(side)
            if target is not None and side not in live and size > 0:
                if len(msg_times) < cfg.risk.max_msgs_per_s:
                    live[side] = ex.place(now, side, target, size)
                    live_px[side] = target
                    msg_times.append(now)
    mid_last = res.mids[-1][1] if res.mids else 0.0
    res.final_position = pos
    res.final_equity = cash + pos * mid_last - fees
    res.fees_paid = fees
    res.halted = bool(gov.halted)
    res.n_msgs = int(ex.n_msgs)
    res.n_rejects = int(ex.n_rejects)
    return res
