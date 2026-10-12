"""RAIM command line. SIMULATION/PAPER only."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from raim import core
from raim.audit import AuditLog, new_run_id
from raim.config import RaimConfig, load_config
from raim.data.recorder import record, replay
from raim.data.synthetic import SyntheticCfg, generate
from raim.data.venue import BinancePublicAdapter
from raim.execution.guard import assert_not_live
from raim.research.gate import adverse_selection_from_trades, first_gate
from raim.research.pipeline import fit_regimes, train_test_run, walk_forward
from raim.sim.engine import run_sim
from raim.types import BookSnapshot, Trade

log = logging.getLogger("raim")


def _events(args: argparse.Namespace, cfg: RaimConfig) -> list:
    if args.data:
        return list(replay(Path(args.data)))
    books, trades, _ = generate(SyntheticCfg(n_steps=args.steps, seed=cfg.seed))
    log.warning("Using SYNTHETIC data: results say nothing about real-market edge")
    return sorted(books + trades, key=lambda e: (e.ts_ms, isinstance(e, BookSnapshot)))


def _adapter(cfg: RaimConfig) -> BinancePublicAdapter:
    v = cfg.venue
    return BinancePublicAdapter(v.symbol, v.ws_url, v.depth_levels, v.depth_speed_ms)


def cmd_record(args, cfg, audit) -> None:
    out = Path(args.out)
    n = asyncio.run(record(_adapter(cfg), out, args.seconds))
    audit.write("record", path=str(out), n_events=n, seconds=args.seconds)


def cmd_sim(args, cfg, audit) -> None:
    rep = train_test_run(_events(args, cfg), cfg, args.train_frac)
    r = rep.result
    summary = {"backend": core.BACKEND, "books": r.n_books, "fills": len(r.fills),
               "final_pos": r.final_position, "equity_usd": round(r.final_equity, 6),
               "fees_usd": round(r.fees_paid, 6), "max_dd_usd": round(r.max_drawdown, 6),
               "halted": r.halted, "msgs": r.n_msgs, "rejects": r.n_rejects,
               "risk_reasons": r.reasons}
    log.info("SIM summary: %s", json.dumps(summary))
    if not rep.attribution.empty:
        log.info("per-regime markouts (bps):\n%s", rep.attribution.round(3).to_string())
    audit.write("sim", **summary, attribution=rep.attribution.round(4).to_dict())


def cmd_walkforward(args, cfg, audit) -> None:
    df = walk_forward(_events(args, cfg), cfg, args.folds)
    log.info("walk-forward (purged+embargoed):\n%s", df.round(4).to_string())
    audit.write("walkforward", folds=df.to_dict("records"))


def cmd_gate(args, cfg, audit) -> None:
    evs = _events(args, cfg)
    books = [e for e in evs if isinstance(e, BookSnapshot)]
    trades = [e for e in evs if isinstance(e, Trade)]
    mts = np.array([b.recv_ms or b.ts_ms for b in books])
    mids = np.array([b.mid for b in books])
    spreads = np.array([(b.best_ask - b.best_bid) / b.mid * 1e4 for b in books])
    adv = adverse_selection_from_trades(mts, mids, np.array([t.recv_ms or t.ts_ms for t in trades]),
                                        np.array([t.price for t in trades]),
                                        np.array([t.sign for t in trades]), args.horizon_ms)
    g = first_gate(spreads, adv, cfg.fees.maker_bps)
    log.info(g.summary())
    audit.write("gate", **g.__dict__)


def cmd_paper(args, cfg, audit) -> None:
    """Live PUBLIC data -> paper executor. No orders ever leave the machine."""
    from raim.runtime import ExternalContext

    assert_not_live(cfg.mode)
    warm = Path(args.warmup_file)
    asyncio.run(record(_adapter(cfg), warm, args.warmup))
    params, scaler = fit_regimes(list(replay(warm)), cfg)
    q: queue.Queue = queue.Queue(maxsize=100_000)
    stop = threading.Event()

    def feed() -> None:
        async def main() -> None:
            async for ev in _adapter(cfg).stream():
                q.put(ev)
                if stop.is_set():
                    return
        asyncio.run(main())

    threading.Thread(target=feed, daemon=True).start()
    ctx = ExternalContext(cfg)
    ctx.start()
    end = time.monotonic() + args.seconds

    def gen():
        while time.monotonic() < end:
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                continue
    res = run_sim(gen(), cfg, params, scaler, ext_multipliers=ctx.multipliers)
    stop.set()
    ctx.stop()
    summary = {"mode": "paper", "books": res.n_books, "fills": len(res.fills),
               "equity_usd": res.final_equity, "max_dd_usd": res.max_drawdown,
               "halted": res.halted, "reasons": res.reasons}
    log.info("PAPER summary: %s", json.dumps(summary, default=str))
    audit.write("paper", **summary)


def cmd_bench(args, cfg, audit) -> None:
    from raim.bench import run_bench

    out = run_bench(args.n)
    for k, v in out.items():
        log.info("bench %-28s %10.1f ns/update", k, v)
    audit.write("bench", **out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="raim", description="RAIM research system (SIM/PAPER only)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("record"); s.add_argument("--seconds", type=float, default=60); s.add_argument("--out", default="data/btcusdt.jsonl")
    for name in ("sim", "walkforward", "gate"):
        s = sub.add_parser(name)
        s.add_argument("--data", default=None, help="recorded JSONL; omit for synthetic")
        s.add_argument("--steps", type=int, default=20_000)
        s.add_argument("--train-frac", type=float, default=0.3)
        s.add_argument("--folds", type=int, default=4)
        s.add_argument("--horizon-ms", type=int, default=5000)
        s.add_argument("--latency-stress", type=float, default=None)
    s = sub.add_parser("paper"); s.add_argument("--seconds", type=float, default=300); s.add_argument("--warmup", type=float, default=120); s.add_argument("--warmup-file", default="data/paper_warmup.jsonl")
    s = sub.add_parser("bench"); s.add_argument("--n", type=int, default=200_000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    over = {}
    if getattr(args, "latency_stress", None):
        over["sim"] = {"latency_stress": args.latency_stress}
    cfg = load_config(args.config if Path(args.config).exists() else None, over)
    assert_not_live(cfg.mode)
    run_id = new_run_id(cfg.hash())
    audit = AuditLog(cfg.audit.dir, run_id, cfg.hash())
    audit.write("start", cmd=args.cmd, backend=core.BACKEND, config=cfg.to_dict())
    log.info("run_id=%s backend=%s mode=%s", run_id, core.BACKEND, cfg.mode)
    {"record": cmd_record, "sim": cmd_sim, "walkforward": cmd_walkforward, "gate": cmd_gate,
     "paper": cmd_paper, "bench": cmd_bench}[args.cmd](args, cfg, audit)
    audit.write("end")


if __name__ == "__main__":
    main()
