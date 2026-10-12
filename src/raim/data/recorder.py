"""Record / replay normalized events as JSONL (optionally parquet via pyarrow)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

from raim.data.venue import VenueAdapter
from raim.types import BookSnapshot, Trade

log = logging.getLogger(__name__)


def event_to_dict(ev: BookSnapshot | Trade) -> dict:
    if isinstance(ev, BookSnapshot):
        return {"t": "b", "ts": ev.ts_ms, "rv": ev.recv_ms, "b": ev.bids, "a": ev.asks}
    return {"t": "t", "ts": ev.ts_ms, "rv": ev.recv_ms, "p": ev.price, "q": ev.qty, "m": ev.is_buyer_maker}


def dict_to_event(d: dict) -> BookSnapshot | Trade:
    if d["t"] == "b":
        return BookSnapshot(d["ts"], tuple(map(tuple, d["b"])), tuple(map(tuple, d["a"])), d["rv"])
    return Trade(d["ts"], d["p"], d["q"], d["m"], d["rv"])


async def record(adapter: VenueAdapter, out_path: Path, seconds: float) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    end = time.monotonic() + seconds

    async def _run() -> None:
        nonlocal n
        with out_path.open("w", encoding="utf-8") as fh:
            async for ev in adapter.stream():
                fh.write(json.dumps(event_to_dict(ev)) + "\n")
                n += 1
                if time.monotonic() >= end:
                    return

    try:
        await asyncio.wait_for(_run(), timeout=seconds + 15)
    except asyncio.TimeoutError:
        log.warning("recording timed out")
    log.info("recorded %d events to %s", n, out_path)
    return n


def replay(path: Path) -> Iterator[BookSnapshot | Trade]:
    """Yield events ordered by RECEIVE time (what the strategy could actually know)."""
    with Path(path).open(encoding="utf-8") as fh:
        evs = [dict_to_event(json.loads(line)) for line in fh if line.strip()]
    evs.sort(key=lambda e: e.recv_ms or e.ts_ms)
    yield from evs


def to_parquet(jsonl: Path, out: Path) -> None:
    import pandas as pd

    rows = [json.loads(x) for x in Path(jsonl).read_text(encoding="utf-8").splitlines() if x]
    pd.DataFrame(rows).to_parquet(out)  # requires pyarrow (extra: parquet)
