"""Append-only JSONL audit log with run ids and config hash."""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def new_run_id(config_hash: str) -> str:
    """Git-style short id: time + config hash + random."""
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{config_hash[:7]}-{secrets.token_hex(3)}"


class AuditLog:
    def __init__(self, directory: str | Path, run_id: str, config_hash: str) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"audit_{run_id}.jsonl"
        self.run_id = run_id
        self.config_hash = config_hash
        self._seq = 0

    def write(self, event: str, **payload: Any) -> None:
        self._seq += 1
        rec = {"run_id": self.run_id, "cfg": self.config_hash, "seq": self._seq,
               "wall_ms": int(time.time() * 1000), "event": event, **payload}
        # open in append mode for every write: append-only, crash-safe
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
