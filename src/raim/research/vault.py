"""R3: sealed hold-out vault. The last `months` of data are sealed per family; open_once() can be
called ONCE per family (a second call raises) and every access is appended to the vault log.
The seal pins the validator code + rules file by hash via FrozenValidator; runners call
guard() so no backtest can read dates inside a sealed window."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from raim.research.registry import FrozenValidator

STATE = Path("registry/vault.json")
LOG = Path("registry/vault_log.jsonl")
RULES = Path("configs/research_rules.yaml")


class VaultError(RuntimeError):
    pass


class Vault:
    def __init__(self, state: Path = STATE, log: Path = LOG, rules: Path = RULES):
        self.state_path, self.log_path, self.rules = Path(state), Path(log), Path(rules)
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else \
            {"seals": {}, "families": {}}

    # ------------------------------------------------------------------ internals
    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def _log(self, **kw):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps({"ts": time.time(), **kw}) + "\n")

    def _pins(self) -> dict:
        rules_sha = hashlib.sha256(self.rules.read_bytes()).hexdigest() if self.rules.exists() else None
        return {"validator_sha": FrozenValidator.code_hash(), "rules_sha": rules_sha}

    # ------------------------------------------------------------------ API
    def seal(self, name: str, start: str, end: str, families: list[str] | str = "*",
             data_sha: str | None = None) -> dict:
        if name in self.state["seals"]:
            raise VaultError(f"seal {name} already exists")
        s = {"start": start, "end": end, "families": families, "data_sha": data_sha,
             "sealed_at": time.time(), **self._pins()}
        self.state["seals"][name] = s
        self._save()
        self._log(event="seal", seal=name, **{k: v for k, v in s.items() if k != "sealed_at"})
        return s

    def mark_contaminated(self, family: str, window: tuple[str, str], reason: str) -> None:
        self.state.setdefault("contaminated", {})[family] = {
            "status": "spent_contaminated", "window": list(window), "reason": reason, "ts": time.time()}
        self._save()
        self._log(event="contaminated", family=family, window=list(window), reason=reason)

    def status(self, family: str) -> dict:
        if family in self.state["families"]:
            return self.state["families"][family]
        return self.state.get("contaminated", {}).get(family, {"status": "sealed"})

    def verify_pins(self, seal: str) -> None:
        s = self.state["seals"][seal]
        p = self._pins()
        if p["validator_sha"] != s["validator_sha"] or p["rules_sha"] != s["rules_sha"]:
            raise VaultError("validator code or rules changed since the vault was sealed")

    def _seals_for(self, family: str):
        for n, s in self.state["seals"].items():
            if s["families"] == "*" or family in s["families"]:
                yield n, s

    def guard(self, family: str, index) -> None:
        """Raise if a backtest touches a sealed window that this family has not opened."""
        if not len(index):
            return
        idx = pd.DatetimeIndex(index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        for n, s in self._seals_for(family):
            st = self.state["families"].get(family, {})
            if st.get("status") == "opened" and st.get("seal") == n:
                continue
            if ((idx >= pd.Timestamp(s["start"])) & (idx < pd.Timestamp(s["end"]))).any():
                raise VaultError(f"{family}: data touches sealed vault {n} [{s['start']},{s['end']})")

    def open_once(self, family: str, seal: str) -> dict:
        st = self.state["families"].get(family)
        if st is not None:
            raise VaultError(f"vault already used for {family}: {st['status']}")
        self.verify_pins(seal)
        s = self.state["seals"][seal]
        self.state["families"][family] = {"status": "opened", "seal": seal, "ts": time.time()}
        self._save()
        self._log(event="open", family=family, seal=seal)
        return s
