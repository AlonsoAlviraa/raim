import json

import pytest

from raim.audit import AuditLog, new_run_id
from raim.config import load_config


def test_yaml_loads_and_hash_stable(tmp_path):
    cfg = load_config("configs/default.yaml")
    assert cfg.mode == "simulation" and cfg.slow_brain.model == "claude-opus-5-5"
    assert cfg.hash() == load_config("configs/default.yaml").hash()
    assert cfg.hash() != load_config("configs/default.yaml", {"seed": 1}).hash()
    with pytest.raises(ValueError):
        load_config(None, {"risk": {"nope": 1}})


def test_audit_append_only(tmp_path):
    a = AuditLog(tmp_path, new_run_id("abc1234"), "abc1234")
    a.write("x", v=1); a.write("y", v=2)
    lines = [json.loads(x) for x in a.path.read_text(encoding="utf-8").splitlines()]
    assert [r["seq"] for r in lines] == [1, 2] and lines[0]["cfg"] == "abc1234"


def test_env_example_has_no_values():
    for line in open(".env.example", encoding="utf-8"):
        if "=" in line and not line.startswith("#"):
            assert line.strip().split("=", 1)[1] == ""
