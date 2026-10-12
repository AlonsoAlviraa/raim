#!/usr/bin/env sh
set -e
uv run raim record --seconds 60 --out data/btcusdt_smoke.jsonl
uv run raim gate --data data/btcusdt_smoke.jsonl
uv run raim sim --data data/btcusdt_smoke.jsonl
uv run raim sim --data data/btcusdt_smoke.jsonl --latency-stress 2
