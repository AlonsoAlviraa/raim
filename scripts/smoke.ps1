# Smoke: graba 60 s de datos públicos de Binance y simula sobre ellos (SIN órdenes reales).
$ErrorActionPreference = "Stop"
uv run raim record --seconds 60 --out data/btcusdt_smoke.jsonl
uv run raim gate --data data/btcusdt_smoke.jsonl
uv run raim sim --data data/btcusdt_smoke.jsonl
uv run raim sim --data data/btcusdt_smoke.jsonl --latency-stress 2
