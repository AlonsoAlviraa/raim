# AGENT_LOG

Registro honesto de quién hizo qué. **Todos los roles de las fases 1 y 2 los ejecutó un único
agente** (Grok Bot, subagente executor), delegado por el agente principal de la conversación.
En esta sesión no había herramienta de sub-agentes (Task) ni CloudAgent, así que **no hubo
revisores independientes**: las "revisiones" son pasadas adversariales separadas del mismo
agente, documentadas en `docs/reviews/`. Una revisión humana o de otro agente sigue pendiente.
Horas en Europe/Madrid (UTC+2).

## Fase 1 — 2026-10-12 ~14:32–14:50 (registro retroactivo)
| Rol | Agente | Ficheros | Tests | Notas |
|---|---|---|---|---|
| ops | executor único | pyproject, configs, audit, cli, bench, scripts, CI | test_config_audit | uv, YAML, audit JSONL |
| data | executor único | data/venue.py, recorder.py, synthetic.py | test_leakage | smoke real 60 s BTCUSDT (1668 eventos) |
| features | executor único | core/features_ref.py | test_leakage, test_parity | causales |
| regime | executor único | core/hmm_ref.py, regime/* | test_regime | Baum-Welch, BOCPD, histéresis |
| quoting | executor único | core/quoter_ref.py | test_parity | AS/GLFT en bps |
| risk | executor único | core/governor_ref.py, execution/guard.py | test_risk | monotonía en ambos backends |
| sim | executor único | core/simexchange_ref.py, sim/engine.py | test_sim | cola top-of-book (ver SIM-1) |
| cpp | executor único | cpp/* | test_parity | compilado con zig c++ en Linux; MSVC no probado |
| research | executor único | research/* | test_leakage, test_sim | gate FAIL en BTCUSDT |
| slowbrain | executor único | slowbrain/* | test_external | mock por defecto |
| jev | executor único | jev/adapter.py | test_external | API verificada en docs.typesafe.ai |
| docs | executor único | README.md, AGENTS.md | — | |
Resultado: 59 passed (C++ construido), 39 passed / 5 skipped en Python 3.11 sin compilador.

## Fase 2 — 2026-10-12 ~14:53–15:10
| Rol | Agente (autor / revisor) | Hallazgos | Tests (primero en rojo) | Resolución |
|---|---|---|---|---|
| entendimiento | executor | — | — | docs/00_entendimiento.md |
| regime | executor / executor (pasada adversarial) | REGIME-1..4 | test_regime1_* | 1 arreglado; 2–4 abiertos |
| risk | executor / executor | RISK-1..4 | test_risk1_* (ambos backends) | 1 arreglado (crítico); 2–3 abiertos |
| quoting+cpp | executor / executor | QUOTE-1..3, CPP-1 | test_quote1_* (ambos backends) | 1 arreglado; 2–3 abiertos |
| sim | executor / executor | SIM-1..5 | test_sim1_*, test_sim2_* | 1–2 arreglados; 3–5 abiertos |
| slowbrain | executor / executor | BRAIN-1..5 | test_brain1_* | 1 arreglado; 2–4 abiertos |
| jev | executor / executor | JEV-1..3 | test_jev1_*, test_jev2_* | 1–2 arreglados; 3 abierto |
Ejecución en rojo antes de arreglar: `docs/reviews/red_run.txt` (22 failed, 5 passed).
Después: 86 passed con RAIM_BACKEND=python y 86 passed con RAIM_BACKEND=cpp.

## Commits (hora, mensaje)
- b60d022 14:49 [ops] project scaffold: pyproject (uv), config YAML, .env.example, MIT license
- da95830 14:49 [ops] append-only JSONL audit log with run ids + config hash
- c511568 14:49 [data] Binance public WS adapter, recorder/replay, synthetic regime generator (tests only)
- 8c3c353 14:49 [features] causal incremental features reference (microprice, OFI, BV vol, VPIN, VR, Hawkes)
- b6482b3 14:49 [regime] HMM forward filter ref, Baum-Welch fit, BOCPD, conservative blending + hysteresis
- dffe34f 14:49 [quoting] Avellaneda-Stoikov / GLFT quoter with inventory skew
- a626567 14:49 [risk] deterministic risk governor: external inputs can only tighten; live guard
- 7724cbd 14:49 [sim] tick-level SimExchange (queue, partials, latency/jitter, fees) + event-driven runner
- 16e0926 14:49 [cpp] C++17 hot path raim_core (pybind11, scikit-build-core), optional with Python fallback
- b42a469 14:49 [research] purged/embargoed walk-forward, markouts, regime attribution, first gate
- bba01ac 14:49 [slowbrain] pydantic SlowState, tradingview-mcp client, Anthropic/mock providers, outage-safe loop
- e62f79d 14:49 [jev] TypeSafe Jev adapter per public API docs + mock + conservative fallback
- 860f8af 14:49 [ops] CLI (record/sim/walkforward/gate/paper/bench), benchmark, scripts
- c23bf6b 14:49 [tests] parity C++/Py, risk monotonicity on both backends, leakage, HMM, sim, outage
- 5490add 14:49 [ops] GitHub Actions CI: Linux+Windows, pure-Python and C++ backends
- 7c79e7c 14:49 [docs] README (es) and AGENTS.md
- 5b72434 14:55 [docs] phase 2: 00_entendimiento (principles, constraints, what-could-be-wrong)
- 81a8dc4 14:55 [tests] phase 2 review findings as failing tests (tests-first, red)
- 1370405 14:57 [risk] RISK-1: non-finite equity/position -> cancel_all 'bad_input'; NaN never touches peak/halt (C+++Py)
- 147ebf0 14:57 [quoting] QUOTE-1: degenerate gamma/kappa/mid/tick -> no quote (NaN), engine skips (C+++Py)
- 4c40cf4 14:57 [cpp] mirror RISK-1, QUOTE-1, SIM-1 (L2 queue, on_book_l2) in raim_core; parity kept
- 4ee556b 14:57 [sim] SIM-1 L2 queue position (unknown depth = infinite queue), SIM-2 single receive clock, wiring of slow context/features
- 7591297 14:57 [slowbrain] BRAIN-1: forward fast features + symbol as MCP/LLM context; expose slow_context for HMM tilt
- 46b5bf3 14:57 [jev] JEV-2: validate/clamp answers, complete+renormalize regime probs, non-finite -> fallback
