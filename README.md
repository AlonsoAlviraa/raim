# RAIM — Regime-Adaptive, Inventory-Aware Market Making (investigación)

> **AVISO IMPORTANTE.** Esto es un sistema de **investigación** en modo **SIMULACIÓN / PAPER**.
> **No es asesoramiento financiero.** El market making puede perder dinero muy rápido.
> No hay ningún camino de órdenes reales: el modo `live` hace *hard-fail* siempre.
> Ningún resultado de este repo (y menos sobre datos sintéticos) demuestra rentabilidad.

## Arquitectura

```
                 ┌──────────── SLOW BRAIN (async, segundos-minutos, ≤1 llamada/s) ───────────┐
                 │ tradingview-mcp (oficial, uvx) ─► LLM (Anthropic, modelo configurable)      │
                 │      └► SlowState validado con pydantic: vix_z, trend_align, vol_expansion, │
                 │         event_risk, universe_score, risk_multiplier m∈[0,1], confidence c   │
                 │ Jev (TypeSafe, System One): preguntas tipadas ─► p(toxic), p(régimen)       │
                 └───────────────┬──────────────── solo multiplicadores m_llm, m_jev ──────────┘
                                 │ (cache, lectura sin bloqueo; si cae ⇒ decae a conservador)
FAST BODY (determinista, sin LLM) ▼
WebSocket L2+trades ─► features ─► HMM 4 estados + BOCPD ─► quoter AS/GLFT ─► paper OMS ─► RISK GOVERNOR
(Binance público)     (C++/Py)      (filtro C++/Py)         (C++/Py)          (SimExchange)  (C++/Py, límites duros)
```

- **Multiplicador aplicado = min(m_llm, m_rules, m_jev)**, recortado a [0,1] (NaN→0). Los
  inputs externos **solo pueden apretar**: nunca aflojan límites, nunca levantan un halt
  (latched), nunca abren posición. Está en código (C++ y Python) y testeado.
- **Regímenes**: R1 calma/mean-reverting, R2 tendencia/informado, R3 alta vol/estrés,
  R4 tóxico/ruptura. Baum-Welch offline (solo sobre train), filtro forward causal, BOCPD
  (Adams-MacKay) como alarma, mezcla θ=Σ p_k θ_k con sesgo conservador hacia R3/R4,
  histéresis (retira quotes si p(R4)>0.7, reanuda <0.4). El contexto lento solo inclina la
  matriz de transición **hacia** estrés.
- **Quoting**: Avellaneda-Stoikov (2008) / Guéant-Lehalle-Fernandez-Tapia (2013) en bps,
  skew por inventario, post-only, nunca cruza el touch.
- **Riesgo duro**: `Qmax=min(B/(z·σ·mid·√h), c·depth)`, exposición bruta máx., halt por
  drawdown (latched), cancel-all por feed obsoleto, backoff por rechazos/mensajes, racha de
  pérdidas: ensanchar → pausar.
- **Simulador**: event-driven tick a tick, posición en cola (conservadora), fills parciales,
  latencia de ack/cancel con jitter, *stress* 2×, fees/rebates, slippage como haircut.
- **Investigación**: walk-forward purgado+embargado, markouts 1s/5s/30s, atribución por régimen,
  *first gate* `maker_fee + E[adverse_selection] < spread/2`.

### Backends C++ / Python
El hot path (order book L1/features, filtro HMM, quoter, governor, SimExchange) existe en
**C++17** (`cpp/`, pybind11 + scikit-build-core) y en **Python puro** (`src/raim/core/*_ref.py`)
con la misma API. Al arrancar verás `RAIM fast-body backend: cpp|python`. Forzar con
`$env:RAIM_BACKEND="python"` o `"cpp"`. Tests de paridad C++ vs Python incluidos.

Benchmark honesto en el box Linux de desarrollo (incluye overhead Python→C++ por llamada):

| componente | Python ns/update | C++ ns/update |
|---|---|---|
| features (on_book+values) | ~3 470 | ~590 |
| HMM step (4×6) | ~7 660 | ~780 |
| governor.evaluate | ~1 720 | ~680 |

El cuello de botella real será el bucle Python que orquesta; la latencia de red (decenas de ms)
domina en cualquier caso. Esto **no** es un sistema HFT de microsegundos.

## Instalación (Windows / PowerShell)

Requisitos: Python 3.10–3.13 (probado 3.11/3.12), `uv`, `git`.

```powershell
cd C:\Users\Mariano\Documents\ALONSOO\TRAD\RAIM
uv sync                      # instala RAIM en modo Python puro (no necesita compilador)
Copy-Item .env.example .env  # rellena solo las claves que uses
uv run pytest -q
```

### (Opcional) compilar el núcleo C++
Necesitas **Visual Studio Build Tools 2022** (workload C++) y **CMake**:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install Kitware.CMake
# abre una terminal nueva ("Developer PowerShell for VS 2022" si cl.exe no está en PATH)
uv pip install ./cpp
uv run raim bench
```
Si no compilas, todo funciona igual con el backend Python (más lento).

## Uso

```powershell
uv run raim record --seconds 60 --out data/btcusdt.jsonl   # datos públicos Binance (sin claves)
uv run raim gate --data data/btcusdt.jsonl                 # first gate PASS/FAIL
uv run raim sim  --data data/btcusdt.jsonl                 # fit HMM en 30% inicial, simula el resto
uv run raim sim  --data data/btcusdt.jsonl --latency-stress 2
uv run raim walkforward --data data/btcusdt.jsonl --folds 4
uv run raim sim                                            # SINTÉTICO (solo para probar el código)
uv run raim paper --warmup 120 --seconds 600               # feed público en vivo + ejecutor PAPER
uv run raim bench
```
Todo deja un log de auditoría append-only en `logs/audit_<run_id>.jsonl` (run id estilo git +
hash de la config). Config en `configs/default.yaml` (semilla incluida → reproducible).

`wss://data-stream.binance.vision` es el endpoint oficial de **solo datos de mercado**. Binance
restringe el acceso por jurisdicción; **comprueba que puedes usarlo legalmente desde España**
y revisa tu tier real de comisiones (`fees.maker_bps`) antes de interpretar el gate.

## Claves API (nunca en git)
Edita `.env` (o variables de entorno de PowerShell, `$env:NOMBRE="..."`):

| Variable | Para qué | Dónde se obtiene |
|---|---|---|
| `ANTHROPIC_API_KEY` | slow brain con `slow_brain.provider: anthropic` | console.anthropic.com |
| `TYPESAFE_API_KEY` | Jev real con `jev.provider: http` | https://console.typesafe.ai/keys |
| `MARKETAUX_API_TOKEN` | noticias/sentimiento en tradingview-mcp (opcional) | marketaux.com |

Sin claves todo corre con **mocks** (por defecto).

### tradingview-mcp (slow brain)
Solo el paquete oficial: PyPI `tradingview-mcp-server` (repo
`github.com/atilaahmettaner/tradingview-mcp`). **Nunca** instales clones de npm (hay malware).
```powershell
uv sync --extra mcp
uvx --from tradingview-mcp-server tradingview-mcp      # si hace timeout: uv tool install --python 3.13 tradingview-mcp-server
```
Es una capa de **análisis polled** (caché 60 s, TA ≤4 concurrentes, ≥0.8 s entre llamadas,
backtests diarios/horarios, sin order book ni órdenes). Nunca entra en el bucle de ticks.

## Qué está mockeado / pendiente
- Slow brain: `MockSlowProvider` por defecto. El proveedor Anthropic está implementado
  (Messages API, `claude-opus-5-5`, verificado en docs de Anthropic el 2026-10-12) pero no
  probado contra la API real (no hay clave en este entorno). El cliente MCP está
  implementado pero **no está aún cableado** a un `context_fn` real en `paper`.
- Jev: `MockJev` (heurística VPIN, **no es Jev**). El cliente HTTP sigue la API documentada
  (`POST https://api.typesafe.ai/v1/systemone`, `jev-latest`, preguntas `noul`/`choice`),
  probado solo con transporte simulado.
- AgenKit (agenkit.xyz): es un producto de suscripción (agentes para Claude Code/Cursor/etc.),
  no una librería ni CLI headless; este código se implementó directamente.
- Datos sintéticos: solo para tests. El smoke real (60 s BTCUSDT) es demasiado corto para
  conclusiones.
- Book: el SimExchange usa la profundidad L2 visible para la posición en cola (fase 2, SIM-1);
  niveles más allá de la profundidad visible = cola desconocida (solo llenan por trade-through).
- Hallazgos abiertos de revisión: ver `docs/reviews/` y `docs/AGENT_LOG.md`. Arquitectura:
  `docs/design/ARCHITECTURE.md`.

## Riesgos y lo que podría estar mal
- Con BTCUSDT el spread típico es 1 tick (~0.0014 bps): **el first gate falla** con
  comisión maker de 1 bps. Hay que elegir otro venue/instrumento/tier antes de seguir.
- El modelo de cola es una aproximación; los fills reales pueden ser peores.
- El HMM puede etiquetar regímenes de forma inestable entre ventanas; revisa la atribución.
- Latencias de simulación son supuestos; mide las tuyas y usa `--latency-stress 2`.

## Legalidad y conducta
Verifica comisiones y legalidad en tu jurisdicción antes de cualquier despliegue. Prohibido
spoofing, layering y wash trading. Paper/live quedan fuera de alcance hasta que pasen los
gates OOS + stress y haya aprobación explícita.
