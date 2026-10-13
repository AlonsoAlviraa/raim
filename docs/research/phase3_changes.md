# Fase 3 — cambios derivados del deep research (2026-10)

Fuente: `/workspace/research/deepresearch_2026-10.md` (en el box, no está en el repo). Todos los
papers son **preprints**. Casi toda su evidencia viene de simulación. Aquí los usamos como
*hipótesis de diseño* que hay que validar con datos propios, no como prueba de edge.

| # | Cambio | Papers | Dónde | Tests |
|---|---|---|---|---|
| 1 | **FlowCPD**: BOCPD Beta-Bernoulli sobre el *signo* del flujo como feature del fast body (`p_change`, `flow_bias`). Acorta el horizonte de quoting tras un cambio | Moret & Lillo, *Deep Learning of Robust Market Making under Regime-Switching Order Flow*, arXiv:2609.11614; Adams & MacKay 2007 | `core/flowcpd_ref.py`, `cpp/src/core.hpp::FlowCPD` | detección, causalidad, paridad C++ |
| 2 | **Asimetría de exposición ajustada por cola**: ensancha o reduce el lado expuesto por el flujo persistente y el lado con menos cola delante. Solo puede ensanchar o encoger, nunca bloquea la reducción de inventario | Moret & Lillo 2609.11614 | `quoter_ref.exposure_adjust`, `exposure_adjust` (C++), `SimExchange.queue_ahead` | monotonía, dirección, NaN, paridad |
| 3 | **Baseline de "dial de estrés" continuo**: PC1 de las observaciones causales (no solo el spread), ≤4 bandas por cuantiles de train, frente al HMM en el mismo walk-forward purgado | Thawait, *Latent Continuum of Regimes in LOB Dynamics*, arXiv:2610.05740 | `research/stress_index.py`, `pipeline.compare_regime_models`, `raim compare-regimes` | orientación, bandas, informe |
| 4 | **Protocolo de cola**: CVaR10/30 por episodio, bootstrap de sesgo de inventario terminal y batería de escenarios de estrés (deriva±, liquidez escasa o densa, Hawkes alto, impacto alto, combinaciones, pico de vol) | Yang & Xu, *Robust MM with Hawkes Order Flow and Price Impact via ARL*, arXiv:2609.22785 | `research/robustness.py`, `sim/scenarios.py`, `raim stress` | cvar, bias test, batería |
| 5 | **Ablación con costes**: solo reglas vs +LLM vs +Jev vs ambos, neto del coste de API | Duan et al., *TradeLens*, arXiv:2607.10286 (v2); Chen & Wang, *The Price of Thought*, arXiv:2609.30705 | `research/ablation.py` | brazos, contabilidad de costes |
| 6 | **Registro de pruebas, Deflated Sharpe y validador congelado** | Bailey & López de Prado 2014 (DSR); Gençay, arXiv:2608.27734; Qu et al., arXiv:2609.27051 | `research/registry.py`; `walkforward`, `stress` y `compare-regimes` registran sus pruebas | DSR penaliza N, append-only, freeze/verify |
| 7 | **Calibración de Jev antes de usarlo** (Brier, ECE, fiabilidad, gate). Un Jev sin calibrar se ignora | Deep research §5 (sin evidencia independiente); Ulesov et al., arXiv:2609.28090 (los jueces LLM no sustituyen los tests) | `research/calibration.py`, `runtime.ExternalContext.set_calibration` | métricas, gate, "no calibrado ⇒ no se usa" |
| 8 | **Cadena de suministro**: `tradingview-mcp-server==0.8.1` fijado; `validate_mcp_command` rechaza npm/npx/node y versiones sin fijar; test que busca instalaciones npm en el repo | Deep research §4–5 (≥3 proyectos homónimos) | `slowbrain/mcp_client.py`, `configs/default.yaml` | rechazos, escaneo del repo |
| R2 | **Unwind tras halt**: determinista, primero pasivo y luego agresivo, solo reduce, con clip y límite de slippage. Solo en simulación | — | `execution/unwind.py`, `SimExchange.take` (C++ y Py), `sim/engine.py` | plan, take L2, halt ⇒ posición plana |
| R3 | **Profundidad para Qmax según el lado** (el lado contra el que se deshace la posición) | — | `sim/engine.unwind_depth` | test |

## Resultados (honestos, solo sintéticos)
- `docs/reports/phase3_stress_battery.txt` y `docs/reports/phase3_hmm_vs_stress_index.txt`.
- **HMM vs dial de estrés** (sintético con regímenes *discretos*, 4 folds): en las medias el HMM
  sale mejor (equity 0,62 vs −0,35 USD, CVaR10 −0,038 vs −0,187), pero lo domina un solo fold
  (fold 0). Además, los datos se generaron con 4 regímenes discretos, lo que favorece al HMM por
  construcción. **No es evidencia** a favor ni en contra en datos reales. El PC1 explica ~46 % de
  la varianza en estos sintéticos, frente al ~84 % que reporta Thawait en FESX: hay que medirlo en datos reales.
- **Batería de estrés**: ver la sección de hallazgos abiertos (PH3-1..3).

## Ajustes hechos al ver los datos (por transparencia)
- FlowCPD `short_run`: el test se escribió con 10. Con un cambio de flujo 0,5→0,9 la masa en
  run<10 apenas sube (máx. 0,19), así que se pasó a 30 (máx. 0,77). El cambio se hizo *antes*
  de usarlo en ningún informe de rendimiento.
- Dos tests propios tenían errores y se corrigieron. El de stress index evaluaba un punto
  justo en el borde del cuantil 75. El de unwind usaba una configuración sin fills, así que el
  halt nunca saltaba; ahora fuerza quotes cerca del touch.
