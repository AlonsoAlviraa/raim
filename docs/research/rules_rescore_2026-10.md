# Re-evaluación de las 14 estrategias previas con las 15 "super reglas" (13-10-2026)

**Solo investigación con capital virtual (backtest / paper). No es consejo financiero. Ninguna estrategia es rentable demostrada.**

Protocolo: `/workspace/research/x_backtest_rules_2026-10.md` (sección C). Implementado en la rama `research-rules` (desde `strategies`).

## Resultado en una línea
**Las 14 estrategias (S1–S6, C1–C8) salen KILL.** Ninguna llega a WATCH ni a CANDIDATE_FOR_PAPER y ninguna supera DSR, SPA ni la bóveda.

## Qué se ha implementado
| Regla | Código | Umbral (en `configs/research_rules.yaml`) |
|---|---|---|
| Veredicto único | `src/raim/research/verdict.py::verdict(report)`. Lee los umbrales SOLO del YAML (su sha va en cada veredicto) y N del fichero de registro | KILL / WATCH / CANDIDATE_FOR_PAPER / PROMOTE |
| R1 preregistro | `research/prereg.py` (`register` → sha y log `registry/prereg_log.jsonl`; `check` exige registro anterior al primer trial y no retroactivo) | ≤50 variantes por familia |
| R2 registro | `verdict.registry_stats`: N = trials únicos por hash (una repetición idéntica no es una variante nueva). La varianza del SR excluye los trials degenerados | N global = 1.508 |
| R3 bóveda | `research/vault.py` (`seal`, `open_once` (el segundo intento lanza `VaultError`), `guard`, `verify_pins` vía `FrozenValidator`, más el código del validador y el sha de las reglas). Integrado en `run_strategies.py` y `run_convex.py`: comprueba los pins al arrancar y aplica `guard()` en cada `save` | 12 meses |
| R4 walk-forward | `strategies/wf.py`: ≥5 folds válidos garantizados, `min_embargo_frac` y purge ≥ horizonte (p95 de la duración de las operaciones) | ≥5 folds, ≥80 % positivos, embargo ≥1 % |
| R5 DSR | `engine.dsr`/`registry.deflated_sharpe` con N del registro | ≥0,95 |
| R6 PBO | `research/pbo.py::cscv_pbo` (CSCV, 16 bloques, 12.870 particiones) | ≤0,25 |
| R7 SPA / White RC | `research/spa.py::hansen_spa` (bootstrap estacionario, B=2000), con todas las variantes y la serie WF frente a BTC buy&hold y frente a efectivo al 4 % | p ≤0,05 frente a ambos |
| R8 MinTRL | `research/mintrl.py` | días OOS ≥ MinTRL(SR*=0, 95 %) y ≥200 operaciones |
| R9 meseta | `research/stability.py` (`neighbors`, `stability`, `heatmap`, `plateau_select`). **`wf.py` usa ahora `selector="plateau"` por defecto** (centro de la meseta con la mediana del vecindario), no el argmax | ≥70 % de los vecinos conservan ≥70 % del SR |
| R10 costes | `research/costs_curve.py`: curva de 0× a 5× y break-even exacto (el neto es lineal en el multiplicador) | break-even ≥3×, SR a 2× >0,5 |
| R11 ejecución | `engine.run(fill="next_open", opens=…)` y `impact=` (slippage σ·√(q/ADV)). En la re-evaluación: pasada completa con +1 barra de retraso | conserva ≥70 % del SR |
| R12 régimen | `research/regime_split.py` (BTC frente a su SMA200 y la pendiente de 20 días, causal) | ≥2/3 regímenes con SR>0, ≥75 % de años, peor DD de régimen ≥−30 % |
| R13 bootstrap | `research/bootstrap.py::block_bootstrap` (10.000 remuestreos, bloque medio de 20 días) | p5 del SR >0, p95 del DD ≥−35 %, racha perdedora p95 ≤15 días |
| R14 datos y capacidad | `src/raim/data/manifest.py` (sha256 de los 26.715 ficheros de `/workspace/data_hist`, manifiesto en `docs/research/rules_rescore_2026-10/`), `research/capacity.py` | manifiesto, universo PIT y ≤1 % del ADV a 100.000 USD |
| R15 paper y kill | `src/raim/paper/forward_gate.py` (≥90 días o ≥100 operaciones, SR ≥50 % del OOS, dentro del IC bootstrap) y `paper/kill_switch.py` (DD > p99 del bootstrap o CUSUM k=0,5σ, h=5σ). **Sin paper, el mejor veredicto posible es CANDIDATE_FOR_PAPER** | — |

Escalera del veredicto: si falla cualquier regla estadística, de robustez o de datos (R2, R4–R14) → **KILL**. Si todas pasan pero falla el proceso (R1 o R3) → **WATCH**. Con R1–R14 aprobadas → **CANDIDATE_FOR_PAPER**. PROMOTE solo con R15.

CI (`.github/workflows/ci.yml`, job `research-rules`):
- `tests/test_verdict.py`: ruido puro → KILL; señal plantada → pasa TODAS las reglas estadísticas (y sin bóveda queda en WATCH; con bóveda limpia, CANDIDATE_FOR_PAPER; solo con paper, PROMOTE).
- Se rechazan los reportes sin `prereg_sha` o sin manifiesto (`scripts/ci/check_reports.py`).
- `scripts/ci/check_rules_separation.py` falla si `configs/research_rules.yaml` cambia en el mismo PR que `reports/`, `docs/research/` o `registry/`.
- Un test comprueba que `verdict()` no tiene umbrales numéricos escritos a mano y que `run_strategies.py`/`run_convex.py` ya no calculan veredictos. Su código duplicado se ha eliminado: ahora solo escriben métricas descriptivas.

## La bóveda (honestidad)
- **Todos los datos disponibles terminan el 01-10-2026, y los últimos 12 meses (01-10-2025 → 01-10-2026) ya se usaron como OOS del walk-forward en las rondas 1 y 2.** Para las 14 familias ese hold-out está **contaminado**: la bóveda histórica está **gastada** (`spent_contaminated`, registrado en `registry/vault.json` y `registry/vault_log.jsonl`). S5 solo tiene datos de 1 minuto de 2024-07 a 2025-07, y su último año también fue su OOS: también está gastada.
- Se ha sellado una **bóveda nueva solo con datos futuros**: `vault_2026-10-13`, del 13-10-2026 al 13-10-2027, para todas las familias. Está fijada por el sha del validador y de las reglas. `guard()` impide que cualquier runner lea esas fechas, y `open_once` permite abrirla UNA sola vez por familia, a partir de oct-2027. Hubo un "reseal" registrado antes de cualquier uso porque el código del validador se terminó después del primer ensayo. La ventana no cambió y nunca se leyó.
- Por eso R3 falla para las 14. Aunque todo lo demás pasara, el techo sería WATCH.

## R1 preregistro (honestidad)
No hubo preregistro antes de bajar datos en las rondas 1 y 2. Se han escrito specs **retroactivas** (`configs/prereg/*.yaml`, `retroactive: true`) solo para la auditoría. Están en el log con fecha posterior a los trials y por diseño **no cumplen R1**. S5 además tiene 60 variantes por fold (más de 50).

## Cómo se re-evaluó
`scripts/rescore_rules.py` vuelve a ejecutar cada familia con el **protocolo nuevo**: selección por meseta, ≥5 folds válidos, embargo ≥1 % de la muestra, costes VIP0 con slippage por tramos y funding, más una segunda pasada con +1 barra de retraso. Las variantes nuevas (pasadas con retraso, folds nuevos) se añaden al registro global: N pasa de 805 a **1.508 trials únicos**. Guarda un JSON por estrategia en `reports/research_rules/` (con copia en `docs/research/rules_rescore_2026-10/`). Los Sharpe difieren algo de las rondas 1 y 2 porque cambian los folds y la selección (meseta en lugar de argmax).

## Tabla: regla por regla (✅ pasa, ❌ falla)
| strategy | oos_sharpe | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_basis_arb | 0.73 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | KILL |
| S2_funding_carry | -0.02 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | KILL |
| S3_donchian_ensemble | 1.07 | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | KILL |
| S4_tsmom_atr_6h | 0.68 | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | KILL |
| S5_leadlag_1m | -0.7 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | KILL |
| S6_pairs_coint | -0.93 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | KILL |
| C1_intraday_tsmom_30m | -3.16 | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | KILL |
| C2_donchian_15m | 0.13 | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | KILL |
| C3_squeeze_4h | 0.27 | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | KILL |
| C4_funding_extremes | 0.31 | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | KILL |
| C5_cascade_1h | -0.27 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | KILL |
| C6_alt_tsmom_weekly_pit | 0.7 | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | KILL |
| C7_S3_donchian_pit20 | 0.53 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | KILL |
| C8_new_listings | -1.35 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | KILL |

## Valores clave
| Estrategia | Sharpe OOS | Folds + | DSR | PBO | SPA p (BTC/cash) | días/MinTRL | trades | estab. meseta | breakeven × | SR 2× | ret. +1 barra | regímenes + | años + | peor DD rég. | boot SR p5 | boot DD p95 | ADV p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_basis_arb | 0.73 | 1/5 | 0.0 | 0.29 | 1.0/1.0 | 1645/3765 | 10 | 0.5 | 2.12 | 0.04 | 0.92 | 1/3 | 0.2 | -0.0 | -0.64 | -0.0 | 0.0006 |
| S2_funding_carry | -0.02 | 3/5 | 0.0 | 0.42 | 1.0/1.0 | 1645/∞ | 126 | 1.0 | 0.92 | -0.21 | 0.0 | 1/3 | 0.4 | -0.05 | -0.59 | -0.1 | 0.00012 |
| S3_donchian_ensemble | 1.07 | 4/5 | 0.0 | 0.47 | 1.0/0.126 | 1391/844 | 711 | 0.67 | 10.31 | 0.96 | 0.95 | 2/3 | 0.6 | -0.14 | 0.03 | -0.23 | 4e-05 |
| S4_tsmom_atr_6h | 0.68 | 3/5 | 0.0 | 0.24 | 1.0/0.258 | 1395/2081 | 1253 | 0.67 | 7.24 | 0.57 | 0.85 | 3/3 | 0.6 | -0.49 | -0.26 | -0.74 | 0.00065 |
| S5_leadlag_1m | -0.7 | 6/50 | 0.0 | — | —/— | 274/∞ | 84 | — | 0.59 | -2.28 | 0.0 | 1/2 | 0.5 | -0.07 | -2.28 | -0.09 | — |
| S6_pairs_coint | -0.93 | 2/5 | 0.0 | 0.42 | 1.0/1.0 | 1395/∞ | 2753 | 0.0 | 0.1 | -1.93 | 0.0 | 0/3 | 0.6 | -0.2 | -1.87 | -0.4 | 0.00049 |
| C1_intraday_tsmom_30m | -3.16 | 0/5 | 0.0 | 0.13 | 1.0/1.0 | 1395/∞ | 517 | 0.0 | -0.49 | -4.75 | 0.0 | 0/3 | 0.0 | -0.16 | -3.84 | -0.38 | 0.00093 |
| C2_donchian_15m | 0.13 | 3/5 | 0.0 | 0.24 | 1.0/0.466 | 1395/61243 | 936 | 0.33 | 1.31 | -0.27 | 1.85 | 1/3 | 0.6 | -0.33 | -0.8 | -0.7 | 0.00055 |
| C3_squeeze_4h | 0.27 | 4/5 | 0.0 | 0.08 | 1.0/0.36 | 1395/13753 | 1005 | 0.5 | 3.84 | 0.17 | 0.71 | 2/3 | 0.8 | -0.4 | -0.45 | -0.73 | 0.00065 |
| C4_funding_extremes | 0.31 | 2/5 | 0.0 | 0.07 | 1.0/0.452 | 1395/10296 | 945 | 0.33 | 2.85 | 0.14 | 1.15 | 3/3 | 0.6 | -0.66 | -0.15 | -0.91 | 0.00153 |
| C5_cascade_1h | -0.27 | 3/5 | 0.0 | 0.56 | 1.0/1.0 | 1395/∞ | 1703 | 0.0 | 0.54 | -0.84 | 0.0 | 2/3 | 0.4 | -0.56 | -1.3 | -0.85 | 0.00305 |
| C6_alt_tsmom_weekly_pit | 0.7 | 3/5 | 0.0 | 0.08 | 1.0/0.3 | 1395/2074 | 780 | 0.33 | 9.52 | 0.61 | 1.05 | 2/3 | 0.8 | -0.3 | -0.23 | -0.54 | 0.00018 |
| C7_S3_donchian_pit20 | 0.53 | 2/5 | 0.0 | 0.5 | 1.0/0.365 | 1395/3511 | 1639 | 0.67 | 3.57 | 0.33 | 1.14 | 2/3 | 0.6 | -0.11 | -0.44 | -0.23 | 2e-05 |
| C8_new_listings | -1.35 | 1/5 | 0.0 | 0.47 | 1.0/1.0 | 1395/∞ | 238 | 0.0 | -12.43 | -1.45 | 0.0 | 0/3 | 0.2 | -0.43 | -2.2 | -0.74 | 0.0014 |

(Columnas: «días/MinTRL» = días OOS frente a la longitud mínima necesaria; «breakeven ×» = múltiplo del coste real que deja el SR en 0; «ret. +1 barra» = SR con un retraso extra dividido por el SR base; «boot DD p95» = el DD que el 95 % de los remuestreos no supera.)

## Qué falla cada estrategia
- **Todas:** R1 (sin preregistro previo), R3 (bóveda gastada), R5 (DSR≈0 con N=1.508), R7 (ninguna bate a BTC buy&hold con p≤0,05: p≈1; frente a efectivo la mejor es S3 con p=0,13) y R15 (sin paper).
- **S1 basis:** además R4 (1/5), R6, R8 (10 operaciones), R9, R10 (break-even 2,1×), R12, R13, R14 (universo fijo). Solo pasa R2 y R11.
- **S2 funding carry:** SR −0,02. Falla R4, R6, R8, R10, R11, R12, R13 y R14. Pasa R9: la meseta es plana… en torno a cero.
- **S3 Donchian (lista fija de 10):** **la más cercana**. Pasa R4 (4/5), R8, R10 (break-even 10×, SR 0,96 a 2×), R11 (95 %) y R13 (p5 del SR 0,03). Falla R6 (PBO 0,47), R9 (estabilidad 0,67 < 0,70), R12 (60 % de años positivos) y R14 (lista fija de 2026, sesgo de supervivencia; con universo PIT es C7 y empeora).
- **S4 TSMOM 6h:** pasa R6, R10 y R11. Falla R4 (3/5), R8, R9, R12 (DD de régimen −49 %), R13 (p95 del DD −74 %) y R14.
- **S5 lead-lag 1m:** falla todo menos R2. PBO, SPA y meseta no se pueden calcular (selección dentro de cada fold, sin matriz de variantes): cuentan como fallo.
- **S6 pares:** SR −0,93. Falla todo menos R2.
- **C1 intradía 30m:** SR −3,2. Pasa R6 (PBO 0,13, pero de variantes todas negativas) y nada útil más.
- **C2 Donchian 15m / C3 squeeze 4h / C4 funding extremo:** pasan R6 y R11. Fallan R4, R8, R9, R10 (break-even 1,3× / 3,8× / 2,85×; SR a 2× −0,27 / 0,17 / 0,14), R12 (peor DD de régimen −33/−40/−66 %), R13 y R14.
- **C5 cascadas:** falla todo menos R2.
- **C6 TSMOM altcoins semanal PIT:** pasa R6 (PBO 0,08), R10 (break-even 9,5×, SR 0,61 a 2×), R11 y **R14** (PIT con monedas que dejaron de cotizar, 0,02 % del ADV). Falla R4 (3/5), R8 (1.395 días frente a MinTRL de 2.074), R9 (0,33), R12 (bear con SR −1,04 y DD −30,4 %), R13 (p5 del SR −0,23, p95 del DD −54 %).
- **C7 (S3 en PIT top-20):** pasa R11 y R14. Falla R4 (2/5), R6 (0,50), R8, R9, R10 (SR 0,33 a 2×), R12 y R13.
- **C8 nuevos listados:** solo pasa R2 y R14.

## Lectura honesta
1. Con el protocolo estricto el resultado es el mismo que ya intuíamos, pero ahora está medido: **no hay ninguna estrategia candidata**. Las que más se acercan (S3, C6) fallan por **estabilidad temporal y de régimen** (R4, R9, R12) y por **significancia frente a BTC** (R5, R7), no por costes.
2. La selección por meseta empeora un poco algunos Sharpe frente al argmax de las rondas 1 y 2 (S3 1,17 → 1,07, C6 0,87 → 0,70, C7 0,70 → 0,53). Es lo esperable si parte del resultado anterior era el pico de la rejilla.
3. **No queda hold-out limpio para estas 14 familias.** La única prueba honesta posible es hacia delante: paper desde hoy (con kill switch) y la bóveda `vault_2026-10-13`, que se podrá abrir una sola vez en oct-2027. Cualquier nueva variante de S3 o C6 debe preregistrarse ANTES (≤50 variantes) y cuenta en N.

## Limitaciones
- R11 se evalúa con +1 barra de retraso (más conservador que «apertura siguiente»). `fill="next_open"` y el impacto σ·√(q/ADV) están en el motor y tienen tests, pero los runners de cada estrategia no se han reescrito para usarlos.
- Capacidad: los perps usan el ADV del spot como proxy; S5 no conserva pesos, así que R14 no se puede calcular (fallo).
- El manifiesto cubre todo `/workspace/data_hist`, no solo los ficheros que usa cada estrategia.
- PBO/SPA se calculan con la rejilla de cada familia (6–18 variantes). Es una familia pequeña y el PBO es ruidoso.
- El número de operaciones de S5 es una aproximación por turnover.

## Reproducir
```
PYTHONPATH=src python scripts/rescore_rules.py            # --skip-done para reanudar
PYTHONPATH=src python scripts/rules_rescore_table.py
python scripts/ci/check_reports.py
pytest -q
```
