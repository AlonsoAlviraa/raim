# Estrategias alternativas al market making (2026-10)

Fecha: 2026-10-12 · Rama `strategies` (desde `phase3-research`) · Código: `src/raim/strategies/`,
`scripts/run_strategies.py`, tests `tests/test_strategies_leakage.py` · Resultados crudos:
`docs/research/strategies_2026-10/` (summary.csv/json, un JSON por estrategia con folds, registro
de trials `trials.jsonl.gz`).

**Solo backtest con capital virtual. No es consejo financiero. Ningún resultado de aquí demuestra
rentabilidad: son evaluaciones fuera de muestra (OOS) con supuestos explícitos que pueden fallar.**

## 1. Por qué

El market making falla el primer filtro (maker_fee + E[adverse selection] < ½ spread) con
comisiones minoristas en los 30 instrumentos grabados. Se buscan estrategias más lentas en las
que una comisión taker VIP0 no sea el factor dominante.

## 2. Papers (enlaces comprobados el 2026-10-12)

| # | Estrategia | Paper | Enlace |
|---|---|---|---|
| S1 | Arbitraje de base "random-maturity" spot-perp | He, Manela, Ross, von Wachter, *Fundamentals of Perpetual Futures* (borrador sept. 2026) | https://arxiv.org/abs/2212.06888 |
| S2 | Carry de funding (largo spot / corto perp) | Baseline "Carry" (Christin et al.) del paper anterior; marco de riesgo de Pindza, *CEX–DEX funding rate arbitrage as a basis trade*, Digital Finance, 19-ago-2026 | https://link.springer.com/article/10.1007/s42521-026-00213-3 |
| S3 | Ensemble de canales Donchian + sizing por volatilidad | Zarattini, Pagani, Barbon, *Catching Crypto Trends* (SSRN, abr-2025) | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907 |
| S4 | TSMOM 6h con trailing stop ATR (AdaptiveTrend simplificado) | Nguyen, *Systematic Trend-Following with Adaptive Portfolio Construction* (arXiv, feb-2026) | https://arxiv.org/abs/2602.11708 |
| S5 | Lead-lag BTC → altcoins a 1 minuto | Kurihara & Matsumoto, *Price Transmission from Bitcoin to Altcoins*, Asia-Pacific Financial Markets, 10-mar-2026 | https://link.springer.com/article/10.1007/s10690-026-09589-z |
| S6 | Pares cointegrados | Danilo, Jamhamed, Martin, CREM WP 2024-11 (RePEc); variante cópula Tadi & Witzany arXiv:2305.06961 | https://ideas.repec.org/p/tut/cremwp/2024-11.html |

Notas de credibilidad: S1 es el más sólido (autores académicos, revisado varias veces, datos de
Binance). S4 es de un único autor de una empresa, con Sharpe 2,41 y una equity curve
sospechosamente lisa: tratar sus cifras como no verificadas. S5 se evalúa en ventanas de 1 semana
OOS con comisión 0,02 % (muy por debajo de la VIP0 spot real de 0,10 %). El SSRN de S3 lo
cité por su abstract (no pude leer el PDF completo). S6 es un working paper de RePEc, no arXiv/SSRN.

## 3. Método (igual para todas)

- **Datos**: archivos públicos de `data.binance.vision`: klines 1h spot y USDT-M de
  2021-01 a 2026-09 para BTC, ETH, SOL, BNB, XRP, DOGE, ADA, LTC, LINK y AVAX; `fundingRate`
  mensual; klines 1m spot de 2024-07 a 2025-06 para BTC, ETH, LTC, PIVX, QKC y GNO. No usé
  Bybit/OKX históricos (sin tiempo en esta ronda). Las grabaciones L2 de `gate_scan*` (2h) son
  demasiado cortas para estas estrategias y no se usan.
- **Motor** (`engine.py`): la estrategia da pesos objetivo con información ≤ t; el motor los
  retrasa **1 barra extra obligatoria** (señal al cierre de t → ejecución al cierre de t+1).
  Coste por |Δpeso|: comisiones **taker VIP0** (spot 0,10 %, perp 0,05 %) + slippage de
  2 bps por lado (10 bps en small caps). Funding real: los largos en perp pagan y los cortos cobran.
  No supongo fills maker.
- **Robustez de costes**: pasada `_stress2x` con el doble de slippage.
- **Walk-forward** purgado y con embargo (expanding, 5 folds; purga 7-60 días según el horizonte).
  El parámetro de cada fold se elige solo con el train purgado. S5 usa un walk-forward mensual
  propio (OLS ajustado en el train y umbral elegido en el train).
- **Deflated Sharpe**: **todas** las variantes probadas se registran en `TrialRegistry`
  (N = 679 trials en total, 601 de ellas del grid de S5). Doy dos valores. El DSR global usa
  N = 679 y la varianza del SR entre trials (excluyendo trials degenerados con menos de 60 días
  o menos del 5 % de días activos). El DSR de familia usa solo los trials de esa estrategia.
- **Métricas** sobre retornos diarios OOS: Sharpe (rf = 0), Sharpe en exceso de un 4 % anual de
  caja (coste de oportunidad del capital), max DD, CVaR 5 % diario, turnover anual (Σ|Δw|/año),
  coste anual y funding anual sobre el capital. **Comparación**: buy-and-hold de BTC en las mismas
  fechas OOS.
- **Veredicto** (regla fijada antes de mirar los números finales, ver `summarize()`):
  PROMOTE si Sharpe exceso > 1, DSR global > 0,95, maxDD > −20 %, ≥ 2/3 de folds positivos y
  Sharpe por encima del de BTC B&H. WATCH si Sharpe exceso > 0,5 y PSR(0) > 0,9. Si no, KILL.

## 4. Resultados OOS (netos de comisiones, slippage y funding)

| Estrategia | Paper | OOS | Sharpe OOS (rf=0) | Sharpe exceso 4 % | PSR(0) | DSR global / familia | Max DD | CVaR5 diario | Rent. anual neta | Turnover/año | Coste/año | Folds + | BTC B&H Sharpe / DD | Veredicto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 Arbitraje de base | He et al. 2212.06888 | 2021-12 → 2026-10 | 0,71 | negativo | 0,86 | 0,00 / 0,16 | −0,1 % | ~0 | 0,0 % | 0,3× | 0,03 % | 1/5 | 0,55 / −69 % | **KILL** |
| S2 Carry de funding | Christin / Pindza 2026 | 2021-12 → 2026-10 | 3,54 | −13 (rinde < caja) | 1,00 | 0,00 / 1,00 | −0,3 % | −0,01 % | 0,9 % | 4,0× | 0,38 % | 3/5 | 0,55 / −69 % | **KILL** (WATCH como overlay) |
| S3 Donchian ensemble + vol | Zarattini et al. 2025 | 2022-12 → 2026-10 | 1,17 | 0,78 | 0,99 | 0,00 / 0,97 | −14,6 % | −1,18 % | 12,1 % | 10× | 1,2 % | 3/4 | 1,15 / −52 % | **WATCH** |
| S4 TSMOM 6h + ATR | Nguyen 2602.11708 | 2022-12 → 2026-10 | 0,49 | 0,38 | 0,83 | 0,00 / 0,62 | −62,3 % | −4,7 % | 12,1 % | 60× | 4,2 % | 3/4 | 1,20 / −51 % | **KILL** |
| S5 Lead-lag BTC→alts 1m | Kurihara & Matsumoto 2026 | 2024-10 → 2025-07 | −0,70 | −1,49 | 0,27 | 0,00 / 0,00 | −6,6 % | −0,68 % | −3,7 % | 45× | 8,8 % | 6/50 | 1,60 / −28 % | **KILL** |
| S6 Pares cointegrados | Danilo et al. 2024 | 2022-12 → 2026-10 | −1,08 | −1,48 | 0,02 | 0,00 / 0,00 | −42,9 % | −1,29 % | −10,7 % | 140× | 9,8 % | 1/4 | 1,20 / −51 % | **KILL** |

Con el doble de slippage (`_stress2x`): S1 pasa a Sharpe −0,61, S2 a 3,09, S3 a 1,15, S4 a 0,46
y S6 a −1,36. Las conclusiones no cambian.

**Ninguna estrategia llega a PROMOTE.** El DSR global es ~0 en todas, porque con 679 trials
y la dispersión de Sharpe entre trials, el máximo esperado por azar supera cualquier Sharpe OOS
observado.

## 5. Lectura por estrategia

- **S1 (base spot-perp).** Con costes taker VIP0, el round-trip de las dos patas es ~38 bps.
  Después de 2022, la desviación ρ casi nunca supera ese umbral: solo hay posición el 2 % de los
  días y el rendimiento es ~0. El Sharpe 3,35 que el paper da para "fees altas" usa comisiones
  *maker* (6,75/1,44 bps) y anualiza solo sobre las horas activas (convención de Lucca-Moench).
  El propio paper documenta que las desviaciones caen ~22 pp/año. Para un minorista taker este
  edge no existe en 2022-2026. KILL.
- **S2 (carry de funding).** El Sharpe es alto porque la volatilidad es mínima, pero la
  rentabilidad es ~0,9 % anual sobre el capital, por debajo de lo que rinde la caja (~4 %).
  El funding cobrado (1,16 %/año) apenas supera el coste de entrar y salir (0,38 %/año). En
  2 de 5 folds no hubo ninguna señal. Como estrategia aislada, KILL por eficiencia de capital.
  Podría servir como *overlay* para capital que ya está en spot (por ejemplo, cubrir parte
  cuando el funding medio de 21 prints supera 1 bp). Riesgos no modelados: contraparte del
  exchange (FTX), liquidación de la pata corta en subidas violentas y tipos de interés.
- **S3 (Donchian ensemble).** [Actualización ronda 2: con universo PIT (C7 en `strategies_convex_2026-10.md`) el Sharpe cae a 0,70 → KILL.] Es la única con algo de evidencia: Sharpe OOS 1,17 (0,78 en exceso
  de caja), DSR de familia 0,97, max DD −15 % frente a −52 % de BTC y 3 de 4 folds positivos.
  **Pero** el Sharpe es casi igual al de mantener BTC en el mismo periodo (1,15) y la rentabilidad
  es mucho menor (12 % frente a 53 % anual): su valor es reducir el drawdown, no generar alpha.
  El último fold (2025-26) es negativo, el universo de 10 monedas se eligió hoy (sesgo de
  supervivencia que favorece la estrategia) y el DSR global es 0. WATCH: candidata a paper
  trading con capital virtual. La siguiente prueba es un universo sin sesgo de supervivencia
  (top-N por volumen en cada fecha).
- **S4 (TSMOM 6h + ATR).** Max DD −62 %, peor que BTC. Turnover de 60× al año, coste 4,2 %/año
  y funding pagado −3,6 %/año. No reproduce en absoluto el Sharpe 2,41 del paper. KILL.
- **S5 (lead-lag 1m).** El efecto existe: en PIVX, la beta del retorno de BTC del minuto
  anterior es estable en ~0,33 en los 10 folds. Pero con comisión spot VIP0 de 10 bps + slippage,
  ningún umbral consigue que el retorno esperado del minuto siguiente supere el coste. ETH y LTC
  apenas operan; PIVX (−16 %) y GNO (−10 %) pierden; QKC (+9 %) gana en un periodo corto.
  El resultado del paper depende de una comisión del 0,02 %. KILL.
- **S6 (pares).** La cointegración entre grandes no es estable a 1h: turnover de 140× al año,
  coste 9,8 %/año y una pérdida bruta incluso antes de costes. KILL.

## 6. Tests de fuga (`tests/test_strategies_leakage.py`, 11 tests)

- Para cada estrategia, perturbar los datos posteriores a t no cambia ningún peso hasta t.
- El motor aplica el retraso obligatorio, cobra el coste en la barra correcta y rechaza
  `delay < 0`. El signo del funding es correcto (el corto cobra el funding positivo).
- Un oráculo deliberadamente con fuga (mira una barra adelante) da Sharpe > 20 sin retraso y
  pierde la ventaja con `delay = 1`, lo que demuestra que el test detecta fugas.
- El walk-forward respeta la purga (`assert_no_leakage`).

## 7. ¿En qué podría estar equivocado?

- **Sesgo de supervivencia** en el universo de 10 monedas actuales. Infla sobre todo S3.
- **Precios de cierre 1h** como precio de ejecución: no modelo el libro de órdenes. El
  slippage de 2 bps es razonable para BTC y ETH, pero optimista en crisis.
- **S1** usa un tipo de interés constante del 5 %, no la serie de Aave del paper.
- **Funding**: asigno cada pago a la hora redondeada. Binance ha cambiado intervalos (4h en
  algunos contratos) y el archivo lo refleja, pero no lo he verificado contrato a contrato.
- **Comisiones VIP0** tomadas del calendario público. Hay que comprobarlas en tu cuenta y
  jurisdicción antes de nada.
- **Grids pequeños** a propósito (para limitar el sobreajuste). Otra parametrización podría
  dar otros números, y por eso cualquier variante nueva debe registrarse en el trial registry.

## 8. Reproducir

```
# descargar datos (públicos) a /workspace/data_hist con los scripts dl.sh / dl1m.sh, o ajustar data.ROOT
python scripts/run_strategies.py                # todas, costes VIP0
python scripts/run_strategies.py --stress       # doble slippage
python scripts/run_strategies.py --summary-only # recalcular DSR/veredictos con el registro actual
pytest -q tests/test_strategies_leakage.py
```
