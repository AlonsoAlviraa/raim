# Ronda 2: estrategias convexas en varios horizontes (2026-10)

Fecha: 2026-10-12 · Rama `strategies` · Código: `src/raim/strategies/convex.py`, `universe.py`,
`convex_metrics.py`, `scripts/run_convex.py` · Tests: `tests/test_convex_leakage.py` ·
Resultados: `docs/research/strategies_convex_2026-10/` (summary, un JSON por estrategia con los
folds y el registro de trials de las dos rondas).

**Solo backtest con capital virtual. No es consejo financiero ni demuestra rentabilidad.**

## 1. Objetivo

Buscar estrategias volátiles con **tasa de acierto baja pero ganancias grandes** (sesgo positivo,
pagos convexos) en cuatro horizontes: intradía (15-30 min), 1-4 h, diario y semanal. Además,
corregir el sesgo de supervivencia de S3 con un universo point-in-time (PIT) que incluye monedas
deslistadas.

## 2. Estrategias y fuentes (enlaces comprobados el 2026-10-12)

| # | Horizonte | Estrategia | Fuente | Tipo |
|---|---|---|---|---|
| C1 | 30 min | Momentum intradía: la primera media hora de la sesión predice la última | Shen, Urquhart, Wang, *Bitcoin intraday time-series momentum*, Financial Review 2022 (doi:10.1111/fire.12290); Wen, Bouri, Xu, Zhao, SSRN 4080253 | Peer-reviewed (2022) |
| C2 | 15 min | Ruptura de canal Donchian con trailing stop ATR (BTC/ETH/SOL perp) | Práctica estándar de trend following; stops adaptativos de Kaminski & Lo (2014), aplicados en Nguyen arXiv:2602.11708 | Practitioner |
| C3 | 4 h | *Squeeze* Bollinger dentro de Keltner + ruptura y trailing ATR (10 perps) | Practitioner (BBKC squeeze, PyQuantLab 2025). No encontré ningún paper académico 2024-26 sobre esto | Practitioner |
| C4 | 1 h / 24 h | Extremos de funding: reversión condicionada (Shek), "squeeze" siempre largo y fade del funding positivo | Shek, *Conditional Arbitrage Capacity*, SSRN 7520320 (sept. 2026); evidencia contraria en markettrace.ai (2026) | SSRN de autor independiente |
| C5 | 1 h | Proxy de cascada de liquidaciones (barra con \|r\| > kσ y volumen > 3× la mediana): continuación frente a reversión | Lim, *Anatomy of a Crypto Cascade*, SSRN 6579278 (2026); arXiv:2608.03616 (2026: la cascada se concentra al principio, el 88 % de las ventas forzadas en 30 min) | SSRN / arXiv |
| C6 | Semanal | TSMOM sobre altcoins con universo PIT, vol targeting y rebalanceo semanal | Han, Kang, Ryu, *Momentum in the Cryptocurrency Market…*, SSRN 4675565 (RAPS accepted; revisado sept. 2026) | Peer-reviewed (aceptado) |
| C7 | Diario | S3 (ensemble Donchian) **re-ejecutada sobre el universo PIT top-20** | Zarattini, Pagani, Barbon, SSRN 5209907 (2025) | SSRN |
| C8 | Diario (evento) | Momentum tras un nuevo listado en Binance spot | Ante (2019), RockawayX (2024), Empirica (2024): estudios practitioner que encuentran un pump inicial seguido de caída | Practitioner |

## 3. Método

Es el mismo harness de la ronda 1, con estos añadidos:

- **Universo PIT** (`universe.py`): todos los pares USDT spot de `data.binance.vision`
  (672 símbolos con datos 2021-2026, **179 deslistados** y 481 listados después de 2021-03).
  Excluyo stablecoins y tokens apalancados. En cada fecha, el universo son los top-N por mediana
  de volumen en USD de los 30 días anteriores (solo pasado), con al menos 60 días de historia y
  cotizando el día anterior. Un test destapó que un deslistado seguía en el universo ~15 días
  por la mediana móvil; está corregido.
- **Slippage escalonado por liquidez** (volumen pasado): ≥ 100 M$/día 3 bps, ≥ 10 M$ 10 bps,
  ≥ 1 M$ 30 bps y 80 bps por debajo, por lado. A esto se suman comisiones taker VIP0 (spot
  0,10 %, perp 0,05 %) y el funding real en perps. Pasada de estrés con el doble de slippage.
- **Retraso obligatorio** de una barra extra en todas las estrategias. Walk-forward purgado y
  con embargo; el parámetro se elige solo con el train.
- **Registro de trials compartido** con la ronda 1: N global = **805** variantes (DSR global).
  El DSR de familia usa solo las variantes de cada estrategia.
- **Métricas nuevas** (`convex_metrics.py`): un trade es un tramo continuo de posición del mismo
  signo en un instrumento, con la comisión de salida incluida. Calculo win rate, payoff (ganancia
  media / pérdida media), profit factor, skew diario y por trade, % del PnL que viene del 5 %
  de mejores trades, racha perdedora más larga, tiempo máximo bajo el agua (días) y CVaR 5 %.
- **Veredicto**: la regla de la ronda 1 **más** skew diario > 0 **y** payoff ≥ 2, tanto para
  PROMOTE como para WATCH.

## 4. Resultados OOS (netos de comisiones, slippage y funding; costes base VIP0)

| Estrategia | OOS | Sharpe | Sharpe −4 % | DSR global / fam. | Max DD | CVaR5 d | Rent. anual | Turnover/año | Trades | Win rate | Payoff | PF | Skew d / trade | Top 5 % del PnL | Racha perd. | Días bajo agua | Folds + | Veredicto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 intradía 30m | 2022-12→2026-10 | −2,69 | −4,09 | 0 / 0 | −25,6 % | −0,47 % | −7,4 % | 80× | 456 | 27 % | 0,83 | 0,31 | −5,4 / −2,2 | n/a (PnL<0) | 25 | 1394 | 0/4 | **KILL** |
| C2 Donchian 15m | 〃 | 0,16 | 0,03 | 0 / 0,08 | −45,4 % | −2,9 % | 0,1 % | 210× | 1203 | 33 % | 2,08 | 1,04 | 1,9 / 2,6 | 1403 %* | 20 | 688 | 3/4 | **KILL** |
| C3 Squeeze 4h | 〃 | 0,09 | −0,02 | 0 / 0,25 | −57,7 % | −4,7 % | −3,6 % | 51× | 999 | 35 % | 1,93 | 1,04 | −0,2 / 3,4 | 1229 %* | 19 | 660 | 1/4 | **KILL** |
| C4 Extremos de funding | 〃 | 0,50 | 0,37 | 0 / 0,15 | −47,9 % | −3,8 % | 11,6 % | 103× | 656 | 55 % | 0,98 | 1,22 | 2,5 / −0,3 | 170 % | 12 | 604 | 2/4 | **KILL** |
| C5 Cascada 1h | 〃 | 0,02 | −0,16 | 0 / 0,10 | −36,3 % | −2,1 % | −2,1 % | 112× | 716 | 49 % | 1,05 | 1,01 | 5,5 / −1,4 | 3342 %* | 14 | 1177 | 1/4 | **KILL** |
| C6 TSMOM alts semanal (PIT) | 2022-12→2026-10 | 0,87 | 0,70 | 0 / 0,80 | −28,7 % | −2,9 % | 19,3 % | 11× | 625 | 42 % | 2,38 | 1,70 | −0,2 / 2,5 | 110 % | 15 | 406 | 3/4 | **KILL** (no cumple skew > 0; si no, sería WATCH) |
| C7 = S3 sobre PIT top-20 | 〃 | 0,70 | 0,20 | 0 / 0,88 | −14,2 % | −0,97 % | 5,4 % | 10× | 1638 | 29 % | 3,44 | 1,44 | 0,1 / 6,4 | 252 % | 45 | 654 | 3/4 | **KILL** (Sharpe −4 % < 0,5) |
| C8 Nuevos listados | 2022-12→2026-10 | −1,38 | −1,71 | 0 / 0 | −50,5 % | −1,7 % | −16,1 % | 1,7× | 178 | 29 % | 0,89 | 0,37 | −1,1 / 1,5 | n/a | 13 | 658 | 0/4 | **KILL** |

\*Si el top 5 % supera el 100 %, todo el beneficio sale de unos pocos trades y el resto pierde
en neto. Las cifras de varios miles de % aparecen cuando el PnL total es casi 0.

En el mismo periodo, BTC buy-and-hold tiene Sharpe 1,15-1,20 y max DD −51 a −52 %.

Con el doble de slippage: C1 −2,76, C2 0,02, C3 0,06, C4 0,43, C5 −0,08, C6 0,83, C7 0,61
y C8 −1,52. Ningún veredicto cambia.

**Resultado: 0 PROMOTE, 0 WATCH y 8 KILL.** El DSR global es ~0 en todas (805 trials).

## 5. Lectura honesta

- **S3 tenía sesgo de supervivencia.** En la ronda 1, con las 10 grandes de hoy, S3 daba Sharpe
  1,17. Con el universo PIT y slippage por liquidez (C7, mismas reglas y mismo grid) baja a
  **0,70** (0,20 en exceso de caja) y la rentabilidad cae del 12 % al 5 % anual. Sí tiene el
  perfil convexo buscado (win rate 29 %, payoff 3,4 y el 5 % de trades aporta 2,5 veces el PnL
  total), pero no deja suficiente después de costes. **S3 pasa de WATCH a KILL.**
- **C6 (TSMOM de altcoins semanal)** es la mejor de la ronda: Sharpe 0,87, rentabilidad 19 %
  anual, DD −29 % (frente a −52 % de BTC), payoff 2,4 y 3 de 4 folds positivos. No llega a WATCH
  porque su skew diario es ligeramente negativo (−0,19), porque sigue por debajo del Sharpe de BTC
  y por el DSR. El último fold (2025-26) es negativo (−0,12). Es lo único que merece una tercera
  prueba (por ejemplo, combinarlo con un filtro de régimen de BTC), registrando cada variante.
- **Patrón común.** En C2, C3, C4, C6 y C7, **el fold de 2025-26 es negativo**. Las señales de
  tendencia y squeeze funcionaron en 2023-24 y se han degradado desde entonces.
- **C4 (funding).** El walk-forward eligió "comprar siempre en funding muy negativo", no la
  versión condicional de Shek. Los dos primeros folds son buenos y los dos últimos negativos. El
  resultado coincide con la crítica de que el edge se concentra en capitulaciones antiguas. Win
  rate 55 % y payoff ~1: no es convexa.
- **C5 (cascadas).** La versión de reversión rápida (4 h) es la que más se eligió, coherente con
  arXiv:2608.03616, pero tras costes no queda nada.
- **C1 (intradía).** El edge es negativo incluso antes de costes (−0,9 bps de media por trade en
  BTC). Lo publicado para 2013-2020 no se mantiene en 2022-2026.
- **C8 (listados).** El momentum tras el listado pierde dinero de forma sistemática (−16 %/año),
  coherente con los estudios practitioner de "pump and dump". Con spot no se puede vender en
  corto, y la versión corta con perps exigiría que el perp exista desde el día 1 (no lo he probado).

## 6. Tests (`tests/test_convex_leakage.py`, 9 tests)

- Para C1, C2, C3, C4, C5, C6 y C7, perturbar los datos posteriores a t no cambia ningún peso
  hasta t. C1 además comprueba que solo se tiene posición en la última media hora.
- El universo PIT solo usa datos pasados, exige historia mínima tras el listado y saca al
  deslistado al día siguiente.
- C8 solo usa información del cierre del día 1.
- Tests unitarios de las métricas de trade (win rate, payoff, racha y días bajo el agua).
- La suite completa pasa (ver commit).

## 7. ¿En qué podría estar equivocado?

- **Deslistados**: si una moneda se deslista con posición abierta, la última barra cuenta y el
  remanente se da por vendido a ese precio. En la realidad suele ser peor, así que C6, C7 y C8
  pueden estar algo inflados.
- **El archivo solo incluye pares que existieron en Binance**: hay sesgo de exchange, pero no de
  supervivencia dentro de Binance.
- **C1 usa precios spot** con costes de perp. **C4 y C5** usan un peso de 0,3 del capital por
  evento. Sin datos de open interest, la cascada es un proxy de precio y volumen, no
  liquidaciones reales.
- **Grids pequeños** a propósito. Otras variantes podrían dar otros números, pero también
  aumentarían N y endurecerían el DSR.

## 8. Reproducir

```
# Datos: lista S3 de data.binance.vision → urls → descarga paralela (ver scripts/data/)
python scripts/run_convex.py               # C1-C8, costes VIP0
python scripts/run_convex.py --stress      # doble slippage
python scripts/run_convex.py --summary-only
pytest -q tests/test_convex_leakage.py
```
