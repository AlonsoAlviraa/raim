# Revisión adversarial — Quoting / microestructura / paridad C++
*Revisor: mismo agente executor, pasada separada (sin sub-agente independiente disponible).* 2026-10-12

| ID | Severidad | Hallazgo | Estado |
|---|---|---|---|
| QUOTE-1 | Alta | γ≤0 ⇒ `ZeroDivisionError` en Python e `inf/NaN` en C++; κ≤0, mid≤0/NaN ⇒ precios NaN que podían llegar al SimExchange. | **Arreglado**: el quoter devuelve NaN = "no cotizar"; el motor salta quotes no finitas. Tests en ambos backends. |
| QUOTE-2 | Media | σ procede de BV por update escalada con EWMA de dt: ruidosa; κ (intensidad de llegada) no se calibra con datos. | Abierto: calibrar κ por nivel con fills/trades reales. |
| QUOTE-3 | Media | Con BTCUSDT (spread 1 tick) el half-spread mínimo de 0.5 bps sitúa las quotes muy lejos del touch: solo hay fills por barridos (máxima selección adversa). | Abierto (de diseño); es coherente con el gate FAIL. |
| CPP-1 | Info | Paridad revisada tras cambios (features, HMM, quote, governor, SimExchange incl. L2): tests verdes. Comparaciones exactas de precio (`==`) idénticas en ambos lados porque los precios salen redondeados igual. | OK, vigilar si se cambia el redondeo. |
