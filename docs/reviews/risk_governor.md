# Revisión adversarial — Risk governor
*Revisor: mismo agente executor, pasada separada (sin sub-agente independiente disponible).* 2026-10-12

| ID | Severidad | Hallazgo | Estado |
|---|---|---|---|
| RISK-1 | **Crítica** | `equity=NaN` hacía que `peak - equity >= max_dd` fuese siempre falso: **el halt por drawdown no saltaba** y se seguía cotizando. `position=NaN` también pasaba. | **Arreglado** en C++ y Python: entradas no finitas ⇒ `cancel_all`, razón `bad_input`; NaN nunca toca peak/halt; un halt latched sigue latched con NaN. Tests en ambos backends. |
| RISK-2 | Alta | El halt cancela quotes pero **deja la posición abierta** (no hay unwind). | **Arreglado (fase 3, solo sim)**: `execution/unwind.py` (pasivo `unwind_passive_s` → taker con clip y límite de slippage, solo reduce), `SimExchange.take` recorre L2 con fee taker (C++ y Py). Test: halt ⇒ posición plana. |
| RISK-3 | Baja | `depth` para Qmax = suma top-5 de ambos lados, no el lado relevante para deshacer. | **Arreglado (fase 3)**: `unwind_depth` (largo→bids, corto→asks, plano→el peor). |
| RISK-4 | Info | Verificado: con m externo ∈ {−5, NaN, inf, …} nunca se afloja (test de rejilla en ambos backends). | OK. |
