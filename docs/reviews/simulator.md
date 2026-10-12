# Revisión adversarial — Simulador / backtester
*Revisor: mismo agente executor, pasada separada (sin sub-agente independiente disponible).* 2026-10-12

| ID | Severidad | Hallazgo | Estado |
|---|---|---|---|
| SIM-1 | **Alta** | Una orden detrás del touch recibía `queue_ahead = 0` (primera de la cola): **optimista**. Solo se usaba top-of-book. | **Arreglado** (C++ y Python): `on_book_l2` con todos los niveles; cola = cantidad visible en nuestro nivel; nivel vacío dentro del rango = 0; más allá de la profundidad visible = cola infinita (solo llena por trade-through). |
| SIM-2 | **Alta** | Mezcla de relojes: trades con tiempo de exchange y depth parcial con tiempo de recepción ⇒ fills/markouts con timestamps incoherentes (posible look-ahead). | **Arreglado**: el motor usa siempre `recv_ms` (fallback `ts_ms`). Test con desfase de 10.037 s. |
| SIM-3 | Media | 22 % de los ticks sintéticos en `rate_backoff`: el tope de mensajes del OMS coincide con el umbral del governor ⇒ backoff auto-inducido. | Abierto: OMS con presupuesto < umbral y cancel/replace agrupado. |
| SIM-4 | Media | Sin liquidez oculta/icebergs, sin impacto propio, snapshots de profundidad cada 100 ms (cambios intra-snapshot invisibles). | Abierto; documentado. |
| SIM-5 | Baja | El stream de depth parcial no trae tiempo de exchange; se usa recepción. | Documentado. |
