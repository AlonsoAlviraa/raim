# 00 — Entendimiento del problema, principios y restricciones

*Autor: agente executor (Grok Bot), 2026-10-12. Documento vivo.*

## Qué es RAIM (y qué no es)
RAIM es un **sistema de investigación** para market making adaptativo a régimen y consciente
del inventario. Su producto no es dinero sino **evidencia**: ¿existe edge neto, fuera de
muestra, después de comisiones, selección adversa, slippage y colas realistas? Si la respuesta
es "no" o "no sabemos", el sistema ha cumplido su función al decirlo.

No es un bot de trading en producción, no es HFT de microsegundos (Python orquesta, la red
añade decenas de ms) y no tiene camino de órdenes reales.

## Principios rectores (orden de prioridad)
1. **Preservar capital** > edge verificable > rendimiento > elegancia.
2. **Determinismo en el camino rápido**: ningún LLM ni modelo externo decide precios,
   tamaños, envíos o cancelaciones. Los modelos externos solo emiten un multiplicador en [0,1].
3. **Monotonía del riesgo**: `m = min(m_llm, m_rules, m_jev)`. Los inputs externos solo
   aprietan; nunca aflojan, nunca levantan un halt, nunca abren exposición.
4. **Causalidad**: toda feature, posterior o decisión en t usa solo información disponible en t
   *en el reloj del receptor* (no en el del exchange).
5. **Validación honesta**: fit solo en train; walk-forward purgado+embargado; stress 2× latencia;
   markouts; atribución por régimen. Reportar números malos tal cual.
6. **Primer gate**: `maker_fee + E[AS] < spread/2`. Si falla, se cambia venue/instrumento; no se
   "optimiza" hasta que pase.
7. **Fallo conservador**: dato ausente/obsoleto/contradictorio ⇒ menos riesgo, nunca más.

## Restricciones duras
- Modo SIMULATION/PAPER. `live` hace hard-fail aunque haya flag + confirmación por entorno.
- tradingview-mcp: polled, caché 60 s, ≤4 concurrentes, ≥0.8 s entre llamadas, sin order book,
  sin tape, sin órdenes. Nunca en el bucle de ticks. Solo paquete oficial PyPI.
- Jev: 70–500 ms, fuera del bucle de ticks, fallback conservador ante timeout/error.
- Claves solo por entorno; `.env` nunca en git.
- Multiplataforma (Windows del usuario sin compilador): el backend C++ es opcional.
- Legalidad/comisiones a verificar por el usuario; prohibido spoofing/layering/wash trading.

## Modelo mental del edge
Ganancia esperada por fill ≈ medio spread capturado − comisión − selección adversa − slippage
− coste de inventario. En BTCUSDT spot el spread típico es 1 tick (~0.0014 bps): con 1 bps de
comisión maker el gate falla por órdenes de magnitud. Cualquier "beneficio" en simulación con
quotes a ≥0.5 bps del mid proviene de barridos (trade-throughs), que son exactamente los fills
con más selección adversa. Por eso la métrica clave es el **markout**, no el PnL.

## ¿En qué podría estar equivocado? (lista adversarial)
- **Relojes**: mezclar tiempo de exchange (trades) y de recepción (depth parcial) crea
  look-ahead sutil en markouts y fills.
- **Cola**: suponer que estamos primeros en un nivel detrás del touch es optimista; la
  profundidad L2 visible debe usarse.
- **Etiquetado de regímenes**: ordenar estados del HMM por "media de observaciones" no garantiza
  semántica R2 (tendencia) vs R3 (vol). Los nombres pueden mentir.
- **No estacionariedad**: parámetros de 30 % inicial pueden no valer al final de la sesión.
- **Datos insuficientes**: 60 s de datos no permiten ninguna conclusión; ni el gate debería
  considerarse estable con n pequeño.
- **Degeneración numérica**: γ o κ ≤ 0, σ = 0, equity NaN ⇒ precios NaN o halts que no saltan.
- **Contexto vacío**: si el slow brain / Jev reciben estado vacío, devuelven valores por defecto
  que parecen "normales" y podrían relajar (dentro de [0,1]) lo que debería ser conservador.
- **Coste operativo del MCP**: lanzar un subproceso `uvx` por llamada es lento y frágil.
- **Inventario tras halt**: un halt cancela quotes pero deja la posición abierta.
- **Sesgo de supervivencia/venue**: elegir el venue *después* de ver resultados es overfitting.
- **Simulación ≠ realidad**: latencias, rechazos y prioridad de cola son supuestos.
