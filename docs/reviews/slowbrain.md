# Revisión adversarial — Slow brain / tradingview-mcp / Anthropic
*Revisor: mismo agente executor, pasada separada (sin sub-agente independiente disponible).* 2026-10-12

| ID | Severidad | Hallazgo | Estado |
|---|---|---|---|
| BRAIN-1 | Alta | El slow brain recibía **contexto vacío `{}`**; el proveedor devolvía valores "por defecto" que parecían normales. | **Arreglado**: `ExternalContext.update_features` alimenta `fast_features` + símbolo al `context_fn`; `paper` lo cablea. Test con proveedor espía. |
| BRAIN-2 | Media | `TradingViewMCP.call` lanza un subproceso `uvx` **por llamada** (lento, frágil; anula la caché de 60 s del servidor). | Abierto: sesión MCP persistente con reconexión. |
| BRAIN-3 | Media | Ninguna herramienta del MCP está mapeada todavía a campos del SlowState (vix_z vía `market_snapshot`, event_risk vía `financial_news`, etc.). | Abierto. |
| BRAIN-4 | Media | Proveedor Anthropic (`claude-opus-5-5`) no probado contra la API real (sin clave). | Abierto. |
| BRAIN-5 | Info | Decaimiento de confianza verificado: nunca sube m por encima del último valor; sin estado ⇒ 0.5. | OK. |
