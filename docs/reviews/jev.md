# Revisión adversarial — Adaptador Jev (TypeSafe)
*Revisor: mismo agente executor, pasada separada (sin sub-agente independiente disponible).* 2026-10-12

| ID | Severidad | Hallazgo | Estado |
|---|---|---|---|
| JEV-1 | Alta | Jev recibía `self.features` que **nunca se actualizaba** ⇒ estado vacío; el mock devolvía p_toxic=0 ⇒ m_jev=1 (sin efecto protector). | **Arreglado**: `run_sim(on_features=...)` publica features con nombre + posterior de régimen + posición cada 10 books. |
| JEV-2 | Media | Respuestas mal formadas: `noul` fuera de [0,1] o no finito, falta la pregunta `regime`, probabilidades incompletas. | **Arreglado**: `parse_answers` valida, recorta, completa R1..R4 y renormaliza; no finito o pregunta ausente ⇒ fallback conservador. |
| JEV-3 | Media | Cliente HTTP probado solo con transporte simulado; preguntas no calibradas; el presupuesto de 500 ms incluye TLS en la primera llamada. | Abierto (requiere `TYPESAFE_API_KEY`). |
