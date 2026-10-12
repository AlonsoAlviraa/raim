# Revisión adversarial — Régimen / HMM
*Revisor: el mismo agente executor (Grok Bot) en una pasada de revisión aparte. No hubo sub-agente independiente (no había herramienta de sub-agentes ni CloudAgent en esta sesión).* 2026-10-12

| ID | Severidad | Hallazgo | Estado |
|---|---|---|---|
| REGIME-1 | Alta | `tilt_transition` existía pero **no estaba conectado**: el contexto lento (vix_z, event_risk) nunca inclinaba el HMM. | **Arreglado**: `run_sim(slow_context=...)` → `hmm.set_transition(tilt(...))`; test `test_regime1_slow_context_tilts_toward_stress` (contexto negativo no afloja; contexto caliente ≥ estrés base). |
| REGIME-2 | Media | Los estados se ordenan por la media de observaciones estandarizadas; eso no garantiza la semántica R2=tendencia vs R3=vol. Las etiquetas pueden mentir. | **Abierto**: hay que validar con la atribución por régimen (markouts, VR, vol) en datos reales; posible etiquetado por features específicas (VR para R2, BV para R3, VPIN/Hawkes para R4). |
| REGIME-3 | Baja | Alarma BOCPD = masa en run-length < 5 con umbral 0.5: heurística no calibrada. | Abierto: calibrar tasa de falsas alarmas OOS. |
| REGIME-4 | Baja | Fit sobre el 30 % inicial: no estacionariedad intra-sesión. | Abierto: re-fit rodante dentro del walk-forward. |

## Fase 3 (2026-10-12)
- Baseline obligatorio de **dial de estrés continuo** (Thawait, arXiv:2610.05740) implementado y comparado en el mismo walk-forward purgado (`raim compare-regimes`). En los sintéticos (que tienen regímenes discretos *por construcción*) el HMM sale mejor en media, pero lo domina un solo fold. **Hay que repetirlo en datos reales**; REGIME-2 (etiquetas cruzadas) sigue abierto y conviene leerlo junto con el argumento de Thawait de que el "régimen discreto" puede ser una cuantización de un continuo.
- El spread no se usa solo: el índice usa spread, vol, VPIN, |OFI|, VR y Hawkes ("spread lies").
