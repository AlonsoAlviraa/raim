# AGENTS.md — reglas para agentes que trabajen en RAIM

Mismo estilo que `TRAD/Pruebas`. Léelo antes de tocar código.

## Principios
1. **Preservar capital y edge verificable** por encima de backtests bonitos.
2. **Features causales**: un valor en t solo usa eventos con timestamp ≤ t. Cualquier
   feature nueva necesita un test de "el futuro no cambia el pasado" (ver `tests/test_leakage.py`).
3. **Validación**: walk-forward purgado + embargado; fit (HMM, estandarizador) solo en train.
4. **Costes**: siempre fees + slippage conservador + latencia (y stress 2×). Sin costes, no hay resultado.
5. **Honestidad OOS**: reporta números reales, incluidos los malos. Nunca afirmes rentabilidad.
6. **Sin datos reales fabricados**: sintéticos solo en tests y marcados como tales.
7. **Live/paper con dinero fuera de alcance** hasta pasar gates OOS + stress y aprobación explícita.
   `execution/guard.py` debe seguir haciendo hard-fail.

## Riesgo (no negociable)
- Límites duros deterministas en `RiskGovernor` (C++ y Python, misma semántica).
- Multiplicador aplicado = `min(m_llm, m_rules, m_jev)`; los externos solo aprietan.
  Cualquier cambio debe mantener verde `tests/test_risk.py` en **ambos** backends.
- Nunca un LLM/Jev en el bucle de ticks ni decidiendo precios/órdenes.

## Código
- Funciones puras + `@dataclass(frozen=True)` para configs; módulos pequeños; type hints.
- `logging`, nunca `print`. Identificadores en inglés, documentación en español.
- Hot path: cambia `src/raim/core/*_ref.py` **y** `cpp/src/core.hpp` a la vez y mantén
  `tests/test_parity.py` verde.
- Multiplataforma (Windows/PowerShell primero): `pathlib`, sin llamadas POSIX.
- Commits etiquetados por rol: `[data]`, `[features]`, `[regime]`, `[quoting]`, `[risk]`,
  `[sim]`, `[research]`, `[slowbrain]`, `[jev]`, `[cpp]`, `[ops]`, `[docs]`, `[tests]`.

## Antes de dar algo por hecho
`uv run pytest -q` verde con `RAIM_BACKEND=python` y (si compilado) `RAIM_BACKEND=cpp`.
Pregúntate: ¿edge positivo tras fees, selección adversa, slippage y cola realista? ¿predictivo OOS?
¿funciona fuera de R1? ¿sobrevive caídas de feed, picos de vol, inventario adverso, rupturas?
¿el slow brain está aislado y nunca aflojó un límite? **¿En qué podría estar equivocado?**
