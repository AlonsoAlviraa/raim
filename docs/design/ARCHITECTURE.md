# Arquitectura de RAIM

```mermaid
flowchart LR
  subgraph FAST["FAST BODY — determinista, sin LLM (C++ raim_core o Python ref)"]
    WS["Binance WS público<br/>depth20@100ms + trade"] --> REC["recorder / replay<br/>(JSONL, reloj de recepción)"]
    REC --> FE["FeatureEngine<br/>microprice, OFI, BV vol, VPIN, VR, Hawkes n"]
    FE --> HMM["HMMFilter 4 estados<br/>+ BOCPD (alarma)"]
    HMM --> BL["blend θ=Σp·θ_k<br/>sesgo R3/R4 + histéresis"]
    BL --> Q["quote() AS/GLFT<br/>skew inventario, post-only"]
    Q --> GOV["RiskGovernor<br/>límites duros, m=min(m_llm,m_rules,m_jev)"]
    GOV --> OMS["paper OMS"] --> SIM["SimExchange L2<br/>cola, parciales, latencia, fees"]
    SIM -->|fills| GOV
  end
  subgraph SLOW["SLOW BRAIN — async, ≤1 llamada/s, nunca en el bucle de ticks"]
    MCP["tradingview-mcp oficial<br/>(uvx, polled)"] --> LLM["LLM Anthropic<br/>(claude-opus-5-5, configurable)"]
    LLM --> SS["SlowState (pydantic)<br/>m∈[0,1], c, vix_z, event_risk…"]
    JEV["Jev /v1/systemone<br/>noul+choice, 70–500 ms"] --> MJ["m_jev"]
  end
  FE -. features (on_features) .-> JEV
  FE -. fast_features .-> LLM
  SS -. "m_llm (decae a ≤0.5)" .-> GOV
  SS -. "vix_z, event_risk (solo inclina a estrés)" .-> HMM
  MJ -. "m_jev (fallback 0.5)" .-> GOV
  GUARD["execution/guard.py<br/>live ⇒ hard-fail"] -.-> OMS
  RES["research: walk-forward purgado+embargado,<br/>markouts 1/5/30 s, atribución, gate"] --- SIM
  AUD["audit JSONL append-only<br/>run_id + hash config"] --- GOV
```

## Invariantes
1. Las flechas punteadas solo transportan **multiplicadores o contexto**; ninguna lleva precios ni órdenes.
2. `m_aplicado = min(m_llm, m_rules, m_jev)` ∈ [0,1]; NaN ⇒ 0; halt latched.
3. Todo el tiempo del motor es **reloj de recepción**.
4. Si el slow brain o Jev caen: m_llm decae hacia ≤0.5, m_jev = 0.5; el fast body sigue.
5. No existe ejecutor live.

## Módulos
| Capa | Python | C++ |
|---|---|---|
| Features | `core/features_ref.py` | `FeatureEngine` |
| Filtro HMM | `core/hmm_ref.py` (fit: `regime/hmm_fit.py`) | `HMMFilter` |
| Quoter | `core/quoter_ref.py` | `quote` |
| Riesgo | `core/governor_ref.py` | `RiskGovernor` |
| Simulador | `core/simexchange_ref.py` (+ `sim/engine.py`) | `SimExchange` |
| Slow brain | `slowbrain/*`, `runtime.py` | — |
| Jev | `jev/adapter.py` | — |
| Investigación | `research/*` | — |
