"""Backend selection for the fast-body hot path.

Uses the compiled C++ extension `raim_core` when importable, otherwise the pure-Python
reference implementation. Both expose the same API. Override with RAIM_BACKEND=python|cpp.
"""
from __future__ import annotations

import logging
import os

from raim.core import features_ref, flowcpd_ref, governor_ref, hmm_ref, quoter_ref, simexchange_ref
from raim.core.features_ref import FEATURE_NAMES

log = logging.getLogger(__name__)

_want = os.environ.get("RAIM_BACKEND", "auto").lower()
_cpp = None
if _want in ("auto", "cpp"):
    try:
        import raim_core as _cpp  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on build
        if _want == "cpp":
            raise ImportError("RAIM_BACKEND=cpp but raim_core is not built") from exc
        _cpp = None

if _cpp is not None:
    BACKEND = "cpp"
    FeatureEngine = _cpp.FeatureEngine
    HMMFilter = _cpp.HMMFilter
    RiskGovernor = _cpp.RiskGovernor
    SimExchange = _cpp.SimExchange
    quote = _cpp.quote
    exposure_adjust = _cpp.exposure_adjust
    FlowCPD = _cpp.FlowCPD
else:
    BACKEND = "python"
    FeatureEngine = features_ref.FeatureEngine
    HMMFilter = hmm_ref.HMMFilter
    RiskGovernor = governor_ref.RiskGovernor
    SimExchange = simexchange_ref.SimExchange
    quote = quoter_ref.quote
    exposure_adjust = quoter_ref.exposure_adjust
    FlowCPD = flowcpd_ref.FlowCPD

log.info("RAIM fast-body backend: %s", BACKEND)

__all__ = ["BACKEND", "FEATURE_NAMES", "FeatureEngine", "HMMFilter", "RiskGovernor",
           "SimExchange", "quote", "exposure_adjust", "FlowCPD"]
