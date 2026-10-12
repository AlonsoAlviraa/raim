import importlib.util

import pytest

from raim.core import features_ref, governor_ref, hmm_ref, quoter_ref, simexchange_ref

HAS_CPP = importlib.util.find_spec("raim_core") is not None
BACKENDS = ["python"] + (["cpp"] if HAS_CPP else [])


def mod(name: str, backend: str):
    if backend == "cpp":
        import raim_core
        return raim_core
    return {"features": features_ref, "governor": governor_ref, "hmm": hmm_ref,
            "quoter": quoter_ref, "sim": simexchange_ref}[name]


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param
