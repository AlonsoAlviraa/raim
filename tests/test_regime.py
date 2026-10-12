import numpy as np

from conftest import mod
from raim.regime.blend import Hysteresis, blend_params, conservative_posterior
from raim.regime.bocpd import BOCPD
from raim.regime.hmm_fit import fit_hmm, tilt_transition


def _sample(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    means = np.array([[-1.5, -1.5], [-0.5, 0.0], [0.8, 0.8], [2.5, 2.5]])
    A = np.full((4, 4), 0.01 / 3); np.fill_diagonal(A, 0.99)
    s, xs, ss = 0, [], []
    for _ in range(n):
        s = rng.choice(4, p=A[s])
        xs.append(means[s] + 0.3 * rng.standard_normal(2)); ss.append(s)
    return np.array(xs), np.array(ss), means


def test_baum_welch_recovers_ordered_states():
    x, s, means = _sample()
    p = fit_hmm(x, 4, seed=0)
    np.testing.assert_allclose(p.means, means, atol=0.25)
    assert np.all(np.diag(p.trans) > 0.9)
    assert np.allclose(p.trans.sum(1), 1)


def test_forward_filter_tracks_truth(backend):
    x, s, _ = _sample(1)
    p = fit_hmm(x[:1500], 4, seed=0)
    f = mod("hmm", backend).HMMFilter(*p.as_lists())
    pred = np.array([np.argmax(f.step(list(r))) for r in x[1500:]])
    assert (pred == s[1500:]).mean() > 0.9


def test_filter_is_causal(backend):
    x, _, _ = _sample(2, 400)
    p = fit_hmm(x, 4, seed=0)
    H = mod("hmm", backend).HMMFilter
    f1, f2 = H(*p.as_lists()), H(*p.as_lists())
    a = [f1.step(list(r)) for r in x[:200]]
    b = [f2.step(list(r)) for r in x][:200]  # extra future data must not change the past
    np.testing.assert_allclose(a, b)


def test_tilt_only_toward_stress():
    A = np.full((4, 4), 0.05); np.fill_diagonal(A, 0.85)
    T = tilt_transition(A, vix_z=3, event_risk=0.5)
    assert np.allclose(T.sum(1), 1) and np.all(T[:, 2:] >= A[:, 2:] - 1e-12)
    assert np.allclose(tilt_transition(A, -5, 0.0), A)


def test_conservative_bias_and_blend():
    p = np.array([0.7, 0.3, 0.0, 0.0])
    q = conservative_posterior(p, 0.15, False)
    assert q[2:].sum() > 0 and np.isclose(q.sum(), 1)
    params = {"R1": [1, 1, 1], "R2": [2, 1.6, 0.6], "R3": [3, 2.5, 0.4], "R4": [6, 5, 0]}
    g0, s0, z0 = blend_params(p, params)
    g1, s1, z1 = blend_params(q, params)
    assert g1 >= g0 and s1 >= s0 and z1 <= z0


def test_hysteresis():
    h = Hysteresis(0.7, 0.4)
    seq = [0.5, 0.71, 0.6, 0.45, 0.39, 0.6]
    assert [h.update(x) for x in seq] == [False, True, True, True, False, False]


def test_bocpd_detects_shift():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 300), rng.normal(5, 1, 100)]
    d = BOCPD(hazard=0.01)
    probs = np.array([d.update(v) for v in x])
    assert probs[300:310].max() > 0.5
    assert np.mean(probs[50:290] > 0.5) < 0.05
