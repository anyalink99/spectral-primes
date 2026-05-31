import numpy as np
import pytest

from spectral_primes.operator import U_batch, U_density_batch, U_psi_batch


def test_U_single_zero_matches_manual():
    gamma = np.array([14.134725], dtype=np.float64)
    Lambda = 100.0
    x = np.array([np.e])
    w = np.exp(-(gamma[0] ** 2) / (2 * Lambda**2))
    expected = w * np.cos(gamma[0] * 1.0) / gamma[0]
    got = U_batch(x, gamma, Lambda)
    assert got.shape == (1,)
    assert abs(got[0] - expected) < 1e-12


def test_U_empty_sum_when_Lambda_below_smallest_gamma():
    gamma = np.array([100.0, 200.0])
    x = np.array([2.0])
    got = U_batch(x, gamma, Lambda=50.0)
    assert got[0] == 0.0


def test_U_raises_on_nonpositive_x():
    with pytest.raises(ValueError):
        U_batch(np.array([0.0]), np.array([14.0]), 100.0)


def test_density_operator_is_minus_cos_no_inverse_gamma():
    gamma = np.array([14.134725, 21.022040], dtype=np.float64)
    Lambda = 100.0
    x = np.array([1234.5])
    w = np.exp(-(gamma**2) / (2 * Lambda**2))
    expected = -np.sum(w * np.cos(gamma * np.log(x[0])))
    got = U_density_batch(x, gamma, Lambda)
    assert abs(got[0] - expected) < 1e-12


def test_psi_operator_uses_sin_over_gamma():
    gamma = np.array([14.134725, 21.022040], dtype=np.float64)
    Lambda = 100.0
    x = np.array([1234.5])
    w = np.exp(-(gamma**2) / (2 * Lambda**2))
    expected = np.sum(w * np.sin(gamma * np.log(x[0])) / gamma)
    got = U_psi_batch(x, gamma, Lambda)
    assert abs(got[0] - expected) < 1e-12


def test_density_and_old_operator_differ_in_sign_phase():
    # On a grid, the corrected density operator and the manuscript operator are
    # genuinely different objects (not proportional).
    rng = np.random.default_rng(0)
    gamma = np.sort(rng.uniform(10, 70, size=15))
    x = np.linspace(1e6, 5e6, 200)
    d = U_density_batch(x, gamma, 80.0)
    u = U_batch(x, gamma, 80.0)
    corr = np.corrcoef(d, u)[0, 1]
    assert abs(corr) < 0.95
