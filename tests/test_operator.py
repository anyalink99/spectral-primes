import numpy as np
import pytest

from spectral_primes.operator import U_batch


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
