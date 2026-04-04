"""Tests for signal-to-noise and rank_anneal interpretation warnings."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from spectral_primes.experiment import _warn_snr, permutation_test_sparse


def test_warn_few_zeros() -> None:
    with pytest.warns(UserWarning, match="Only 5 zeros"):
        _warn_snr(Lambda=20.0, x_hi=1_000_000, n_zeros_used=5)


def test_warn_low_lambda_high_x() -> None:
    with pytest.warns(UserWarning, match="truncation noise"):
        _warn_snr(Lambda=80.0, x_hi=100_000_000, n_zeros_used=50)


def test_no_warn_reasonable_regime() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _warn_snr(Lambda=200.0, x_hi=8_000_000, n_zeros_used=100)


@pytest.fixture
def small_gammas() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.sort(rng.uniform(10, 35, 40))


@mock.patch(
    "spectral_primes.experiment.prime_density_per_1e5",
    lambda c, w: float((c + w) % 251) / 10.0,
)
def test_permutation_warns_on_rank_anneal_interpretation(small_gammas: np.ndarray) -> None:
    with pytest.warns(UserWarning, match="not part of the original preprint"):
        permutation_test_sparse(
            small_gammas,
            Lambda=50.0,
            delta=80.0,
            n_t=24,
            x_lo=5000,
            x_hi=20000,
            n_per_group=6,
            window_half=100,
            n_perm=2,
            pool_factor=400,
            seed=1,
            rank_anneal=0.05,
            ref_grid_points=80,
        )
