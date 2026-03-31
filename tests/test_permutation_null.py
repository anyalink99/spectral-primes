"""Guards for degenerate γ-shuffle null when rank_anneal=0."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from spectral_primes.experiment import permutation_test_sparse


@pytest.fixture
def small_gammas() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.sort(rng.uniform(10, 35, 40))


def test_permutation_rejects_rank_anneal_zero_without_flag(small_gammas: np.ndarray) -> None:
    with pytest.raises(ValueError, match="degenerate"):
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
            rank_anneal=0.0,
            allow_degenerate_null=False,
            ref_grid_points=80,
        )


@mock.patch(
    "spectral_primes.experiment.prime_density_per_1e5",
    lambda c, w: float((c + w) % 251) / 10.0,
)
def test_permutation_accepts_rank_anneal_zero_with_flag(small_gammas: np.ndarray) -> None:
    out = permutation_test_sparse(
        small_gammas,
        Lambda=50.0,
        delta=80.0,
        n_t=24,
        x_lo=5000,
        x_hi=20000,
        n_per_group=6,
        window_half=100,
        n_perm=4,
        pool_factor=400,
        seed=2,
        rank_anneal=0.0,
        allow_degenerate_null=True,
        ref_grid_points=80,
    )
    assert "p_two_sided" in out
    assert out["rank_anneal"] == 0.0
    assert out["n_perm"] == 4
