import numpy as np

from spectral_primes.operator import U_batch as U_full_batch
from spectral_primes.subset import (
    energies_on_interval,
    relative_var_weights,
    subset_mask,
    U_sparse_at,
)


def test_relative_var_weights_sum_to_one():
    E = np.array([0.1, 0.3, 0.6])
    V = relative_var_weights(E)
    assert abs(V.sum() - 1.0) < 1e-12


def test_subset_mask_includes_at_least_one():
    N = 5
    Var = np.ones(N) / N  # none strictly > 1/N
    m = subset_mask(Var, N)
    assert m.sum() >= 1


def test_rank_anneal_makes_shuffle_change_U_sparse():
    rng = np.random.default_rng(0)
    g = np.sort(rng.uniform(10, 40, size=12))
    Lambda = 50.0
    delta = 50.0
    x0 = 1_000_000.0
    n_t = 32
    ra = 0.08
    u0, _, _ = U_sparse_at(x0, g, Lambda, delta, n_t, rank_anneal=ra)
    gp = g.copy()
    rng.shuffle(gp)
    u1, _, _ = U_sparse_at(x0, gp, Lambda, delta, n_t, rank_anneal=ra)
    assert abs(u0 - u1) > 1e-10


def test_U_sparse_invariant_under_gamma_shuffle_when_rank_anneal_zero():
    rng = np.random.default_rng(2)
    g = np.sort(rng.uniform(10, 40, size=10))
    gp = g.copy()
    rng.shuffle(gp)
    x0 = 900_000.0
    u0, _, _ = U_sparse_at(x0, g, 50.0, 40.0, 24, rank_anneal=0.0)
    u1, _, _ = U_sparse_at(x0, gp, 50.0, 40.0, 24, rank_anneal=0.0)
    assert abs(u0 - u1) < 1e-9


def test_full_U_invariant_under_gamma_shuffle():
    """Plain U is sum f(gamma); order of summation unchanged by permuting the array."""
    rng = np.random.default_rng(1)
    g = np.sort(rng.uniform(10, 30, size=8))
    gp = g.copy()
    rng.shuffle(gp)
    x = np.array([500_000.0])
    L = 40.0
    assert np.allclose(U_full_batch(x, g, L), U_full_batch(x, gp, L))


def test_energies_positive():
    g = np.array([14.134725, 21.022040, 25.010858])
    x0 = 500_000.0
    gg, E, _slots = energies_on_interval(x0, 200.0, g, Lambda=80.0, n_t=24)
    assert gg.size == E.size
    assert np.all(E >= 0)
