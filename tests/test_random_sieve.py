import numpy as np

from spectral_primes.random_sieve import (
    build_cramer_field,
    cramer_count_interval,
    cramer_probability,
    compare_prime_vs_cramer,
)


def test_cramer_probability_small():
    assert cramer_probability(1) == 0.0
    assert cramer_probability(2) == min(1.0, 1.0 / np.log(2))


def test_prefix_matches_direct():
    x_min, x_max = 50, 120
    seed = 7
    pref, xmin = build_cramer_field(x_min, x_max, seed)
    assert xmin == x_min
    rng = np.random.default_rng(seed)
    direct = []
    for n in range(x_min, x_max):
        p = cramer_probability(n)
        direct.append(rng.random() < p)
    assert pref[-1] == sum(direct)
    for lo, hi in [(55, 70), (x_min, x_max), (60, 61)]:
        s = sum(direct[lo - x_min : hi - x_min])
        assert cramer_count_interval(pref, x_min, lo, hi) == s


def test_compare_runs():
    r = compare_prime_vs_cramer(
        x_lo=5000,
        x_hi=8000,
        half_width=80,
        n_windows=15,
        seed=0,
        universes=10,
    )
    assert r.n_windows == 15
    assert 0.0 <= r.empirical_p_real_exceeds_random <= 1.0
    assert r.real_mean > 0
