"""
Cramér-style «random sieve»: each integer n >= 2 is marked independently with
probability 1 / ln n (and 0 for n < 2). This is a classical stochastic null for
prime-like sparsity; it ignores divisibility structure.

One independent realization is built on [x_min, x_max) with a prefix-sum array
for O(1) window counts.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from spectral_primes.primes import prime_density_per_1e5


def cramer_probability(n: int) -> float:
    if n < 2:
        return 0.0
    return min(1.0, 1.0 / float(np.log(n)))


def build_cramer_field(x_min: int, x_max: int, seed: int) -> tuple[np.ndarray, int]:
    """
    Half-open range [x_min, x_max). Returns (prefix, x_min) with
    prefix[k] = number of pseudo-primes in [x_min, x_min + k), prefix[0] = 0.
    """
    if x_max <= x_min:
        raise ValueError("need x_max > x_min")
    length = x_max - x_min
    rng = np.random.default_rng(seed)
    n_arr = np.arange(x_min, x_max, dtype=np.float64)
    p = np.where(n_arr >= 2.0, 1.0 / np.log(n_arr), 0.0)
    p = np.minimum(p, 1.0)
    flags = rng.random(length) < p
    pref = np.empty(length + 1, dtype=np.int64)
    pref[0] = 0
    np.cumsum(flags, dtype=np.int64, out=pref[1:])
    return pref, x_min


def cramer_count_interval(pref: np.ndarray, x_min: int, lo: int, hi: int) -> int:
    """Count pseudo-primes in [lo, hi) (half-open)."""
    if hi <= lo:
        return 0
    i0 = lo - x_min
    i1 = hi - x_min
    if i0 < 0 or i1 > pref.size - 1:
        raise ValueError("interval outside precomputed field")
    return int(pref[i1] - pref[i0])


def cramer_density_per_1e5(
    pref: np.ndarray,
    x_min: int,
    center: int,
    half_width: int,
) -> float:
    lo = center - half_width
    hi = center + half_width
    width = max(hi - lo, 1)
    c = cramer_count_interval(pref, x_min, lo, hi)
    return 1e5 * c / width


@dataclass
class CramerComparisonResult:
    n_windows: int
    half_width: int
    seed: int
    universes: int
    real_mean: float
    real_std: float
    random_mean_universe0: float
    random_std_within_universe0: float
    mean_of_universe_means: float
    std_of_universe_means: float
    empirical_p_real_exceeds_random: float


def compare_prime_vs_cramer(
    x_lo: int,
    x_hi: int,
    half_width: int,
    n_windows: int,
    seed: int,
    universes: int = 80,
) -> CramerComparisonResult:
    """
    Draw the same random window centres for all trials. For each universe,
    build one independent Cramér field covering all windows, then record mean
    pseudo-prime density (per 1e5) across windows. True primes use SymPy on
    the same centres.

    empirical_p_real_exceeds_random: fraction of universe *means* that are
    below real_mean (one-sided: are true primes denser on average than this
    random model in these windows?).
    """
    if universes < 2:
        raise ValueError("universes must be at least 2")
    rng = random.Random(seed)
    span_lo = x_lo + half_width
    span_hi = x_hi - half_width
    if span_hi <= span_lo:
        raise ValueError("x range too narrow for half_width")
    centers = [rng.randrange(span_lo, span_hi) for _ in range(n_windows)]
    w_lo = min(c - half_width for c in centers)
    w_hi = max(c + half_width for c in centers)
    x_min = min(w_lo, x_lo)
    x_max = max(w_hi, x_hi)

    real = np.array(
        [prime_density_per_1e5(c, half_width) for c in centers], dtype=np.float64
    )
    real_mean = float(real.mean())
    real_std = float(real.std(ddof=1)) if n_windows > 1 else 0.0

    universe_means = np.empty(universes, dtype=np.float64)
    random_mean_universe0 = 0.0
    random_std_within_universe0 = 0.0
    for u in range(universes):
        pref, xmin = build_cramer_field(x_min, x_max, seed + 10_000 + u)
        rnd = np.array(
            [cramer_density_per_1e5(pref, xmin, c, half_width) for c in centers],
            dtype=np.float64,
        )
        universe_means[u] = float(rnd.mean())
        if u == 0:
            random_mean_universe0 = universe_means[0]
            random_std_within_universe0 = float(rnd.std(ddof=1)) if n_windows > 1 else 0.0
    mean_of_um = float(universe_means.mean())
    std_of_um = float(universe_means.std(ddof=1))
    # one-sided: how often random *global mean* is below observed prime mean
    empirical_p = float(np.mean(universe_means < real_mean))

    return CramerComparisonResult(
        n_windows=n_windows,
        half_width=half_width,
        seed=seed,
        universes=universes,
        real_mean=real_mean,
        real_std=real_std,
        random_mean_universe0=random_mean_universe0,
        random_std_within_universe0=random_std_within_universe0,
        mean_of_universe_means=mean_of_um,
        std_of_universe_means=std_of_um,
        empirical_p_real_exceeds_random=empirical_p,
    )
