"""
Run from repository root:

    python -m experiments.wild_theories.demo

Quick sanity checks (small n): twin mask vs noise, residues mod 4 along primes, scaling rows.
"""
from __future__ import annotations

import numpy as np

from .fourier import compare_to_random_trials, magnitude_spectrum
from .scaling import scaling_table
from .signals import (
    consecutive_prime_residues,
    empirical_prime_density,
    segment_prime_indicator,
    segment_twin_lower_indicator,
    twin_gap_indicator_along_primes,
)


def main() -> None:
    rng = np.random.default_rng(42)

    lo, hi = 2, 5000
    twin = segment_twin_lower_indicator(lo, hi)
    prime_mask = segment_prime_indicator(lo, hi)
    print("--- Twin lower-member indicator vs Bernoulli null (same segment) ---")
    print(compare_to_random_trials(twin, rng, n_trials=80, p_match=twin.mean()))
    print("--- Prime indicator vs Bernoulli null ---")
    print(compare_to_random_trials(prime_mask, rng, n_trials=80))

    print("\n--- Consecutive primes mod 4 (first 2000 gaps); FFT peak index (excluding DC) ---")
    res = consecutive_prime_residues(2000, 4)
    mag = magnitude_spectrum(res)
    if mag.size > 1:
        k = 1 + int(np.argmax(mag[1:]))
        print(f"peak k={k}, |mag|={mag[k]:.4f}, total energy={float(np.sum(mag**2)):.4f}")

    print("\n--- Twin gap indicator along primes (first 5000 gaps) ---")
    tg = twin_gap_indicator_along_primes(5000)
    print(
        "twin gap fraction",
        float(tg.mean()),
        "FFT energy",
        float(np.sum(magnitude_spectrum(tg) ** 2)),
    )

    print("\n--- Scaling table (small windows; extend in your own runs) ---")
    wins = [(10_000, 20_000), (100_000, 110_000), (1_000_000, 1_010_000)]
    for kind in ("prime", "twin"):
        rows = scaling_table(wins, kind=kind)
        print(kind)
        for r in rows:
            print(
                f"  [{r.lo}, {r.hi}) n={r.n} peak_k={r.peak_k} peak={r.peak_mag:.4f} E={r.total_energy:.4f}"
            )
        p = empirical_prime_density(wins[0][0], wins[0][1])
        print(f"  empirical pi density in first window ~ {p:.6f}")


if __name__ == "__main__":
    main()
