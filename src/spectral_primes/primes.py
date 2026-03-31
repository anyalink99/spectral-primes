from __future__ import annotations

from sympy import isprime


def prime_count_interval(a: int, b: int) -> int:
    """Count primes p with a ≤ p < b (half-open). Uses SymPy (deterministic for n < 2^64)."""
    if b <= a:
        return 0
    return sum(1 for n in range(a, b) if isprime(n))


def prime_density_per_1e5(center: int, half_width: int) -> float:
    """Primes per 10^5 integers in [center - half_width, center + half_width)."""
    lo = center - half_width
    hi = center + half_width
    width = max(hi - lo, 1)
    c = prime_count_interval(lo, hi)
    return 1e5 * c / width
