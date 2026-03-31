from __future__ import annotations

import warnings
from typing import Callable

from sympy import isprime as sympy_isprime

try:
    import gmpy2
except ImportError:
    gmpy2 = None  # type: ignore[misc, assignment]

# SymPy isprime per integer in [a,b) becomes costly for very large endpoints; warn once per call site.
_LARGE_INTERVAL_WARN_AT = 10**9


def _choose_isprime() -> Callable[[int], bool]:
    if gmpy2 is not None:

        def _p(n: int) -> bool:
            return bool(gmpy2.is_prime(n))

        return _p
    return sympy_isprime


_isprime: Callable[[int], bool] = _choose_isprime()


def prime_count_interval(a: int, b: int) -> int:
    """Count primes p with a ≤ p < b (half-open).

    Uses SymPy by default (deterministic for n < 2^64). With optional ``gmpy2``
    (``pip install "spectral-primes[fast-primes]"``), uses ``gmpy2.is_prime`` instead.
    """
    if b <= a:
        return 0
    hi = b - 1
    if hi >= _LARGE_INTERVAL_WARN_AT or a >= _LARGE_INTERVAL_WARN_AT:
        warnings.warn(
            "prime_count_interval: endpoints are large; each integer in [a,b) is tested "
            "individually (SymPy or gmpy2). For x ~ 1e14-scale work, prefer a segment sieve "
            "or install gmpy2 via extras [fast-primes].",
            UserWarning,
            stacklevel=2,
        )
    ip = _isprime
    return sum(1 for n in range(a, b) if ip(n))


def prime_density_per_1e5(center: int, half_width: int) -> float:
    """Primes per 10^5 integers in [center - half_width, center + half_width)."""
    lo = center - half_width
    hi = center + half_width
    width = max(hi - lo, 1)
    c = prime_count_interval(lo, hi)
    return 1e5 * c / width
