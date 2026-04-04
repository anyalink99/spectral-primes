from __future__ import annotations

import math
import warnings

# ---------------------------------------------------------------------------
# Segment-sieve implementation (replaces per-integer isprime for intervals).
# Complexity: O(√b · log log √b) for small primes + O(b-a) marking.
# ---------------------------------------------------------------------------

# For intervals ending above this threshold we warn that the sieve may
# allocate a non-trivial amount of memory (though still far faster than
# per-element primality testing).
_LARGE_INTERVAL_WARN_AT = 10**14


def _small_primes(limit: int) -> list[int]:
    """Sieve of Eratosthenes up to *limit* (inclusive). Returns list of primes."""
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i, v in enumerate(sieve) if v]


def _segment_sieve_count(a: int, b: int) -> int:
    """Count primes in [a, b) using a segmented sieve of Eratosthenes."""
    if b <= a:
        return 0
    if a < 2:
        a = 2
    if b <= a:
        return 0

    length = b - a
    is_prime = bytearray(b"\x01") * length  # is_prime[i] ↔ (a+i) is prime

    # We need small primes up to √(b-1).
    sp = _small_primes(int(math.isqrt(b - 1)) + 1)
    for p in sp:
        # First multiple of p in [a, b).
        start = ((a + p - 1) // p) * p
        if start == p:          # p itself is prime, don't mark it
            start += p
        for j in range(start - a, length, p):
            is_prime[j] = 0

    return sum(is_prime)


def prime_count_interval(a: int, b: int) -> int:
    """Count primes p with a <= p < b (half-open).

    Uses a segmented sieve of Eratosthenes — O(sqrt(b) log log sqrt(b)) time,
    dramatically faster than per-integer primality tests for intervals above ~1000.
    """
    if b <= a:
        return 0
    if b - 1 >= _LARGE_INTERVAL_WARN_AT or a >= _LARGE_INTERVAL_WARN_AT:
        warnings.warn(
            "prime_count_interval: endpoints >= 1e14. The segment sieve is still correct "
            "but may allocate substantial memory for the marking array. Consider chunking "
            "the interval if memory is tight.",
            UserWarning,
            stacklevel=2,
        )
    return _segment_sieve_count(a, b)


def prime_density_per_1e5(center: int, half_width: int) -> float:
    """Primes per 10^5 integers in [center - half_width, center + half_width)."""
    lo = center - half_width
    hi = center + half_width
    width = max(hi - lo, 1)
    c = prime_count_interval(lo, hi)
    return 1e5 * c / width
