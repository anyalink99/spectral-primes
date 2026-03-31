"""
Build 1D sequences for discrete spectral experiments.

Ideas from exploratory notes: twin-prime indicator, Bernoulli baseline matching prime density,
consecutive-prime residues mod m (Lemke Oliver–style correlations are visible in time domain;
FFT is one way to hunt periodic structure).
"""
from __future__ import annotations

import numpy as np
from sympy import isprime, prime


def segment_twin_lower_indicator(lo: int, hi: int) -> np.ndarray:
    """
    For integers n in [lo, hi), value 1 iff n and n+2 are both prime (lower twin member).
    Length = hi - lo.
    """
    if hi <= lo:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(hi - lo, dtype=np.float64)
    for n in range(lo, hi):
        if isprime(n) and isprime(n + 2):
            out[n - lo] = 1.0
    return out


def _sieve_primes_upto(n: int) -> np.ndarray:
    """Boolean mask: True iff k is prime for 0 <= k <= n."""
    if n < 0:
        return np.zeros(0, dtype=bool)
    if n < 2:
        b = np.zeros(n + 1, dtype=bool)
        return b
    is_p = np.ones(n + 1, dtype=bool)
    is_p[0] = is_p[1] = False
    lim = int(n**0.5) + 1
    for i in range(2, lim):
        if is_p[i]:
            is_p[i * i : n + 1 : i] = False
    return is_p


def segment_twin_lower_indicator_sieve(lo: int, hi: int) -> np.ndarray:
    """
    Same meaning as segment_twin_lower_indicator; uses Eratosthenes up to hi+1.
    Use for long intervals (e.g. 1e5+) where per-n SymPy isprime is slow.
    """
    if hi <= lo:
        return np.zeros(0, dtype=np.float64)
    is_p = _sieve_primes_upto(hi + 1)
    out = np.zeros(hi - lo, dtype=np.float64)
    for offset, n in enumerate(range(lo, hi)):
        if is_p[n] and is_p[n + 2]:
            out[offset] = 1.0
    return out


def segment_prime_indicator(lo: int, hi: int) -> np.ndarray:
    """1 at primes in [lo, hi), 0 otherwise. Length = hi - lo."""
    if hi <= lo:
        return np.zeros(0, dtype=np.float64)
    return np.array([1.0 if isprime(n) else 0.0 for n in range(lo, hi)], dtype=np.float64)


def empirical_prime_density(lo: int, hi: int) -> float:
    """Fraction of integers in [lo, hi) that are prime."""
    w = hi - lo
    if w <= 0:
        return 0.0
    return float(np.count_nonzero(segment_prime_indicator(lo, hi))) / w


def bernoulli_same_density(
    length: int,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """i.i.d. Bernoulli(p), same shape as a length-N binary prime mask."""
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    p = float(np.clip(p, 0.0, 1.0))
    return rng.binomial(1, p, size=length).astype(np.float64)


def consecutive_prime_residues(n_terms: int, modulus: int) -> np.ndarray:
    """
    Sequence (p_i mod m) for i = 1..n_terms, p_i the i-th prime.
    """
    if modulus < 2:
        raise ValueError("modulus must be >= 2")
    if n_terms <= 0:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(n_terms, dtype=np.float64)
    for i in range(n_terms):
        out[i] = float(prime(i + 1) % modulus)
    return out


def twin_gap_indicator_along_primes(n_gaps: int) -> np.ndarray:
    """
    For the first (n_gaps + 1) primes, mark gap 1 where p_{i+1} - p_i == 2, else 0.
    Length n_gaps (one value per consecutive pair).
    """
    if n_gaps <= 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(n_gaps, dtype=np.float64)
    for i in range(n_gaps):
        a, b = prime(i + 1), prime(i + 2)
        if b - a == 2:
            out[i] = 1.0
    return out


def demean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    return x - x.mean()


def _big_omega_upto(n_max: int) -> np.ndarray:
    """Ω(k) = total prime factors with multiplicity for 2 ≤ k ≤ n_max; Ω(0)=Ω(1)=0."""
    if n_max < 2:
        return np.zeros(max(n_max, 0) + 1, dtype=np.int32)
    omega = np.zeros(n_max + 1, dtype=np.int32)
    for i in range(2, n_max + 1):
        if omega[i] != 0:
            continue
        for j in range(i, n_max + 1, i):
            x = j
            while x % i == 0:
                omega[j] += 1
                x //= i
    return omega


def segment_liouville(lo: int, hi: int) -> np.ndarray:
    """
    λ(n) on integers n in [lo, hi): completely multiplicative, λ(p)=−1 on primes.
    Here λ(n) = (−1)^Ω(n) with Ω total prime factors with multiplicity; λ(1)=1.
    """
    if hi <= lo:
        return np.zeros(0, dtype=np.float64)
    omg = _big_omega_upto(hi - 1)
    out = np.empty(hi - lo, dtype=np.float64)
    for idx, n in enumerate(range(lo, hi)):
        if n <= 1:
            out[idx] = 1.0
        else:
            out[idx] = 1.0 if (int(omg[n]) % 2 == 0) else -1.0
    return out
