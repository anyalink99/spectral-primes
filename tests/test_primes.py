import warnings

import pytest

from spectral_primes.primes import _segment_sieve_count, _small_primes, prime_count_interval


def test_prime_count_interval_small() -> None:
    assert prime_count_interval(2, 10) == 4  # 2,3,5,7


def test_prime_count_warns_on_large_interval() -> None:
    with pytest.warns(UserWarning, match="1e14"):
        prime_count_interval(10**14, 10**14 + 3)


def test_prime_count_small_raises_no_userwarning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        prime_count_interval(100, 200)


# ---- segment sieve correctness ----

def test_small_primes_up_to_30() -> None:
    assert _small_primes(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_sieve_matches_known_counts() -> None:
    # π(100) = 25, π(10) = 4 → primes in [10, 100) = 25 - 4 = 21
    assert _segment_sieve_count(10, 100) == 21


def test_sieve_empty_interval() -> None:
    assert _segment_sieve_count(10, 10) == 0
    assert _segment_sieve_count(10, 5) == 0


def test_sieve_single_prime() -> None:
    assert _segment_sieve_count(7, 8) == 1  # just 7


def test_sieve_single_composite() -> None:
    assert _segment_sieve_count(8, 9) == 0


def test_sieve_large_interval() -> None:
    # There are 65 primes in [999000, 1000000) (verified via sympy.primerange).
    assert _segment_sieve_count(999_000, 1_000_000) == 65


def test_sieve_from_below_2() -> None:
    # interval starting below 2 should still work
    assert _segment_sieve_count(0, 10) == 4  # 2,3,5,7


def test_sieve_pi_1e7() -> None:
    """π(10^7) = 664579. Validates the sieve at moderate scale."""
    assert _segment_sieve_count(2, 10**7) == 664_579
