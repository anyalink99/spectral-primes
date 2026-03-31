import warnings

import pytest

from spectral_primes.primes import prime_count_interval


def test_prime_count_interval_small() -> None:
    assert prime_count_interval(2, 10) == 4  # 2,3,5,7


def test_prime_count_warns_on_large_interval() -> None:
    with pytest.warns(UserWarning, match="large"):
        prime_count_interval(10**9, 10**9 + 3)


def test_prime_count_small_raises_no_userwarning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        prime_count_interval(100, 200)
