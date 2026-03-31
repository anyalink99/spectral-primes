# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-04-01

### Added

- `permutation_test_sparse(..., allow_degenerate_null=False)`: calling with `rank_anneal=0` now raises `ValueError` unless `allow_degenerate_null=True` (degenerate γ-shuffle null is opt-in).
- CLI `permute`: `--allow-degenerate-null` to run with `--rank-anneal 0`; otherwise exit with an error message.
- `UserWarning` from `prime_count_interval` when interval endpoints are ≥ 10⁹ (per-integer primality tests are costly at large scales).
- Optional dependency extra `[fast-primes]` (`gmpy2`): when installed, `prime_count_interval` uses `gmpy2.is_prime` instead of SymPy.
- `tests/test_permutation_null.py`, `tests/test_primes.py`.
- GitHub Actions workflow running `pytest` on push and pull request.
- This changelog and a short reproducibility note in the README files.

### Changed

- README / RU: `permute` options and `spectral_primes.primes` description updated for the above behaviour.

## [0.3.0] - earlier

Packaging and docs as tagged in `pyproject.toml` before this changelog; see git history for details.
