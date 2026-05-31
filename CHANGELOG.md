# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-06-01

Correctness release. See [`ERRATA.md`](ERRATA.md) for the full audit with measured numbers.

### Fixed

- **Operator phase/sign.** The manuscript operator `Σ ω cos(γ ln x)/γ` is the wrong object for prime *density* and its sign is backwards (it *anti*-correlates: `r ≈ −0.031`, `Z ≈ −14`). Added the correct density operator `U_density_batch` = `−Σ ω cos(γ ln x)` (high value ⟺ more primes; `r ≈ +0.059`, `Z ≈ +26`) and the counting-phase `U_psi_batch` = `Σ ω sin(γ ln x)/γ`.
- **`|S| = O(√N log N)` reinterpreted.** Shown empirically to be the Riemann–von Mangoldt zero count below `Λ = O(√N)`, not a property of the variance-selection criterion (with `Λ` fixed, `|S|` saturates).
- **Speedup claim withdrawn.** Ranking windows by the operator captures primes at the uniform-baseline rate; only ~2% of windows can be skipped at <2% prime loss (= random). The "15–20% sieve reduction" is not supported.

### Added

- `experiments/verification/verify.py` + `results.txt`: reproducible audit (real segmented sieve + real zeros) for phase/sign, `|S|` scaling, and the speedup test.
- `scripts/build_manuscript.py`: regenerates the corrected manuscript (`spectral_primes.docx` + `MANUSCRIPT.ru.md`) from a single source.
- `ERRATA.md`; `U_density_batch`, `U_psi_batch`, `reference_stats_density`; tests for the new operators.
- Segmented Eratosthenes sieve replacing per-integer `sympy.isprime` in `prime_count_interval` (orders of magnitude faster at `x ~ 10^7+`). (#1)
- Runtime SNR warnings for noisy regimes (few zeros / small Λ with large x) and for the non-preprint `rank_anneal > 0` permutation null; `tests/test_warnings.py`. (#1)
- `ruff` lint job in CI. (#1)

### Changed

- README / RU: corrected operator, honesty notes, and a one-line reproduction command.
- Original manuscript preserved as `spectral_primes_v1_original.docx`; `spectral_primes.docx` is now the corrected v0.5 text.

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
