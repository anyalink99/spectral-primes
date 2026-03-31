# Spectral primes

**Language:** English · [Русский](README.ru.md)

A small **Python toolkit** for experiments with a *spectral* construction tied to the imaginary parts `γ_n` of nontrivial zeros of the Riemann zeta function. It is a **computational companion** to the preprint *Spectral approximation of local maxima of prime density* (Russian manuscript: `spectral_primes.docx` in the repo root).

This software is **not** a proof of any claim in that document. It reproduces **definitions and pipelines** so you can run them on a laptop, extend them, and inspect behaviour at moderate scales (e.g. `x ~ 10^6–10^8`). The preprint’s registered Monte Carlo uses much larger `x` and many more zeros; **do not expect the same effect sizes or p-values** here.

---

## Table of contents

1. [Background](#background)
2. [What this repository contains](#what-this-repository-contains)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Reproducibility](#reproducibility)
6. [Data: zeros and SQLite](#data-zeros-and-sqlite)
7. [Command-line interface](#command-line-interface)
8. [Python package layout](#python-package-layout)
9. [Experiment design (three groups)](#experiment-design-three-groups)
10. [Random sieve baseline (Cramer model)](#random-sieve-baseline-cramer-model)
11. [Sparse operator, Varₙ, and permutation tests](#sparse-operator-varₙ-and-permutation-tests)
12. [Limitations and honesty notes](#limitations-and-honesty-notes)
13. [Running tests](#running-tests)
14. [C++ MPFR tool](#c-mpfr-tool-spectral_u)
15. [Discrete FFT sandbox (`experiments/wild_theories`)](#discrete-fft-sandbox-experimentswild_theories)
16. [License](#license)

---

## Background

The von Mangoldt explicit formula links prime counting to zeros `ρ = ½ + iγ` of `ζ(s)`. A simplified *oscillatory* object used in the preprint is the **regularized spectral operator** (phase `φ_n ≡ 0` as in the manuscript):

$$
U(x,\Lambda) = \sum_{\gamma_n \le \Lambda} \omega(\gamma_n,\Lambda)\,\frac{\cos(\gamma_n \ln x)}{\gamma_n}
\qquad
\omega(\gamma,\Lambda) = \exp\left(-\frac{\gamma^2}{2\Lambda^2}\right).
$$

Only zeros with `γ_n ≤ Λ` enter the sum. The Python code evaluates `U` in a **vectorized** way over many `x` at once (`numpy`). An optional **C++** utility [`spectral_u`](cpp/README.md) uses **GNU MPFR** (and GMP) for arbitrary-precision values of the same sum—useful for large `x` or cross-checks.

---

## What this repository contains

| Area | Description |
|------|-------------|
| **Data** | Seed file `data/zeros.csv` (indexed `γ_n`; default build has hundreds of zeros — regenerate for more). |
| **Database** | `sql/schema.sql` + `scripts/init_db.py` → SQLite `data/zeta_zeros.sqlite3` (gitignored after build). |
| **Core library** | `spectral_primes`: `U(x)`, optional **Varₙ-based sparse** `U_sparse`, prime density, **Cramer random-sieve** baseline (`random_sieve`), demos, permutation helper. |
| **CLI** | `spectral-primes`: `demo`, `demo-sparse`, `permute`, `random-compare`, `curve`. |
| **Scripts** | `scripts/build_zeros_csv.py` fills CSV via `mpmath.zetazero`. |
| **Tests** | `pytest` under `tests/`. |
| **C++ (optional)** | `cpp/spectral_u` — MPFR evaluation of `U(x,Λ)`; see [`cpp/README.md`](cpp/README.md). |
| **FFT sandbox** | [`experiments/wild_theories/`](experiments/wild_theories/README.md) — informal `numpy.fft` experiments on prime/twin indicators, null models (shuffle, Welch, surrogates, block bootstrap); **not** the zeta-zero operator `U(x,Λ)`. |

---

## Discrete FFT sandbox (`experiments/wild_theories`)

Separate from the main `spectral_primes` pipeline, the `experiments/wild_theories` folder explores **discrete** spectra of binary or residue sequences built from primes (twin masks, Liouville on integers, Ulam layout plots, and several statistical baselines). This is exploratory tooling and does not claim results about the Riemann zeros or the preprint’s `U(x,Λ)` object. Commands and module list: [`experiments/wild_theories/README.md`](experiments/wild_theories/README.md).

---

## Requirements

- **Python** 3.10+
- **Dependencies** (see `pyproject.toml` / `requirements.txt`): `numpy`, `sympy`, `mpmath`
- **Optional:** `pip install -e ".[fast-primes]"` for `gmpy2`-accelerated interval prime checks (see [`CHANGELOG.md`](CHANGELOG.md))
- **Dev / tests**: `pip install -e ".[dev]"` pulls `pytest`
- **C++ (optional):** CMake 3.16+, a C++17 compiler, and **MPFR** (which depends on **GMP**)

---

## Installation

```bash
git clone https://github.com/anyalink99/spectral-primes.git
cd spectral-primes

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install -e ".[dev]"
```

Optional: `pip install -e ".[analysis]"` if you add SciPy-based code later (`pyproject.toml` lists optional extras).

---

## Reproducibility

To tie a computational run to a **specific preprint revision**, record:

- The **package version** from `pyproject.toml` (or `import importlib.metadata; importlib.metadata.version("spectral-primes")` after install).
- The **git commit** (or release tag such as `v0.4.0`) of this repository.

[`CHANGELOG.md`](CHANGELOG.md) summarizes API and CLI changes between versions. CI runs `pytest` on each push and pull request (see `.github/workflows/ci.yml`).

---

## Data: zeros and SQLite

1. **CSV**  
   Columns: `n`, `gamma`. The repository ships a starter CSV. To grow it (slow for large `N`):

   ```bash
   python scripts/build_zeros_csv.py -n 5000 -o data/zeros.csv
   ```

2. **SQLite**  
   ```bash
   python scripts/init_db.py
   ```  
   Creates `data/zeta_zeros.sqlite3` from `data/zeros.csv`. The CLI prefers this file when present; otherwise it falls back to CSV.

---

## Command-line interface

Global options (before the subcommand):

| Option | Default | Meaning |
|--------|---------|---------|
| `--csv` | `data/zeros.csv` | Path to zeros CSV |
| `--sqlite` | `data/zeta_zeros.sqlite3` | Used if the file exists |

### `demo` — full `U`, three groups

Compares mean prime density (per `10^5` integers) in windows centred at **high** `U`, **random** centres, and **low** `U`, relative to mean ± `sigma_u`·std of `U` on a reference grid in `[x_lo, x_hi]`.

| Option | Default | Description |
|--------|---------|-------------|
| `--Lambda` | `80` | Cutoff `Λ` |
| `--x-lo`, `--x-hi` | `2_000_000`, `8_000_000` | Sampling range for centres |
| `--half-width` | `400` | Half-width of prime-counting window |
| `--n` | `40` | Samples per group |
| `--sigma-u` | `1.5` | Threshold in units of reference std |
| `--seed` | `42` | RNG seed |

### `demo-sparse` — `U_sparse` + Varₙ subset

Same three-group design, but `U` is replaced by **U_sparse** (see below). Integrates squared contributions on `[x − δ, x + δ]` with `n_t` sample points.

| Option | Default | Description |
|--------|---------|-------------|
| `--delta` | `300` | Half-width `δ` for local energy |
| `--n-t` | `48` | Points for time average |
| `--rank-anneal` | `0` | Slot factor `exp(rank_anneal·(slot+1)/N))`; `0` = γ-only rule |
| *(same as `demo`)* | | `Lambda`, `x-lo`, `x-hi`, `half-width`, `n`, `sigma-u`, `seed` |

### `permute` — permutation null for sparse contrast

Fixes a pool of centres and a **fixed** random control set **B**. Shuffles the **ordered** vector of `γ` (Fisher–Yates). Statistic: `mean(density | A_high) − mean(density | B)`. Reports a two-sided **empirical** p-value.

| Option | Default | Description |
|--------|---------|-------------|
| `--reps` | `200` | Number of shuffles |
| `--pool-factor` | `80` | Pool size ≥ `n * pool_factor` |
| `--rank-anneal` | `0.05` | Must be `> 0` for a non-degenerate γ-shuffle null; `0` requires `--allow-degenerate-null` |
| `--allow-degenerate-null` | off | Allow `rank_anneal=0` (degenerate null; diagnostics only) |
| *(overlap with `demo-sparse`)* | | |

### `curve` — export `U(x)` (full operator)

Log-spaced grid; uses **full** `U`, not sparse.

| Option | Default | Description |
|--------|---------|-------------|
| `--x-lo`, `--x-hi` | `1e6`, `1e7` | Range (float) |
| `--points` | `50` | Number of points |
| `--Lambda` | `80` | Cutoff |

### `random-compare` — primes vs Cramer random sieve

Does **not** use zeta zeros. Draws `n` random window centres in `[x_lo, x_hi]`, computes **true** prime density in each window, and compares to **independent** *Cramer-style* realizations: each integer `m ≥ 2` is marked with probability `1 / ln m` (independently), giving a pseudo-prime set with similar *mean* density but **no** arithmetic structure. One field is built per “universe”; the mean density across windows is recorded for many universes. Reports a one-sided empirical **P**(mean random &lt; mean true) — how often the random model’s global mean falls below the observed prime mean on the same windows.

| Option | Default | Description |
|--------|---------|-------------|
| `--n` | `60` | Number of random window centres |
| `--universes` | `100` | Independent Cramer fields |
| `--x-lo`, `--x-hi`, `--half-width`, `--seed` | same spirit as `demo` | |

Example:

```bash
python scripts/init_db.py
spectral-primes demo --n 30
spectral-primes demo-sparse --delta 300 --n-t 48 --n 30
spectral-primes permute --reps 200 --n 25
spectral-primes random-compare --n 80 --universes 120
spectral-primes curve --points 100 > u_curve.tsv
pytest -q
```

---

## Python package layout

| Module | Role |
|--------|------|
| `spectral_primes.operator` | `U_batch`, `reference_stats` |
| `spectral_primes.subset` | Varₙ energies, subset mask, `U_sparse_at` / `U_sparse_batch`, `reference_stats_sparse` |
| `spectral_primes.primes` | Prime counting / density (`sympy` by default; optional `gmpy2` via `[fast-primes]`; warns for large endpoints) |
| `spectral_primes.experiment` | Three-group runs, Z helper, fixed-**B** permutation test (`allow_degenerate_null` for `rank_anneal=0`) |
| `spectral_primes.random_sieve` | Cramér field builder, `compare_prime_vs_cramer` |
| `spectral_primes.io_data` | Load `γ` from SQLite or CSV |
| `spectral_primes.cli` | Argument parsing and entry point |

Import example:

```python
import numpy as np
from spectral_primes.operator import U_batch, reference_stats

gammas = ...  # 1D float64 array
x = np.linspace(1e6, 2e6, 100)
u = U_batch(x, gammas, Lambda=80.0)
```

---

## Experiment design (three groups)

1. Choose an interval `[x_lo, x_hi]` and draw a **pool** of random window centres (integers).
2. On a **reference grid** of `x` values in the same interval, compute mean `μ` and std `σ` of the chosen operator (`U` or `U_sparse`).
3. **Group A:** centres where the operator exceeds `μ + kσ` (default `k = 1.5`).
4. **Group B:** uniform random centres from the pool (independent of `U` for the standard `demo`; same fixed **B** in `permute`).
5. **Group C:** centres where the operator is below `μ − kσ`.
6. For each centre, estimate **prime density** (primes per `10^5` integers) in `[centre − half_width, centre + half_width)`.

Prime checks use **SymPy** by default (deterministic for `n < 2^64`). For endpoints ≥ 10⁹, `prime_count_interval` emits a **warning**; install **`[fast-primes]`** (`gmpy2`) for faster per-integer tests, or use an external sieve for very large `x`. This stack is illustrative at moderate scales, not a replacement for a production sieve at preprint-scale `x`.

---

## Random sieve baseline (Cramer model)

To separate “**sparse like primes**” from “**actual primes**”, the package implements a textbook **stochastic null** (often associated with Cramer): each `n ≥ 2` is labelled independently with probability `1 / ln n` (and `n < 2` never). This matches the *first-order* expected density `~ 1 / ln x` but ignores divisibility (so it is **not** a claim that primes behave exactly like this process).

The CLI command `random-compare` uses the **same random windows** for true primes and for many independent Cramer universes. If the empirical P-value is high, the observed mean prime density in those windows is **not unusual** for this null; if low, true primes are **systematically denser** (or rarer, depending on tail) than the model in that slice — a hint that the “fingerprint” goes beyond i.i.d. thinning. The model is still crude (known issues for fine statistics, e.g. Goldbach-type heuristics); treat results as **exploratory**.

---

## Sparse operator, Varₙ, and permutation tests

**Local energy.** On `[x − δ, x + δ]`, approximate

$$
E_n(x) \approx \text{mean}_t\,\bigl(r_n\,\omega(\gamma_n,\Lambda)\,\cos(\gamma_n \ln t)/\gamma_n\bigr)^2
$$

on a uniform grid in `t`. **Relative weights** `Var_n = E_n / \sum_k E_k`. **Subset** `S`: indices with `Var_n > 1/N` (with a fallback if none qualify). **Sparse value at `x`:**

$$
U_{\text{sparse}}(x) = \sum_{n \in S} r_n\,\omega(\gamma_n,\Lambda)\,\frac{\cos(\gamma_n \ln x)}{\gamma_n}.
$$

**Rank factor.** `r_n = exp(rank_anneal · (slot_n+1)/N))` uses the zero’s **position** in the input list (`slot_n`), not only `γ_n`. With `rank_anneal = 0`, `r_n ≡ 1` (closest to the γ-only Var rule in the preprint).

**Why `rank_anneal` matters for `permute`.** If only `γ_n` enters `E_n` and the amplitude is the same for every slot, then **shuffling γ across slots does not change** `U_sparse(x)` (the selected multiset of contributing frequencies is unchanged). The permutation null would be **degenerate**. A small `rank_anneal > 0` breaks that symmetry so shuffles change the operator; this is a **simulation device**, not a claim that the `.docx` includes that factor. The plain sum `U(x,Λ)` is permutation-invariant under γ shuffle (commutative sum); see `tests/test_subset.py`. The library **rejects** `permutation_test_sparse(..., rank_anneal=0)` unless `allow_degenerate_null=True`; the CLI requires **`--allow-degenerate-null`** with `--rank-anneal 0`.

---

## Limitations and honesty notes

- **Scale:** Effects in the preprint are stated for very large `x` and many zeros; modest `x` and few zeros produce **noise** and different prime density scale.
- **Statistics:** CLI Z-scores and permutation p-values are **illustrative**; they are not a replication of the preprint’s full registered protocol.
- **Permutation:** Interpretation depends on `rank_anneal` and on the fixed-pool / fixed-**B** design documented above.
- **Cramer model:** A convenient null for *density*, not a faithful model of all prime correlations.
- **No crypto claims:** None of this is offered as a factorization or cryptanalysis tool.

---

## Running tests

```bash
pytest -q
```

---

## C++ MPFR tool (`spectral_u`)

The [`cpp/`](cpp/) directory builds a small program that computes the **same** full operator `U(x,Λ)` as Python, with **rounded arbitrary precision** via [GNU MPFR](https://www.mpfr.org/) (built on [GMP](https://gmplib.org/)).

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
./cpp/build/spectral_u --prec 512 --lambda 80 --x 1e14
```

Full dependency list, Windows/vcpkg/MSYS2 notes, and flags are in **[`cpp/README.md`](cpp/README.md)**. This does **not** reimplement zeta-zero finding (still use Python/`mpmath` for `build_zeros_csv.py`).

---

## License

[MIT](LICENSE)

---

## Source manuscript

The preprint text (DOCX) is included as `spectral_primes.docx` for provenance. If you prefer a slimmer clone, remove it from the tree or exclude it from version control.
