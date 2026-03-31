# Spectral primes

Computational companion to the preprint *«Спектральная аппроксимация локальных максимумов плотности простых чисел»* (`spectral_primes.docx`): a **regularized spectral operator** built from imaginary parts `γ_n` of nontrivial Riemann zeta zeros,

\[
U(x,\Lambda) = \sum_{\gamma_n \le \Lambda} \omega(\gamma_n,\Lambda)\,\frac{\cos(\gamma_n \ln x)}{\gamma_n},
\qquad
\omega(\gamma,\Lambda) = \exp\!\Big(-\frac{\gamma^2}{2\Lambda^2}\Big).
\]

The repo provides:

- **SQLite** database schema and loader for zeros `γ_n` (`sql/schema.sql`, `scripts/init_db.py`).
- **Seed CSV** `data/zeros.csv` (400 zeros; extend locally with `scripts/build_zeros_csv.py`).
- **Python package** `spectral_primes`: vectorized `U(x)`, reference statistics, and a **three-group demo** (high-`U` vs random vs low-`U` windows) with prime density in short intervals.
- **CLI** `spectral-primes` and **tests** (`pytest`).

This is **not** a proof of the claims in the preprint. It is a reproducible toolbox: the published Monte Carlo design uses very large `x` (e.g. `10^14`) and many zeros; here you can run the **same pipeline** on smaller scales for teaching and debugging.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"

python scripts/init_db.py       # builds data/zeta_zeros.sqlite3 from data/zeros.csv
spectral-primes demo --n 30     # uses CSV or SQLite if present
spectral-primes curve --points 20 > u_curve.tsv
pytest -q
```

### More zeros

Computing zeros is slow in pure `mpmath`; run overnight for large `N`:

```bash
python scripts/build_zeros_csv.py -n 5000 -o data/zeros.csv
python scripts/init_db.py
```

---

## RU

Репозиторий воспроизводит **формулу оператора** и **схему эксперимента** (группы A/B/C, порог `mean ± 1.5·std` для `U` на референсной сетке по `x`). На умеренных `x` (миллионы) **не ждите тех же Z-статистик**, что в препринте для `x ~ 10^14`: плотность простых и осцилляции другого масштаба. Смысл проекта — прозрачный код, база нулей и CLI для дальнейших экспериментов.

Исходный текст работы лежит в корне: `spectral_primes.docx`.

---

## Layout

| Path | Role |
|------|------|
| `sql/schema.sql` | Table `zeta_zeros(n, gamma)` |
| `data/zeros.csv` | Seed zeros |
| `src/spectral_primes/` | Library + CLI |
| `scripts/build_zeros_csv.py` | Fill CSV via `mpmath.zetazero` |
| `scripts/init_db.py` | CSV → SQLite |

---

## Permutation caveat

For the **plain** sum `U(x,Λ)`, permuting the multiset `{γ_n}` does not change `U` (commutative sum). The preprint’s permutation check is meaningful in their **full** pipeline (e.g. variance-based subset `S(x,Λ)` and registered thresholds). Extending this repo with that subset rule is a natural next step.

---

## License

MIT — see `LICENSE`.
