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
- **Python package** `spectral_primes`: vectorized `U(x)`, **Varₙ subset** / `U_sparse(x)` (§2.2), reference statistics, **three-group demos** (full `U` and sparse), and a **permutation test** with fixed centers and control windows.
- **CLI** `spectral-primes` (`demo`, `demo-sparse`, `permute`, `curve`) and **tests** (`pytest`).

This is **not** a proof of the claims in the preprint. It is a reproducible toolbox: the published Monte Carlo design uses very large `x` (e.g. `10^14`) and many zeros; here you can run the **same pipeline** on smaller scales for teaching and debugging.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"

python scripts/init_db.py       # builds data/zeta_zeros.sqlite3 from data/zeros.csv
spectral-primes demo --n 30
spectral-primes demo-sparse --delta 300 --n-t 48 --n 30
spectral-primes permute --reps 200 --n 25
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

Дополнительно: **отбор нулей по Varₙ** (относительная доля локальной энергии на `[x−δ,x+δ]`) и команда **`permute`**: фиксированный пул центров и фиксированная контрольная группа B, перестановка вектора γ по слотам (Fisher–Yates).

Репозиторий воспроизводит **формулу оператора** и **схему эксперимента** (группы A/B/C, порог `mean ± 1.5·std` для `U` на референсной сетке по `x`). На умеренных `x` (миллионы) **не ждите тех же Z-статистик**, что в препринте для `x ~ 10^14`: плотность простых и осцилляции другого масштаба. Смысл проекта — прозрачный код, база нулей и CLI для дальнейших экспериментов.

Исходный текст работы лежит в корне: `spectral_primes.docx`.

---

## Layout

| Path | Role |
|------|------|
| `sql/schema.sql` | Table `zeta_zeros(n, gamma)` |
| `data/zeros.csv` | Seed zeros |
| `src/spectral_primes/` | Library + CLI (`operator`, `subset`, `experiment`) |
| `scripts/build_zeros_csv.py` | Fill CSV via `mpmath.zetazero` |
| `scripts/init_db.py` | CSV → SQLite |

---

## Varₙ subset and permutation (honest note)

If **only** `γ_n` enters both the energy `E_n` and the threshold `Var_n = E_n / \sum_k E_k`, then the selected set of frequencies is a function of the **multiset** `{γ_n}`: shuffling which γ sits in which database slot does **not** change `U_sparse(x)` when amplitudes are otherwise identical. So a naive “shuffle γ” null for that model is degenerate.

The CLI therefore supports an optional **slot rank factor** `r_n = exp(rank_anneal · (slot_n+1)/N))` multiplying each term (default `rank_anneal=0` in `demo-sparse` for a paper-close Var rule; default `0.05` in `permute` so the null is non-degenerate). This is a **technical device for simulation**, not part of the original closed form in the `.docx`.

The **plain** sum `U(x,Λ)` is still fully permutation-invariant under γ shuffle (commutative sum); see `tests/test_subset.py`.

---

## License

MIT — see `LICENSE`.
