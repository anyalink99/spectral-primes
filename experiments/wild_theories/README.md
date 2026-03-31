# Wild theories (sandbox)

Идеи вроде «FFT по маске простых / близнецов», сравнение с белым шумом той же плотности, окна разного масштаба, раскладка по спирали Улама — **не** часть основного пайплайна `U(x, Λ)` из нулей дзета. Здесь только дискретные последовательности и `numpy.fft`.

## Запуск

Из корня репозитория:

```bash
python -m experiments.wild_theories.demo
```

Графики и `summary.txt` (нужен `matplotlib`):

```bash
pip install matplotlib
python -m experiments.wild_theories.plot_results
```

Файлы появляются в `experiments/wild_theories/output/`.

Дополнительные проверки (shuffle, масштаб 50k, Лиувилль):

```bash
python -m experiments.wild_theories.followup_checks
```

Углублённые нули: циклический сдвиг, Welch на ~100k, суррогат 1/log n:

```bash
python -m experiments.wild_theories.level2_spectral
```

Блочный bootstrap (перемешивание блоков длины B, тот же ряд близнецов n≈10⁵):

```bash
python -m experiments.wild_theories.block_bootstrap
```

Длинные отрезки близнецов считаются через решето (`segment_twin_lower_indicator_sieve`), не через `sympy` по каждому n.

## Модули

| Файл | Назначение |
|------|------------|
| `signals.py` | Маска близнецов (нижний член), маска простых, Bernoulli с той же `p`, остатки `p_i mod m`, индикатор «зазор = 2» вдоль простых |
| `fourier.py` | Нормированный `rfft`, энергия, `spectrum_peakiness`, сравнение с случайными масками |
| `scaling.py` | Таблица по списку окон `(lo, hi)` |
| `ulam.py` | Индекс → `(x, y)` на спирали Улама для визуализации |
| `demo.py` | Короткие примеры |
| `followup_checks.py` | Shuffle, сравнение длин, Лиувилль |
| `level2_spectral.py` | Welch, суррогаты, инвариант циклического сдвига для полного FFT |
| `block_bootstrap.py` | Moving-block bootstrap для `spectrum_peakiness` |

## Связь с основным пакетом

`src/spectral_primes` по-прежнему про спектральный оператор и плотность простых в окнах. При желании можно подставлять свои метрики плотности вместо `prime_density_per_1e5`, но это уже отдельный рефакторинг.
