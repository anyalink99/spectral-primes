"""
Build PNG figures + summary.txt under experiments/wild_theories/output/

  python -m experiments.wild_theories.plot_results

Requires matplotlib (pip install matplotlib).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sympy import isprime

from .fourier import compare_to_random_trials, magnitude_spectrum
from .scaling import scaling_table
from .signals import (
    consecutive_prime_residues,
    segment_prime_indicator,
    segment_twin_lower_indicator,
    twin_gap_indicator_along_primes,
)
from .ulam import array_to_spiral_coords


def _out_dir() -> Path:
    return Path(__file__).resolve().parent / "output"


def _random_spectrum_mean_std(
    n: int,
    p: float,
    rng: np.random.Generator,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean and std of |rfft| magnitudes across Bernoulli(p) masks."""
    y0 = rng.binomial(1, p, size=n).astype(np.float64)
    mag0 = magnitude_spectrum(y0)
    klen = mag0.size
    acc = np.zeros(klen, dtype=np.float64)
    acc2 = np.zeros(klen, dtype=np.float64)
    for _ in range(n_trials):
        y = rng.binomial(1, p, size=n).astype(np.float64)
        m = magnitude_spectrum(y)
        acc += m
        acc2 += m * m
    mean = acc / n_trials
    var = np.maximum(acc2 / n_trials - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean, std, klen


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    out = _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    lines: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        print(msg)

    # --- Text stats (same spirit as demo) ---
    lo, hi = 2, 5000
    twin = segment_twin_lower_indicator(lo, hi)
    prime_mask = segment_prime_indicator(lo, hi)
    log("=== Segment [2, 5000) ===")
    t1 = compare_to_random_trials(twin, rng, n_trials=100, p_match=float(twin.mean()))
    log(f"twin vs Bernoulli(p=twin_density): {t1}")
    rng2 = np.random.default_rng(42)
    t2 = compare_to_random_trials(prime_mask, rng2, n_trials=100)
    log(f"prime vs Bernoulli(p=mean): {t2}")

    res = consecutive_prime_residues(2000, 4)
    mag_res = magnitude_spectrum(res)
    if mag_res.size > 1:
        kpk = 1 + int(np.argmax(mag_res[1:]))
        log(
            f"consecutive primes mod 4 (n=2000): peak_k={kpk}, "
            f"|mag|={mag_res[kpk]:.4f}, E={float(np.sum(mag_res**2)):.4f}"
        )

    tg = twin_gap_indicator_along_primes(5000)
    log(
        f"twin gaps along primes (5000 gaps): mean={float(tg.mean()):.4f}, "
        f"FFT E={float(np.sum(magnitude_spectrum(tg)**2)):.4f}"
    )

    log("\n=== Scaling (windows) ===")
    wins = [(10_000, 20_000), (100_000, 110_000), (1_000_000, 1_010_000)]
    for kind in ("prime", "twin"):
        rows = scaling_table(wins, kind=kind)
        log(kind)
        for r in rows:
            log(
                f"  [{r.lo}, {r.hi}) peak_k={r.peak_k} peak={r.peak_mag:.4f} E={r.total_energy:.4f}"
            )

    summary_path = out / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nWrote {summary_path}")

    # --- Figure 1: prime mask spectrum vs random mean band ---
    n = hi - lo
    p = float(prime_mask.mean())
    mag_p = magnitude_spectrum(prime_mask)
    mean_r, std_r, _ = _random_spectrum_mean_std(n, p, rng, n_trials=120)
    k = np.arange(mag_p.size)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.plot(k, mag_p, color="#1a5276", lw=0.8, label="|FFT| простых (индикатор)")
    ax.fill_between(
        k,
        np.maximum(mean_r - 2 * std_r, 0),
        mean_r + 2 * std_r,
        color="#c0392b",
        alpha=0.25,
        label="Bernoulli(p) ±2σ (120 проб)",
    )
    ax.plot(k, mean_r, color="#c0392b", lw=0.9, ls="--", label="среднее Bernoulli")
    ax.set_xlim(0, min(800, mag_p.size - 1))
    ax.set_xlabel("индекс rFFT k")
    ax.set_ylabel("|амплитуда| / √n")
    ax.set_title("Спектр маски простых vs случайная маска той же плотности")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p1 = out / "spectra_prime_vs_noise.png"
    fig.savefig(p1)
    plt.close(fig)
    log(f"Wrote {p1}")

    # --- Figure 2: twin indicator spectrum (full k range zoom low) ---
    mag_t = magnitude_spectrum(twin)
    k2 = np.arange(mag_t.size)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=120)
    ax.plot(k2, mag_t, color="#117a3d", lw=0.7, label="близнецы (нижний член)")
    ax.set_xlim(0, min(600, mag_t.size - 1))
    ax.set_xlabel("k")
    ax.set_ylabel("|амплитуда| / √n")
    ax.set_title("Спектр индикатора близнецов [2, 5000)")
    ax.legend()
    fig.tight_layout()
    p2 = out / "spectra_twin.png"
    fig.savefig(p2)
    plt.close(fig)
    log(f"Wrote {p2}")

    # --- Figure 3: mod-4 residues along primes ---
    fig, ax = plt.subplots(figsize=(9, 4), dpi=120)
    kk = np.arange(mag_res.size)
    ax.plot(kk, mag_res, color="#6c3483", lw=0.6)
    ax.set_xlim(1, min(400, mag_res.size - 1))
    ax.set_xlabel("k")
    ax.set_ylabel("|FFT|")
    ax.set_title("Спектр последовательности (p_i mod 4), i=1..2000")
    fig.tight_layout()
    p3 = out / "spectra_residues_mod4.png"
    fig.savefig(p3)
    plt.close(fig)
    log(f"Wrote {p3}")

    # --- Figure 4: Ulam spiral, primes up to L ---
    L = 4000
    vals = np.array([1.0 if isprime(i) else 0.0 for i in range(1, L + 1)], dtype=np.float64)
    xs, ys, v = array_to_spiral_coords(vals)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    m = v > 0.5
    ax.scatter(xs[~m], ys[~m], s=4, c="#e8e8e8", linewidths=0)
    ax.scatter(xs[m], ys[m], s=6, c="#c0392b", linewidths=0, label="простое")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Спираль Улама: простые ≤ {L}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    p4 = out / "ulam_primes.png"
    fig.savefig(p4)
    plt.close(fig)
    log(f"Wrote {p4}")

    # --- Figure 5: FFT magnitude on spiral (first K bins of prime mask spectrum) ---
    Kspiral = min(2500, mag_p.size)
    spiral_amp = mag_p[:Kspiral].copy()
    xs2, ys2, va = array_to_spiral_coords(spiral_amp)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    sc = ax.scatter(xs2, ys2, c=va, s=8, cmap="viridis", linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.046, label="|FFT[k]|")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Амплитуды спектра маски простых на спирали Улама (k=0..)")
    fig.tight_layout()
    p5 = out / "ulam_fft_amplitudes.png"
    fig.savefig(p5)
    plt.close(fig)
    log(f"Wrote {p5}")

    # --- Figure 6: scaling energies bar ---
    rows_p = scaling_table(wins, kind="prime")
    rows_t = scaling_table(wins, kind="twin")
    labels = [f"{a}–{b}" for a, b in wins]
    Ep = [r.total_energy for r in rows_p]
    Et = [r.total_energy for r in rows_t]
    xpos = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.bar(xpos - w / 2, Ep, w, label="простые", color="#1a5276")
    ax.bar(xpos + w / 2, Et, w, label="близнецы", color="#117a3d")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Σ |FFT|² (норм.)")
    ax.set_title("Энергия спектра по окнам (одинаковая длина 10⁴)")
    ax.legend()
    fig.tight_layout()
    p6 = out / "scaling_energy_bars.png"
    fig.savefig(p6)
    plt.close(fig)
    log(f"Wrote {p6}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
