"""
Follow-up checks: shuffle null, length scaling on nu=k/n, Liouville lambda(n) spectrum.

  python -m experiments.wild_theories.followup_checks

Writes PNG + summary under experiments/wild_theories/output/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .fourier import magnitude_spectrum, spectrum_peakiness
from .signals import segment_liouville, segment_twin_lower_indicator


def _out() -> Path:
    return Path(__file__).resolve().parent / "output"


def shuffle_spectrum_stats(
    x: np.ndarray,
    rng: np.random.Generator,
    n_shuffles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of |rfft| over permutations of x (same multiset)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    m0 = magnitude_spectrum(x)
    klen = m0.size
    acc = np.zeros(klen)
    acc2 = np.zeros(klen)
    for _ in range(n_shuffles):
        perm = rng.permutation(n)
        m = magnitude_spectrum(x[perm])
        acc += m
        acc2 += m * m
    mean = acc / n_shuffles
    var = np.maximum(acc2 / n_shuffles - mean * mean, 0.0)
    return mean, np.sqrt(var)


def main() -> int:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib", file=sys.stderr)
        return 1

    out = _out()
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(s: str) -> None:
        lines.append(s)
        print(s)

    rng = np.random.default_rng(2026)

    # --- 1) Shuffle test: twin [2, 5000) ---
    lo, hi = 2, 5000
    log(f"Building twin indicator [{lo}, {hi}) ...")
    twin5 = segment_twin_lower_indicator(lo, hi)
    L5 = twin5.size
    mag5 = magnitude_spectrum(twin5)
    log(f"  n={L5}, sum(twin)={int(twin5.sum())}, density={twin5.mean():.5f}")

    log(f"Shuffle test: {200} permutations ...")
    sh_mean, sh_std = shuffle_spectrum_stats(twin5, rng, 200)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    k = np.arange(mag5.size)
    nu = k / L5
    ax.plot(nu, mag5, color="#117a3d", lw=0.9, label="|FFT| twins (true order)")
    ax.fill_between(
        nu,
        np.maximum(sh_mean - 2 * sh_std, 0),
        sh_mean + 2 * sh_std,
        color="#7f8c8d",
        alpha=0.35,
        label="shuffled values +/-2sd (200)",
    )
    ax.plot(nu, sh_mean, color="#34495e", ls="--", lw=0.9, label="mean after shuffle")
    ax.set_xlim(0, min(0.12, nu[-1]))
    ax.set_xlabel("nu = k / n")
    ax.set_ylabel("|amplitude| / sqrt(n)")
    ax.set_title("Shuffle: twin indicator vs same multiset, random order")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p_shuffle = out / "followup_shuffle_twin.png"
    fig.savefig(p_shuffle)
    plt.close(fig)
    log(f"Wrote {p_shuffle}")

    def idx_for_nu(target_nu: float) -> int:
        return int(np.clip(round(target_nu * L5), 0, mag5.size - 1))

    for name, nu_t in [("nu~330/5000", 330 / 5000), ("nu~500/5000", 500 / 5000)]:
        ik = idx_for_nu(nu_t)
        o = mag5[ik]
        sm = sh_mean[ik]
        ss = max(sh_std[ik], 1e-15)
        log(f"  {name}: k={ik} orig={o:.4f} shuffle_mean={sm:.4f} z~{(o-sm)/ss:.2f}")

    # --- 2) Scale: twin [2, 50_000) vs [2, 5000) on nu = k/n ---
    hi_big = 50_000
    log(f"\nBuilding twin indicator [{lo}, {hi_big}) ... (SymPy isprime)")
    twin50 = segment_twin_lower_indicator(lo, hi_big)
    L50 = twin50.size
    mag50 = magnitude_spectrum(twin50)
    log(f"  n={L50}, sum(twin)={int(twin50.sum())}")

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    nu5 = np.arange(mag5.size) / L5
    nu50 = np.arange(mag50.size) / L50
    ax.plot(nu5, mag5, color="#117a3d", lw=0.8, alpha=0.9, label=f"n={L5}")
    ax.plot(nu50, mag50, color="#1a5276", lw=0.7, alpha=0.85, label=f"n={L50}")
    ax.set_xlim(0, 0.12)
    ax.set_xlabel("nu = k / n")
    ax.set_ylabel("|FFT| / sqrt(n)")
    ax.set_title("Twins: spectra for n=5k vs n=50k (x = k/n)")
    ax.legend()
    fig.tight_layout()
    p_scale = out / "followup_twin_scale_5k_vs_50k.png"
    fig.savefig(p_scale)
    plt.close(fig)
    log(f"Wrote {p_scale}")

    i5 = 1 + int(np.argmax(mag5[1: min(len(mag5), L5 // 2)]))
    i50 = 1 + int(np.argmax(mag50[1: min(len(mag50), L50 // 2)]))
    log(f"  argmax nu (low-half): n={L5} -> k={i5}, nu={i5/L5:.6f}")
    log(f"  argmax nu (low-half): n={L50} -> k={i50}, nu={i50/L50:.6f}")

    log("  fixed nu (bins 330/5000 and 500/5000):")
    for nu_star, label in [(330 / 5000, "330/5000"), (500 / 5000, "500/5000")]:
        k5b = int(np.clip(round(nu_star * L5), 0, mag5.size - 1))
        k50b = int(np.clip(round(nu_star * L50), 0, mag50.size - 1))
        log(
            f"    {label}: n={L5} k={k5b} |FFT|={mag5[k5b]:.4f} ; "
            f"n={L50} k={k50b} |FFT|={mag50[k50b]:.4f}"
        )

    # --- 3) Liouville lambda(n) on same index range vs twin (overlay) ---
    lio = segment_liouville(lo, hi)
    mag_lio = magnitude_spectrum(lio)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    k5 = np.arange(mag5.size)
    ax.plot(k5 / L5, mag5, color="#117a3d", lw=0.8, label="|FFT| twins")
    ax.plot(k5 / L5, mag_lio, color="#8e44ad", lw=0.6, alpha=0.9, label="|FFT| lambda(n), same [2,5000)")
    ax.set_xlim(0, 0.12)
    ax.set_xlabel("nu = k / n")
    ax.set_ylabel("|FFT| / sqrt(n)")
    ax.set_title("Twins vs Liouville lambda(n) on integers, same interval")
    ax.legend()
    fig.tight_layout()
    p_lio = out / "followup_twin_vs_liouville.png"
    fig.savefig(p_lio)
    plt.close(fig)
    log(f"Wrote {p_lio}")

    e_t = float(np.sum(mag5**2))
    e_l = float(np.sum(mag_lio**2))
    log(f"  total sum|FFT|^2 twin={e_t:.2f} liouville={e_l:.2f}")
    log(
        f"  peakiness max/median (k>=1): twin={spectrum_peakiness(mag5):.3f} "
        f"liouville={spectrum_peakiness(mag_lio):.3f}"
    )

    summary = out / "followup_summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nWrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
