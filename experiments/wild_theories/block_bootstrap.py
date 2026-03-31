"""
Moving-block bootstrap on the twin-prime indicator: permute contiguous blocks
to preserve within-block dependence while breaking long-range alignment.

Answers (empirically): does spectrum_peakiness drop vs the ~187 z from i.i.d. 1/log n?

  python -m experiments.wild_theories.block_bootstrap

Requires matplotlib. Output: experiments/wild_theories/output/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .fourier import magnitude_spectrum, spectrum_peakiness
from .signals import segment_twin_lower_indicator_sieve


def _out() -> Path:
    return Path(__file__).resolve().parent / "output"


def block_shuffle(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Split x into contiguous blocks of length `block_size`; shuffle block order.
    Remainder (if n not divisible) is left appended unchanged at the end.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    nb = n // block_size
    if nb <= 1:
        return x.copy()
    head = x[: nb * block_size].reshape(nb, block_size).copy()
    tail = x[nb * block_size :].copy()
    order = rng.permutation(nb)
    return np.concatenate([head[order].ravel(), tail])


def bootstrap_peakiness_distribution(
    x: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
    n_boot: int,
) -> np.ndarray:
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        y = block_shuffle(x, block_size, rng)
        out[i] = spectrum_peakiness(magnitude_spectrum(y))
    return out


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

    def log(msg: str) -> None:
        lines.append(msg)
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"))

    rng = np.random.default_rng(20260401)

    lo, hi = 2, 100_002
    log(f"twin mask [{lo}, {hi}) sieve ...")
    twin = segment_twin_lower_indicator_sieve(lo, hi)
    n = twin.size
    mag0 = magnitude_spectrum(twin)
    pk_obs = spectrum_peakiness(mag0)
    log(f"  n={n}, mean={twin.mean():.6f}, spectrum_peakiness(data)={pk_obs:.4f}")

    block_sizes = [50, 100, 500, 1000, 2000]
    n_boot = 450
    log(f"\nBlock bootstrap: n_boot={n_boot} per block size")

    hist_B = 500
    hist_samples: np.ndarray | None = None
    z_by_B: list[float] = []

    for B in block_sizes:
        samp = bootstrap_peakiness_distribution(twin, B, rng, n_boot)
        m, s = float(samp.mean()), float(samp.std(ddof=1))
        z = (pk_obs - m) / s if s > 1e-15 else float("nan")
        z_by_B.append(z)
        p_right = float(np.mean(samp >= pk_obs))
        log(
            f"  B={B:4d}: boot mean={m:.3f} std={s:.3f}  z(obs vs boot)={z:.2f}  P(boot>=obs)={p_right:.4f}"
        )
        if B == hist_B:
            hist_samples = samp

    summary_path = out / "block_bootstrap_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nWrote {summary_path}")

    if hist_samples is not None:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
        ax.hist(
            hist_samples,
            bins=32,
            color="#3498db",
            alpha=0.85,
            density=True,
            label=f"block bootstrap B={hist_B}",
        )
        ax.axvline(
            pk_obs,
            color="#c0392b",
            lw=2,
            label=f"observed twins peakiness={pk_obs:.1f}",
        )
        ax.set_xlabel("spectrum_peakiness (max/median |FFT|, k>=1)")
        ax.set_ylabel("density")
        ax.set_title("Twin indicator on n=1e5: block-shuffled null (B=500)")
        ax.legend()
        fig.tight_layout()
        pfig = out / "block_bootstrap_peakiness_B500.png"
        fig.savefig(pfig)
        plt.close(fig)
        log(f"Wrote {pfig}")

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.bar(
        range(len(block_sizes)),
        z_by_B,
        tick_label=[str(b) for b in block_sizes],
        color="#2c3e50",
    )
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("block size B")
    ax.set_ylabel("z: (obs peakiness - mean_boot) / sd_boot")
    ax.set_title("How extreme is twin peakiness vs moving-block null?")
    fig.tight_layout()
    p2 = out / "block_bootstrap_z_vs_B.png"
    fig.savefig(p2)
    plt.close(fig)
    log(f"Wrote {p2}")

    log(
        "\nInterpretation: i.i.d. 1/log surrogate (level2_spectral) had mean peakiness ~4 vs obs ~48 (z~187)."
        " Block bootstrap keeps local runs of 0/1 inside each block, so boot mean stays ~43-45."
        " Residual z~20-30 means true order still boosts peakiness a bit vs shuffled blocks,"
        " but most elevation vs i.i.d. is explained by local twin clustering, not global alignment alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
