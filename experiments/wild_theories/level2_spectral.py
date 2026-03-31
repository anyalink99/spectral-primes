"""
Deeper spectral null checks (not the zeta U(x) operator):

1) Cyclic shifts vs full-length |FFT| (invariance) + Welch stability under random roll.
2) Welch-averaged spectrum on a long twin segment (~1e5).
3) Inhomogeneous Bernoulli surrogate ~ 1/log(n) with matched mean; peakiness distribution.

  python -m experiments.wild_theories.level2_spectral

Requires matplotlib. Writes under experiments/wild_theories/output/
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from .fourier import magnitude_spectrum, spectrum_peakiness
from .signals import segment_twin_lower_indicator, segment_twin_lower_indicator_sieve


def _out() -> Path:
    return Path(__file__).resolve().parent / "output"


def welch_mean_magnitude(
    x: np.ndarray,
    seg_len: int,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Average |rfft(windowed segment)| / sqrt(seg_len) over Hann-windowed segments.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    hop = max(int(round(seg_len * (1.0 - overlap))), 1)
    w = np.hanning(seg_len)
    acc = None
    count = 0
    for start in range(0, n - seg_len + 1, hop):
        seg = x[start : start + seg_len]
        xc = (seg - seg.mean()) * w
        m = np.abs(np.fft.rfft(xc)) / math.sqrt(seg_len)
        if acc is None:
            acc = np.zeros_like(m, dtype=np.float64)
        acc += m
        count += 1
    if count == 0 or acc is None:
        return np.zeros(max(seg_len // 2 + 1, 0), dtype=np.float64)
    return acc / count


def cyclic_shift_full_fft_max_error(x: np.ndarray, shifts: list[int]) -> float:
    """Max L_inf diff of magnitude spectrum vs unshifted (should be ~0 for true circular roll)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    m0 = magnitude_spectrum(x)
    worst = 0.0
    for s in shifts:
        mr = magnitude_spectrum(np.roll(x, s))
        worst = max(worst, float(np.max(np.abs(mr - m0))))
    return worst


def surrogate_inhomogeneous_log(
    lo: int,
    hi: int,
    target_mean: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independent Bernoulli with p(n) proportional to 1/log(n), mean matched to target_mean."""
    idx = np.arange(lo, hi, dtype=np.float64)
    w = 1.0 / np.log(np.maximum(idx, 2.0))
    w = w / float(np.mean(w))
    p = w * float(target_mean)
    p = np.clip(p, 0.0, 1.0)
    return rng.binomial(1, p, size=hi - lo).astype(np.float64)


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

    rng = np.random.default_rng(42)

    # Sanity: sieve matches SymPy on a short range
    a = segment_twin_lower_indicator(2, 8000)
    b = segment_twin_lower_indicator_sieve(2, 8000)
    if not np.allclose(a, b):
        raise RuntimeError("sieve twin mask mismatch vs SymPy")
    log("sieve vs SymPy twin mask [2,8000): OK")

    lo, hi = 2, 100_002
    log(f"Building twin mask [{lo}, {hi}) via sieve ...")
    twin = segment_twin_lower_indicator_sieve(lo, hi)
    L = twin.size
    mu = float(twin.mean())
    log(f"  n={L}, count_ones={int(twin.sum())}, mean={mu:.6f}")

    # --- 1) Full FFT invariance under cyclic shift ---
    shifts = [0, 1, 17, L // 3, L // 2, L - 1]
    err = cyclic_shift_full_fft_max_error(twin, shifts)
    log(f"\nCyclic shift |FFT| invariance (full length): max diff over shifts {shifts[:4]}... = {err:.3e}")
    log(
        "  Note: for length-N DFT, |FFT(roll(x,s))| == |FFT(x)|; phase randomization by roll"
        " does not change magnitude spectrum of the whole vector."
    )

    seg_len = 5000
    hop = max(int(round(seg_len * 0.5)), 1)
    base_welch = welch_mean_magnitude(twin, seg_len, overlap=0.5)
    roll_stds = []
    roll_max = []
    n_roll_trials = 40
    for _ in range(n_roll_trials):
        s = int(rng.integers(0, L))
        wmag = welch_mean_magnitude(np.roll(twin, s), seg_len, overlap=0.5)
        roll_stds.append(float(np.std(wmag - base_welch)))
        roll_max.append(float(np.max(wmag[1:])))
    log(
        f"\nWelch seg_len={seg_len} hop={hop}: under random cyclic roll,"
        f" std(mean mag diff vs no roll) ~ {np.mean(roll_stds):.4f} (mean over {n_roll_trials} rolls)"
    )
    log(f"  max Welch bin (k>=1) no roll: {float(np.max(base_welch[1:])):.4f}")
    log(
        f"  max Welch bin over rolls: mean={np.mean(roll_max):.4f} std={np.std(roll_max):.4f}"
    )

    # --- 2) Welch on 100k: twin vs surrogate mean band ---
    seg2 = 10_000
    w_twin = welch_mean_magnitude(twin, seg2, overlap=0.5)
    nu2 = np.arange(w_twin.size) / seg2

    n_sur = 80
    sur_stack = []
    pk_sur = []
    for _ in range(n_sur):
        s = surrogate_inhomogeneous_log(lo, hi, mu, rng)
        wm = welch_mean_magnitude(s, seg2, overlap=0.5)
        sur_stack.append(wm)
        pk_sur.append(spectrum_peakiness(magnitude_spectrum(s)))
    sur_arr = np.stack(sur_stack, axis=0)
    sur_mean = sur_arr.mean(axis=0)
    sur_std = sur_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
    ax.plot(nu2, w_twin, color="#117a3d", lw=1.0, label=f"twins Welch avg (Lseg={seg2})")
    ax.fill_between(
        nu2,
        np.maximum(sur_mean - 2 * sur_std, 0),
        sur_mean + 2 * sur_std,
        color="#7f8c8d",
        alpha=0.35,
        label="surrogate 1/log n +/-2sd (80 draws)",
    )
    ax.plot(nu2, sur_mean, color="#34495e", ls="--", lw=0.9, label="surrogate mean")
    ax.set_xlim(0, 0.15)
    ax.set_xlabel("nu = k / Lseg")
    ax.set_ylabel("mean |rFFT| / sqrt(Lseg)")
    ax.set_title("Welch-averaged spectrum: twins vs inhomogeneous Bernoulli (mean matched)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p_w = out / "level2_welch_twin_vs_surrogate.png"
    fig.savefig(p_w)
    plt.close(fig)
    log(f"\nWrote {p_w}")

    seg_small = 5000
    w_twin_s = welch_mean_magnitude(twin, seg_small, overlap=0.5)
    nu_s = np.arange(w_twin_s.size) / seg_small
    sur_stack_s = []
    for _ in range(50):
        s = surrogate_inhomogeneous_log(lo, hi, mu, rng)
        sur_stack_s.append(welch_mean_magnitude(s, seg_small, overlap=0.5))
    sur_s = np.stack(sur_stack_s, axis=0)
    sm_m, sm_sd = sur_s.mean(axis=0), sur_s.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
    ax.plot(nu_s, w_twin_s, color="#117a3d", lw=1.0, label=f"twins Lseg={seg_small}")
    ax.fill_between(
        nu_s,
        np.maximum(sm_m - 2 * sm_sd, 0),
        sm_m + 2 * sm_sd,
        color="#7f8c8d",
        alpha=0.35,
        label="surrogate +/-2sd (50)",
    )
    ax.plot(nu_s, sm_m, color="#34495e", ls="--", lw=0.9)
    ax.set_xlim(0, 0.15)
    ax.set_xlabel("nu = k / Lseg")
    ax.set_ylabel("mean |rFFT| / sqrt(Lseg)")
    ax.set_title(f"Welch (Lseg={seg_small}): twins vs surrogate")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p_ws = out / "level2_welch_L5000.png"
    fig.savefig(p_ws)
    plt.close(fig)
    log(f"Wrote {p_ws}")

    # --- 3) Peakiness: full-length FFT on whole 100k window ---
    mag_full = magnitude_spectrum(twin)
    pk_twin = spectrum_peakiness(mag_full)
    pk_surr = []
    for _ in range(200):
        s = surrogate_inhomogeneous_log(lo, hi, mu, rng)
        pk_surr.append(spectrum_peakiness(magnitude_spectrum(s)))
    pk_surr = np.array(pk_surr, dtype=np.float64)
    z = (pk_twin - pk_surr.mean()) / max(pk_surr.std(ddof=1), 1e-15)

    log(f"\nPeakiness max/median (full FFT, n={L}): twin={pk_twin:.3f}")
    log(
        f"  surrogate (200): mean={pk_surr.mean():.3f} std={pk_surr.std(ddof=1):.3f} z={z:.2f}"
    )

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.hist(pk_surr, bins=25, color="#95a5a6", alpha=0.85, density=True, label="surrogate")
    ax.axvline(pk_twin, color="#c0392b", lw=2, label=f"twin (peakiness={pk_twin:.2f})")
    ax.set_xlabel("peakiness = max/median |FFT|, k>=1")
    ax.set_ylabel("density")
    ax.set_title("Twin vs 1/log(n) surrogate (full-length spectrum)")
    ax.legend()
    fig.tight_layout()
    p_h = out / "level2_peakiness_hist.png"
    fig.savefig(p_h)
    plt.close(fig)
    log(f"Wrote {p_h}")

    summary = out / "level2_summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nWrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
