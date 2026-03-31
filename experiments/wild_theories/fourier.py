"""Normalized FFT energy for real sequences; compare to random baseline."""
from __future__ import annotations

import math

import numpy as np


def magnitude_spectrum(x: np.ndarray) -> np.ndarray:
    """
    One-sided magnitudes for real x (demeaned inside).
    Returns |RFFT(x - mean)| / sqrt(n) so scales are comparable across n.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    xc = x - x.mean()
    return np.abs(np.fft.rfft(xc)) / math.sqrt(n)


def spectrum_peakiness(magnitude: np.ndarray) -> float:
    """max/median of |FFT| bins for k >= 1 (DC excluded). Exploratory spike metric."""
    m = np.asarray(magnitude, dtype=np.float64).ravel()[1:]
    if m.size == 0:
        return float("nan")
    med = float(np.median(m))
    return float(np.max(m) / med) if med > 1e-12 else float("inf")


def integrated_energy_band(mag: np.ndarray, k_lo: int, k_hi: int) -> float:
    """Sum of mag[k]^2 for k in [k_lo, k_hi) (indices along rfft axis)."""
    if mag.size == 0:
        return 0.0
    k_lo = max(0, k_lo)
    k_hi = min(mag.size, k_hi)
    if k_hi <= k_lo:
        return 0.0
    return float(np.sum(mag[k_lo:k_hi] ** 2))


def compare_to_random_trials(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    n_trials: int = 50,
    p_match: float | None = None,
) -> dict:
    """
    x is typically a 0/1 sequence. If p_match is None, uses mean(x) as Bernoulli p for null.
    Returns mean/std of total spectral energy of random masks vs energy of x.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return {"n": 0, "energy_data": 0.0, "energy_random_mean": float("nan")}
    p = float(x.mean()) if p_match is None else float(p_match)
    mag_d = magnitude_spectrum(x)
    energy_data = float(np.sum(mag_d**2))

    trials = []
    for _ in range(n_trials):
        y = rng.binomial(1, p, size=n).astype(np.float64)
        mag_y = magnitude_spectrum(y)
        trials.append(float(np.sum(mag_y**2)))
    arr = np.array(trials, dtype=np.float64)
    return {
        "n": n,
        "p_bernoulli": p,
        "energy_data": energy_data,
        "energy_random_mean": float(arr.mean()),
        "energy_random_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "z_vs_random": float((energy_data - arr.mean()) / arr.std(ddof=1))
        if arr.size > 1 and arr.std(ddof=1) > 0
        else float("nan"),
    }
