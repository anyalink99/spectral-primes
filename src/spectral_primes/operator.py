"""
Regularized spectral operator from the preprint:

    U(x, Λ) = Σ_{n : γ_n ≤ Λ}  ω(γ_n, Λ) · cos(γ_n ln x) / γ_n,
    ω(γ, Λ) = exp(−γ² / (2Λ²)).

Phase φ_n ≡ 0 as in the document.
"""
from __future__ import annotations

import numpy as np


def U_batch(
    x: np.ndarray,
    gammas: np.ndarray,
    Lambda: float,
) -> np.ndarray:
    """
    Vectorized U at points x > 0. Only zeros with γ_n ≤ Λ enter the sum.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("x must be positive")
    g = np.asarray(gammas, dtype=np.float64)
    g = g[g <= Lambda]
    if g.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    w = np.exp(-(g**2) / (2.0 * Lambda**2))
    ln_x = np.log(x)
    phase = g[:, np.newaxis] * ln_x[np.newaxis, :]
    terms = (w[:, np.newaxis] * np.cos(phase)) / g[:, np.newaxis]
    return terms.sum(axis=0)


def reference_stats(
    x_grid: np.ndarray,
    gammas: np.ndarray,
    Lambda: float,
) -> tuple[float, float]:
    """Sample mean μ and standard deviation σ of U on x_grid."""
    u = U_batch(x_grid, gammas, Lambda)
    return float(u.mean()), float(u.std(ddof=1))
