r"""
Spectral operators built from the imaginary parts γ_n of the nontrivial zeta zeros.

Two distinct objects come out of the explicit formula. Getting them straight is the
whole point of the v0.5 correction (see ERRATA.md):

1. ``U_batch`` — the operator literally written in the original manuscript:

       U(x, Λ) = Σ_{γ_n ≤ Λ}  ω(γ_n, Λ) · cos(γ_n ln x) / γ_n,
       ω(γ, Λ) = exp(−γ² / (2Λ²)).

   This is kept for provenance, but it is **not** the right object for predicting
   local prime *density*, and its sign convention is backwards for that purpose
   (empirically it ANTI-correlates with window density — see experiments/verification).
   The 1/γ weight also makes it dominated by the few lowest zeros.

2. ``U_density_batch`` — the term that actually governs the *density* of primes.
   Differentiating the von Mangoldt explicit formula ψ(x) = x − Σ_ρ x^ρ/ρ − … gives

       ψ'(x) = 1 − Σ_ρ x^{ρ−1} = 1 − 2 x^{−1/2} Σ_{γ>0} cos(γ ln x)            (RH)

   so the prime density near x is ≈ (1/ln x)·(1 − 2 x^{−1/2} Σ cos(γ ln x)).
   The oscillatory driver of density is therefore **cos with NO 1/γ weight**, and
   primes are LOCALLY DENSER where Σ cos(γ ln x) is negative. We define

       D(x, Λ) = − Σ_{γ_n ≤ Λ} ω(γ_n, Λ) · cos(γ_n ln x)

   with the leading minus sign so that **high D ⟺ more primes**. The Gaussian ω is
   the regulator that makes the truncated sum well-behaved.

The historical phase confusion in the manuscript ("φ_n = arg(ρ_n) = π/2 ⇒ φ_n = 0")
was wrong: Re(x^ρ/ρ) ∝ ((1/2)cos + γ sin)/(1/4+γ²) ≈ sin(γ ln x)/γ for the *counting*
function, while the *density* uses cos with no 1/γ. Neither matches the manuscript's
+cos/γ. ``U_psi_batch`` provides the counting-phase object for reference.
"""
from __future__ import annotations

import numpy as np


def _filter_weights(gammas: np.ndarray, Lambda: float):
    g = np.asarray(gammas, dtype=np.float64)
    g = g[g <= Lambda]
    w = np.exp(-(g**2) / (2.0 * Lambda**2)) if g.size else g
    return g, w


def U_batch(x: np.ndarray, gammas: np.ndarray, Lambda: float) -> np.ndarray:
    """Original manuscript operator U(x,Λ) = Σ ω cos(γ ln x)/γ (provenance only).

    NOTE: kept faithful to the preprint. For predicting prime *density* prefer
    :func:`U_density_batch`; ``U_batch`` has the wrong phase/weight and its sign is
    backwards (it anti-correlates with density). Only zeros with γ_n ≤ Λ enter.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("x must be positive")
    g, w = _filter_weights(gammas, Lambda)
    if g.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    phase = g[:, np.newaxis] * np.log(x)[np.newaxis, :]
    terms = (w[:, np.newaxis] * np.cos(phase)) / g[:, np.newaxis]
    return terms.sum(axis=0)


def U_density_batch(x: np.ndarray, gammas: np.ndarray, Lambda: float) -> np.ndarray:
    """Density operator D(x,Λ) = −Σ ω cos(γ ln x). High D ⟺ locally more primes.

    This is the (regularized) oscillatory part of ψ'(x) from the explicit formula and
    is the operator that genuinely tracks local prime density. The correlation is real
    but tiny (it is an O(x^{-1/2}) fluctuation); see experiments/verification.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("x must be positive")
    g, w = _filter_weights(gammas, Lambda)
    if g.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    phase = g[:, np.newaxis] * np.log(x)[np.newaxis, :]
    terms = -(w[:, np.newaxis] * np.cos(phase))
    return terms.sum(axis=0)


def U_psi_batch(x: np.ndarray, gammas: np.ndarray, Lambda: float) -> np.ndarray:
    """Counting-phase operator Σ ω sin(γ ln x)/γ (≈ oscillation of (ψ(x)−x)/√x).

    Provided for reference: this is the correct phase/weight for the *counting*
    function ψ(x), as opposed to the *density* object :func:`U_density_batch`.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("x must be positive")
    g, w = _filter_weights(gammas, Lambda)
    if g.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    phase = g[:, np.newaxis] * np.log(x)[np.newaxis, :]
    terms = (w[:, np.newaxis] * np.sin(phase)) / g[:, np.newaxis]
    return terms.sum(axis=0)


def reference_stats(
    x_grid: np.ndarray,
    gammas: np.ndarray,
    Lambda: float,
) -> tuple[float, float]:
    """Sample mean μ and standard deviation σ of U on x_grid (original operator)."""
    u = U_batch(x_grid, gammas, Lambda)
    return float(u.mean()), float(u.std(ddof=1))


def reference_stats_density(
    x_grid: np.ndarray,
    gammas: np.ndarray,
    Lambda: float,
) -> tuple[float, float]:
    """Sample mean μ and std σ of the density operator D on x_grid."""
    u = U_density_batch(x_grid, gammas, Lambda)
    return float(u.mean()), float(u.std(ddof=1))
