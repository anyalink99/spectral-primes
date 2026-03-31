"""
Variance-based zero selection (preprint §2.2).

For interval [x - δ, x + δ] define

    E_n(x) = (1/(2δ)) ∫ [ r_n · ω(γ_n,Λ) cos(γ_n ln t) / γ_n ]^2 dt,

approximated by the mean of the squared term on a uniform grid in t.
Here r_n = exp(rank_anneal · (slot_n+1)/N_slots) is optional (default 1): it depends on
the **index** of the zero in the ordered list, not only on γ_n. With rank_anneal=0,
only γ_n enters; then shuffling the multiset of γ across slots leaves the multiset
{Var(γ)} and hence U_sparse(x) invariant — a permutation test on γ alone is degenerate.
A small rank_anneal>0 breaks that symmetry for null simulations (see README).
"""
from __future__ import annotations

import numpy as np


def _filtered_slots(gammas: np.ndarray, Lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """γ values with γ≤Λ and their original slot indices (0-based)."""
    g = np.asarray(gammas, dtype=np.float64)
    m = g <= Lambda
    return g[m], np.nonzero(m)[0].astype(np.float64)


def _rank_mul(slot_idx: np.ndarray, n_slots: int, rank_anneal: float) -> np.ndarray:
    if rank_anneal == 0.0 or n_slots <= 0:
        return np.ones(slot_idx.shape[0], dtype=np.float64)
    return np.exp(rank_anneal * (slot_idx + 1.0) / float(n_slots))


def energies_on_interval(
    x_center: float,
    delta: float,
    gammas: np.ndarray,
    Lambda: float,
    n_t: int,
    rank_anneal: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (g_filtered, E, slot_idx) where E[n] is mean_t of (r_n ω cos(γ ln t)/γ)^2 on [x±δ].
    """
    if x_center <= delta or n_t < 2:
        raise ValueError("need x_center > delta and n_t >= 2")
    g, slot_idx = _filtered_slots(gammas, Lambda)
    if g.size == 0:
        return g, np.zeros(0, dtype=np.float64), slot_idx
    n_all = int(gammas.shape[0])
    rk = _rank_mul(slot_idx, n_all, rank_anneal)
    t = np.linspace(x_center - delta, x_center + delta, n_t, dtype=np.float64)
    ln_t = np.log(t)
    w = np.exp(-(g**2) / (2.0 * Lambda**2))
    phase = g[:, np.newaxis] * ln_t[np.newaxis, :]
    terms = (rk[:, np.newaxis] * w[:, np.newaxis] * np.cos(phase)) / g[:, np.newaxis]
    E = np.mean(terms**2, axis=1)
    return g, E, slot_idx


def relative_var_weights(E: np.ndarray) -> np.ndarray:
    s = float(E.sum())
    if s <= 0:
        return np.ones_like(E) / max(E.size, 1)
    return E / s


def subset_mask(Var: np.ndarray, N: int) -> np.ndarray:
    """Var_n > 1/N; if empty, keep argmax Var (single term)."""
    if N <= 0 or Var.size == 0:
        return np.zeros(Var.size, dtype=bool)
    thr = 1.0 / N
    mask = Var > thr
    if not np.any(mask):
        mask = np.zeros_like(mask)
        mask[int(np.argmax(Var))] = True
    return mask


def U_sparse_at(
    x_center: float,
    gammas: np.ndarray,
    Lambda: float,
    delta: float,
    n_t: int,
    rank_anneal: float = 0.0,
) -> tuple[float, int, int]:
    """U_sparse at x_center; returns (value, |S|, N)."""
    if x_center <= 0:
        raise ValueError("x_center must be positive")
    g, E, slot_idx = energies_on_interval(
        x_center, delta, gammas, Lambda, n_t, rank_anneal
    )
    N = g.size
    if N == 0:
        return 0.0, 0, 0
    Var = relative_var_weights(E)
    mask = subset_mask(Var, N)
    n_all = int(gammas.shape[0])
    rk = _rank_mul(slot_idx, n_all, rank_anneal)
    w = np.exp(-(g**2) / (2.0 * Lambda**2))
    ln_x = np.log(x_center)
    phase = g * ln_x
    contrib = (rk * w * np.cos(phase)) / g
    u = float(np.sum(contrib[mask]))
    return u, int(mask.sum()), N


def U_sparse_batch(
    x: np.ndarray,
    gammas: np.ndarray,
    Lambda: float,
    delta: float,
    n_t: int,
    rank_anneal: float = 0.0,
) -> np.ndarray:
    """U_sparse at each x[i] (own subset per point)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        out[i], _, _ = U_sparse_at(float(x[i]), gammas, Lambda, delta, n_t, rank_anneal)
    return out


def reference_stats_sparse(
    x_grid: np.ndarray,
    gammas: np.ndarray,
    Lambda: float,
    delta: float,
    n_t: int,
    rank_anneal: float = 0.0,
) -> tuple[float, float]:
    u = U_sparse_batch(x_grid, gammas, Lambda, delta, n_t, rank_anneal)
    return float(u.mean()), float(u.std(ddof=1))
