"""
Blind-style comparison (preprint §3): Max-U vs random vs Min-U windows.

Uses moderate x (e.g. 10^6–10^7) so prime counting stays feasible; the
methodology mirrors the paper without claiming the same p-values.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from spectral_primes.operator import U_batch, reference_stats
from spectral_primes.primes import prime_density_per_1e5


@dataclass
class GroupResult:
    name: str
    mean_density: float
    std_density: float
    n: int


def run_three_group_demo(
    gammas: np.ndarray,
    Lambda: float,
    x_lo: int,
    x_hi: int,
    n_per_group: int,
    window_half: int,
    u_threshold_sigma: float = 1.5,
    ref_grid_points: int = 500,
    pool_factor: int = 80,
    seed: int = 42,
) -> tuple[GroupResult, GroupResult, GroupResult, dict, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    ref_x = np.linspace(x_lo, x_hi, ref_grid_points, dtype=np.float64)
    mu_u, sig_u = reference_stats(ref_x, gammas, Lambda)
    if sig_u <= 0:
        raise ValueError("Reference σ_U is zero; widen x range or adjust Λ")

    pool_size = max(n_per_group * pool_factor, 2000)
    span_lo = x_lo + window_half
    span_hi = x_hi - window_half
    if span_hi <= span_lo:
        raise ValueError("x range too narrow for window half-width")
    centers = [rng.randrange(span_lo, span_hi) for _ in range(pool_size)]
    x_arr = np.array(centers, dtype=np.float64)
    u_vals = U_batch(x_arr, gammas, Lambda)

    thr_hi = mu_u + u_threshold_sigma * sig_u
    thr_lo = mu_u - u_threshold_sigma * sig_u
    idx_hi = [i for i, u in enumerate(u_vals) if u > thr_hi]
    idx_lo = [i for i, u in enumerate(u_vals) if u < thr_lo]
    idx_all = list(range(pool_size))

    rng.shuffle(idx_hi)
    rng.shuffle(idx_lo)
    if len(idx_hi) < n_per_group or len(idx_lo) < n_per_group:
        raise ValueError(
            f"Not enough extreme-U samples (high={len(idx_hi)}, low={len(idx_lo)}). "
            "Increase pool_factor, relax σ threshold, or expand [x_lo, x_hi]."
        )

    a_idx = idx_hi[:n_per_group]
    c_idx = idx_lo[:n_per_group]
    b_idx = rng.sample(idx_all, n_per_group)

    def densities(idxs: list[int]) -> list[float]:
        return [prime_density_per_1e5(centers[i], window_half) for i in idxs]

    da, db, dc = densities(a_idx), densities(b_idx), densities(c_idx)
    ar = np.array(da, dtype=np.float64)
    br = np.array(db, dtype=np.float64)
    cr = np.array(dc, dtype=np.float64)

    meta = {
        "mu_U": mu_u,
        "sigma_U": sig_u,
        "Lambda": Lambda,
        "u_threshold_sigma": u_threshold_sigma,
        "pool_size": pool_size,
    }
    return (
        GroupResult("A_Max_U", float(ar.mean()), float(ar.std(ddof=1)), n_per_group),
        GroupResult("B_Control_random", float(br.mean()), float(br.std(ddof=1)), n_per_group),
        GroupResult("C_Min_U", float(cr.mean()), float(cr.std(ddof=1)), n_per_group),
        meta,
        ar,
        br,
        cr,
    )


def z_test_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    se = (va / na + vb / nb) ** 0.5
    if se == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / se)
