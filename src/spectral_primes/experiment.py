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
from spectral_primes.subset import U_sparse_batch, reference_stats_sparse


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

    a_idx, b_idx, c_idx = _sample_groups_from_pool(
        rng, u_vals, centers, mu_u, sig_u, u_threshold_sigma, n_per_group, pool_size
    )

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


def _a_indices_high_u_sorted(
    u_vals: np.ndarray,
    mu_u: float,
    sig_u: float,
    u_threshold_sigma: float,
    n_per_group: int,
) -> list[int]:
    """Deterministic A: smallest pool indices among those above the high-U threshold."""
    thr = mu_u + u_threshold_sigma * sig_u
    idx_hi = sorted(i for i, u in enumerate(u_vals) if u > thr)
    if len(idx_hi) < n_per_group:
        raise ValueError(
            f"Not enough high-U samples ({len(idx_hi)} < {n_per_group}). "
            "Increase pool_factor or relax sigma threshold."
        )
    return idx_hi[:n_per_group]


def density_diff_sparse_fixed_b(
    gammas: np.ndarray,
    centers: list[int],
    b_idx: list[int],
    ref_x: np.ndarray,
    *,
    Lambda: float,
    delta: float,
    n_t: int,
    window_half: int,
    u_threshold_sigma: float,
    n_per_group: int,
    rank_anneal: float = 0.0,
) -> float:
    """
    mean density at A(high-U_sparse) minus mean at fixed control indices B.
    A = first n_per_group pool indices (sorted) with U > mu + k*sigma on ref stats.
    """
    mu_u, sig_u = reference_stats_sparse(
        ref_x, gammas, Lambda, delta, n_t, rank_anneal
    )
    if sig_u <= 0:
        return float("nan")
    x_arr = np.array(centers, dtype=np.float64)
    u_vals = U_sparse_batch(x_arr, gammas, Lambda, delta, n_t, rank_anneal)
    a_idx = _a_indices_high_u_sorted(u_vals, mu_u, sig_u, u_threshold_sigma, n_per_group)
    da = np.array([prime_density_per_1e5(centers[i], window_half) for i in a_idx])
    db = np.array([prime_density_per_1e5(centers[i], window_half) for i in b_idx])
    return float(da.mean() - db.mean())


def _sample_groups_from_pool(
    rng: random.Random,
    u_vals: np.ndarray,
    centers: list[int],
    mu_u: float,
    sig_u: float,
    u_threshold_sigma: float,
    n_per_group: int,
    pool_size: int,
) -> tuple[list[int], list[int], list[int]]:
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
            "Increase pool_factor, relax sigma threshold, or expand [x_lo, x_hi]."
        )
    a_idx = idx_hi[:n_per_group]
    c_idx = idx_lo[:n_per_group]
    b_idx = rng.sample(idx_all, n_per_group)
    return a_idx, b_idx, c_idx


def run_three_group_demo_sparse(
    gammas: np.ndarray,
    Lambda: float,
    delta: float,
    n_t: int,
    x_lo: int,
    x_hi: int,
    n_per_group: int,
    window_half: int,
    u_threshold_sigma: float = 1.5,
    ref_grid_points: int = 500,
    pool_factor: int = 80,
    seed: int = 42,
    rank_anneal: float = 0.0,
) -> tuple[GroupResult, GroupResult, GroupResult, dict, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    ref_x = np.linspace(x_lo, x_hi, ref_grid_points, dtype=np.float64)
    mu_u, sig_u = reference_stats_sparse(
        ref_x, gammas, Lambda, delta, n_t, rank_anneal
    )
    if sig_u <= 0:
        raise ValueError("Reference sigma_U is zero; adjust range, Lambda, or delta.")

    pool_size = max(n_per_group * pool_factor, 2000)
    span_lo = x_lo + window_half
    span_hi = x_hi - window_half
    if span_hi <= span_lo:
        raise ValueError("x range too narrow for window half-width")
    centers = [rng.randrange(span_lo, span_hi) for _ in range(pool_size)]
    x_arr = np.array(centers, dtype=np.float64)
    u_vals = U_sparse_batch(x_arr, gammas, Lambda, delta, n_t, rank_anneal)

    a_idx, b_idx, c_idx = _sample_groups_from_pool(
        rng, u_vals, centers, mu_u, sig_u, u_threshold_sigma, n_per_group, pool_size
    )

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
        "delta": delta,
        "n_t": n_t,
        "rank_anneal": rank_anneal,
        "u_threshold_sigma": u_threshold_sigma,
        "pool_size": pool_size,
        "operator": "U_sparse",
    }
    return (
        GroupResult("A_Max_U_sparse", float(ar.mean()), float(ar.std(ddof=1)), n_per_group),
        GroupResult("B_Control_random", float(br.mean()), float(br.std(ddof=1)), n_per_group),
        GroupResult("C_Min_U_sparse", float(cr.mean()), float(cr.std(ddof=1)), n_per_group),
        meta,
        ar,
        br,
        cr,
    )


def permutation_test_sparse(
    gammas: np.ndarray,
    Lambda: float,
    delta: float,
    n_t: int,
    x_lo: int,
    x_hi: int,
    n_per_group: int,
    window_half: int,
    n_perm: int,
    u_threshold_sigma: float = 1.5,
    ref_grid_points: int = 500,
    pool_factor: int = 80,
    seed: int = 42,
    rank_anneal: float = 0.05,
    allow_degenerate_null: bool = False,
) -> dict:
    """
    Fixed pool of centers and fixed control group B; only gammas are shuffled (Fisher–Yates).
    Statistic: mean(density at A_high-U_sparse) - mean(density at B). Two-sided empirical p-value.

    With rank_anneal=0, U_sparse is invariant under γ permutation (same multiset of frequencies);
    the shuffle null is degenerate unless allow_degenerate_null=True (diagnostics only). See README.
    """
    if rank_anneal == 0.0 and not allow_degenerate_null:
        raise ValueError(
            "permutation_test_sparse with rank_anneal=0: the γ-shuffle null is degenerate "
            "(U_sparse is invariant; see README sparse operator / permutation tests). "
            "Use rank_anneal > 0, or allow_degenerate_null=True for diagnostics only."
        )
    rng = random.Random(seed)
    pool_size = max(n_per_group * pool_factor, 2000)
    span_lo = x_lo + window_half
    span_hi = x_hi - window_half
    if span_hi <= span_lo:
        raise ValueError("x range too narrow for window half-width")
    centers = [rng.randrange(span_lo, span_hi) for _ in range(pool_size)]
    ref_x = np.linspace(x_lo, x_hi, ref_grid_points, dtype=np.float64)
    idx_all = list(range(pool_size))
    b_idx = rng.sample(idx_all, n_per_group)
    g = np.asarray(gammas, dtype=np.float64).copy()

    d_obs = density_diff_sparse_fixed_b(
        g,
        centers,
        b_idx,
        ref_x,
        Lambda=Lambda,
        delta=delta,
        n_t=n_t,
        window_half=window_half,
        u_threshold_sigma=u_threshold_sigma,
        n_per_group=n_per_group,
        rank_anneal=rank_anneal,
    )
    if d_obs != d_obs:
        raise ValueError("Observed statistic is NaN (sigma_U or design failure).")

    extremes = 0
    valid = 0
    for k in range(n_perm):
        gp = g.copy()
        prng = random.Random(seed + 1_000_003 + k)
        for i in range(len(gp) - 1, 0, -1):
            j = prng.randint(0, i)
            gp[i], gp[j] = gp[j], gp[i]
        try:
            d = density_diff_sparse_fixed_b(
                gp,
                centers,
                b_idx,
                ref_x,
                Lambda=Lambda,
                delta=delta,
                n_t=n_t,
                window_half=window_half,
                u_threshold_sigma=u_threshold_sigma,
                n_per_group=n_per_group,
                rank_anneal=rank_anneal,
            )
        except ValueError:
            continue
        if d != d:
            continue
        valid += 1
        if abs(d) >= abs(d_obs):
            extremes += 1

    p_two_sided = (1 + extremes) / (1 + valid) if valid else float("nan")
    return {
        "d_observed": d_obs,
        "n_perm": n_perm,
        "n_perm_valid": valid,
        "p_two_sided": p_two_sided,
        "pool_size": pool_size,
        "rank_anneal": rank_anneal,
    }
