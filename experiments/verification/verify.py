#!/usr/bin/env python3
"""
Honest empirical verification of the spectral-primes claims.

Three questions, answered with REAL primes (segmented sieve, not Cramer) and the
REAL zeta zeros shipped in data/zeros.csv:

  A. PHASE / SIGN. Which spectral object actually tracks local prime density?
     - U_old   = +Σ ω(γ) cos(γ ln x)/γ          (the manuscript's operator)
     - U_sin   = +Σ ω(γ) sin(γ ln x)/γ          (ψ(x)-counting phase, ~explicit formula)
     - D_dens  = -Σ ω(γ) cos(γ ln x)            (ψ'(x) density term: high D ⟺ more primes)
     We correlate each with window prime density and report Pearson r + A/B/C means.

  B. COMPLEXITY CLAIM |S| = O(√N log N). We measure |S| (Var_n>1/N subset size)
     as N grows, with Λ=c√N (paper's regime) and with Λ fixed, and compare to the
     plain zero-count below Λ. The point: is |S| a property of the *selection*, or
     just the Riemann–von Mangoldt zero count N(Λ)?

  C. SPEEDUP / APPLIED CLAIM. If we mark "resonance zones" (top operator values) and
     only sieve those, what fraction of primes do we capture vs fraction of x scanned?
     This is the honest version of "15-20% sieve reduction at <2% loss".
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


# ---------------------------------------------------------------------------
# Data + fast real primes
# ---------------------------------------------------------------------------
def load_gammas(path: str) -> np.ndarray:
    g = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                g.append(float(row[1]))
    return np.array(sorted(g), dtype=np.float64)


def sieve_prefix(x_max: int) -> np.ndarray:
    """prefix[k] = number of primes < k, for k in [0, x_max]. O(x_max log log x_max)."""
    is_p = np.ones(x_max, dtype=bool)
    is_p[:2] = False
    for i in range(2, int(math.isqrt(x_max - 1)) + 1):
        if is_p[i]:
            is_p[i * i :: i] = False
    pref = np.zeros(x_max + 1, dtype=np.int64)
    np.cumsum(is_p, out=pref[1:])
    return pref


def window_density(pref: np.ndarray, centers: np.ndarray, half: int) -> np.ndarray:
    lo = (centers - half).astype(np.int64)
    hi = (centers + half).astype(np.int64)
    return 1e5 * (pref[hi] - pref[lo]) / (hi - lo)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def _wg(gammas: np.ndarray, Lambda: float):
    g = gammas[gammas <= Lambda]
    w = np.exp(-(g**2) / (2.0 * Lambda**2))
    return g, w


def op_value(x: np.ndarray, gammas: np.ndarray, Lambda: float, kind: str) -> np.ndarray:
    g, w = _wg(gammas, Lambda)
    if g.size == 0:
        return np.zeros_like(x)
    phase = g[:, None] * np.log(x)[None, :]
    if kind == "U_old":  # manuscript: +cos / gamma
        terms = (w[:, None] * np.cos(phase)) / g[:, None]
    elif kind == "U_sin":  # psi(x) explicit-formula phase: +sin / gamma
        terms = (w[:, None] * np.sin(phase)) / g[:, None]
    elif kind == "D_dens":  # psi'(x) density term: -cos, no 1/gamma. high D <=> more primes
        terms = -(w[:, None] * np.cos(phase))
    else:
        raise ValueError(kind)
    return terms.sum(axis=0)


# ---------------------------------------------------------------------------
# Var_n subset (preprint 2.2) — operator-agnostic energy
# ---------------------------------------------------------------------------
def subset_size(
    x_center: float, gammas: np.ndarray, Lambda: float, delta: float, n_t: int, kind: str
) -> int:
    g, w = _wg(gammas, Lambda)
    if g.size == 0:
        return 0
    t = np.linspace(x_center - delta, x_center + delta, n_t)
    ln_t = np.log(t)
    phase = g[:, None] * ln_t[None, :]
    if kind == "D_dens":
        base = -(w[:, None] * np.cos(phase))
    elif kind == "U_sin":
        base = (w[:, None] * np.sin(phase)) / g[:, None]
    else:
        base = (w[:, None] * np.cos(phase)) / g[:, None]
    E = np.mean(base**2, axis=1)
    s = E.sum()
    if s <= 0:
        return 0
    Var = E / s
    N = g.size
    return int(np.count_nonzero(Var > 1.0 / N))


def riemann_von_mangoldt_count(Lambda: float) -> float:
    """Asymptotic number of zeros with 0 < gamma <= Lambda."""
    if Lambda < 2 * math.pi:
        return 0.0
    t = Lambda / (2 * math.pi)
    return t * math.log(t) - t + 7.0 / 8.0


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def group_means(op: np.ndarray, dens: np.ndarray, k: float = 1.5):
    mu, sig = op.mean(), op.std(ddof=1)
    hi = dens[op > mu + k * sig]
    lo = dens[op < mu - k * sig]
    mid = dens
    return (
        (hi.mean() if hi.size else float("nan"), hi.size),
        (mid.mean(), mid.size),
        (lo.mean() if lo.size else float("nan"), lo.size),
    )


def experiment_A(gammas, pref, rng, x_lo=2_000_000, x_hi=9_000_000,
                 half=500, n=200000):
    centers = rng.integers(x_lo + half, x_hi - half, size=n)
    cx = centers.astype(np.float64)
    dens = window_density(pref, centers, half)
    print(f"\n=== A. PHASE / SIGN  (x∈[{x_lo:,},{x_hi:,}], half={half}, "
          f"windows={n:,}) ===")
    print(f"mean window density = {dens.mean():.3f} per 1e5  "
          f"(≈ 1/ln x ⇒ {1e5/math.log((x_lo+x_hi)/2):.1f})")
    print("r = Pearson corr(operator, density); Z = r·√n (significance vs r=0);")
    print("high-op/low-op = mean density in the |op−μ|>1.5σ tails.\n")
    for Lambda in (80.0, 200.0):
        n_zeros = int(np.count_nonzero(gammas <= Lambda))
        print(f"  Λ={Lambda:.0f}  ({n_zeros} zeros γ≤Λ):")
        print(f"    {'operator':9s} {'r':>9s} {'Z':>7s}   "
              f"{'high-op':>10s} {'all':>9s} {'low-op':>10s}")
        for kind in ("U_old", "U_sin", "D_dens"):
            ov = op_value(cx, gammas, Lambda, kind)
            r = pearson(ov, dens)
            z = r * math.sqrt(n)
            (a_m, _), (m_m, _), (c_m, _) = group_means(ov, dens)
            print(f"    {kind:9s} {r:>+9.4f} {z:>+7.2f}   "
                  f"{a_m:>10.3f} {m_m:>9.3f} {c_m:>10.3f}")
    return dens


def experiment_B(gammas, Lambda_fixed=80.0, delta=300.0, n_t=64, x0=5_000_000.0):
    print(f"\n=== B. |S| SCALING  (Var_n>1/N subset, x0={x0:,.0f}, δ={delta}, "
          f"n_t={n_t}) ===")
    Ns = [100, 200, 400, 800, 1600, 2400]
    Ns = [N for N in Ns if N <= gammas.size]
    print("regime Λ = c·√N  (paper's claim |S| = O(√N log N)):")
    print(f"{'N':>6s} {'Λ=c√N':>8s} {'#γ≤Λ':>6s} {'|S|':>6s} {'√N·lnN':>9s} "
          f"{'RvM N(Λ)':>9s}")
    c = 80.0 / math.sqrt(800.0)  # calibrate so Λ≈80 at N=800 (matches default)
    rows = []
    for N in Ns:
        g = gammas[:N]
        Lam = c * math.sqrt(N)
        n_below = int(np.count_nonzero(g <= Lam))
        s = subset_size(x0, g, Lam, delta, n_t, "U_old")
        rvm = riemann_von_mangoldt_count(Lam)
        snl = math.sqrt(N) * math.log(N)
        rows.append((N, Lam, n_below, s, snl, rvm))
        print(f"{N:>6d} {Lam:>8.2f} {n_below:>6d} {s:>6d} {snl:>9.1f} {rvm:>9.1f}")
    # fit log|S| = a*log N + b  → exponent a
    Ns_arr = np.array([r[0] for r in rows], dtype=float)
    S_arr = np.array([max(r[3], 1) for r in rows], dtype=float)
    nb_arr = np.array([r[2] for r in rows], dtype=float)
    A = np.vstack([np.log(Ns_arr), np.ones_like(Ns_arr)]).T
    slope_S = float(np.linalg.lstsq(A, np.log(S_arr), rcond=None)[0][0])
    slope_nb = float(np.linalg.lstsq(A, np.log(nb_arr), rcond=None)[0][0])
    print(f"fitted exponent  d log|S|/d log N = {slope_S:.3f}   "
          f"(√N log N ⇒ ≈0.5 + small log correction)")
    print(f"fitted exponent  d log(#γ≤Λ)/d log N = {slope_nb:.3f}   "
          f"← |S| just tracks the zero count below the cutoff")

    print("\nregime Λ = 80 FIXED (what the shipped CLI actually does):")
    print(f"{'N':>6s} {'Λ':>6s} {'#γ≤Λ':>6s} {'|S|':>6s}")
    for N in Ns:
        g = gammas[:N]
        n_below = int(np.count_nonzero(g <= Lambda_fixed))
        s = subset_size(x0, g, Lambda_fixed, delta, n_t, "U_old")
        print(f"{N:>6d} {Lambda_fixed:>6.0f} {n_below:>6d} {s:>6d}")
    print("→ with Λ fixed, |S| SATURATES: extra zeros are above the cutoff (ω≈0).")


def experiment_C(gammas, pref, rng, dens_pre=None, Lambda=80.0,
                 x_lo=2_000_000, x_hi=9_000_000, half=1000, n=40000):
    print(f"\n=== C. SPEEDUP / 'resonance zone' prime capture "
          f"(Λ={Lambda}, windows={n}) ===")
    centers = rng.integers(x_lo + half, x_hi - half, size=n)
    cx = centers.astype(np.float64)
    dens = window_density(pref, centers, half)
    total_primes = dens.sum()  # proportional to captured primes (equal-width windows)

    def capture_curve(score, label):
        order = np.argsort(-score)  # scan highest-score windows first
        cum_primes = np.cumsum(dens[order]) / total_primes
        print(f"  {label}")
        for frac in (0.10, 0.20, 0.50, 0.80):
            k = int(frac * n)
            print(f"    scan top {frac*100:4.0f}% windows → capture "
                  f"{cum_primes[k-1]*100:5.2f}% of primes "
                  f"(uniform baseline = {frac*100:.0f}%)")
        return cum_primes

    # Best operator chosen by experiment A (D_dens). Also show U_old (manuscript).
    for kind in ("D_dens", "U_old"):
        sc = op_value(cx, gammas, Lambda, kind)
        capture_curve(sc, f"rank by {kind}:")
    # Oracle (rank by true density) and random, for context
    capture_curve(dens.copy(), "oracle (rank by TRUE density — upper bound):")
    capture_curve(rng.random(n), "random ordering (sanity baseline):")

    # The headline number: to LOSE <2% of primes, how much can you skip?
    print("\n  'Skip low-resonance zones, lose <2% of primes' — how much can we skip?")
    for kind in ("D_dens", "U_old"):
        sc = op_value(cx, gammas, Lambda, kind)
        order = np.argsort(-sc)
        cum = np.cumsum(dens[order]) / total_primes
        # smallest top-fraction that still captures >=98%
        idx = np.searchsorted(cum, 0.98) + 1
        skip = 100.0 * (1 - idx / n)
        print(f"    {kind:7s}: can skip {skip:5.2f}% of windows "
              f"(paper claims ~15–20%). Uniform random would skip ~2%.")


def main():
    csv_path = os.path.join(ROOT, "data", "zeros.csv")
    gammas = load_gammas(csv_path)
    print(f"loaded {gammas.size} zeros, γ_1={gammas[0]:.4f} … γ_N={gammas[-1]:.2f}")
    x_max = 9_200_000
    print(f"sieving primes up to {x_max:,} …")
    pref = sieve_prefix(x_max)
    print(f"π({x_max:,}) = {pref[x_max]:,}")
    rng = np.random.default_rng(12345)

    experiment_A(gammas, pref, rng)
    experiment_B(gammas)
    experiment_C(gammas, pref, rng)


if __name__ == "__main__":
    main()
