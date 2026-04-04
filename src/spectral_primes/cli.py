from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from spectral_primes.experiment import (
    permutation_test_sparse,
    run_three_group_demo,
    run_three_group_demo_sparse,
    z_test_two_sample,
)
from spectral_primes.io_data import load_gammas_from_csv, load_gammas_from_sqlite
from spectral_primes.operator import U_batch
from spectral_primes.random_sieve import compare_prime_vs_cramer


def _load_gammas(args: argparse.Namespace) -> np.ndarray:
    if args.sqlite and Path(args.sqlite).is_file():
        return load_gammas_from_sqlite(args.sqlite)
    if args.csv and Path(args.csv).is_file():
        return load_gammas_from_csv(args.csv)
    print("Provide --csv or existing --sqlite (run scripts/init_db.py).", file=sys.stderr)
    sys.exit(1)


def cmd_demo(args: argparse.Namespace) -> None:
    g = _load_gammas(args)
    A, B, C, meta, da, db, _dc = run_three_group_demo(
        g,
        Lambda=args.Lambda,
        x_lo=args.x_lo,
        x_hi=args.x_hi,
        n_per_group=args.n,
        window_half=args.half_width,
        u_threshold_sigma=args.sigma_u,
        seed=args.seed,
    )
    print("Reference U on grid: mean = {:.6g}, std = {:.6g}".format(meta["mu_U"], meta["sigma_U"]))
    print("Lambda = {}, threshold = mean + {} * std".format(meta["Lambda"], meta["u_threshold_sigma"]))
    print("Pool size: {}".format(meta["pool_size"]))
    print()
    for gr in (A, B, C):
        print(
            f"{gr.name:20s}  n={gr.n}  mean density (per 1e5) = {gr.mean_density:.4f}  "
            f"sd = {gr.std_density:.4f}"
        )
    z = z_test_two_sample(da, db)
    print()
    print(f"Two-sample Z (A vs B, illustrative): {z:.3f}")


def cmd_demo_sparse(args: argparse.Namespace) -> None:
    g = _load_gammas(args)
    A, B, C, meta, da, db, _dc = run_three_group_demo_sparse(
        g,
        Lambda=args.Lambda,
        delta=args.delta,
        n_t=args.n_t,
        x_lo=args.x_lo,
        x_hi=args.x_hi,
        n_per_group=args.n,
        window_half=args.half_width,
        u_threshold_sigma=args.sigma_u,
        seed=args.seed,
        rank_anneal=args.rank_anneal,
    )
    print("Operator: U_sparse (Var_n subset, preprint sec. 2.2)")
    print(
        "Reference U_sparse on grid: mean = {:.6g}, std = {:.6g}".format(
            meta["mu_U"], meta["sigma_U"]
        )
    )
    print(
        "Lambda = {}, delta = {}, n_t = {}, threshold = mean + {} * std".format(
            meta["Lambda"],
            meta["delta"],
            meta["n_t"],
            meta["u_threshold_sigma"],
        )
    )
    print("rank_anneal = {} (0 = paper-pure Var; >0 breaks slot symmetry)".format(meta["rank_anneal"]))
    print("Pool size: {}".format(meta["pool_size"]))
    print()
    for gr in (A, B, C):
        print(
            f"{gr.name:22s}  n={gr.n}  mean density (per 1e5) = {gr.mean_density:.4f}  "
            f"sd = {gr.std_density:.4f}"
        )
    z = z_test_two_sample(da, db)
    print()
    print(f"Two-sample Z (A vs B, illustrative): {z:.3f}")


def cmd_permute(args: argparse.Namespace) -> None:
    g = _load_gammas(args)
    if args.rank_anneal == 0.0 and not args.allow_degenerate_null:
        print(
            "permute: rank_anneal=0 gives a degenerate permutation null (γ shuffle does not "
            "change U_sparse). Use --rank-anneal > 0 or --allow-degenerate-null for diagnostics.",
            file=sys.stderr,
        )
        sys.exit(1)
    out = permutation_test_sparse(
        g,
        Lambda=args.Lambda,
        delta=args.delta,
        n_t=args.n_t,
        x_lo=args.x_lo,
        x_hi=args.x_hi,
        n_per_group=args.n,
        window_half=args.half_width,
        n_perm=args.reps,
        u_threshold_sigma=args.sigma_u,
        pool_factor=args.pool_factor,
        seed=args.seed,
        rank_anneal=args.rank_anneal,
        allow_degenerate_null=args.allow_degenerate_null,
    )
    print("Permutation test (fixed centers + fixed B; shuffle gamma multiset)")
    print(f"Observed mean(A) - mean(B) density: {out['d_observed']:.6g}")
    print(f"Valid permutations: {out['n_perm_valid']} / {out['n_perm']}")
    print(f"Two-sided p-value (empirical): {out['p_two_sided']:.6g}")
    print(f"Pool size: {out['pool_size']}")
    print(f"rank_anneal: {out['rank_anneal']}")


def cmd_random_compare(args: argparse.Namespace) -> None:
    r = compare_prime_vs_cramer(
        x_lo=args.x_lo,
        x_hi=args.x_hi,
        half_width=args.half_width,
        n_windows=args.n,
        seed=args.seed,
        universes=args.universes,
    )
    print("Prime vs Cramer random sieve (independent 1/ln n marks on integers)")
    print(f"Range [x_lo, x_hi] = [{args.x_lo}, {args.x_hi}], half_width = {r.half_width}")
    print(f"Windows: {r.n_windows}, seed = {r.seed}, Cramer universes = {r.universes}")
    print()
    print(f"True primes:   mean density (per 1e5) = {r.real_mean:.4f}  sd = {r.real_std:.4f}")
    print(
        f"Cramer u=0:    mean over windows = {r.random_mean_universe0:.4f}  "
        f"sd across windows = {r.random_std_within_universe0:.4f}"
    )
    print(
        f"Cramer means:  mean of universe-averages = {r.mean_of_universe_means:.4f}  "
        f"sd between universes = {r.std_of_universe_means:.4f}"
    )
    print()
    print(
        "Empirical P( mean_random_universe < mean_real ) = "
        f"{r.empirical_p_real_exceeds_random:.4f}"
    )
    print("(one-sided: how often the random model's global mean is below true prime mean)")


def cmd_curve(args: argparse.Namespace) -> None:
    """Print U(x) along a grid (sanity check / plotting source)."""
    g = _load_gammas(args)
    xs = np.logspace(np.log10(args.x_lo), np.log10(args.x_hi), args.points)
    u = U_batch(xs, g, args.Lambda)
    for x, v in zip(xs, u):
        print(f"{x:.12g}\t{v:.12g}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Spectral primes — zeta-zero operator & demos")
    p.add_argument("--csv", default=str(Path("data/zeros.csv")))
    p.add_argument("--sqlite", default=str(Path("data/zeta_zeros.sqlite3")))
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Three-group prime-density demo (small x)")
    d.add_argument("--Lambda", type=float, default=80.0, help="spectral cutoff Λ")
    d.add_argument("--x-lo", type=int, default=2_000_000, dest="x_lo")
    d.add_argument("--x-hi", type=int, default=8_000_000, dest="x_hi")
    d.add_argument("--half-width", type=int, default=400, dest="half_width")
    d.add_argument("--n", type=int, default=40, help="per group")
    d.add_argument("--sigma-u", type=float, default=1.5, dest="sigma_u")
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=cmd_demo)

    ds = sub.add_parser("demo-sparse", help="Three-group demo with U_sparse + Var_n subset")
    ds.add_argument("--Lambda", type=float, default=80.0)
    ds.add_argument("--delta", type=float, default=300.0, help="half-interval delta for Var_n integral")
    ds.add_argument("--n-t", type=int, default=48, dest="n_t", help="grid points on [x-delta,x+delta]")
    ds.add_argument("--x-lo", type=int, default=2_000_000, dest="x_lo")
    ds.add_argument("--x-hi", type=int, default=8_000_000, dest="x_hi")
    ds.add_argument("--half-width", type=int, default=400, dest="half_width")
    ds.add_argument("--n", type=int, default=40, help="per group")
    ds.add_argument("--sigma-u", type=float, default=1.5, dest="sigma_u")
    ds.add_argument("--seed", type=int, default=42)
    ds.add_argument(
        "--rank-anneal",
        type=float,
        default=0.0,
        dest="rank_anneal",
        help="slot rank factor exp(a*(slot+1)/N); 0 matches pure gamma-only Var",
    )
    ds.set_defaults(func=cmd_demo_sparse)

    pm = sub.add_parser("permute", help="Permutation test on U_sparse density contrast (fixed B)")
    pm.add_argument("--Lambda", type=float, default=80.0)
    pm.add_argument("--delta", type=float, default=300.0)
    pm.add_argument("--n-t", type=int, default=48, dest="n_t")
    pm.add_argument("--x-lo", type=int, default=2_000_000, dest="x_lo")
    pm.add_argument("--x-hi", type=int, default=8_000_000, dest="x_hi")
    pm.add_argument("--half-width", type=int, default=400, dest="half_width")
    pm.add_argument("--n", type=int, default=30, help="per group (A and B)")
    pm.add_argument("--reps", type=int, default=200, help="number of gamma shuffles")
    pm.add_argument("--pool-factor", type=int, default=80, dest="pool_factor")
    pm.add_argument("--sigma-u", type=float, default=1.5, dest="sigma_u")
    pm.add_argument("--seed", type=int, default=42)
    pm.add_argument(
        "--rank-anneal",
        type=float,
        default=0.05,
        dest="rank_anneal",
        help="must be >0 for a non-degenerate γ-shuffle null (default 0.05); 0 needs --allow-degenerate-null",
    )
    pm.add_argument(
        "--allow-degenerate-null",
        action="store_true",
        dest="allow_degenerate_null",
        help="allow rank_anneal=0 (degenerate null; illustrative only)",
    )
    pm.set_defaults(func=cmd_permute, allow_degenerate_null=False)

    rc = sub.add_parser(
        "random-compare",
        help="Compare window prime density to Cramer random-sieve baseline",
    )
    rc.add_argument("--x-lo", type=int, default=2_000_000, dest="x_lo")
    rc.add_argument("--x-hi", type=int, default=8_000_000, dest="x_hi")
    rc.add_argument("--half-width", type=int, default=400, dest="half_width")
    rc.add_argument("--n", type=int, default=60, help="number of random window centres")
    rc.add_argument("--universes", type=int, default=100, help="independent Cramer fields")
    rc.add_argument("--seed", type=int, default=42)
    rc.set_defaults(func=cmd_random_compare)

    c = sub.add_parser("curve", help="Export U(x) on a log-spaced grid")
    c.add_argument("--Lambda", type=float, default=80.0)
    c.add_argument("--x-lo", type=float, default=1e6)
    c.add_argument("--x-hi", type=float, default=1e7)
    c.add_argument("--points", type=int, default=50)
    c.set_defaults(func=cmd_curve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
