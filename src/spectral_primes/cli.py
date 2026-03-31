from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from spectral_primes.experiment import run_three_group_demo, z_test_two_sample
from spectral_primes.io_data import load_gammas_from_csv, load_gammas_from_sqlite
from spectral_primes.operator import U_batch


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
