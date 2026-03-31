#!/usr/bin/env python3
"""Compute imaginary parts γ_n of the first N nontrivial zeta zeros and write CSV."""
from __future__ import annotations

import argparse
import csv
import sys

try:
    import mpmath as mp
except ImportError:
    print("Install mpmath: pip install mpmath", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Build data/zeros.csv via mpmath.zetazero")
    p.add_argument("-n", "--count", type=int, default=2000, help="number of zeros")
    p.add_argument("-o", "--output", type=str, default="data/zeros.csv")
    p.add_argument("--dps", type=int, default=30, help="mpmath decimal precision")
    args = p.parse_args()
    mp.mp.dps = args.dps
    rows: list[tuple[int, float]] = []
    for k in range(1, args.count + 1):
        z = mp.zetazero(k)
        rows.append((k, float(z.imag)))
        if k % 200 == 0:
            print(k, file=sys.stderr)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "gamma"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
