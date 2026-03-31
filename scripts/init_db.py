#!/usr/bin/env python3
"""Create SQLite DB from data/zeros.csv (or another CSV with columns n,gamma)."""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=root / "data" / "zeros.csv")
    p.add_argument("--db", type=Path, default=root / "data" / "zeta_zeros.sqlite3")
    args = p.parse_args()
    schema = root / "sql" / "schema.sql"
    if not schema.is_file():
        print("Missing sql/schema.sql", file=sys.stderr)
        sys.exit(1)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.executescript(schema.read_text(encoding="utf-8"))
    with open(args.csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [(int(row["n"]), float(row["gamma"])) for row in r]
    conn.executemany("INSERT OR REPLACE INTO zeta_zeros (n, gamma) VALUES (?, ?)", rows)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM zeta_zeros").fetchone()[0]
    conn.close()
    print(f"Loaded {n} zeros into {args.db}", file=sys.stderr)


if __name__ == "__main__":
    main()
