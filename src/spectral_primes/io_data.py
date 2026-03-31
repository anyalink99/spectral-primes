from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


def load_gammas_from_sqlite(db_path: str | Path) -> np.ndarray:
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT gamma FROM zeta_zeros ORDER BY n")
    g = np.array([row[0] for row in cur.fetchall()], dtype=np.float64)
    conn.close()
    if g.size == 0:
        raise ValueError("No rows in zeta_zeros")
    return g


def load_gammas_from_csv(csv_path: str | Path) -> np.ndarray:
    import csv

    gammas: list[float] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            gammas.append(float(row["gamma"]))
    return np.array(gammas, dtype=np.float64)
