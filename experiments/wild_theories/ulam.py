"""
Map a 1D array along the Ulam spiral: index t -> (x, y) integer lattice.

Useful for plotting spectral amplitudes or binary masks in polar/spiral layout.
"""
from __future__ import annotations

import numpy as np


def ulam_xy(n: int) -> tuple[int, int]:
    """
    Coordinates of integer n >= 1 on the standard Ulam spiral (n=1 at origin).
    Axes: right +x, up +y; first step goes to (1,0) for n=2.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 0, 0
    x, y = 0, 0
    step_len = 1
    direction = 0  # 0=E, 1=N, 2=W, 3=S
    m = 1
    while m < n:
        for _ in range(2):
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]
            for _ in range(step_len):
                m += 1
                x += dx
                y += dy
                if m == n:
                    return x, y
            direction = (direction + 1) % 4
        step_len += 1
    return x, y


def array_to_spiral_coords(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    values[k] is placed at ulam_xy(k+1). Returns (xs, ys, vals) aligned arrays.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    L = v.size
    if L == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    xs = np.empty(L, dtype=np.int64)
    ys = np.empty(L, dtype=np.int64)
    for k in range(L):
        x, y = ulam_xy(k + 1)
        xs[k] = x
        ys[k] = y
    return xs, ys, v
