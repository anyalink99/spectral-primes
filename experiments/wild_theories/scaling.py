"""
Repeat FFT summaries on windows of different lengths / positions (scaling probe).

Heavy ranges use SymPy isprime — keep hi - lo moderate for interactive runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fourier import magnitude_spectrum
from .signals import segment_prime_indicator, segment_twin_lower_indicator


@dataclass
class WindowSpectrum:
    lo: int
    hi: int
    kind: str
    n: int
    peak_k: int
    peak_mag: float
    total_energy: float


def summarize_window(lo: int, hi: int, kind: str = "prime") -> WindowSpectrum:
    if kind == "prime":
        seq = segment_prime_indicator(lo, hi)
    elif kind == "twin":
        seq = segment_twin_lower_indicator(lo, hi)
    else:
        raise ValueError("kind must be 'prime' or 'twin'")
    mag = magnitude_spectrum(seq)
    n = seq.size
    if mag.size <= 1:
        pk, pv = 0, 0.0
    else:
        k = 1 + int(np.argmax(mag[1:]))
        pk, pv = k, float(mag[k])
    te = float(np.sum(mag**2))
    return WindowSpectrum(lo, hi, kind, n, pk, pv, te)


def scaling_table(
    windows: list[tuple[int, int]],
    kind: str = "prime",
) -> list[WindowSpectrum]:
    return [summarize_window(lo, hi, kind) for lo, hi in windows]
