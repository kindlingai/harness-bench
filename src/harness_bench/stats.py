"""Intervals for the summary: Wilson for pass rates, Student t for means of per-run values."""

from __future__ import annotations

import math
import statistics

# two-sided 95% Student t critical values by degrees of freedom; 1.96 beyond the table
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def t95(df: int) -> float:
    if df <= 0:
        return math.inf
    keys = [k for k in _T95 if k <= df]
    return _T95[max(keys)] if df <= 120 else 1.96


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Pass rate and its 95% Wilson interval; well behaved at 0 and n, unlike the normal approximation."""
    if n == 0:
        return (math.nan, 0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def mean_ci(xs: list[float]) -> dict:
    """Mean, 95% t-interval half width, n and relative half width (half / mean)."""
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "ci95": None, "rel": None}
    m = statistics.fmean(xs)
    half = t95(n - 1) * statistics.stdev(xs) / math.sqrt(n) if n > 1 else math.inf
    rel = half / abs(m) if m else math.inf
    return {
        "n": n,
        "mean": round(m, 3),
        "ci95": round(half, 3) if math.isfinite(half) else None,
        "rel": round(rel, 3) if math.isfinite(rel) else None,
    }
