"""
analysis_utils.py — Generic numerical analysis helpers for validation scripts.

fit_slope(x, y) -> tuple[float, float]
    OLS fit of log2(y) = slope * x + intercept.  Useful for measuring
    convergence rates from dyadic-refinement experiments.

compare_levelwise(sig_a, sig_b) -> list[dict]
    Per-level absolute and factorial-scaled error statistics between two
    signature tuples.  Works with any pair of JAX or NumPy signature tuples.
"""

from __future__ import annotations

import math

import numpy as np


def fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit log2(y) = slope * x + intercept via OLS.

    Parameters
    ----------
    x : 1-D array of predictor values (e.g. dyadic orders 0, 1, 2, ...).
    y : 1-D array of positive response values (e.g. per-level errors).
        Non-positive or non-finite values are silently excluded.

    Returns
    -------
    (slope, intercept) — both NaN when fewer than 2 finite points remain.
    """
    log_y = np.log2(np.where(y > 0, y, np.nan))
    valid = np.isfinite(log_y)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    coeffs = np.polyfit(x[valid], log_y[valid], 1)
    return float(coeffs[0]), float(coeffs[1])


def compare_levelwise(sig_a, sig_b) -> list[dict]:
    """Return per-level error statistics between two signature tuples.

    Parameters
    ----------
    sig_a, sig_b : tuples of arrays
        Signature tuples to compare (JAX or NumPy arrays).  Level k is the
        k-th element (level 0 is the constant term).

    Returns
    -------
    list of dicts, one per level, with keys:
        level        int    — level index (0-based)
        mean_abs     float  — mean absolute error
        max_abs      float  — max absolute error
        mean_scaled  float  — k! · mean_abs
        max_scaled   float  — k! · max_abs
    """
    rows = []
    for k, (Sa, Sb) in enumerate(zip(sig_a, sig_b)):
        err = np.abs(np.asarray(Sa) - np.asarray(Sb))
        fac = math.factorial(k)
        rows.append({
            "level":       k,
            "mean_abs":    float(err.mean()),
            "max_abs":     float(err.max()),
            "mean_scaled": float(fac * err.mean()),
            "max_scaled":  float(fac * err.max()),
        })
    return rows
