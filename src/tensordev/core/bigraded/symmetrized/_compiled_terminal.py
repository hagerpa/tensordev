"""Shared terminal-block arithmetic for compiled quotient plans."""

from __future__ import annotations

import numpy as np
from numba import njit
from numba.extending import register_jitable


@register_jitable
def _merge_terminal_coefficient_impl(
    left: np.ndarray,
    right: np.ndarray,
    merged: np.ndarray,
    binomial: np.ndarray,
) -> np.uint64:
    """Merge two terminal multiplicity blocks and return their coefficient."""
    coefficient = np.uint64(1)
    for letter in range(left.size):
        left_value = int(left[letter])
        right_value = int(right[letter])
        merged[letter] = left_value + right_value
        coefficient *= binomial[left_value + right_value, left_value]
    return coefficient


_merge_terminal_coefficient = njit(cache=True, nogil=True)(
    _merge_terminal_coefficient_impl
)


__all__ = ["_merge_terminal_coefficient"]
