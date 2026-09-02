"""Small exact host-storage helpers shared by quotient plan builders."""

from __future__ import annotations

import numpy as np


def _coefficient_dtype(maximum: int):
    if maximum < 0:
        raise ValueError("coefficient maximum must be non-negative")
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if maximum <= np.iinfo(dtype).max:
            return dtype
    raise OverflowError(f"coefficient maximum {maximum} exceeds uint64")


def _placement_tuple(row) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, block)) for block in row)


__all__ = ["_coefficient_dtype", "_placement_tuple"]
