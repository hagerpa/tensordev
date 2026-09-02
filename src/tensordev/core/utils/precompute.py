"""Small NumPy storage helpers shared by host-side plan builders."""

from __future__ import annotations

import numpy as np


def _readonly(array: np.ndarray) -> np.ndarray:
    """Mark an authoritative plan buffer immutable and return it."""
    array.setflags(write=False)
    return array


def _unsigned_index_dtype(maximum: int):
    """Smallest practical unsigned NumPy dtype storing ``maximum``."""
    for dtype in (np.uint8, np.uint16, np.uint32):
        if maximum <= np.iinfo(dtype).max:
            return dtype
    return np.uint64


__all__ = ["_readonly", "_unsigned_index_dtype"]
