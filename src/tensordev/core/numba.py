# ---- Numba backend ----
from __future__ import annotations

import numpy as np
import numba as nb

from .shuffle import ShuffleCore
from .universal import Universal, Array, DenseElem, DenseElemFirstOn, Callable


# ---------------------------------------------------------------------------
# Module-level numba kernel (lazy: compiled on first call, not at import time)
# ---------------------------------------------------------------------------

@nb.njit(fastmath=True, cache=True)
def _sparse_einsum_nb(Ai, Bj, segment_ids, rows, cols, data):
    """
    Numba-JIT sparse bilinear contraction (inner loop).

    Computes  res[b, segment_ids[k]] += Ai[b, rows[k]] * Bj[b, cols[k]] * data[k]
    for all k, accumulating into a zero-initialised output of shape
    (batch, d_i * d_j).
    """
    res = np.zeros((Ai.shape[0], Ai.shape[1] * Bj.shape[1]), dtype=Ai.dtype)
    for k in range(len(segment_ids)):
        res[:, segment_ids[k]] += Ai[:, rows[k]] * Bj[:, cols[k]] * data[k]
    return res


# ---------------------------------------------------------------------------
# Numba backend class
# ---------------------------------------------------------------------------

class Numba(Universal[np.ndarray]):
    """
    Numba-accelerated backend.

    Inherits all graded tensor operations from ``Universal`` (numpy fallback).
    Overrides ``sparse_einsum`` with a ``@nb.njit``-compiled loop for the
    shuffle product inner kernel.

    Other operations (tensor_product, tensor_exponential, …) remain as the
    numpy implementations from ``Universal``; JIT-compiling them with numba is
    left for future work (see TODOs below).

    TODO: compile main tensor ops (product, sum, fmexp, log, …) with numba.
    TODO: investigate whether dummy_jit can auto-generate one compiled function
          per (trunc, input-degree) combo to eliminate Python-level dispatch.
    """

    def __init__(self) -> None:
        super().__init__(np)

    def sparse_einsum(self, Ai, Bj, d: int, i: int, j: int):
        """
        Numba-JIT override of ``Universal.sparse_einsum``.

        Retrieves the operator from cache and delegates to the module-level
        ``_sparse_einsum_nb`` kernel, which is compiled by numba on first call.

        Parameters
        ----------
        Ai : np.ndarray
            Left factor with shape ``(batch, d**i)``.
        Bj : np.ndarray
            Right factor with shape ``(batch, d**j)``.
        d : int
            Base dimension of the tensor algebra.
        i : int
            Degree of ``Ai`` (must satisfy ``i >= j``).
        j : int
            Degree of ``Bj``.

        Returns
        -------
        np.ndarray
            Result with shape ``(batch, d**(i+j))``.
        """
        # Retrieve operator from cache
        operator = self._shuffle_cache[d].operators[(i, j)]
        _, Q = operator
        segment_ids, rows, cols, data = Q
        return _sparse_einsum_nb(
            np.asarray(Ai, dtype=np.float64),
            np.asarray(Bj, dtype=np.float64),
            segment_ids, rows, cols, data,
        )


class NumbaShuffleCore(ShuffleCore[np.ndarray]):
    """
    Numba-accelerated shuffle product engine.

    Operators are precomputed at construction; ``sparse_einsum`` dispatches
    to the module-level ``@nb.njit`` kernel.
    """

    def __init__(self, d: int, trunc: int) -> None:
        super().__init__(np, d, trunc)

    def sparse_einsum(self, Ai, Bj, i: int, j: int):
        _, Q = self.operators[(i, j)]
        segment_ids, rows, cols, data = Q
        return _sparse_einsum_nb(
            np.asarray(Ai, dtype=np.float64),
            np.asarray(Bj, dtype=np.float64),
            segment_ids, rows, cols, data,
        )
