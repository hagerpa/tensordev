import numpy as np
import numba as nb

from typing import Tuple


@nb.njit(fastmath=True)
def index_add_numba(a: np.ndarray, index: np.ndarray) -> np.ndarray:
    """
    Accepts an 2d-array of shape (N_sample, n). For each sample the values are 
    added up based on the indices specified in target_idx.

    Example:

    a = np.array([[1,2,3,4,1,4]]) # shape (1,6)
    target_idx = np.array([1,2,2,1]) # shape (4,) with sum 6
    >>> np.array([1,5,5,4]) # =(1,2+3,4+1,4)
    """

    assert np.sum(index) == a.shape[-1]
    out = np.zeros((a.shape[0], len(index))).astype(a.dtype)
    start = 0
    end = index[0]
    for i in range(len(index)):
          out[:,i] = np.sum(a[:,start:end], axis=1)
          start += index[i]
          if i < len(index) - 1: end += index[i+1]
    return out



