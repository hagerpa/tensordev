import numpy as np
import numba as nb

@nb.njit(
    nb.float64[:,:](
        nb.float64[:,:],
        nb.float64[:,:],
        nb.types.Tuple((nb.types.Tuple((nb.int64, nb.int64, nb.int64)), nb.types.Tuple((nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:]))))
    ),
    fastmath=True   
)
def sparse_einsum(Ai, Bj, operator):
    """
    Performs Q-product of tensor Ai and Bj. Roughly corresponds to `np.einsum('bi,bj,ijl->bl', Ai, Q, Bj)` where 
    we allow for sparse representation of Q.

    ATTENTION:
    - Requires Ai and Bi to have a batch dimension.
    - Meta data of shuffle tuple Q must align with tensor shapes of Ai and Bj.
    - Batch dimension of Ai and Bj must match.
    - Data type of Ai and Bj must match.
    """
    # Unpack operator
    _, Q = operator
    segment_ids, rows, cols, data = Q

    # Init output
    res = np.zeros((Ai.shape[0], Ai.shape[1]*Bj.shape[1]), dtype=Ai.dtype)

    # Run sparse einsum
    for i in range(len(segment_ids)):
        res[:,segment_ids[i]] += Ai[:,rows[i]] * Bj[:,cols[i]] * data[i]
    return res