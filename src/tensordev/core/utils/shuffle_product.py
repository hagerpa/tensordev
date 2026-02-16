import numba as nb
import numpy as np


@nb.njit(
    nb.float64[:,:](
        nb.types.Tuple((nb.types.Tuple((nb.int64, nb.int64, nb.int64)), nb.types.Tuple((nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:])))),
        nb.float64[:,:],
        nb.float64[:,:]
    ),
    fastmath=True   
)
def sparse_einsum_numba(Q, Ai, Bj): # rename to sparse_einsum
    """
    Performs Q-product of tensor Ai and Bj. Roughly corresponds to `np.einsum('bi,bj,ijl->bl', Ai, Q, Bj)` where 
    we allow for sparse representation of Q.

    ATTENTION:
    - Requires Ai and Bi to have a batch dimension.
    - Meta data of shuffle tuple Q must align with tensor shapes of Ai and Bj.
    - Batch dimension of Ai and Bj must match.
    - Data type of Ai and Bj must match.
    """
    # Unpack Q
    d, n, m = Q[0]
    size, rows, cols, vals = Q[1]

    # Initalize output 
    batch_size = Ai.shape[0]
    res = np.empty((batch_size,d**(n+m)), dtype=Ai.dtype)

    # Run sparse einsum
    offset = 0
    for i in range(len(size)):
        for j in range(size[i]):
            res[:,i] += Ai[:,rows[offset+j]] * Bj[:,cols[offset+j]] * vals[offset+j]
        offset += size[i]
    return res


def shuffle_product(shuffle_algebra, A, B):
    """
    Graded (Cauchy-type) product ``C = A shuffle B`` in the free tensor algebra.
    """
    # Recast A and B as tuples/lists for indexing
    A = tuple(A)
    B = tuple(B)

    # Get truncation levels of A and B
    NA, NB = len(A) - 1, len(B) - 1
    batch_size = A[0].shape[0]

    # Unpack Shuffle Algebra
    d, N = shuffle_algebra['metadata']
    assert max(NA,NB) <= N, "Precomputed shuffles not sufficient."

    operators = shuffle_algebra['operators']
    out = []

    for n in range(NA + NB + 1):
        # Determine the range of indices i + j = n
        # i must be in [0, NA] and j must be in [0, NB]
        i_min = max(0, n - NB)
        i_max = min(n, NA)
        
        # Initialize the result for this level (batch, d**n)
        term = np.zeros((batch_size, d**n), dtype=A[0].dtype)

        for i in range(i_min, i_max + 1):
            j = n - i
            
            # Use commutativity to ensure we access the precomputed (I, j) where I >= j
            if i >= j:
                Q = operators[(i, j)]
                term += sparse_einsum_numba(Q, A[i], B[j]) # REPLACE BY GENERIC SPARSE EINSUM (NUMBA, NUMPY, TORCH, JAX, ...)
            else:
                # Swap A and B because only (j, i) is in the cache
                Q = operators[(j, i)]
                term += sparse_einsum_numba(Q, B[j], A[i])
        
        out.append(term)

    return tuple(out)