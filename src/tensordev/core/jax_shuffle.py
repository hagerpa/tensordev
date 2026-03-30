import jax
import numpy as np

from functools import partial

from tensordev.core.utils.shuffle_precalculation import assemble_shuffle_algebra_homogeneous


def assemble_shuffle_algebra_jax(d: int, N: int):
    """
    Assembles dictionary containing all precomputed shuffles. Maximum allowed shuffle is of to tensors of level N.
    """
    # Create empty dictionary
    operators = {}
    
    for i in range(N + 1):
        for j in range(i + 1):
            meta, vals = assemble_shuffle_algebra_homogeneous(d, i, j)
            idx, idy, idz, data = vals
            # alter idx
            idi = np.arange(len(idx))
            segment_ids = np.repeat(idi, idx)
            operators[(i, j)] = meta, (segment_ids, idy, idz, data)
    
    # Bundle metadata with the operators
    shuffle_algebra_jax = {
        "metadata": (d, N),
        "operators": operators
    }
    return shuffle_algebra_jax


@partial(jax.jit, static_argnums=(2, 3, 4))
def sparse_einsum_jax(Ai, Bj, d, NAi, NBj, operator_jax):
    # Unpack operator
    _, Q = operator_jax
    idx, idy, idz, data = Q

    def single_batch_op(Ai_s, Bj_s):
        vals = data * Ai_s[idy] * Bj_s[idz]
        return jax.ops.segment_sum(vals, idx, num_segments=d ** (NAi + NBj))

    # Use vmap to handle batch dimension
    return jax.vmap(single_batch_op)(Ai, Bj)


@partial(jax.jit, static_argnums=(2,3))
def shuffle_product_jax(A: tuple, B: tuple, d: int, N: int, shuffle_algebra_jax: dict):
    """
    A, B: Tuples of jnp.arrays (the algebra elements)
    shuffle_algebra_jax: The dict containing 'operators'
    d, N: Static integers for truncation and dimension
    """
    operators = shuffle_algebra_jax['operators']
    out = []

    # total level n = i + j
    for n in range(N + N + 1):
        i_min = max(0, n - N)
        i_max = min(n, N)
        
        # Sum the terms for this specific tensor level
        current_grade_terms = []

        for i in range(i_min, i_max + 1):
            j = n - i
            
            # Symmetric lookup logic
            if i >= j:
                # Q contains (idx, idy, idz, data)
                op = operators[(i, j)]
                term = sparse_einsum_jax(A[i], B[j], d, i, j, op)
            else:
                # Swap A and B because only (i, j) where i >= j is cached
                op = operators[(j, i)]
                term = sparse_einsum_jax(B[j], A[i], d, j, i, op)
            
            current_grade_terms.append(term)
        
        # Combine all i,j pairs for this tensor level
        out.append(sum(current_grade_terms))

    return tuple(out)


