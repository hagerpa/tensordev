from tensordev.core.utils.einsum_numba import sparse_einsum

#import jax
#from tensordev.core.utils.einsum_jax import sparse_einsum_jax as sparse_einsum


#jax.jit
def shuffle_product(A, B, shuffle_algebra):
    """
    A, B: Tuples of arrays (the tensors)
    """

    # Extract length of A and B 
    NA = len(A) - 1 # Subtract 1 to get truncation level
    NB = len(B) - 1
    N = shuffle_algebra['metadata'][1]
    assert max(NA,NB) <= N, "Precomputed shuffles not sufficient."
    
    # Unpack and initiate output
    operators = shuffle_algebra['operators']
    out = []

    for n in range(NA + NB + 1):
        # Determine the range of indices i + j = n; i must be in [0, NA] and j must be in [0, NB]
        i_min = max(0, n - NB)
        i_max = min(n, NA)

        # Sum the terms for this specific tensor level in the end
        current_grade_terms = []

        for i in range(i_min, i_max + 1):
            j = n - i
            
            # Symmetric lookup logic
            if i >= j:
                op = operators[(i, j)]
                term = sparse_einsum(A[i], B[j], op) # To adapt to JAX only have to change the sparse einsum
            else:
                # Swap A and B because only (i, j) where i >= j is cached
                op = operators[(j, i)]
                term = sparse_einsum(B[j], A[i], op)
            
            current_grade_terms.append(term)
        
        # Combine all i,j pairs for this tensor level
        out.append(sum(current_grade_terms))

    return tuple(out)