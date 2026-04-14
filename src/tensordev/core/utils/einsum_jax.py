import jax

jax.jit
def sparse_einsum_jax(Ai, Bj, operator):
    # Unpack operator
    _, Q = operator
    idx, idy, idz, data = Q

    def single_batch_op(Ai_s, Bj_s):
        vals = data * Ai_s[idy] * Bj_s[idz]
        return jax.ops.segment_sum(vals, idx, num_segments=Ai.shape[1]*Bj.shape[1])

    # Use vmap to handle batch dimension
    return jax.vmap(single_batch_op)(Ai, Bj)
