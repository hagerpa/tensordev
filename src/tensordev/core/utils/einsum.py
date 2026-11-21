"""Backend-neutral ``einsum`` at full float32 precision under JAX."""


def full_precision_einsum(xp, subscripts, *operands):
    """Return ``xp.einsum(subscripts, *operands)``.

    Under JAX the contraction requests full float32 precision: on Ampere and
    newer GPUs XLA otherwise uses TF32, which keeps about three digits.
    """
    if getattr(xp, "__name__", "").startswith("jax"):
        return xp.einsum(subscripts, *operands, precision="highest")
    return xp.einsum(subscripts, *operands)
