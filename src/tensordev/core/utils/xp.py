"""The array namespace of the JAX cores."""

from __future__ import annotations

from jax import numpy as jnp


class _FullPrecisionJax:
    """``jax.numpy`` whose ``einsum`` and ``matmul`` contract at full float32 precision.

    On recent NVIDIA GPUs XLA otherwise contracts float32 operands in TF32, which keeps
    about three of their seven significant digits.
    """

    __name__ = "jax.numpy"

    def __getattr__(self, name):
        return getattr(jnp, name)

    @staticmethod
    def einsum(subscripts, *operands, **kwargs):
        return jnp.einsum(subscripts, *operands, precision="highest", **kwargs)

    @staticmethod
    def matmul(a, b):
        return jnp.matmul(a, b, precision="highest")


xp = _FullPrecisionJax()

__all__ = ["xp"]
