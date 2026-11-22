"""Shared helpers of the Volterra kernel gradient tests."""

import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from tensordev.volterra import FractionalKernel, GammaKernel


def parameter_kernel(name, params, A):
    """Kernel with ``beta = params[0]``.

    ``"fractional"`` is a :class:`FractionalKernel` with kernel matrices ``A``;
    ``"fractional_q2"`` has the second exponent ``params[1]`` and the kernel
    matrices ``A`` and ``A[:, ::-1]``; any other name gives a
    :class:`GammaKernel` with ``(scale, rate) = (params[1], params[2])``.
    """
    if name == "fractional":
        return FractionalKernel(beta=params[:1], A=A)
    if name == "fractional_q2":
        return FractionalKernel(beta=params[:2], A=jnp.concatenate([A, A[:, ::-1]]))
    return GammaKernel(beta=params[0], scale=params[1], rate=params[2], A=A)


def check_grads_each(f, args, **kwargs):
    """Reverse-mode ``check_grads`` in every coordinate of every argument separately."""
    args = tuple(jnp.asarray(a) for a in args)
    for i, a in enumerate(args):
        for j in np.ndindex(a.shape):
            check_grads(
                lambda x: f(*args[:i], a.at[j].set(x), *args[i + 1:]), (a[j],),
                order=1, modes=("rev",), **kwargs,
            )
