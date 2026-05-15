r"""Reference FSSK kernel approximation via inner product of Volterra signatures.

The FSSK signature kernel k(X, Y) admits a series expansion

    k(X, Y) = sum_{n=0}^{infty} <VSig^n(X), VSig^n(Y)>

where VSig^n(X) is the degree-n Volterra signature of X computed via the exact
SSS recursion (tensordev.sss.state_update.fssk_vsig) and <.,.> is the Euclidean
inner product on the flattened tensor level.

Truncating at level N gives an approximation k_N(X, Y) that converges to the
true kernel as N -> inf. The truncation error decays like the tail of the
signature series; for well-behaved paths it decreases super-geometrically in N.

This module provides:
  vsig_kernel      -- batchwise or pairwise kernel values for batched paths
  vsig_kernel_pair -- single pair k(X, Y)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tensordev.sss.kernel import FSSK
from tensordev.sss.state_update import fssk_vsig

Array = jax.Array


def vsig_kernel_pair(
    X: Array,
    Y: Array,
    *,
    kernel: FSSK,
    dt_x: float | Array,
    dt_y: float | Array,
    trunc: int,
    dtype=None,
) -> Array:
    """Kernel between a single pair of paths via truncated VSig inner product.

    Parameters
    ----------
    X : (nodes_x, d)
    Y : (nodes_y, d)
    kernel : FSSK
    dt_x, dt_y : step sizes for X and Y respectively
    trunc : signature truncation level
    dtype : optional output dtype

    Returns
    -------
    scalar Array
    """
    X = jnp.asarray(X, dtype=dtype)
    Y = jnp.asarray(Y, dtype=dtype)
    vx = fssk_vsig(X, kernel=kernel, dt=dt_x, trunc=trunc, tau_dt=0.0)
    vy = fssk_vsig(Y, kernel=kernel, dt=dt_y, trunc=trunc, tau_dt=0.0)
    return sum(jnp.dot(a.ravel(), b.ravel()) for a, b in zip(vx, vy))


def vsig_kernel(
    X: Array,
    Y: Array,
    *,
    kernel: FSSK,
    dt_x: float | Array,
    dt_y: float | Array,
    trunc: int,
    pairwise: bool = False,
    dtype=None,
) -> Array:
    """FSSK kernel approximation via truncated Volterra signature inner products.

    Parameters
    ----------
    X : (batch_x, nodes_x, d)
    Y : (batch_y, nodes_y, d)
    kernel : FSSK
    dt_x, dt_y : step sizes for X and Y
    trunc : signature truncation level N
    pairwise : if True return (batch_x, batch_y) Gram matrix;
               if False return (batch,) batchwise values (requires batch_x == batch_y)
    dtype : optional output dtype

    Returns
    -------
    Array of shape (batch_x, batch_y) if pairwise, else (batch,)
    """
    X = jnp.asarray(X, dtype=dtype)
    Y = jnp.asarray(Y, dtype=dtype)

    # Compute feature vectors for all paths in each batch.
    # fssk_vsig handles batched inputs: level n has shape (batch, m^n).
    vx = fssk_vsig(X, kernel=kernel, dt=dt_x, trunc=trunc, tau_dt=0.0)
    vy = fssk_vsig(Y, kernel=kernel, dt=dt_y, trunc=trunc, tau_dt=0.0)

    if pairwise:
        # k[i, j] = sum_n  <vx_n[i], vy_n[j]>
        return sum(
            jnp.einsum("bi,ci->bc", lvl_x.reshape(lvl_x.shape[0], -1),
                                    lvl_y.reshape(lvl_y.shape[0], -1))
            for lvl_x, lvl_y in zip(vx, vy)
        )
    else:
        # k[i] = sum_n  <vx_n[i], vy_n[i]>
        return sum(
            jnp.einsum("bi,bi->b", lvl_x.reshape(lvl_x.shape[0], -1),
                                   lvl_y.reshape(lvl_y.shape[0], -1))
            for lvl_x, lvl_y in zip(vx, vy)
        )


__all__ = ["vsig_kernel_pair", "vsig_kernel"]